import os
import re
import torch
import random
import time
import logging
import argparse
import numpy as np
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler
import torch.nn.functional as F
from transformers import (BertConfig, AdamW, get_linear_schedule_with_warmup)

from modeling import Observer, Reranker
from dataset import MSMARCODataset, get_collate_function
from utils import generate_rank, eval_results

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s-%(levelname)s-%(name)s- %(message)s',
                        datefmt = '%d %H:%M:%S',
                        level = logging.INFO)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def save_model(model, output_dir, save_name, args):
    save_dir = os.path.join(output_dir, save_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_to_save = model.module if hasattr(model, 'module') else model  
    model_to_save.save_pretrained(save_dir)
    torch.save(args, os.path.join(save_dir, 'training_args.bin'))


def train(args, observer, model):
    """ Train the model """
    tb_writer = SummaryWriter(args.log_dir)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_dataset = MSMARCODataset("train", 
            args.collection_memmap_dir, args.tokenize_dir,
            args.max_query_length, args.max_doc_length)

    # NOTE: Must Sequential! Pos, Neg, Pos, Neg, ...
    train_sampler = SequentialSampler(train_dataset) 
    collate_fn = get_collate_function()
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
        batch_size=args.train_batch_size, num_workers=args.data_num_workers, 
        collate_fn=collate_fn)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    model_optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    model_scheduler = get_linear_schedule_with_warmup(model_optimizer, num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total)    # ##############################

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in observer.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in observer.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    observer_optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    observer_scheduler = get_linear_schedule_with_warmup(observer_optimizer, num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total)    # ##############################

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=args.device)
        observer = torch.nn.DataParallel(observer, device_ids=args.device)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    model_tr_loss, model_tr_loss_weight, model_logging_loss, model_logging_loss_weight = 0.0, 0.0, 0.0, 0.0
    observer_tr_loss, observer_tr_loss_weight, observer_logging_loss, observer_logging_loss_weight = 0.0, 0.0, 0.0, 0.0
    model.zero_grad()
    observer.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for epoch_idx, _ in enumerate(train_iterator):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, (batch, _, _) in enumerate(epoch_iterator):
            batch = {k: v.to(args.device[0]) for k, v in batch.items()}

            # ####### observation probability
            observer.eval()
            with torch.no_grad():
                observer_score = observer(**batch, weight=None, is_training=False)  # [B, 1]
                observer_score = torch.cat((observer_score[0::2], observer_score[1::2]), dim=1)  # [bs//2, 2]
                observer_score = 1.0 / (1.0 - F.sigmoid(observer_score[:, 1] / args.debias_ratio))  # [bs//2]

            # ####### relevance probability
            model.eval()
            with torch.no_grad():
                relevance_score = model(**batch, weight=None, is_training=False)  # [B, 1]
                relevance_score = torch.cat((relevance_score[0::2], relevance_score[1::2]), dim=1)  # [bs//2, 2]
                relevance_score = 1.0 / (1.0 - F.sigmoid(relevance_score[:, 1] / args.debias_ratio))  # [bs//2]

            # ####### ranker
            model.train()
            model_outputs = model(**batch, weight=observer_score, is_training=True)  # #############################################
            model_loss, model_loss_weight, _ = model_outputs    # ###########
            if args.n_gpu > 1:
                model_loss = model_loss.mean()
                model_loss_weight = model_loss_weight.mean()
            if args.gradient_accumulation_steps > 1:
                model_loss = model_loss / args.gradient_accumulation_steps
                model_loss_weight = model_loss_weight / args.gradient_accumulation_steps
            model_loss_weight.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            # ####### observer
            observer.train()
            observer_outputs = observer(**batch, weight=relevance_score, is_training=True)  # #############################################
            observer_loss, observer_loss_weight, _ = observer_outputs   # ###########
            if args.n_gpu > 1:
                observer_loss = observer_loss.mean()
                observer_loss_weight = observer_loss_weight.mean()
            if args.gradient_accumulation_steps > 1:
                observer_loss = observer_loss / args.gradient_accumulation_steps
                observer_loss_weight = observer_loss_weight / args.gradient_accumulation_steps
            observer_loss_weight.backward()
            torch.nn.utils.clip_grad_norm_(observer.parameters(), args.max_grad_norm)

           # #################
            model_tr_loss += model_loss.item()
            model_tr_loss_weight += model_loss_weight.item()
            observer_tr_loss += observer_loss.item()
            observer_tr_loss_weight += observer_loss_weight.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                model_optimizer.step()
                model_scheduler.step()
                model.zero_grad()
                
                observer_optimizer.step()
                observer_scheduler.step()
                observer.zero_grad()

                global_step += 1
                # logging
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_writer.add_scalar('train/model_lr', model_scheduler.get_lr()[0], global_step)
                    cur_loss = (model_tr_loss - model_logging_loss) / args.logging_steps
                    tb_writer.add_scalar('train/model_loss', cur_loss, global_step)
                    model_logging_loss = model_tr_loss
                    cur_loss = (model_tr_loss_weight - model_logging_loss_weight) / args.logging_steps
                    tb_writer.add_scalar('train/model_loss_weight', cur_loss, global_step)
                    model_logging_loss_weight = model_tr_loss_weight
                    
                    tb_writer.add_scalar('train/observer_lr', observer_scheduler.get_lr()[0], global_step)
                    cur_loss = (observer_tr_loss - observer_logging_loss) / args.logging_steps
                    tb_writer.add_scalar('train/observer_loss', cur_loss, global_step)
                    observer_logging_loss = observer_tr_loss
                    cur_loss = (observer_tr_loss_weight - observer_logging_loss_weight) / args.logging_steps
                    tb_writer.add_scalar('train/observer_loss_weight', cur_loss, global_step)
                    observer_logging_loss_weight = observer_tr_loss_weight
                # model save
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    save_model(model, args.model_save_dir, 'ckpt-{}'.format(global_step), args)
                    save_model(observer, args.observer_save_dir, 'ckpt-{}'.format(global_step), args)
                # evaluate
                if args.evaluate_during_training and (global_step % args.training_eval_steps == 0):
                    mrr = evaluate(args, model, mode="dev", prefix="step_{}".format(global_step))
                    tb_writer.add_scalar('dev/MRR@10', mrr, global_step)


def evaluate(args, model, mode, prefix):
    eval_output_dir = args.eval_save_dir
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)
  
    eval_dataset = MSMARCODataset(mode, 
            args.collection_memmap_dir, args.tokenize_dir,
            args.max_query_length, args.max_doc_length)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    collate_fn = get_collate_function()
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size,
        num_workers=args.data_num_workers, collate_fn=collate_fn)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    output_file_path = f"{eval_output_dir}/{prefix}.{mode}.score.tsv"
    with open(output_file_path, 'w') as outputfile:
        for batch, qids, docids in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            with torch.no_grad():
                batch = {k: v.to(args.device[0]) for k, v in batch.items()}
                outputs = model(**batch, weight=None, is_training=False)
                scores = outputs.squeeze(dim=1).detach().cpu().numpy()  # ################## (bs,)
                assert len(qids) == len(docids) == len(scores)
                for qid, docid, score in zip(qids, docids, scores):
                    outputfile.write(f"{qid}\t{docid}\t{score}\n")

    rank_output = f"{eval_output_dir}/{prefix}.{mode}.rank.tsv"
    generate_rank(output_file_path, rank_output)

    if mode == "dev":
        mrr = eval_results(rank_output)
        return mrr



def run_parse_args():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--task", choices=["train", "dev", "eval", "test"], required=True)
    parser.add_argument("--output_dir", type=str, default=f"../passage_exp/ANCE_IPW")  # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    
    parser.add_argument("--msmarco_dir", type=str, default=f"../passage_exp/marco_passage_data/msmarco-passage")
    parser.add_argument("--collection_memmap_dir", type=str, default="../passage_exp/marco_passage_data/collection_memmap")
    parser.add_argument("--tokenize_dir", type=str, default="../passage_exp/marco_passage_data/tokenize")
    parser.add_argument("--max_query_length", type=int, default=32)
    parser.add_argument("--max_doc_length", type=int, default=256)  
    parser.add_argument("--debias_ratio", type=float, default=2.0) 

    ## Other parameters
    parser.add_argument("--eval_ckpt", type=int, default=None)
    parser.add_argument("--per_gpu_eval_batch_size", default=64, type=int)
    parser.add_argument("--per_gpu_train_batch_size", default=64, type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument("--evaluate_during_training", action="store_true")
    parser.add_argument("--training_eval_steps", type=int, default=20000)

    parser.add_argument("--save_steps", type=int, default=20000)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--data_num_workers", default=5, type=int)

    parser.add_argument("--learning_rate", default=3e-6, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--warmup_steps", default=10000, type=int)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_train_epochs", default=1, type=int)

    args = parser.parse_args()

    time_stamp = time.strftime("%b-%d_%H:%M:%S", time.localtime())
    args.log_dir = f"{args.output_dir}/log/{time_stamp}"
    args.model_save_dir = f"{args.output_dir}/models"
    args.observer_save_dir = f"{args.output_dir}/observer_models"
    args.eval_save_dir = f"{args.output_dir}/rerank100_eval_results"
    return args


def main():
    args = run_parse_args()
    logger.info(args)
    
    # Setup CUDA, GPU 
    device = [1]
    args.n_gpu = len(device)

    args.device = device

    # Setup logging
    logger.warning("Device: %s, n_gpu: %s", device, args.n_gpu)

    # Set seed
    set_seed(args)

    if args.task == "train":
        load_model_path = f"bert-base-uncased"
        load_observer_path = f"bert-base-uncased"
    else:
        assert args.eval_ckpt is not None
        load_model_path = f"{args.model_save_dir}/ckpt-{args.eval_ckpt}"
        load_observer_path = f"{args.observer_save_dir}/ckpt-{args.eval_ckpt}"

    model_config = BertConfig.from_pretrained(load_model_path)
    model = Reranker.from_pretrained(load_model_path, config=model_config)
    model.to(args.device[0])

    if args.task == "train":
        observer_config = BertConfig.from_pretrained(load_observer_path)
        observer = Observer.from_pretrained(load_observer_path, config=observer_config)
        observer.to(args.device[0])

    logger.info("Training/evaluation parameters %s", args)
    # Evaluation
    if args.task == "train":
        train(args, observer, model)
    else:
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model, device_ids=args.device)
        result = evaluate(args, model, args.task, prefix=f"ckpt-{args.eval_ckpt}")
        print(result)
    


if __name__ == "__main__":
    main()
