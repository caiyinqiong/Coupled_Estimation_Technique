import os
import math
import json
import torch
import logging
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer
from torch.utils.data import Dataset
logger = logging.getLogger(__name__)


class CollectionDataset:
    def __init__(self, collection_memmap_dir):
        self.pids = np.memmap(f"{collection_memmap_dir}/pids.memmap", dtype='int32',)
        self.lengths = np.memmap(f"{collection_memmap_dir}/lengths.memmap", dtype='int32',)
        self.collection_size = len(self.pids)
        self.token_ids = np.memmap(f"{collection_memmap_dir}/token_ids.memmap", 
                dtype='int32', shape=(self.collection_size, 512))
    
    def __len__(self):
        return self.collection_size

    def __getitem__(self, item):
        assert self.pids[item] == item
        return self.token_ids[item, :self.lengths[item]].tolist()


def load_queries(tokenize_dir, mode):
    queries = dict()
    for line in tqdm(open(f"{tokenize_dir}/queries.{mode}.json"), desc="queries"):
        data = json.loads(line)
        queries[int(data['id'])] = data['ids']
    return queries


def load_querydoc_pairs(mode):
    qids, pids, labels = [], [], []
    if mode == "train":
        for line in tqdm(open("../passage_exp/marco_passage_data/ANCE/qidpidtriples.train.small.ance.tsv"), desc="load train triples"):
            qid, pos_pid, neg_pid, _, _ = line.split("\t")
            qid, pos_pid, neg_pid = int(qid), int(pos_pid), int(neg_pid)
            qids.append(qid)
            pids.append(pos_pid)
            labels.append(1)
            qids.append(qid)
            pids.append(neg_pid)
            labels.append(0)
    else:
        for line in open("../passage_exp/dl_passage_data/ANCE/dltest_top1000_rank.tsv"):
            qid, pid, rank = line.split("\t")
            qids.append(int(qid))
            pids.append(int(pid))
        labels = None
    return qids, pids, labels


class MSMARCODataset(Dataset):
    def __init__(self, mode, 
            collection_memmap_dir, tokenize_dir,
            max_query_length=20, max_doc_length=256):

        self.collection = CollectionDataset(collection_memmap_dir)
        self.queries = load_queries(tokenize_dir, mode)
        self.qids, self.pids, self.labels = load_querydoc_pairs(mode)
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length

    def __len__(self):  
        return len(self.qids)

    def __getitem__(self, item):
        qid, pid = self.qids[item], self.pids[item]
        query_input_ids, doc_input_ids = self.queries[qid], self.collection[pid]
        query_input_ids = query_input_ids[:self.max_query_length]
        query_input_ids = [self.cls_id] + query_input_ids + [self.sep_id]
        doc_input_ids = doc_input_ids[:self.max_doc_length]
        doc_input_ids = doc_input_ids + [self.sep_id]

        ret_val = {
            "query_input_ids": query_input_ids,
            "doc_input_ids": doc_input_ids,
            "qid": qid,
            "docid": pid
        }
        return ret_val


def pack_tensor_2D(lstlst, default, dtype, length=None):
    batch_size = len(lstlst)
    length = length if length is not None else max(len(l) for l in lstlst)
    tensor = default * torch.ones((batch_size, length), dtype=dtype)
    for i, l in enumerate(lstlst):
        tensor[i, :len(l)] = torch.tensor(l, dtype=dtype)
    return tensor


def get_collate_function():
    def collate_function(batch):
        input_ids_lst = [x["query_input_ids"] + x["doc_input_ids"] for x in batch]
        token_type_ids_lst = [[0]*len(x["query_input_ids"]) + [1]*len(x["doc_input_ids"]) 
            for x in batch]
        valid_mask_lst = [[1]*len(input_ids) for input_ids in input_ids_lst]
        position_ids_lst = [list(range(len(input_ids))) for input_ids in input_ids_lst]
        data = {
            "input_ids": pack_tensor_2D(input_ids_lst, default=0, dtype=torch.int64),
            "token_type_ids": pack_tensor_2D(token_type_ids_lst, default=0, dtype=torch.int64),
            "valid_mask": pack_tensor_2D(valid_mask_lst, default=0, dtype=torch.int64),
            "position_ids": pack_tensor_2D(position_ids_lst, default=0, dtype=torch.int64),
        }
        qid_lst = [x['qid'] for x in batch]
        docid_lst = [x['docid'] for x in batch]

        return data, qid_lst, docid_lst
    return collate_function
