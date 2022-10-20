python train.py \
--task train \
--output_dir '../passage_exp/ANCE_duallearning_IPW(negative)_2' \
--msmarco_dir '../passage_exp/marco_passage_data/msmarco-passage' \
--collection_memmap_dir '../passage_exp/marco_passage_data/collection_memmap' \
--tokenize_dir '../passage_exp/marco_passage_data/tokenize' \
--max_query_length 32 \
--max_doc_length 256 \
--debias_ratio 10.0 \
--per_gpu_eval_batch_size 512 \
--per_gpu_train_batch_size 64 \
--gradient_accumulation_steps 1 \
--seed 42 \
--evaluate_during_training \
--training_eval_steps 10000 \
--save_steps 5000 \
--logging_steps 100 \
--data_num_workers 10 \
--learning_rate 3e-6 \
--warmup_steps 10000 \
--num_train_epochs 1




# python train.py \
# --task dev \
# --output_dir '../passage_exp/ANCE_duallearning_IPW' \
# --msmarco_dir '../passage_exp/marco_passage_data/msmarco-passage' \
# --collection_memmap_dir '../passage_exp/marco_passage_data/collection_memmap' \
# --tokenize_dir '../passage_exp/marco_passage_data/tokenize' \
# --max_query_length 32 \
# --max_doc_length 256 \
# --eval_ckpt 10000 \
# --per_gpu_eval_batch_size 512 \
# --seed 42 \
# --data_num_workers 10 \
# --learning_rate 3e-6 \
# --warmup_steps 10000 \
# --num_train_epochs 1
