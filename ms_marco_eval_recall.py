import collections
import json
import numpy as np
import random
from tqdm import tqdm
import math
import os
import sys


def Recall(qrels, pred, threshold):
    if len(qrels) == 0:
        return 0.0

    score = 0.0
    for item in qrels:
        if item in pred[:threshold]:
            score += 1.0
    return score / len(qrels)


def cal_metrics(qrels_file, pred_file, topk):
    test_qid2qrel = collections.defaultdict(set)
    with open(qrels_file) as f:
        for _, line in enumerate(f):
            qid1, _, qid2, label = line.strip().split()    ##########################
            if int(label) > 0:
                qid1, qid2 = int(qid1), int(qid2)
                test_qid2qrel[qid1].add(qid2)
    print('test avg pos sample number', np.mean([len(qrel) for qrel in test_qid2qrel.values()]))

    test_qid2pred = collections.defaultdict(list)
    with open(pred_file) as f:
        lines = f.readlines()
        for line in tqdm(lines, total=len(lines)):
            qid1, qid2, rank = line.strip().split()   ##########################
            qid1, qid2, rank = int(qid1), int(qid2), int(rank)
            test_qid2pred[qid1].append((qid2, rank))
    metric = []
    for qid in tqdm(test_qid2pred.keys(), total=len(test_qid2pred)):
        qrels = test_qid2qrel[qid]

        pred_out = test_qid2pred[qid]
        pred_out.sort(key=lambda x:x[1])
        pred = [qid2 for (qid2, _) in pred_out]

        metric.append(Recall(qrels, pred, topk))

    print('recall@{}: '.format(sys.argv[3]), np.mean(metric))
    print('QueriesRanked: ', len(test_qid2pred.keys()))


def main():
    """Command line:
    python test_trec_eval.py <path_to_reference_file> <path_to_candidate_file>
    """
    print("Eval Started")
    if len(sys.argv) == 4:
        cal_metrics(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    else:
        print('Usage: test_trec_eval.py <reference ranking> <candidate ranking>')
        exit()
    

if __name__ == '__main__':
    main()