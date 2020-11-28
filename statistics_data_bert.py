import argparse
import random
import jsonlines
import os
from kobert.utils import get_tokenizer
from gluonnlp.data import SentencepieceTokenizer


def main(args):
    for arg in vars(args):
        print(arg, getattr(args, arg))
    tokenizer = SentencepieceTokenizer(get_tokenizer())

    lines_len = 0
    src_docs = []
    with jsonlines.open(args.train_file) as f:

        for line in f.iter():
            lines_len += 1
            sentences = []
            for sentence in line['article_original']:
                sentences.append(sentence)
            src_docs.append(" ".join(sentences).replace('\n', '') + "\n")

    lens = []
    tr_max_src = 0
    for i, src_doc in enumerate(src_docs):
        if i % 100 == 0:
            print(i, len(src_docs))
        tokens = tokenizer(src_doc)
        cur_len = len(tokens)
        lens.append(cur_len)
        if tr_max_src < cur_len:
            tr_max_src = cur_len

    src_docs = []
    with jsonlines.open(args.test_file) as f:

        for line in f.iter():
            lines_len += 1
            sentences = []
            for sentence in line['article_original']:
                sentences.append(sentence)
            src_docs.append(" ".join(sentences).replace('\n', '') + "\n")

    max_src = 0
    test_lens = []
    for i, src_doc in enumerate(src_docs):
        if i % 100 == 0:
            print(i, len(src_docs))
        tokens = tokenizer(src_doc)
        cur_len = len(tokens)
        test_lens.append(cur_len)
        if max_src < cur_len:
            max_src = cur_len
    print("max source length train", tr_max_src)
    print("max source length test", max_src)
    print(sum(lens) / len(lens))
    print(sum(test_lens) / len(test_lens))
    import numpy as np
    print(np.median(np.array(lens)))
    print(np.median(np.array(test_lens)))
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_file', type=str,
                        default='/media/irelin/data_disk/dataset/dacon_summury/extractive/train.jsonl')
    parser.add_argument('--test_file', type=str,
                        default='/media/irelin/data_disk/dataset/dacon_summury/extractive/extractive_test_v2.jsonl')

    main(parser.parse_args())
