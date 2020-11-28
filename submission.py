import argparse
import random
from kobert.pytorch_kobert import get_pytorch_kobert_model
import jsonlines
from torch.utils import data
from gluonnlp.data import SentencepieceTokenizer
from kobert.utils import get_tokenizer
import time
import numpy as np
import torch

from torch.nn.utils.rnn import pad_sequence
import os
from model import ExtractiveModel
import csv


class SentenceDataset(data.Dataset):
    """__init__ and __len__ functions are the same as in TorchvisionDataset"""

    def __init__(self, test_file, vocab, max_token_cnt=300):
        self.tokenizer = SentencepieceTokenizer(get_tokenizer())
        self.vocab = vocab

        self.max_token_cnt = max_token_cnt

        self.media_map = {'경기일보': 0, '광양신문': 1, '광주매일신문': 2, '광주일보': 3, '국제신문': 4, '기호일보': 5, '남도일보': 6, '당진시대': 7,
                          '대구신문': 8, '대구일보': 9, '대전일보': 10}
        print("medias", self.media_map)

        samples = []
        with jsonlines.open(test_file) as f:
            for line in f.iter():
                media = line['media']
                id = line['id']
                sentences = []
                for i, sentence in enumerate(line['article_original']):
                    sentences.append(sentence.replace('\n', '').strip())
                samples.append([sentences, media, id])
        self.samples = samples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        sentences, media, id = self.samples[index]
        media = self.media_map[media]

        token_ids_batch = []
        pos_idx_batch = list(range(len(sentences)))
        media_batch = [media] * len(sentences)
        for sen in sentences:
            tokens = self.tokenizer(sen)
            token_ids = self.vocab.to_indices(tokens)
            if len(token_ids) > self.max_token_cnt:
                token_ids = token_ids[:self.max_token_cnt]
            token_ids_batch.append(torch.tensor(token_ids, dtype=torch.long))

        token_ids_batch = pad_sequence(token_ids_batch, batch_first=True, padding_value=0)

        return token_ids_batch, torch.tensor(pos_idx_batch, dtype=torch.long), \
               torch.tensor(media_batch, dtype=torch.long)  # , np.array(sentences)

    def __len__(self):
        return len(self.samples)


def submit(args):
    bert_model, vocab = get_pytorch_kobert_model()
    test_dataset = SentenceDataset(args.test_file, vocab, max_token_cnt=args.max_token_cnt)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=args.num_workers, shuffle=False)

    model = ExtractiveModel(bert_model, 100, 11, 768,
                            use_bert_sum_words=args.use_bert_sum_words,
                            use_pos=args.use_pos,
                            use_media=args.use_media, num_classes=4, simple_model=args.simple_model,
                            dim_feedforward=args.dim_feedforward,
                            dropout=args.dropout)

    if args.checkpoint_path is not None and os.path.isfile(args.checkpoint_path):
        state_dict = torch.load(args.checkpoint_path)[0]
        model.load_state_dict(state_dict)

    model.eval()  # Set model to evaluate mode
    device = 'cuda'
    model.to(device)

    ids = []
    summaries = []
    for step, (token_ids_batch, pos_idx_batch, media_batch) in enumerate(test_loader):
        if step % 10 == 0:
            print(step, len(test_loader))
        token_ids_batch = token_ids_batch[0].to(device)
        pos_idx_batch = pos_idx_batch[0].to(device)
        media_batch = media_batch[0].to(device)
        sentences, _, id = test_dataset.samples[step]
        ids.append(id)
        sentences = np.array(sentences)
        with torch.set_grad_enabled(False):
            outputs = model(token_ids_batch, pos_idx_batch, media_batch)
            indices = torch.argsort(outputs[:, 0], dim=0)
            sentences = sentences[indices[:3].cpu().numpy()]
            summaries.append("\n".join(sentences))

    os.makedirs(args.output_dir, exist_ok=True)
    rows = zip(ids, summaries)
    with open(os.path.join(args.output_dir, "submission.csv"), "w+") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "summary"])
        for row in rows:
            writer.writerow(row)
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./', required=False, help='checkpoint path')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None, required=False, help='checkpoint path')
    parser.add_argument('--test_file',
                        default='/media/irelin/data_disk/dataset/dacon_summury/extractive/extractive_test_v2.jsonl',
                        type=str)

    parser.add_argument('--max_token_cnt', type=int, default=300)
    parser.add_argument('--use_bert_sum_words', action='store_true', default=False)
    parser.add_argument('--use_media', action='store_true', default=False)
    parser.add_argument('--use_pos', action='store_true', default=False)
    parser.add_argument('--simple_model', action='store_true', default=False)
    parser.add_argument('--dim_feedforward', type=int, default=1024)

    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()

    for arg in vars(args):
        print(arg, getattr(args, arg))

    if args.seed:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    submit(args)
