import argparse
import random
from kobert.pytorch_kobert import get_pytorch_kobert_model
import jsonlines
from torch.utils import data
from gluonnlp.data import SentencepieceTokenizer
from kobert.utils import get_tokenizer
from torch.optim import lr_scheduler
import datetime
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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
                sentences = []
                for i, sentence in enumerate(line['article_original']):
                    sentences.append(sentence.replace('\n', '').strip())
                samples.append([sentences, media])
        self.samples = samples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        sentence, target, pos_idx, media = self.samples[index]
        media = self.media_map[media]
        tokens = self.tokenizer(sentence)
        token_ids = self.vocab.to_indices(tokens)
        if random.random() < self.word_dropout_prob:
            dropout_cnt = round(self.max_word_dropout_ratio * len(token_ids))
            for i in range(dropout_cnt):
                dropout_idx = random.randint(0, len(token_ids) - 1)
                del token_ids[dropout_idx]

        if len(token_ids) > self.max_token_cnt:
            token_ids = token_ids[:self.max_token_cnt]

        return torch.tensor(token_ids, dtype=torch.long), target, pos_idx, media

    def __len__(self):
        return len(self.samples)


def test_accuracy(model, use_multi_class, max_token_cnt, device):
    samples_dict = {}
    medias = set()
    with jsonlines.open(train_file) as f:
        for line in f.iter():
            media = line['media']
            medias.add(media)

    medias = list(medias)
    medias.sort()
    media_map = {m: i for i, m in enumerate(medias)}
    print("medias", media_map)

    samples_dict = {}
    medias = set()
    with jsonlines.open(args.train_file) as f:
        for line in f.iter():
            media = line['media']
            medias.add(media)
            extractive = line['extractive']
            for i, sentence in enumerate(line['article_original']):
                if i in extractive:
                    if use_multi_class:
                        label = extractive.index(i)
                    else:
                        label = 0
                else:
                    if use_multi_class:
                        label = 3
                    else:
                        label = 1
                if label not in samples_dict:
                    samples_dict[label] = []
                samples_dict[label].append([sentence.replace('\n', '').strip(), label, i, media])

    medias = list(medias)
    medias.sort()
    media_map = {m: i for i, m in enumerate(medias)}
    print("medias", media_map)

    train_samples = []
    val_samples = []
    class_cnt = []
    num_classes = 4 if use_multi_class else 2
    for label in range(num_classes):
        random.shuffle(samples_dict[label])
        val_cnt = round(len(samples_dict[label]) * args.val_ratio)
        val_samples += samples_dict[label][:val_cnt]
        tmp_train_samples = samples_dict[label][val_cnt:]
        class_cnt.append(len(tmp_train_samples))
        if args.use_all_train:
            train_samples += samples_dict[label]
        elif args.train_val_data:
            train_samples += val_samples
        else:
            train_samples += tmp_train_samples

    print('class_cnt', class_cnt)

    random.shuffle(train_samples)
    train_targets = [t[1] for t in train_samples]
    print("total samples", len(train_samples) + len(val_samples))
    print("train samples", len(train_samples))
    print("val samples", len(val_samples))

    _, vocab = get_pytorch_kobert_model()
    val_dataset = SentenceDataset(val_samples, vocab, media_map, max_token_cnt=max_token_cnt)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size * 2,
                                             num_workers=args.num_workers,
                                             shuffle=False, pin_memory=args.val_pin_memory, collate_fn=pad_collate)

    model.eval()  # Set model to evaluate mode
    epoch_start_time = time.time()
    epoch_preds = []
    epoch_labels = []

    for step, (token_ids_batch, labels, pos_idx_batch, media_batch) in enumerate(val_loader):
        batch_start_time = time.time()
        epoch_labels += list(labels.numpy())
        token_ids_batch = token_ids_batch.to(device)
        pos_idx_batch = pos_idx_batch.to(device)
        media_batch = media_batch.to(device)

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            start = time.time()
            outputs = model(token_ids_batch, pos_idx_batch, media_batch)
            # print("batch speed", time.time() - start)
            _, preds = torch.max(outputs, 1)
            epoch_preds += list(preds.cpu().numpy())

    current_datetime = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')

    epoch_acc = accuracy_score(epoch_labels, epoch_preds)
    epoch_f1 = f1_score(epoch_labels, epoch_preds, average='macro')
    epoch_elapsed = time.time() - epoch_start_time
    print(
        "[result_val,%s] epoch_elapsed: %s, acc: %f, f1: %f" % (
            current_datetime, epoch_elapsed, epoch_acc, epoch_f1))

    cls_report = classification_report(epoch_labels, epoch_preds)  # , target_names=classes)
    print(cls_report)
    epoch_cm = confusion_matrix(epoch_labels, epoch_preds)
    print("confusion matrix")
    print(epoch_cm)
    # np.save(os.path.join(log_dir, "confusion_matrix_%s_epoch_%d.npy" % (val_name, epoch)), epoch_cm)
    epoch_cm = epoch_cm.astype('float') / epoch_cm.sum(axis=1)[:, np.newaxis]
    epoch_cm = epoch_cm.diagonal()
    print("each accuracies")
    print(epoch_cm)

    return epoch_f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-l', '--work_dir', type=str, default='./log')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None, required=False, help='checkpoint path')
    parser.add_argument('--test_file',
                        default='/media/irelin/data_disk/dataset/dacon_summury/extractive/extractive_test_v2.jsonl',
                        type=str)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--lr_restart_step', type=int, default=1)
    parser.add_argument('-e', '--num_epochs', type=int, default=100)
    parser.add_argument('--log_step_interval', type=int, default=100)

    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--val_batch_size', type=int, default=64)
    parser.add_argument('-w', '--num_workers', type=int, default=8)

    parser.add_argument('--max_token_cnt', type=int, default=300)

    parser.add_argument('--lr_decay_gamma', type=float, default=0.9)
    # parser.add_argument('-d', '--weight_decay', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('-t', '--train', default=False, action="store_true")
    parser.add_argument('-v', '--val', default=False, action="store_true")

    parser.add_argument('--use_all_train', default=False, action="store_true")
    parser.add_argument('--train_val_data', default=False, action="store_true")
    parser.add_argument('--data_parallel', default=False, action="store_true")

    parser.add_argument('--train_pin_memory', default=False, action="store_true")
    parser.add_argument('--val_pin_memory', default=False, action="store_true")
    parser.add_argument('--use_benchmark', default=False, action="store_true")
    parser.add_argument('--nesterov', default=False, action="store_true")
    parser.add_argument('--gpus_per_trial', type=int, default=2)

    parser.add_argument('--num_tune_samples', type=int, default=1)

    args = parser.parse_args()

    for arg in vars(args):
        print(arg, getattr(args, arg))

    if args.seed:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        if args.use_benchmark:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = True

    main(args)
