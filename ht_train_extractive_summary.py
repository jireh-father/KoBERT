import argparse
import random
from kobert.pytorch_kobert import get_pytorch_kobert_model
import jsonlines
from torch.utils import data
from gluonnlp.data import SentencepieceTokenizer
from kobert.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.optim import lr_scheduler
import datetime
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import itertools
from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from model import ExtractiveModel

default_config = {
    "optimizer": tune.choice(['adam', 'sgd']),  # tune.grid_search(['adam', 'sgd']),
    "lr": tune.loguniform(1e-4, 1e-1),  # tune.loguniform(1e-4, 1e-1),
    "weight_decay": tune.loguniform(1e-6, 1e-3),
    "scheduler": tune.choice(['step', 'cosine']),  # tune.grid_search(['cosine', 'step']),
    "max_word_dropout_ratio": tune.quniform(0.1, 0.5, 0.05),  # tune.choice([0.1, 0.2, 0.3]),
    "word_dropout_prob": tune.quniform(0.0, 1.0, 0.1),
    "label_smoothing": tune.choice([0.1, 0.0]),  # tune.grid_search([0.1, 0.0]),
    "use_multi_class": False,  # tune.grid_search([True, False]),
    "freeze_bert": tune.choice([False, True]),
    "use_bert_sum_words": tune.choice([True, False]),  # tune.grid_search([True, False]),
    "use_pos": tune.choice([True, False]),  # True,  # tune.grid_search([True, False]),
    "use_media": tune.choice([True, False]),  # tune.grid_search([True, False]),
    "simple_model": tune.choice([False, True]),  # tune.grid_search([True, False]),
    "max_token_cnt": tune.choice([200, 300, 400, 500]),
    "dim_feedforward": tune.choice([512, 768, 1024]),
    "dropout": tune.choice([0.0, 0.1, 0.2, 0.3, 0.4]),
    "weighted_loss": tune.quniform(1.0, 10.0, 1.0)
}


def freeze_params(model):
    """Set requires_grad=False for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = False


def non_freeze_params(model):
    """Set requires_grad=False for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = False


def init_optimizer(optimizer_name, model, lr, wd, lr_restart_step=1, lr_decay_gamma=0.9,
                   scheduler="step", nesterov=False, num_epochs=None, steps_per_epoch=None):
    if optimizer_name == "sgd":
        optimizer_ft = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd, nesterov=nesterov)
    elif optimizer_name == "adam":
        optimizer_ft = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_name == "adamp":
        from adamp import AdamP
        optimizer_ft = AdamP(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=wd)  # 1e-2)
    elif optimizer_name == "sgdp":
        from adamp import SGDP
        optimizer_ft = SGDP(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9, nesterov=nesterov)
    # else:
    #     opt_attr = getattr(toptim, optimizer_name)
    #     if opt_attr:
    #         optimizer_ft = opt_attr(model.parameters())
    #     else:
    #         raise Exception("unknown optimizer name", optimizer_name)

    if scheduler == "cosine":
        exp_lr_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer_ft, lr_restart_step)
        use_lr_schedule_steps = True
    elif scheduler == "cycle":
        exp_lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer_ft, max_lr=lr, steps_per_epoch=steps_per_epoch,
                                                               epochs=num_epochs, pct_start=0.1)
        use_lr_schedule_steps = False
    elif scheduler == "step":
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=lr_restart_step, gamma=lr_decay_gamma)
        use_lr_schedule_steps = False

    return optimizer_ft, exp_lr_scheduler, use_lr_schedule_steps


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon=0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)


class SentenceDataset(data.Dataset):
    """__init__ and __len__ functions are the same as in TorchvisionDataset"""

    def __init__(self, samples, vocab, media_map, word_dropout_prob=0.0, max_word_dropout_ratio=0.0, max_token_cnt=300):
        self.tokenizer = SentencepieceTokenizer(get_tokenizer())
        self.vocab = vocab

        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.media_map = media_map
        self.word_dropout_prob = word_dropout_prob
        self.max_word_dropout_ratio = max_word_dropout_ratio
        self.max_token_cnt = max_token_cnt
        # self.classes = classes
        # self.class_to_idx = class_to_idx

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


def pad_collate(batch):
    token_ids_batch, target_batch, pos_idx_batch, media_batch = zip(*batch)
    token_ids_batch = pad_sequence(token_ids_batch, batch_first=True, padding_value=0)

    return token_ids_batch, torch.tensor(target_batch, dtype=torch.long), \
           torch.tensor(pos_idx_batch, dtype=torch.long), torch.tensor(media_batch, dtype=torch.long),


def save_model(model, model_path):
    if hasattr(model, 'module'):
        model = model.module
    print("save model", model_path)
    torch.save(model.state_dict(), model_path)


def train(config, args):
    # 문장 최대 갯수 100개

    # import unicodedata
    # unicodedata.normalize('NFKC', '한국어로는 안되?ㅋ')

    samples_dict = {}
    medias = set()
    with jsonlines.open(args.train_file) as f:
        for line in f.iter():
            media = line['media']
            medias.add(media)
            extractive = line['extractive']
            for i, sentence in enumerate(line['article_original']):
                if i in extractive:
                    if config['use_multi_class']:
                        label = extractive.index(i)
                    else:
                        label = 0
                else:
                    if config['use_multi_class']:
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

    os.makedirs(os.path.join(args.work_dir, "saved_models"), exist_ok=True)

    train_samples = []
    val_samples = []
    class_cnt = []
    num_classes = 4 if config['use_multi_class'] else 2
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

    bert_model, vocab = get_pytorch_kobert_model()
    if config['freeze_bert']:
        freeze_params(bert_model.embeddings)
        freeze_params(bert_model.encoder)
        freeze_params(bert_model.pooler)
    else:
        non_freeze_params(bert_model.embeddings)
        non_freeze_params(bert_model.encoder)
        non_freeze_params(bert_model.pooler)

    train_dataset = SentenceDataset(train_samples, vocab, media_map, word_dropout_prob=config['word_dropout_prob'],
                                    max_word_dropout_ratio=config['max_word_dropout_ratio'],
                                    max_token_cnt=config['max_token_cnt'])
    val_dataset = SentenceDataset(val_samples, vocab, media_map, max_token_cnt=config['max_token_cnt'])

    weights = 1. / torch.tensor(class_cnt, dtype=torch.float)
    print('weights', weights)
    samples_weights = weights[train_targets]
    sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weights, len(train_samples))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size,
                                               num_workers=args.num_workers, pin_memory=args.train_pin_memory,
                                               collate_fn=pad_collate,
                                               sampler=sampler
                                               )

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size * 2,
                                             num_workers=args.num_workers,
                                             shuffle=False, pin_memory=args.val_pin_memory, collate_fn=pad_collate)

    model = ExtractiveModel(bert_model, 100, 11, 768,
                            use_bert_sum_words=config["use_bert_sum_words"],
                            use_pos=config["use_pos"],
                            use_media=config['use_media'],
                            simple_model=config['simple_model'],
                            num_classes=num_classes,
                            dim_feedforward=config['dim_feedforward'],
                            dropout=config['dropout'])

    if args.checkpoint_path is not None and os.path.isfile(args.checkpoint_path):
        state_dict = torch.load(args.checkpoint_path)
        model.load_state_dict(state_dict)

    if torch.cuda.device_count() > 1 and args.data_parallel:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    device = "cuda"
    model.to(device)

    steps_per_epoch = len(train_samples) // args.train_batch_size
    if len(train_samples) % args.train_batch_size > 0:
        steps_per_epoch += 1
    optimizer, scheduler, use_lr_schedule_steps = init_optimizer(config['optimizer'], model,
                                                                 config['lr'], config['weight_decay'],
                                                                 args.lr_restart_step,
                                                                 args.lr_decay_gamma,
                                                                 config['scheduler'],
                                                                 nesterov=args.nesterov,
                                                                 num_epochs=args.num_epochs,
                                                                 steps_per_epoch=steps_per_epoch)

    if config['weighted_loss'] > 0:
        weights = [config['weighted_loss'], 1.]
        class_weights = torch.FloatTensor(weights).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    elif config['label_smoothing'] > 0:
        criterion = LabelSmoothingCrossEntropy(epsilon=config['label_smoothing'])
    else:
        criterion = torch.nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    os.makedirs(args.work_dir, exist_ok=True)

    print('train_loader', len(train_loader))
    for epoch in range(args.num_epochs):

        if args.train:
            print("Epoch %d/%d, LR: %f" % (epoch, args.num_epochs, np.array(scheduler.get_lr()).mean()))
            epoch_start_time = time.time()
            model.train()
            epoch_labels = []
            epoch_preds = []
            epoch_loss = 0.
            for step, (token_ids_batch, labels, pos_idx_batch, media_batch) in enumerate(train_loader):
                batch_start_time = time.time()
                epoch_labels += list(labels.numpy())
                labels = labels.to(device)
                token_ids_batch = token_ids_batch.to(device)
                pos_idx_batch = pos_idx_batch.to(device)
                media_batch = media_batch.to(device)

                if use_lr_schedule_steps:
                    scheduler.step(epoch - 1 + step / len(train_loader))
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(True):
                    outputs = model(token_ids_batch, pos_idx_batch, media_batch)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)
                    epoch_preds += list(preds.cpu().numpy())

                    # backward + optimize only if in training phase
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * token_ids_batch.size(0)

                batch_elapsed_time = time.time() - batch_start_time
                if step >= 0 and (step + 1) % args.log_step_interval == 0:
                    current_datetime = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')

                    f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
                    acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())

                    print("[train-epoch:%02d/%02d,step:%d/%d,%s] batch_elapsed: %f" %
                          (epoch, args.num_epochs, step, len(train_loader), current_datetime, batch_elapsed_time))
                    print("loss: %f, acc: %f, f1: %f, lr: %f" % (
                        loss.item(), acc, f1, np.array(scheduler.get_lr()).mean()))

            if not use_lr_schedule_steps:
                scheduler.step()

            epoch_loss = epoch_loss / len(train_loader.dataset)
            epoch_elapsed_time = time.time() - epoch_start_time
            current_datetime = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
            epoch_f1 = f1_score(epoch_labels, epoch_preds, average='macro')
            epoch_acc = accuracy_score(epoch_labels, epoch_preds)

            print(
                "[result:train-epoch:%02d/%02d,%s] epoch_elapsed: %s, loss: %f, acc: %f, f1: %f, lr: %f" % (
                    epoch, args.num_epochs, current_datetime, epoch_elapsed_time, epoch_loss, epoch_acc, epoch_f1,
                    scheduler.get_lr()[0]))

        if args.val:
            model.eval()  # Set model to evaluate mode
            epoch_start_time = time.time()
            epoch_preds = []
            epoch_labels = []
            epoch_loss = 0.

            for step, (token_ids_batch, labels, pos_idx_batch, media_batch) in enumerate(val_loader):
                batch_start_time = time.time()
                epoch_labels += list(labels.numpy())
                labels = labels.to(device)
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
                    loss = criterion(outputs, labels)

                # statistics
                epoch_loss += loss.item() * token_ids_batch.size(0)
                batch_elapsed_time = time.time() - batch_start_time

                if step >= 0 and (step + 1) % args.log_step_interval == 0:
                    current_datetime = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')

                    f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
                    acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())

                    print("[val-epoch:%d, step:%d/%d,%s] batch_elapsed: %f" %
                          (epoch, step, len(val_loader), current_datetime, batch_elapsed_time))
                    print("loss: %f, acc: %f, f1: %f" % (loss.item(), acc, f1))

            epoch_loss = epoch_loss / len(val_loader.dataset)

            current_datetime = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')

            epoch_acc = accuracy_score(epoch_labels, epoch_preds)
            epoch_f1 = f1_score(epoch_labels, epoch_preds, pos_label=0)
            epoch_precision = precision_score(epoch_labels, epoch_preds, pos_label=0)
            epoch_recall = recall_score(epoch_labels, epoch_preds, pos_label=0)
            epoch_elapsed = time.time() - epoch_start_time

            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((model.state_dict(), optimizer.state_dict()), path)

            epoch_cm = confusion_matrix(epoch_labels, epoch_preds)
            np_epoch_labels = np.unique(np.array(epoch_labels))
            np_epoch_labels.sort()
            print("confusion matrix")
            print(epoch_cm)

            tune.report(loss=epoch_loss, f1=epoch_f1, acc=epoch_acc, pos_acc=epoch_cm[0], neg_acc=epoch_cm[1],
                        precision=epoch_precision, recall=epoch_recall)

            print(
                "[result_val-epoch:%d,%s] epoch_elapsed: %s, loss: %f, acc: %f, f1: %f, pre: %f, recall: %f" % (
                    epoch, current_datetime, epoch_elapsed, epoch_loss, epoch_acc, epoch_f1, epoch_precision,
                    epoch_recall))

            cls_report = classification_report(epoch_labels, epoch_preds)  # , target_names=classes)
            print(cls_report)

            # np.save(os.path.join(log_dir, "confusion_matrix_%s_epoch_%d.npy" % (val_name, epoch)), epoch_cm)
            epoch_cm = epoch_cm.astype('float') / epoch_cm.sum(axis=1)[:, np.newaxis]
            epoch_cm = epoch_cm.diagonal()
            print("each accuracies")
            print(epoch_cm)

        if not args.train and args.val:
            print("The end of evaluation.")
            break


def test_accuracy(model, use_multi_class, max_token_cnt, device, args):
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


def main(args=None):
    # config = {
    #     "optimizer": 'adam',  # tune.grid_search(['adam', 'sgd']),
    #     "lr": 0.001,  # tune.loguniform(1e-4, 1e-1),
    #     "scheduler": 'step',  # tune.grid_search(['cosine', 'step']),
    #     "max_word_dropout_ratio": 0.0,
    #     "word_dropout_prob": 0.0,
    #     "label_smoothing": tune.grid_search([0.1, 0.0]),
    #     "use_multi_class": False,  # tune.grid_search([True, False]),
    #     "freeze_bert": tune.grid_search([True, False]),
    #     "use_bert_sum_words": tune.grid_search([True, False]),
    #     "use_pos": tune.grid_search([True, False]),
    #     "use_media": tune.grid_search([True, False]),
    #     "simple_model": tune.grid_search([True, False])
    # }
    config = {
        "optimizer": tune.grid_search(['adam', 'sgd']),  # tune.grid_search(['adam', 'sgd']),
        "lr": tune.grid_search([0.1, 0.01, 0.001, 0.0001]),  # tune.loguniform(1e-4, 1e-1),
        "scheduler": tune.grid_search(['step', 'cosine']),  # tune.grid_search(['cosine', 'step']),
        "max_word_dropout_ratio": tune.grid_search([0.1, 0.2, 0.3]),
        "word_dropout_prob": tune.grid_search([0.5, 0.0, 0.25, 0.75, 1.0]),
        "label_smoothing": 0.0,  # tune.grid_search([0.1, 0.0]),
        "use_multi_class": False,  # tune.grid_search([True, False]),
        "freeze_bert": tune.grid_search([True, False]),
        "use_bert_sum_words": False,  # tune.grid_search([True, False]),
        "use_pos": True,  # tune.grid_search([True, False]),
        "use_media": False,  # tune.grid_search([True, False]),
        "simple_model": True,  # tune.grid_search([True, False])
    }
    # config = {
    #     "optimizer": tune.grid_search(['adam', 'sgd']),  # tune.grid_search(['adam', 'sgd']),
    #     "lr": tune.grid_search([0.001, 0.0001, 0.1, 0.01]),  # tune.loguniform(1e-4, 1e-1),
    #     "scheduler": tune.grid_search(['cosine', 'step']),  # tune.grid_search(['cosine', 'step']),
    #     "max_word_dropout_ratio": tune.grid_search([0.3, 0.2, 0.1]),
    #     "word_dropout_prob": tune.grid_search([1.0, 0.75, 0.25, 0.0, 0.5]),
    #     "label_smoothing": tune.grid_search([0.0, 0.1]),
    #     "use_multi_class": False,  # tune.grid_search([True, False]),
    #     "freeze_bert": tune.grid_search([False, True]),
    #     "use_bert_sum_words": False,  # tune.grid_search([True, False]),
    #     "use_pos": tune.grid_search([False, True]),
    #     "use_media": tune.grid_search([False, True]),
    #     "simple_model": tune.grid_search([False, True])
    # }
    scheduler = ASHAScheduler(
        metric="f1",
        mode="max",
        max_t=args.num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "f1", "training_iteration"])
    result = tune.run(
        partial(train, args=args),
        resources_per_trial={"cpu": args.num_workers, "gpu": args.gpus_per_trial},
        config=config,
        num_samples=args.num_tune_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir=args.work_dir)

    best_trial = result.get_best_trial("f1", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    bert_model, vocab = get_pytorch_kobert_model()
    num_classes = 4 if best_trial.config["use_multi_class"] else 2
    best_trained_model = ExtractiveModel(bert_model, 100, 11, 768,
                                         use_bert_sum_words=best_trial.config["use_bert_sum_words"],
                                         use_pos=best_trial.config["use_pos"],
                                         use_media=best_trial.config['use_media'],
                                         simple_model=best_trial.config['simple_model'],
                                         num_classes=num_classes,
                                         dim_feedforward=best_trial.config['dim_feedforward'],
                                         dropout=best_trial.config['dropout'])

    if torch.cuda.is_available():
        device = "cuda"
        # if args.gpus_per_trial > 1:
        #     best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    test_acc = test_accuracy(best_trained_model, best_trial.config["use_multi_class"], device)
    print("Best trial test set f1: {}".format(test_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-l', '--work_dir', type=str, default='./log')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None, required=False, help='checkpoint path')
    parser.add_argument('--train_file', default='/media/irelin/data_disk/dataset/dacon_summury/abstractive/train.jsonl',
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
