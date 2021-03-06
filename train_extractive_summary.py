import os
import argparse
import random
import numpy as np
import torch
from kobert.pytorch_kobert import get_pytorch_kobert_model
import jsonlines
from torch.utils import data
from gluonnlp.data import SentencepieceTokenizer
from kobert.utils import get_tokenizer
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import datetime
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
import itertools
from model import ExtractiveModel


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


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    figure.canvas.draw()
    return np.array(figure.canvas.renderer._renderer)


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
      cm (array, shape = [n, n]): a confusion matrix of integer classes
      class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


def log_confusion_matrix(writer, epoch, cm, class_names=None):
    # Log the confusion matrix as an image summary.
    figure = plot_confusion_matrix(cm, class_names=class_names)
    cm_image = plot_to_image(figure)

    writer.add_image('confusion_matrix', cm_image, epoch, dataformats='HWC')


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


def train(args):
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
                    if args.use_multi_class:
                        label = extractive.index(i)
                    else:
                        label = 0
                else:
                    if args.use_multi_class:
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
    num_classes = 4 if args.use_multi_class else 2
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
    if args.freeze_bert:
        freeze_params(bert_model.embeddings)
        freeze_params(bert_model.encoder)
        freeze_params(bert_model.pooler)
    else:
        non_freeze_params(bert_model.embeddings)
        non_freeze_params(bert_model.encoder)
        non_freeze_params(bert_model.pooler)
    train_dataset = SentenceDataset(train_samples, vocab, media_map, word_dropout_prob=args.word_dropout_prob,
                                    max_word_dropout_ratio=args.max_word_dropout_ratio,
                                    max_token_cnt=args.max_token_cnt)
    val_dataset = SentenceDataset(val_samples, vocab, media_map, max_token_cnt=args.max_token_cnt)

    weights = 1. / torch.tensor(class_cnt, dtype=torch.float)
    print('weights', weights)
    samples_weights = weights[train_targets]
    sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weights, len(train_samples))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size,
                                               num_workers=args.num_workers, pin_memory=args.train_pin_memory,
                                               collate_fn=pad_collate,
                                               sampler=sampler
                                               )

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size, num_workers=args.num_workers,
                                             shuffle=False, pin_memory=args.val_pin_memory, collate_fn=pad_collate)

    model = ExtractiveModel(bert_model, 100, 11, 768,
                            use_bert_sum_words=args.use_bert_sum_words,
                            use_pos=args.use_pos,
                            use_media=args.use_media, num_classes=num_classes, simple_model=args.simple_model,
                            dim_feedforward=args.dim_feedforward,
                            dropout=args.dropout)

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
    optimizer, scheduler, use_lr_schedule_steps = init_optimizer(args.optimizer, model,
                                                                 args.lr, args.weight_decay,
                                                                 args.lr_restart_step,
                                                                 args.lr_decay_gamma,
                                                                 args.scheduler,
                                                                 nesterov=args.nesterov,
                                                                 num_epochs=args.num_epochs,
                                                                 steps_per_epoch=steps_per_epoch)

    if args.weighted_loss > 0:
        weights = [args.weighted_loss, 1.]
        class_weights = torch.FloatTensor(weights).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    elif args.label_smoothing > 0:
        criterion = LabelSmoothingCrossEntropy(epsilon=args.label_smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    os.makedirs(args.work_dir, exist_ok=True)
    train_writer = SummaryWriter(os.path.join(args.work_dir, 'train'))
    val_writer = SummaryWriter(os.path.join(args.work_dir, 'val'))

    print('train_loader', len(train_loader))
    best_f1 = 0.
    best_acc = 0.
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
                with torch.autograd.set_detect_anomaly(True):
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

                    train_writer.add_scalar('Loss', loss.item(), step + len(train_loader) * epoch)
                    train_writer.add_scalar('Acc', acc, step + len(train_loader) * epoch)
                    train_writer.add_scalar('F1', f1, step + len(train_loader) * epoch)
                    train_writer.add_scalar('LR', np.array(scheduler.get_lr()).mean(),
                                            step + len(train_loader) * epoch)

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
            train_writer.add_scalar('Loss/epoch', epoch_loss, epoch)
            train_writer.add_scalar('Acc/epoch', epoch_acc, epoch)
            train_writer.add_scalar('F1/epoch', epoch_f1, epoch)
            save_model(model, os.path.join(args.work_dir, "saved_models", "epoch_%d.pth" % epoch))

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
            epoch_f1 = f1_score(epoch_labels, epoch_preds, average='macro')
            epoch_elapsed = time.time() - epoch_start_time
            print(
                "[result_val-epoch:%d,%s] epoch_elapsed: %s, loss: %f, acc: %f, f1: %f" % (
                    epoch, current_datetime, epoch_elapsed, epoch_loss, epoch_acc, epoch_f1))

            cls_report = classification_report(epoch_labels, epoch_preds)  # , target_names=classes)
            print(cls_report)
            epoch_cm = confusion_matrix(epoch_labels, epoch_preds)
            np_epoch_labels = np.unique(np.array(epoch_labels))
            np_epoch_labels.sort()
            log_confusion_matrix(val_writer, epoch, epoch_cm, np_epoch_labels)
            print("confusion matrix")
            print(epoch_cm)
            # np.save(os.path.join(log_dir, "confusion_matrix_%s_epoch_%d.npy" % (val_name, epoch)), epoch_cm)
            epoch_cm = epoch_cm.astype('float') / epoch_cm.sum(axis=1)[:, np.newaxis]
            epoch_cm = epoch_cm.diagonal()
            print("each accuracies")
            print(epoch_cm)

            val_writer.add_scalar('Loss/epoch', epoch_loss, epoch)
            val_writer.add_scalar('Acc/epoch', epoch_acc, epoch)
            val_writer.add_scalar('F1/epoch', epoch_f1, epoch)
            if epoch_f1 > best_f1 or (epoch_f1 >= best_f1 and epoch_acc > best_acc):
                best_f1 = epoch_f1
                best_acc = epoch_acc
                save_model(model, os.path.join(args.work_dir, "saved_models", "best_model.pth"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-l', '--work_dir', type=str, default='./log')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None, required=False, help='checkpoint path')
    parser.add_argument('--train_file', default='/media/irelin/data_disk/dataset/dacon_summury/abstractive/train.jsonl',
                        type=str)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('-z', '--optimizer', type=str, default='adam')  # adam')
    parser.add_argument('--scheduler', type=str, default='cosine')  # cosine, step')
    parser.add_argument('--lr_restart_step', type=int, default=1)
    parser.add_argument('-e', '--num_epochs', type=int, default=100)
    parser.add_argument('--log_step_interval', type=int, default=100)

    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--val_batch_size', type=int, default=64)
    parser.add_argument('-w', '--num_workers', type=int, default=8)

    parser.add_argument('--max_token_cnt', type=int, default=300)

    parser.add_argument('--use_bert_sum_words', action='store_true', default=False)
    parser.add_argument('--use_media', action='store_true', default=False)
    parser.add_argument('--use_pos', action='store_true', default=False)
    parser.add_argument('--simple_model', action='store_true', default=False)
    parser.add_argument('--dim_feedforward', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--word_dropout_prob', type=float, default=0.0)
    parser.add_argument('--max_word_dropout_ratio', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_decay_gamma', type=float, default=0.9)
    parser.add_argument('-d', '--weight_decay', type=float, default=1e-5)
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('-t', '--train', default=False, action="store_true")
    parser.add_argument('-v', '--val', default=False, action="store_true")

    parser.add_argument('--use_multi_class', default=False, action="store_true")

    parser.add_argument('--use_all_train', default=False, action="store_true")
    parser.add_argument('--train_val_data', default=False, action="store_true")
    parser.add_argument('--data_parallel', default=False, action="store_true")

    parser.add_argument('--train_pin_memory', default=False, action="store_true")
    parser.add_argument('--val_pin_memory', default=False, action="store_true")
    parser.add_argument('--use_benchmark', default=False, action="store_true")
    parser.add_argument('--nesterov', default=False, action="store_true")
    parser.add_argument('--freeze_bert', default=False, action="store_true")

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

    train(args)
