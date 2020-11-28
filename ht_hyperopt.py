import argparse
import random
from kobert.pytorch_kobert import get_pytorch_kobert_model
from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import HyperBandScheduler
import ht_train_extractive_summary as trainer_util
import ray
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch


def main(args=None):
    current_best_params = [
        {
            "optimizer": 0,  # tune.grid_search(['adam', 'sgd']),
            "lr": 0.001,  # tune.loguniform(1e-4, 1e-1),
            "weight_decay": 1e-5,
            "scheduler": 1,  # tune.grid_search(['cosine', 'step']),
            "max_word_dropout_ratio": 0.2,  # tune.choice([0.1, 0.2, 0.3]),
            "word_dropout_prob": 0.2,
            "label_smoothing": 0.1,  # tune.grid_search([0.1, 0.0]),
            "freeze_bert": 0,
            "use_bert_sum_words": 0,  # tune.grid_search([True, False]),
            "use_pos": 0,  # True,  # tune.grid_search([True, False]),
            "use_media": 0,  # tune.grid_search([True, False]),
            "simple_model": 0,  # tune.grid_search([True, False])
        },
        {
            "optimizer": 0,  # tune.grid_search(['adam', 'sgd']),
            "lr": 0.001,  # tune.loguniform(1e-4, 1e-1),
            "weight_decay": 1e-5,
            "scheduler": 1,  # tune.grid_search(['cosine', 'step']),
            "max_word_dropout_ratio": 0.0,  # tune.choice([0.1, 0.2, 0.3]),
            "word_dropout_prob": 0.0,
            "label_smoothing": 0.1,  # tune.grid_search([0.1, 0.0]),
            "freeze_bert": 0,
            "use_bert_sum_words": 0,  # tune.grid_search([True, False]),
            "use_pos": 0,  # True,  # tune.grid_search([True, False]),
            "use_media": 0,  # tune.grid_search([True, False]),
            "simple_model": 0,  # tune.grid_search([True, False])
        },
        {
            "optimizer": 0,  # tune.grid_search(['adam', 'sgd']),
            "lr": 0.001,  # tune.loguniform(1e-4, 1e-1),
            "weight_decay": 1e-5,
            "scheduler": 0,  # tune.grid_search(['cosine', 'step']),
            "max_word_dropout_ratio": 0.0,  # tune.choice([0.1, 0.2, 0.3]),
            "word_dropout_prob": 0.0,
            "label_smoothing": 0.1,  # tune.grid_search([0.1, 0.0]),
            "freeze_bert": 1,
            "use_bert_sum_words": 0,  # tune.grid_search([True, False]),
            "use_pos": 0,  # True,  # tune.grid_search([True, False]),
            "use_media": 0,  # tune.grid_search([True, False]),
            "simple_model": 0,  # tune.grid_search([True, False])
        }
    ]

    tune_kwargs = {
        "num_samples": args.num_tune_samples,
        "config": {
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
            "simple_model": tune.choice([False, True]),  # tune.grid_search([True, False])
        }
    }

    algo = HyperOptSearch(points_to_evaluate=current_best_params)
    algo = ConcurrencyLimiter(algo, max_concurrent=args.num_avail_gpus)

    scheduler = AsyncHyperBandScheduler()
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "f1", "training_iteration", 'acc', 'pos_acc', 'neg_acc', 'precision', 'recall'])
    result = tune.run(
        partial(trainer_util.train, args=args),
        resources_per_trial={"cpu": args.cpus_per_trial, "gpu": args.gpus_per_trial},
        scheduler=scheduler,
        search_alg=algo,
        metric='precision',
        mode='max',
        progress_reporter=reporter,
        local_dir=args.work_dir,
        **tune_kwargs)

    best_trial = result.get_best_trial("precision", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
    print("Best trial final validation precision: {}".format(best_trial.last_result["precision"]))
    print("Best trial final validation recall: {}".format(best_trial.last_result["recall"]))
    print("Best trial final validation acc: {}".format(best_trial.last_result["acc"]))
    print("Best trial final validation pos_acc: {}".format(best_trial.last_result["pos_acc"]))
    print("Best trial final validation neg_acc: {}".format(best_trial.last_result["neg_acc"]))

    bert_model, vocab = get_pytorch_kobert_model()
    best_trained_model = trainer_util.ExtractiveModel(bert_model, 100, 11, 768,
                                                      use_bert_sum_words=best_trial.config["use_bert_sum_words"],
                                                      use_pos=best_trial.config["use_pos"],
                                                      use_media=best_trial.config['use_media'],
                                                      simple_model=best_trial.config['simple_model'])

    if torch.cuda.is_available():
        device = "cuda:0"
        # if args.gpus_per_trial > 1:
        #     best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    test_acc = trainer_util.test_accuracy(best_trained_model, best_trial.config["use_multi_class"], device)
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
    parser.add_argument('--cpus_per_trial', type=int, default=8)
    parser.add_argument('--num_avail_gpus', type=int, default=8)

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
