import sys
sys.path.append('../..')

import os
import random

import numpy as np
import torch
from torch.optim import Adamax, SGD, Adam
from reproduction.matching.test.adamW import AdamW
from torch.optim.lr_scheduler import StepLR
from torch import nn
from fastNLP import cache_results
from fastNLP.core import Trainer, Tester, AccuracyMetric, Const
from fastNLP.core.callback import GradientClipCallback, LRScheduler, FitlogCallback
import fastNLP.modules.encoder.embedding as Embed

from reproduction.matching.data.MatchingDataLoader import SNLILoader, RTELoader, QNLILoader
from reproduction.matching.model.bert import BertForNLI
from reproduction.matching.model.esim import ESIMModel
from reproduction.matching.model.CIN import CINModel, ParamResetCallback

import fitlog

fitlog.set_log_dir("Logs")

import argparse

argument = argparse.ArgumentParser()
argument.add_argument('--embedding', choices=['glove', 'elmo', 'none'], default='glove')
argument.add_argument('--model', choices=['esim', 'bert', 'cin'], default='cin')
argument.add_argument('--batch-size-per-gpu', type=int, default=128)
argument.add_argument('--n-epochs', type=int, default=100)
argument.add_argument('--lr', type=float, default=1e-4)
argument.add_argument('--wd', type=float, default=1e-5)
argument.add_argument('--k-sz', type=int, default=3)
argument.add_argument('--seq-len-type', choices=['bert', 'mask', 'seq_len'], default='seq_len')
argument.add_argument('--task', choices=['snli', 'rte', 'qnli'], default='snli')
argument.add_argument('--testset-name', type=str, default='test')
argument.add_argument('--seed', type=int, default=42)
arg = argument.parse_args()


random.seed(arg.seed)
np.random.seed(arg.seed)
torch.manual_seed(arg.seed)

n_gpu = torch.cuda.device_count()
if n_gpu > 0:
    torch.cuda.manual_seed_all(arg.seed)


for k in arg.__dict__:
    print(k, arg.__dict__[k], type(arg.__dict__[k]))

# bert_dirs = 'path/to/bert/dir'
bert_dirs = '/remote-home/hyan01/fastnlp_caches/bert-base-uncased'

os.environ['FASTNLP_BASE_URL'] = 'http://10.141.222.118:8888/file/download/'
os.environ['FASTNLP_CACHE_DIR'] = '/mnt/cephfs_hl/mlnlp/gongjingjing/.fastnlp_cache'

# load data set
if arg.task == 'snli':
    @cache_results('snli_upper_tr.pkl')
    def read_snli():
        data_info = SNLILoader().process(
            paths='./data/snli', to_lower=False, seq_len_type=arg.seq_len_type, bert_tokenizer=None,
            get_index=True, concat=False,
        )
        return data_info
    data_info = read_snli()
elif arg.task == 'rte':
    @cache_results('rte.pkl')
    def read_rte():
        data_info = RTELoader().process(
            paths='./data/rte', to_lower=False, seq_len_type=arg.seq_len_type, bert_tokenizer=None,
            get_index=True, concat=False,
        )
        return data_info
    data_info = read_rte()
elif arg.task == 'qnli':
    @cache_results('qnli.pkl')
    def read_qnli():
        data_info = QNLILoader().process(
            paths='./data/qnli', to_lower=False, seq_len_type=arg.seq_len_type, bert_tokenizer=None,
            get_index=True, concat=False, cut_text=512,
        )
        return data_info
    data_info = read_qnli()
else:
    raise RuntimeError(f'NOT support {arg.task} task yet!')

# print([data_info.vocabs['words'].idx2word[w] for w in data_info.datasets['dev'][0]['words']])
print(data_info)


if arg.embedding == 'elmo':
    embedding = Embed.ElmoEmbedding(data_info.vocabs[Const.INPUT], requires_grad=True)
elif arg.embedding == 'glove':
    embedding = Embed.StackEmbedding(
        [Embed.StaticEmbedding(data_info.vocabs[Const.INPUT], requires_grad=True, normalize=False),
         Embed.CNNCharEmbedding(data_info.vocabs[Const.INPUT], dropout=0.0)])


    # embedding.embedding.weight.data = embedding.embedding.weight.data / embedding.embedding.weight.data.std()

elif arg.embedding == 'none':
    embedding = None
else:
    raise RuntimeError(f'NOT support {arg.embedding} embedding yet!')


if arg.model == 'bert':
    model = BertForNLI(bert_dir=bert_dirs, class_num=len(data_info.vocabs[Const.TARGET]))
elif arg.model == 'esim':
    model = ESIMModel(embedding, num_labels=len(data_info.vocabs[Const.TARGET]))
elif arg.model == 'cin':
    model = CINModel(embedding, num_labels=len(data_info.vocabs[Const.TARGET]), k_sz=arg.k_sz)
else:
    raise RuntimeError(f'NOT support {arg.model} model yet!')


optimizer = Adamax(lr=arg.lr, params=model.parameters())
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

callbacks = [
    GradientClipCallback(clip_value=0.5), LRScheduler(scheduler), ParamResetCallback()
]
if arg.task in ['snli']:
    callbacks.append(FitlogCallback(data_info.datasets[arg.testset_name], verbose=1))

trainer = Trainer(train_data=data_info.datasets['train'], model=model,
                  optimizer=optimizer, num_workers=2,
                  batch_size=torch.cuda.device_count() * arg.batch_size_per_gpu,
                  n_epochs=arg.n_epochs, print_every=-1,
                  dev_data=data_info.datasets['dev'],
                  metrics=AccuracyMetric(), metric_key='acc',
                  device=[i for i in range(torch.cuda.device_count())],
                  # validate_every=30,
                  check_code_level=-1, callbacks=callbacks)
trainer.train(load_best_model=True)

tester = Tester(
    data=data_info.datasets[arg.testset_name],
    model=model,
    metrics=AccuracyMetric(),
    batch_size=torch.cuda.device_count() * arg.batch_size_per_gpu,
    device=[i for i in range(torch.cuda.device_count())],
)
tester.test()
