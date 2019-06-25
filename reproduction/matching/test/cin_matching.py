import os,sys
sys.path.append('../..')

import torch
from torch.optim import Adam
from fastNLP.core import Trainer, Tester, AccuracyMetric, Const
from fastNLP.core.callback import GradientClipCallback, LRScheduler
from reproduction.matching.data.MatchingDataLoader import SNLILoader, RTELoader, QNLILoader
from fastNLP.modules.encoder.embedding import ElmoEmbedding, StaticEmbedding
from reproduction.matching.model.bert import BertForNLI
from reproduction.matching.model.CIN import CINModel
from torch.optim.lr_scheduler import StepLR
import argparse


os.environ['FASTNLP_BASE_URL'] = 'http://10.141.222.118:8888/file/download/'
os.environ['FASTNLP_CACHE_DIR'] = '/remote-home/hyan01/fastnlp_caches'

argument = argparse.ArgumentParser()
argument.add_argument('--data-path', type=str, default='./data/snli')
argument.add_argument('--embedding', choices=['glove', 'elmo', 'none'], default='glove')
argument.add_argument('--batch-size-per-gpu', type=int, default=128)
argument.add_argument('--n-epochs', type=int, default=100)
argument.add_argument('--lr', type=float, default=2e-3)
argument.add_argument('--seq-len-type', choices=['bert', 'mask', 'seq_len'], default='seq_len')
argument.add_argument('--task', choices=['snli', 'rte', 'qnli'], default='snli')
argument.add_argument('--testset-name', type=str, default='test')
arg = argument.parse_args()

bert_dirs = '/remote-home/hyan01/fastnlp_caches/bert-base-uncased'
# load data set
if arg.task == 'snli':
    data_info = SNLILoader().process(
        paths=arg.data_path, to_lower=True, seq_len_type=arg.seq_len_type, bert_tokenizer=None,
        get_index=True, concat=False,
    )
elif arg.task == 'rte':
    data_info = RTELoader().process(
        paths=arg.data_path, to_lower=True, seq_len_type=arg.seq_len_type, bert_tokenizer=bert_dirs,
        get_index=True, concat='bert'
    )
elif arg.task == 'qnli':
    data_info = QNLILoader().process(
        paths=arg.data_path, to_lower=True, seq_len_type=arg.seq_len_type, bert_tokenizer=bert_dirs,
        get_index=True, concat='bert', cut_text=512,
    )
else:
    raise RuntimeError(f'NOT support {arg.task} task yet!')


embedding = StaticEmbedding(data_info.vocabs[Const.INPUT], model_dir_or_name='en-glove-840b-300', requires_grad=True)

# model = BertForNLI(bert_dir=bert_dirs)
model = CINModel(embedding)

optimizer = Adam(lr=2e-3, params=model.parameters())
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

callbacks = [
    GradientClipCallback(clip_value=10), LRScheduler(scheduler)
]

trainer = Trainer(train_data=data_info.datasets['train'], model=model,
                  optimizer=optimizer,
                  batch_size=torch.cuda.device_count() * arg.batch_size_per_gpu,
                  n_epochs=arg.n_epochs, print_every=50,
                  dev_data=data_info.datasets['dev'],
                  metrics=AccuracyMetric(), metric_key='acc',
                  device=[i for i in range(torch.cuda.device_count())],
                  check_code_level=-1, callbacks=callbacks)
trainer.train(load_best_model=True)

tester = Tester(
    data=data_info.datasets['test'],
    model=model,
    metrics=AccuracyMetric(),
    batch_size=torch.cuda.device_count() * 12,
    device=[i for i in range(torch.cuda.device_count())],
)
tester.test()


