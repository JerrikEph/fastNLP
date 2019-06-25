import os,sys
sys.path.append('../..')

import torch

from fastNLP.core import Trainer, Tester, Adam, AccuracyMetric, Const
from fastNLP.core.callback import GradientClipCallback, LRScheduler
from reproduction.matching.data.MatchingDataLoader import SNLILoader
from fastNLP.modules.encoder.embedding import ElmoEmbedding, StaticEmbedding
from reproduction.matching.model.bert import BertForNLI
from reproduction.matching.model.CIN import CINModel
from torch.optim.lr_scheduler import StepLR
import argparse


os.environ['FASTNLP_BASE_URL'] = 'http://10.141.222.118:8888/file/download/'
os.environ['FASTNLP_CACHE_DIR'] = '/remote-home/hyan01/fastnlp_caches'

argument = argparse.ArgumentParser()
argument.add_argument('--data-path', type=str, default='./data/snli')
arg = argument.parse_args()

data_info = SNLILoader().process(
    paths=arg.data_path, to_lower=True, seq_len_type='seq_len', bert_tokenizer=None,
    get_index=True, concat=False,
)


embedding = StaticEmbedding(data_info.vocabs[Const.INPUT], model_dir_or_name='en-glove-840b-300', requires_grad=True)

# model = BertForNLI(bert_dir=bert_dirs)
model = CINModel(embedding)

optimizer = Adam(lr=2e-3, model_params=model.parameters())
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

callbacks = [
    GradientClipCallback(clip_value=10), LRScheduler(scheduler)
]

trainer = Trainer(train_data=data_info.datasets['train'], model=model,
                  optimizer=optimizer,
                  batch_size=torch.cuda.device_count() * 256,
                  n_epochs=100, print_every=50,
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


