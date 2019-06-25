import os,sys
sys.path.append('../..')

import torch

from fastNLP.core import Trainer, Tester, Adam, AccuracyMetric, Const

from reproduction.matching.data.MatchingDataLoader import SNLILoader
from fastNLP.modules.encoder.embedding import ElmoEmbedding, StaticEmbedding

from reproduction.matching.model.bert import BertForNLI
from reproduction.matching.model.CIN import CINModel
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

trainer = Trainer(train_data=data_info.datasets['train'], model=model,
                  optimizer=Adam(lr=5e-4, model_params=model.parameters()),
                  batch_size=torch.cuda.device_count() * 168,
                  n_epochs=100, print_every=50,
                  dev_data=data_info.datasets['dev'],
                  metrics=AccuracyMetric(), metric_key='acc',
                  device=[i for i in range(torch.cuda.device_count())],
                  check_code_level=-1)
trainer.train(load_best_model=True)

tester = Tester(
    data=data_info.datasets['test'],
    model=model,
    metrics=AccuracyMetric(),
    batch_size=torch.cuda.device_count() * 12,
    device=[i for i in range(torch.cuda.device_count())],
)
tester.test()


