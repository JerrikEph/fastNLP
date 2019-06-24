

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import numpy as np
from torch.nn.parameter import Parameter
from fastNLP.models import BaseModel
from fastNLP.modules.encoder.embedding import TokenEmbedding
from fastNLP.modules.encoder.lstm import LSTM
from fastNLP.core.const import Const
from fastNLP.core.utils import seq_len_to_mask

#args = type('args', (), {})()


class CINModel(BaseModel):
    def __init__(self, init_embedding: TokenEmbedding, hidden_size=None, num_labels=3, dropout_rate=0.3,
                 dropout_embed=0.1):
        super(CINModel, self).__init__()

        self.embedding = init_embedding
        self.dropout_embed = EmbedDropout(p=dropout_embed)
        if hidden_size is None:
            hidden_size = self.embedding.embed_size
        self.rnn = BiRNN(self.embedding.embed_size, hidden_size, dropout_rate=dropout_rate)
        # self.rnn = LSTM(self.embedding.embed_size, hidden_size, dropout=dropout_rate, bidirectional=True)

        self.interfere = nn.Sequential(nn.Dropout(p=dropout_rate),
                                       nn.Linear(8 * hidden_size, hidden_size),
                                       nn.ReLU())
        nn.init.xavier_uniform_(self.interfere[1].weight.data)

        self.cin_conv = CINConv(hidden_size=600, k_size=3)

        self.rnn_high = BiRNN(self.embedding.embed_size, hidden_size, dropout_rate=dropout_rate)
        # self.rnn_high = LSTM(hidden_size, hidden_size, dropout=dropout_rate, bidirectional=True)

        self.classifier = nn.Sequential(nn.Dropout(p=dropout_rate),
                                        nn.Linear(8 * hidden_size, hidden_size),
                                        nn.Tanh(),
                                        nn.Dropout(p=dropout_rate),
                                        nn.Linear(hidden_size, num_labels))
        nn.init.xavier_uniform_(self.classifier[1].weight.data)
        nn.init.xavier_uniform_(self.classifier[4].weight.data)

    def forward(self, words1, words2, seq_len1, seq_len2, target=None):
        mask1 = seq_len_to_mask(seq_len1)
        mask2 = seq_len_to_mask(seq_len2)
        a0 = self.embedding(words1)  # B * len * emb_dim
        b0 = self.embedding(words2)
        a0, b0 = self.dropout_embed(a0), self.dropout_embed(b0)
        a = self.rnn(a0, mask1.byte())  # a: [B, PL, 2 * H]
        b = self.rnn(b0, mask2.byte())

        ai, bi = self.cin_conv(a, mask1, b, mask2)

        a_ = torch.cat((a, ai, a - ai, a * ai), dim=2)  # ma: [B, PL, 8 * H]
        b_ = torch.cat((b, bi, b - bi, b * bi), dim=2)
        a_f = self.interfere(a_)
        b_f = self.interfere(b_)

        a_h = self.rnn_high(a_f, mask1.byte())  # ma: [B, PL, 2 * H]
        b_h = self.rnn_high(b_f, mask2.byte())

        a_avg = self.mean_pooling(a_h, mask1, dim=1)
        a_max, _ = self.max_pooling(a_h, mask1, dim=1)
        b_avg = self.mean_pooling(b_h, mask2, dim=1)
        b_max, _ = self.max_pooling(b_h, mask2, dim=1)

        out = torch.cat((a_avg, a_max, b_avg, b_max), dim=1)  # v: [B, 8 * H]
        logits = torch.tanh(self.classifier(out))

        if target is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, target)

            return {Const.LOSS: loss, Const.OUTPUT: logits}
        else:
            return {Const.OUTPUT: logits}

    def predict(self, **kwargs):
        return self.forward(**kwargs)

    # input [batch_size, len , hidden]
    # mask  [batch_size, len] (111...00)
    @staticmethod
    def mean_pooling(input, mask, dim=1):
        masks = mask.view(mask.size(0), mask.size(1), -1).float()
        return torch.sum(input * masks, dim=dim) / torch.sum(masks, dim=1)

    @staticmethod
    def max_pooling(input, mask, dim=1):
        my_inf = 10e12
        masks = mask.view(mask.size(0), mask.size(1), -1)
        masks = masks.expand(-1, -1, input.size(2)).float()
        return torch.max(input + masks.le(0.5).float() * -my_inf, dim=dim)


class EmbedDropout(nn.Dropout):

    def forward(self, sequences_batch):
        ones = sequences_batch.data.new_ones(sequences_batch.shape[0], sequences_batch.shape[-1])
        dropout_mask = nn.functional.dropout(ones, self.p, self.training, inplace=False)
        return dropout_mask.unsqueeze(1) * sequences_batch


class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate=0.3):
        super(BiRNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.rnn = nn.LSTM(input_size, hidden_size,
                           num_layers=1,
                           bidirectional=True,
                           batch_first=True)

    def forward(self, x, x_mask):
        # Sort x
        lengths = x_mask.data.eq(1).long().sum(1).squeeze()
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        lengths = list(lengths[idx_sort])

        x = x.index_select(0, idx_sort)
        # Pack it up
        rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        # Apply dropout to input
        if self.dropout_rate > 0:
            dropout_input = F.dropout(rnn_input.data, p=self.dropout_rate, training=self.training)
            rnn_input = nn.utils.rnn.PackedSequence(dropout_input, rnn_input.batch_sizes)
        output = self.rnn(rnn_input)[0]
        # Unpack everything
        output = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)[0]
        output = output.index_select(0, idx_unsort)
        if output.size(1) != x_mask.size(1):
            padding = torch.zeros(output.size(0),
                                  x_mask.size(1) - output.size(1),
                                  output.size(2)).type(output.data.type())
            output = torch.cat([output, padding], 1)
        return output


def masked_softmax(tensor, mask):
    tensor_shape = tensor.size()
    reshaped_tensor = tensor.view(-1, tensor_shape[-1])

    # Reshape the mask so it matches the size of the input tensor.
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(tensor).contiguous().float()
    reshaped_mask = mask.view(-1, mask.size()[-1])
    result = F.softmax(reshaped_tensor * reshaped_mask, dim=-1)
    result = result * reshaped_mask
    # 1e-13 is added to avoid divisions by zero.
    result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)
    return result.view(*tensor_shape)


def weighted_sum(tensor, weights, mask):
    w_sum = weights.bmm(tensor)
    while mask.dim() < w_sum.dim():
        mask = mask.unsqueeze(1)
    mask = mask.transpose(-1, -2)
    mask = mask.expand_as(w_sum).contiguous().float()
    return w_sum * mask



class CINConv(nn.Module):

    def __init__(self, hidden_size, k_size):
        self.pConv = InterativeConv(hidden_size, k_size)
        self.hConv = InterativeConv(hidden_size, k_size)

    @staticmethod
    def mean_pooling(input, mask, dim=1):
        masks = mask.view(mask.size(0), mask.size(1), -1).float()
        return torch.sum(input * masks, dim=dim) / torch.sum(masks, dim=1)

    @staticmethod
    def max_pooling(input, mask, dim=1):
        my_inf = 10e12
        masks = mask.view(mask.size(0), mask.size(1), -1)
        masks = masks.expand(-1, -1, input.size(2)).float()
        return torch.max(input + masks.le(0.5).float() * -my_inf, dim=dim)

    def forward(self, premise_batch, premise_mask, hypothesis_batch, hypothesis_mask):
        p_rep = self.max_pooling(premise_batch, mask=premise_mask)
        h_rep = self.max_pooling(hypothesis_batch, mask=hypothesis_mask)

        p_out = self.pConv(premise_batch, filter_rep=h_rep)
        h_out = self.hConv(hypothesis_batch, filter_rep=p_rep)

        return p_out, h_out


class InterativeConv(nn.Module):
    def __init(self, hidden_size, k_sz):
        super(InterativeConv, self).__init__()
        self.h_sz = hidden_size
        self.k_sz = k_sz
        in_features = hidden_size
        out_features = hidden_size*k_sz
        self.P = Parameter(torch.Tensor(out_features, in_features))   # shape(h_sz*k, h_sz) -> (b_sz, k*h_sz, h_sz) -> (b_sz, k, h_sz, h_sz)

        self.Q = Parameter(torch.Tensor(out_features, in_features))   # shape(k*h_sz, h_sz) -> (1, k, h_sz, h_sz) -> (b_sz, k, h_sz, h_sz)

        self.B = Parameter(torch.Tensor(k_sz, hidden_size, hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.P, a=math.sqrt(5), mode='fan_in')
        nn.init.kaiming_uniform_(self.Q, a=math.sqrt(5), mode='fan_in')
        nn.init.zeros_(self.B)

    def forward(self, inputs, filter_rep):
        '''

        :param inputs:
        :param filter_rep:
        :param k_sz:
        :return:
        '''

        kernel = self.filterGen(filter_rep=filter_rep)  # shape(b_sz, k*h_sz, h_sz)
        fan_in, fan_out = self.k_sz*self.h_sz, self.h_sz
        kernel = nn.LayerNorm([self.h_sz, self.k_sz*self.h_sz])(kernel)
        kernel = kernel/math.sqrt(fan_in)

        out = self.hyperConv(inputs, kernel, k_sz=self.k_sz)
        return out

    def filterGen(self, filter_rep):
        '''

        :param filter_rep: shape(b_sz, h_sz)
        :param k_sz:
        :return:
        '''
        b_sz = list(filter_rep.size())[0]
        z = filter_rep.view(size=[b_sz, 1, self.h_sz])
        P = self.P.view(size=[1, self.k_sz*self.h_sz, self.h_sz])
        Q = self.Q.view(size=[1, self.k_sz, self.h_sz, self.h_sz])
        B = self.B.view(size=[1, self.k_sz, self.h_sz, self.h_sz])

        Pz = P*z   # shape(b_sz, k*h, h)
        Pz = Pz.view(size=[b_sz, self.k_sz, self.h_sz, self.h_sz])
        PzQ = Pz.matmul(Q) + B
        kernel = PzQ.view(size=[b_sz, self.k_sz*self.h_sz, self.h_sz])   # shape(b_sz, k*h_sz, h_sz)
        return nn.LeakyReLU()(kernel)

    @staticmethod
    def hyperConv(inputs, kernel, k_sz):
        '''

        :param inptus: shape(b_sz, tstp, h_sz)
        :param kernel: shape(b_sz, k_sz*h_sz, o_sz)
        :param k_sz: could only be odd number
        :return:
        '''

        outs = InterativeConv.gather(inputs, k_sz)    # shape(b_sz, tstp, k_sz*h_sz)
        return torch.matmul(outs, kernel)   # shape(b_sz, tstp, o_sz)


    @staticmethod
    def gather(inputs, k_size):
        '''

        Args:
            inputs:  shape(b_sz, tstp, h_sz)
            seqLen:  shape(b_sz)
        Returns:
            ret: shape(b_sz, _tstp, k_sz*h_sz)
        '''
        b_sz, tstp, emb_sz = list(inputs.size())
        pad = torch.zeros(b_sz, 1, emb_sz)
        padidx = [i-(k_size-1)/2 for i in range(k_size)]

        collect = []
        for i in padidx:
            if i < 0:
                pdNum = np.abs(i)
                o = torch.cat([pad]*pdNum + [inputs], dim=1)
                collect.append(o[:, :, :tstp])
            elif i == 0:
                collect.append(inputs)
            else:
                pdNum = np.abs(i)
                o = torch.cat([inputs] + [pad] * pdNum, dim=1)
                collect.append(o[:, :, -tstp:])
        out = torch.cat(collect, dim=-1)
        return out