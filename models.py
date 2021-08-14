from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn import functional as F

class Embedding(nn.Module):
    def __init__(self, num_ent, parameter):
        super(Embedding, self).__init__()
        self.device = parameter['device']
        self.es = parameter['embed_dim']
        
        self.embedding = nn.Embedding(num_ent + 1, self.es)
        nn.init.xavier_uniform_(self.embedding.weight)


    def forward(self, triples):
        idx = [[[t[0], t[2]] for t in batch] for batch in triples]
        idx = torch.LongTensor(idx).to(self.device)
        return self.embedding(idx)

class MetaLearner(nn.Module):
    def __init__(self, K, embed_size=100, num_hidden1=500, num_hidden2=200, out_size=100, dropout_p=0.5):
        super(MetaLearner, self).__init__()
        self.embed_size = embed_size
        self.K = K
        self.out_size = out_size
        self.rel_fc1 = nn.Sequential(OrderedDict([
            ('fc',   nn.Linear(2*embed_size, num_hidden1)),
            ('bn',   nn.BatchNorm1d(K)),
            ('relu', nn.LeakyReLU()),
            ('drop', nn.Dropout(p=dropout_p)),
        ]))
        self.rel_fc2 = nn.Sequential(OrderedDict([
            ('fc',   nn.Linear(num_hidden1, num_hidden2)),
            ('bn',   nn.BatchNorm1d(K)),
            ('relu', nn.LeakyReLU()),
            ('drop', nn.Dropout(p=dropout_p)),
        ]))
        self.rel_fc3 = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(num_hidden2, out_size)),
            ('bn', nn.BatchNorm1d(K)),
        ]))
        nn.init.xavier_normal_(self.rel_fc1.fc.weight)
        nn.init.xavier_normal_(self.rel_fc2.fc.weight)
        nn.init.xavier_normal_(self.rel_fc3.fc.weight)

    def forward(self, inputs):
        size = inputs.shape
        x = inputs.contiguous().view(size[0], size[1], -1)
        x = self.rel_fc1(x)
        x = self.rel_fc2(x)
        x = self.rel_fc3(x)
        x = torch.mean(x, 1)

        return x.view(size[0], 1, 1, self.out_size)


class EmbeddingLearner(nn.Module):
    def __init__(self):
        super(EmbeddingLearner, self).__init__()

    def forward(self, h, t, r, pos_num):
        score = -torch.norm(h + r - t, 2, -1).squeeze(2)
        p_score = score[:, :pos_num]
        n_score = score[:, pos_num:]
        return p_score, n_score


class MetaTL(nn.Module):
    def __init__(self, itemnum, parameter):
        super(MetaTL, self).__init__()
        self.device = parameter['device']
        self.beta = parameter['beta']
        self.dropout_p = parameter['dropout_p']
        self.embed_dim = parameter['embed_dim']
        self.margin = parameter['margin']
        self.embedding = Embedding(itemnum, parameter)

        self.relation_learner = MetaLearner(parameter['K'] - 1, embed_size=100, num_hidden1=500,
                                                    num_hidden2=200, out_size=100, dropout_p=self.dropout_p)

        self.embedding_learner = EmbeddingLearner()
        self.loss_func = nn.MarginRankingLoss(self.margin)
        self.rel_q_sharing = dict()

    def split_concat(self, positive, negative):
        pos_neg_e1 = torch.cat([positive[:, :, 0, :],
                                negative[:, :, 0, :]], 1).unsqueeze(2)
        pos_neg_e2 = torch.cat([positive[:, :, 1, :],
                                negative[:, :, 1, :]], 1).unsqueeze(2)
        return pos_neg_e1, pos_neg_e2

    def forward(self, task, iseval=False, curr_rel=''):
        # transfer task string into embedding
        support, support_negative, query, negative = [self.embedding(t) for t in task]

        K = support.shape[1]              # num of K
        num_sn = support_negative.shape[1]  # num of support negative
        num_q = query.shape[1]              # num of query
        num_n = negative.shape[1]           # num of query negative

        rel = self.relation_learner(support)
        rel.retain_grad()

        rel_s = rel.expand(-1, K+num_sn, -1, -1)

        if iseval and curr_rel != '' and curr_rel in self.rel_q_sharing.keys():
            rel_q = self.rel_q_sharing[curr_rel]
        else:
            sup_neg_e1, sup_neg_e2 = self.split_concat(support, support_negative)

            p_score, n_score = self.embedding_learner(sup_neg_e1, sup_neg_e2, rel_s, K)

            y = torch.Tensor([1]).to(self.device)
            self.zero_grad()
            loss = self.loss_func(p_score, n_score, y)
            loss.backward(retain_graph=True)

            grad_meta = rel.grad
            rel_q = rel - self.beta*grad_meta


            self.rel_q_sharing[curr_rel] = rel_q

        rel_q = rel_q.expand(-1, num_q + num_n, -1, -1)

        que_neg_e1, que_neg_e2 = self.split_concat(query, negative)  
        p_score, n_score = self.embedding_learner(que_neg_e1, que_neg_e2, rel_q, num_q)

        return p_score, n_score