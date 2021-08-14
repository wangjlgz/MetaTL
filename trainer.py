from models import *
import os
import sys
import torch
import shutil
import logging
import numpy as np


class Trainer:
    def __init__(self, data_loaders, itemnum, parameter):
        self.parameter = parameter
        # data loader
        self.train_data_loader = data_loaders[0]
        self.dev_data_loader = data_loaders[1]
        self.test_data_loader = data_loaders[2]
        # parameters
        self.batch_size = parameter['batch_size']
        self.learning_rate = parameter['learning_rate']
        self.epoch = parameter['epoch']
        self.print_epoch = parameter['print_epoch']
        self.eval_epoch = parameter['eval_epoch']
        self.device = parameter['device']

        self.MetaTL = MetaTL(itemnum, parameter)
        self.MetaTL.to(self.device)

        self.optimizer = torch.optim.Adam(self.MetaTL.parameters(), self.learning_rate)

            
    def rank_predict(self, data, x, ranks):
        # query_idx is the idx of positive score
        query_idx = x.shape[0] - 1
        # sort all scores with descending, because more plausible triple has higher score
        _, idx = torch.sort(x, descending=True)
        rank = list(idx.cpu().numpy()).index(query_idx) + 1
        ranks.append(rank)
        # update data
        if rank <= 10:
            data['Hits@10'] += 1
            data['NDCG@10'] += 1 / np.log2(rank + 1)
        if rank <= 5:
            data['Hits@5'] += 1
            data['NDCG@5'] += 1 / np.log2(rank + 1)
        if rank == 1:
            data['Hits@1'] += 1
            data['NDCG@1'] += 1 / np.log2(rank + 1)
        data['MRR'] += 1.0 / rank

    def do_one_step(self, task, iseval=False, curr_rel=''):
        loss, p_score, n_score = 0, 0, 0
        if not iseval:
            self.optimizer.zero_grad()
            p_score, n_score = self.MetaTL(task, iseval, curr_rel)
            y = torch.Tensor([1]).to(self.device)
            loss = self.MetaTL.loss_func(p_score, n_score, y)
            loss.backward()
            self.optimizer.step()
        elif curr_rel != '':
            p_score, n_score = self.MetaTL(task, iseval, curr_rel)
            y = torch.Tensor([1]).to(self.device)
            loss = self.MetaTL.loss_func(p_score, n_score, y)
        return loss, p_score, n_score

    def train(self):
        # initialization
        best_epoch = 0
        best_value = 0
        bad_counts = 0

        # training by epoch
        for e in range(self.epoch):
            # sample one batch from data_loader
            train_task, curr_rel = self.train_data_loader.next_batch()
            loss, _, _ = self.do_one_step(train_task, iseval=False, curr_rel=curr_rel)
            # print the loss on specific epoch
            if e % self.print_epoch == 0:
                loss_num = loss.item()
                print("Epoch: {}\tLoss: {:.4f}".format(e, loss_num))
            # do evaluation on specific epoch
            if e % self.eval_epoch == 0 and e != 0:
                print('Epoch  {} Validating...'.format(e))
                valid_data = self.eval(istest=False, epoch=e)

                print('Epoch  {} Testing...'.format(e))
                test_data = self.eval(istest=True, epoch=e)
                
        print('Finish')

    def eval(self, istest=False, epoch=None):
        self.MetaTL.eval()
        
        self.MetaTL.rel_q_sharing = dict()

        if istest:
            data_loader = self.test_data_loader
        else:
            data_loader = self.dev_data_loader
        data_loader.curr_tri_idx = 0

        # initial return data of validation
        data = {'MRR': 0, 'Hits@1': 0, 'Hits@5': 0, 'Hits@10': 0, 'NDCG@1': 0, 'NDCG@5': 0, 'NDCG@10': 0}
        ranks = []

        t = 0
        temp = dict()
        while True:
            # sample all the eval tasks
            eval_task, curr_rel = data_loader.next_one_on_eval()
            # at the end of sample tasks, a symbol 'EOT' will return
            if eval_task == 'EOT':
                break
            t += 1

            _, p_score, n_score = self.do_one_step(eval_task, iseval=True, curr_rel=curr_rel)

            x = torch.cat([n_score, p_score], 1).squeeze()

            self.rank_predict(data, x, ranks)

            # print current temp data dynamically
            for k in data.keys():
                temp[k] = data[k] / t
            sys.stdout.write("{}\tMRR: {:.3f}\tNDCG@10: {:.3f}\tNDCG@5: {:.3f}\tNDCG@1: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
                t, temp['MRR'], temp['NDCG@10'], temp['NDCG@5'], temp['NDCG@1'], temp['Hits@10'], temp['Hits@5'], temp['Hits@1']))
            sys.stdout.flush()

        # print overall evaluation result and return it
        for k in data.keys():
            data[k] = round(data[k] / t, 3)

        
        if istest:
            print("TEST: \tMRR: {:.3f}\tNDCG@10: {:.3f}\tNDCG@5: {:.3f}\tNDCG@1: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
                    temp['MRR'], temp['NDCG@10'], temp['NDCG@5'], temp['NDCG@1'], temp['Hits@10'], temp['Hits@5'], temp['Hits@1']))
        else:
            print("VALID: \tMRR: {:.3f}\tNDCG@10: {:.3f}\tNDCG@5: {:.3f}\tNDCG@1: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
                    temp['MRR'], temp['NDCG@10'], temp['NDCG@5'], temp['NDCG@1'], temp['Hits@10'], temp['Hits@5'], temp['Hits@1']))

        return data