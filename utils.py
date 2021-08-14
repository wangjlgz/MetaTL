import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict, Counter
from multiprocessing import Process, Queue
# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

# train/val/test data generation
def data_load(fname, num_sample):
    usernum = 0
    itemnum = 0
    user_train = defaultdict(list)

    # assume user/item index starting from 1
    f = open('data/%s/%s_train.csv' % (fname, fname), 'r')
    for line in f:
        u, i, t = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        user_train[u].append(i)
    f.close()

    # read in new users for testing
    user_input_test = {}
    user_input_valid = {}
    user_valid = {}
    user_test = {}


    User_test_new = defaultdict(list)
    f = open('data/%s/%s_test_new_user.csv' % (fname, fname), 'r')
    for line in f:
        u, i, t = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        User_test_new[u].append(i)
    f.close()

    for user in User_test_new:
        if len(User_test_new[user]) > num_sample:
            if random.random()<0.3:
                user_input_valid[user] = User_test_new[user][:num_sample]
                user_valid[user] = []
                user_valid[user].append(User_test_new[user][num_sample])
            else:
                user_input_test[user] = User_test_new[user][:num_sample]
                user_test[user] = []
                user_test[user].append(User_test_new[user][num_sample])
    

    return [user_train, usernum, itemnum, user_input_test, user_test, user_input_valid, user_valid]







class DataLoader(object):
    def __init__(self, user_train, user_test, itemnum, parameter):
        self.curr_rel_idx = 0
        
        self.bs = parameter['batch_size']
        self.maxlen = parameter['K']

        self.valid_user = []
        for u in user_train:
            if len(user_train[u]) < self.maxlen or len(user_test[u]) < 1: continue
            self.valid_user.append(u)
        
        self.num_tris = len(self.valid_user)

        self.train = user_train
        self.test = user_test
        
        self.itemnum = itemnum

    def next_one_on_eval(self):
        if self.curr_tri_idx == self.num_tris:
            return "EOT", "EOT"

        u = self.valid_user[self.curr_tri_idx]

        self.curr_tri_idx += 1
        
        seq = np.zeros([self.maxlen], dtype=np.int32)
        pos = np.zeros([self.maxlen - 1], dtype=np.int32)
        neg = np.zeros([self.maxlen - 1], dtype=np.int32)
        
        idx = self.maxlen - 1

        ts = set(self.train[u])
        for i in reversed(self.train[u]):
            seq[idx] = i
            if idx > 0:
                pos[idx - 1] = i
                if i != 0: neg[idx - 1] = random_neq(1, self.itemnum + 1, ts)
            idx -= 1
            if idx == -1: break

        curr_rel = u
        support_triples, support_negative_triples, query_triples, negative_triples = [], [], [], []
        for idx in range(self.maxlen-1):
            support_triples.append([seq[idx],curr_rel,pos[idx]])
            support_negative_triples.append([seq[idx],curr_rel,neg[idx]])

        rated = ts
        rated.add(0)
        query_triples.append([seq[-1],curr_rel,self.test[u][0]])
        for _ in range(100):
            t = np.random.randint(1, self.itemnum + 1)
            while t in rated: t = np.random.randint(1, self.itemnum + 1)
            negative_triples.append([seq[-1],curr_rel,t])

        support_triples = [support_triples]
        support_negative_triples = [support_negative_triples]
        query_triples = [query_triples]
        negative_triples = [negative_triples]

        return [support_triples, support_negative_triples, query_triples, negative_triples], curr_rel