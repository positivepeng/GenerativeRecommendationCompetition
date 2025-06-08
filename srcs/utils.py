import sys
import copy
import paddle
import random
import numpy as np
from collections import defaultdict


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)

def data_partition_mine(fname):
    User = defaultdict(list)
    user_train = {}
    user_valid = {}

    ad2index = {"pad":0}
    user2index = {"pad":0}

    # assume user/item index starting from 1
    with open(fname, 'r') as f:
        for line in f:
            user, ad_list = line.rstrip().split('\t')
            ad_list = ad_list.split(' ')
            if user not in user2index:
                user2index[user] = len(user2index)
            for ad in ad_list:
                if ad not in ad2index:
                    ad2index[ad] = len(ad2index)
            user_index = user2index[user]
            User[user_index] = [ad2index[ad] for ad in ad_list]

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 2:
            user_train[user] = User[user]
            user_valid[user] = []
        else:
            user_train[user] = User[user][:-1]
            user_valid[user] = [User[user][-1]]
    return [user_train, user_valid, None, len(user2index)-1, len(ad2index)]

# train/val/test data generation
def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    with open(fname, 'r') as f:
        for line in f:
            u, i = line.rstrip().split(' ')
            u = int(u)
            i = int(i)
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]
