import random
import copy
import os
from tqdm import tqdm

import paddle
import numpy as np

from srcs.data import WarpSampler



def evaluate_batch(dataset, model, epoch_train, batch_train, args, is_val=True, random_size=-1):
    model.eval()

    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0

    sampler = WarpSampler(train, usernum, itemnum, batch_size=batch_train, maxlen=args.maxlen, n_workers=0)

    tot_batch = 0
    num_batch = len(train) // batch_train
    for i_batch in tqdm(range(num_batch)):
        tot_batch += 1
        u, seq, pos, neg = sampler.next_batch(is_validation=True)  # tuples to ndarray
        item_indexs = []
        for user in u:
            trian_set = set(train[user])
            if random_size > 0:
                item_index = valid[user]
                for _ in range(random_size):
                    t = np.random.randint(1, itemnum + 1)
                    while t in item_index or t in trian_set or t == 0:
                        t = np.random.randint(1, itemnum + 1)
                    item_index.append(t)
            else:
                item_index = list(range(1, itemnum))
            item_indexs.append(item_index)

        predictions = -model.predict_batch(pos, item_indexs)
        rank = predictions.argsort().argsort()[:, 0].numpy()
        valid_user += len(rank)
        rank10_mask = rank < 10
        NDCG += ((1 / np.log2(rank + 2)) * rank10_mask).sum()
        HT += sum(rank10_mask)

    NDCG /= valid_user
    HT /= valid_user

    model.train()
    print('\nEpoch {} Evaluation - NDCG: {:.4f}  HIT@10: {:.4f}'.format(epoch_train,  NDCG, HT))

    return (HT, NDCG)


def evaluate(dataset, model, epoch_train, batch_train, args, is_val=True):
    model.eval()

    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    before = train
    now = valid if is_val else test

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0
    users = range(1, usernum+1)

    for u in tqdm(users):
        if len(before[u]) < 1 or len(now[u]) < 1:
            continue
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        if not is_val:
            seq[idx] = valid[u][0]
            idx -= 1
        for i in reversed(before[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(before[u])
        rated.add(0)
        item_idx = [now[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
        predictions = -model.predict(*[paddle.to_tensor(l) for l in [[seq], item_idx]])
        predictions = predictions[0]  # - for 1st argsort DESC
        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1
        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1

    NDCG /= valid_user
    HT /= valid_user

    model.train()
    print('\nEpoch {} Evaluation - NDCG: {:.4f}  HIT@10: {:.4f}'.format(epoch_train,  NDCG, HT))
    if args.log_result and is_val:
        with open(os.path.join(args.save_folder, 'result.csv'), 'a') as r:
            r.write('\n{:d},{:d},{:.4f},{:.4f}'.format(epoch_train, batch_train, NDCG, HT))
    return (HT, NDCG)
