import os
import json
import numpy as np
import pandas as pd
import threadpool
import settings

from random import randint, shuffle

LABELS = ['forwards', 'backwards', 'right', 'left', 'walking_upstairs', 'walking_downstairs']
n_class = len(LABELS)

class DataLoader:

    def __init__(self):
        self.train_ptr = 0
        self.test_ptr = 0
        self.mean = 0
        self.std = 1
        self.addresses = []
        for info in os.walk('./pro_data/'):
            dir = info[0]
            end_dir = dir.strip('/').split('/')[-1]
            for fname in info[2]:
                label =[int(i == LABELS.index(end_dir)) for i in range(n_class)]
                path = dir + '/' + fname
                self.addresses.append({'path': path, 'label': label})
        self.train_addresses = [self.addresses[i] for i in range(0, len(self.addresses), 2)]
        self.test_addresses = [self.addresses[i] for i in range(1, len(self.addresses), 2)]
        shuffle(self.addresses)
        # df = pd.DataFrame()
        # for address in self.addresses:
        #     df = df.append(pd.read_csv(address['path']), sort=False)
        # self.mean = df.mean()
        # self.std = df.std()

    def load_df(self, item, xs, ys):
        # sequence = pd.read_csv(open(item['path'])).loc[:, ['x', 'y', 'z', 'mean', 'std', 'min', 'max', 'median', 'quater', 'triquater']]
        sequence = pd.read_csv(open(item['path'])).loc[:, ['x', 'y', 'z']]
        # sequence = (sequence - self.mean) / self.std
        xs.append(sequence.values)
        ys.append(item['label'])

    def next_batch(self, batch_size, mode='train', max_frame=settings.n_frames):
        if mode == 'train':
            choices = np.random.choice(self.train_addresses, batch_size)
        else:
            choices = np.random.choice(self.test_addresses, batch_size)
       
        xs = []
        ys = []

        args = []
        pool = threadpool.ThreadPool(5)
        for item in choices:
            args.append(([item, xs, ys], None))
        reqs = threadpool.makeRequests(self.load_df, args)
        [pool.putRequest(req) for req in reqs]
        pool.wait()

        min_len = min([len(batch) for batch in xs])
        min_s_len = min(max_frame, min_len)
        xs = [batch[randint(0, len(batch) - min_s_len - 1):] for batch in xs]
        xs = [batch[:min_s_len] for batch in xs]
        xs = np.array(xs)
        ys = np.array(ys)
        fs = np.array([len(batch) for batch in xs])
        return xs, ys, fs

import matplotlib.pyplot as plt

if __name__ == '__main__':
    loader = DataLoader()
    for i in range(10):
        x, y, f = loader.next_batch(settings.batch_size)
        plt.plot(range(f[0]), x[0])
        plt.show()
