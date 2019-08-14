import os
import json
import numpy as np
import pandas as pd
import threadpool

from scipy.fftpack import fft

LABELS = ['forwards', 'backwards', 'right', 'left', 'walking_upstairs', 'walking_downstairs']
n_class = len(LABELS)

class DataProcessor:

    def __init__(self):
        self.addresses = []
        for info in os.walk('./data/'):
            dir = info[0]
            end_dir = dir.strip('/').split('/')[-1]
            for fname in info[2]:
                label =[int(i == LABELS.index(end_dir)) for i in range(n_class)]
                path = dir + '/' + fname
                self.addresses.append({'path': path, 'label': label})
        self.train_addresses = [self.addresses[i] for i in range(0, len(self.addresses), 2)]
        self.test_addresses = [self.addresses[i] for i in range(1, len(self.addresses), 2)]
        self.all_dfs = []
        for address in self.addresses:
            df_temp = pd.read_csv(address['path'])
            df_temp = df_temp.loc[df_temp['type'] == 1, :]
            df_temp.reset_index(drop=True)
            self.all_dfs.append(df_temp)
    
    @staticmethod
    def get_autocorrelation(sequence, offset):
        mean = np.mean(sequence)
        var_sum = np.var(sequence) * len(sequence)
        ac_value = 0
        for x, y in zip(sequence[:-offset], sequence[offset:]):
            ac_value += (x - mean) * (y - mean) / var_sum
        return ac_value

    def featurize(self, interval=64):
        for i in range(len(self.all_dfs)):
            features = {
                'mean': [],
                'std': [],
                'min': [],
                'max': [],
                'median': [],
                'quater': [],
                'triquater': [],
                'integral': [],
                'fft': [],
                'autocorrelation': []
            }
            for j in range(0, len(self.all_dfs[i]), interval):
                origin = self.all_dfs[i].loc[:, ['x', 'y', 'z']].values[j:j + interval]
                time = self.all_dfs[i].loc[:, 'time'].values[j:j+interval]
                sqr = origin ** 2
                sqr_sum = np.sum(sqr, axis=-1)
                mode = np.sqrt(sqr_sum)
                features['fft'].extend(fft(mode).real)
                features['autocorrelation'].extend([DataProcessor.get_autocorrelation(mode, offset=len(mode)//2) for _ in range(interval)])
                features['mean'].extend([np.mean(mode) for _ in range(interval)])
                features['std'].extend([np.std(mode) for _ in range(interval)])
                features['min'].extend([mode.min() for _ in range(interval)])
                features['max'].extend([mode.max() for _ in range(interval)])
                integral = 0
                for (t0, m0), (t1, m1) in zip([(t, m) for t, m in zip(time, mode)][:-1], [(t, m) for t, m in zip(time, mode)][1:]):
                    integral += (0.5 * (m0 + m1) * (t1 - t0)) * 10e-9
                features['integral'].extend([integral for _ in range(interval)])
                mode.sort()
                features['median'].extend([mode[len(mode)//2] for _ in range(interval)])
                features['quater'].extend([mode[len(mode)//4] for _ in range(interval)])
                features['triquater'].extend([mode[3*len(mode)//4] for _ in range(interval)])
            for key in features:
                while len(features[key]) > len(self.all_dfs[i]):
                    features[key].pop()
            for key in features:
                self.all_dfs[i].insert(0, key, features[key])
    
    def commit(self):
        for df, addr in zip(self.all_dfs, self.addresses):
            new_path = addr['path']
            new_path = new_path.split('/')
            new_path[1] = 'pro_data'
            new_path = '/'.join(new_path)
            df.to_csv(new_path, encoding='utf-8')

processor = DataProcessor()
processor.featurize(interval=16)
processor.commit()
