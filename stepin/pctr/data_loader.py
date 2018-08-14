import logging
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from stepin.ml.ds_split import config_persistent_kfold, CrossValidationColumnStorage
from stepin.ml.encoders import LabelEncoder


# noinspection PyAttributeOutsideInit
class DataLoader(object):
    def __init__(self, config):
        self.def_num_cols = config['num_cols']
        self.def_cat_cols = config['cat_cols']

        self.target_col = 'y'
        self.config = config
        self.cvcs = None
        self.col_names = None
        self.sample_weight_proc = None

    def prepare_data(self):
        train_path = self.config['train_path']
        test_path = self.config['test_path']
        if os.path.exists(train_path) and os.path.exists(test_path):
            train = np.load(train_path)
            self.train_x = pd.DataFrame.from_records(train['x'])
            self.train_y = train['y']
            test = np.load(test_path)
            self.test_x = pd.DataFrame.from_records(test['x'])
            self.test_y = test['y']
        else:
            data = pd.read_csv(self.config['data_path'], sep=';')
            already_int = {'age', 'duration', 'campaign', 'pdays', 'previous'}
            for n in (
                'job', 'marital', 'education', 'default', 'housing', 'loan',
                'contact', 'month', 'day_of_week',
                'poutcome',
                'y'
            ):
                if n in already_int or not np.issubdtype(data.dtypes[n], np.integer):
                    print('convert ' + n)
                    data[n] = pd.factorize(data[n])[0]
            y = data.y
            # data.drop('y', axis=1, inplace=True)
            sss = StratifiedShuffleSplit(n_splits=1, train_size=0.9)
            split = sss.split(np.empty(y.shape, dtype=np.int8), y)
            train_indices, test_indices = next(iter(split))
            self.train_x = data.iloc[train_indices]
            self.train_y = y.iloc[train_indices].values
            np.savez_compressed(train_path, x=self.train_x.to_records(index=False), y=self.train_y)
            self.test_x = data.iloc[test_indices]
            self.test_y = y.iloc[test_indices].values
            np.savez_compressed(test_path, x=self.test_x.to_records(index=False), y=self.test_y)

    def _build_cs(self):
        self.prepare_data()
        self.loaded_names = set(self.train_x.columns)
        # logging.debug('Loaded columns: %s', self.loaded_names)
        self.col_names = list(self.train_x.columns)
        self.col_names.remove(self.target_col)

        self.cat_cols = self.def_cat_cols
        for col in self.def_cat_cols:
            encoder = LabelEncoder()
            encoder.fit(self.train_x[col])
            self.train_x[col] = encoder.transform(self.train_x[col])
            self.test_x[col] = encoder.transform(self.test_x[col])
        # for n in (
        #     'campaign', 'pdays', 'previous',
        #     'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'nr.employed',
        # ):
        #     if np.issubdtype(self.train_x.dtypes[n], np.integer):
        #         print('convert to log: ' + n)
        #         nl = n + '_log'
        #         self.train_x[nl] = np.log(self.train_x[n])
        #         self.train_x.drop(n, axis=1, inplace=True)
        #         self.test_x[nl] = np.log(self.test_x[n])
        #         self.test_x.drop(n, axis=1, inplace=True)
        #         self.col_names.remove(n)
        #         self.col_names.append(nl)

    def _build_weights(self, target):
        target_vals, target_counts = np.unique(target, return_counts=True)
        # logging.debug('target: %s', target_vals)
        weights = self.config.get('class_weights')
        if weights is None:
            ww = np.ones(target_vals.shape[0], dtype=np.float32)
            classes = target_vals
        else:
            ww = np.array(weights['weights'])
            classes = np.array(weights['classes'])
        class_weights = pd.Series(ww, index=classes)
        # logging.debug('class_weights:\n%s', class_weights)
        self.sample_weight_proc = lambda x_, y_: class_weights.get(y_).values

    def load(self):
        self._build_cs()
        self._build_weights(self.train_y)
        if 'cv_split' in self.config:
            logging.info('Building CV split')
            cv_split = config_persistent_kfold(self.config['cv_split'])
            cv_split.save_if_empty(y=self.train_y)
            self.cvcs = CrossValidationColumnStorage(cv_split)

    def feature_sets_iter(self, col_names=None):
        if col_names is None:
            col_names = self.col_names
        return self.cvcs.feature_sets_iter(
            self.train_x,
            dict(
                x=(col_names, 'df'),
                y=([self.target_col], 'dense'),
            )
        )

    def cv_iter(self, col_names=None):
        for train_df, test_df in self.feature_sets_iter(col_names=col_names):
            yield train_df['x'], np.ravel(train_df['y']), test_df['x'], np.ravel(test_df['y'])
