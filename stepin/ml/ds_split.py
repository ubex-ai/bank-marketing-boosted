# coding=utf-8

import os
# noinspection PyProtectedMember
from pydoc import locate

import numpy as np
import pandas as pd
import scipy.sparse as sp

from stepin.ml.file_storage import FileStorage


# noinspection PyPep8Naming
class PersistentOofKFold(object):
    def __init__(self, storage, main_cv_factory, oof_cv_factory):
        self.main_cv_factory = main_cv_factory
        self.oof_cv_factory = oof_cv_factory
        self.storage = storage

    def exists(self):
        return os.path.exists(self.storage.npz_path('0'))

    def save(self, X, y=None):
        cv = self.main_cv_factory()
        i = 0
        for train_idx, test_idx in cv.split(X, y=y):
            self.storage.save_npz(str(i), train=train_idx, test=test_idx)
            j = 0
            if self.oof_cv_factory is not None:
                oof_cv = self.oof_cv_factory()
                for oof_train_idx, oof_test_idx in oof_cv.split(train_idx, y=y[train_idx]):
                    oof_train_idx = train_idx[oof_train_idx]
                    oof_test_idx = train_idx[oof_test_idx]
                    self.storage.save_npz(str(i) + '_' + str(j), train=oof_train_idx, test=oof_test_idx)
                    j += 1
            i += 1

    def save_if_empty(self, X=None, y=None):
        if self.exists():
            return
        if X is None:
            X = np.zeros(len(y))
        self.save(X, y=y)

    def oof_iter(self, i):
        j = 0
        while True:
            splits = self.storage.try_load_npz(str(i) + '_' + str(j))
            if splits is None:
                break
            yield splits['train'], splits['test'],
            j += 1

    def splits(self):
        i = 0
        while True:
            splits = self.storage.try_load_npz(str(i))
            if splits is None:
                break
            yield splits['train'], splits['test']
            i += 1

    def oof_splits(self):
        i = 0
        while True:
            splits = self.storage.try_load_npz(str(i))
            if splits is None:
                break
            yield splits['train'], splits['test'], self.oof_iter(i)
            i += 1


class _CreateObject(object):
    def __init__(self, class_name, kwargs):
        self.klass = locate(class_name)
        self.kwargs = kwargs

    def __call__(self):
        return self.klass(**self.kwargs)


def config_persistent_kfold(config):
    return PersistentOofKFold(
        FileStorage(config['path']),
        _CreateObject(config['main_class'], config['main_args']),
        None if config.get('oof_class') is None else
        _CreateObject(config['oof_class'], config['oof_args'])
    )


def _build_features(names, columns, data_format):
    if data_format == 'df':
        df_cols = {}
        col_names = []
        for name, data in zip(names, columns):
            if len(data.shape) > 1:
                if data.shape[-1] > 1:
                    for ci in range(data.shape[1]):
                        col_name = name + '_' + str(ci)
                        col_names.append(col_name)
                        d = data[:, ci]
                        # # for sparse data
                        # if len(d.shape) > 1:
                        #     d = np.ravel(d)
                        df_cols[col_name] = d
                else:
                    col_names.append(name)
                    df_cols[name] = np.ravel(data)
            else:
                col_names.append(name)
                df_cols[name] = data
        ret_data = pd.DataFrame(df_cols, columns=col_names)
    elif data_format == 'dense':
        ret_data = np.column_stack(columns)
    elif data_format == 'sparse':
        ret_data = sp.hstack(columns).tocsr()
    else:
        raise Exception('Invalid data_format "' + data_format + '"')
    return ret_data


# noinspection PyPep8Naming
class CrossValidationColumnStorage(object):
    def __init__(self, fold_manager):
        super(CrossValidationColumnStorage, self).__init__()
        self.fold_manager = fold_manager

    @staticmethod
    def _train_name(i):
        return '_' + str(i) + '_train'

    @staticmethod
    def _test_name(i):
        return '_' + str(i) + '_test'

    def feature_iter(self, storage, name):
        data = storage[name]
        for train_idx, test_idx in self.fold_manager.splits():
            yield data.iloc[train_idx], data.iloc[test_idx]

    def features_iter(self, storage, names, data_format):
        iters = []
        for name in names:
            iters.append(self.feature_iter(storage, name))
        while True:
            split = [next(it) for it in iters]
            train_split, test_split = zip(*split)
            train = _build_features(names, train_split, data_format)
            test = _build_features(names, test_split, data_format)
            yield train, test

    def feature_sets_iter(self, storage, name_sets):
        iter_sets = {}
        for sname, (names, data_format) in name_sets.items():
            iters = []
            for name in names:
                iters.append(self.feature_iter(storage, name))
            iter_sets[sname] = iters
        while True:
            train = {}
            test = {}
            for sname, (names, data_format) in name_sets.items():
                iters = iter_sets[sname]
                split = [next(it) for it in iters]
                train_split, test_split = zip(*split)
                train[sname] = _build_features(names, train_split, data_format)
                test[sname] = _build_features(names, test_split, data_format)
            yield train, test
