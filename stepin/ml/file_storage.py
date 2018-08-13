# coding=utf-8

import os
import json

import numpy as np
import scipy.sparse as sp
import pandas as pd
from sklearn.externals import joblib


def save_csr(filename, array, prefix=''):
    args = {
        prefix + 'data': array.data,
        prefix + 'indices': array.indices,
        prefix + 'indptr': array.indptr,
        prefix + 'shape': array.shape
    }
    np.savez_compressed(filename, **args)


def load_csr(filename, prefix=''):
    loader = np.load(filename)
    return sp.csr_matrix(
        (loader[prefix + 'data'], loader[prefix + 'indices'], loader[prefix + 'indptr']),
        shape=loader['shape']
    )


class FileStorage(object):
    def __init__(self, root):
        self.root = root

    def path(self, pp):
        return os.path.join(self.root, *pp)

    def path_exists(self, pp):
        return os.path.exists(self.path(pp))

    def list_items(self):
        return os.listdir(self.path([]))

    def remove(self, pp):
        os.remove(self.path(self.fix_pp(pp)))

    @staticmethod
    def fix_pp(pp):
        if isinstance(pp, (str, bytes)):
            return [pp]
        else:
            return list(pp)

    def ext_path(self, pp, ext):
        pp = self.fix_pp(pp)
        pp[-1] += '.' + ext
        return os.path.join(self.root, *pp)

    def ext_path_exists(self, pp, ext):
        return os.path.exists(self.ext_path(pp, ext))

    def _ensure_path(self, pp):
        pp = self.fix_pp(pp)
        path = self.path(pp[:-1])
        if not os.path.exists(path):
            os.makedirs(path)

    def json_path(self, pp):
        return self.ext_path(pp, 'json')

    def json_exists(self, pp):
        return os.path.exists(self.ext_path(pp, 'json'))

    def save_json(self, obj, pp):
        self._ensure_path(pp)
        with open(self.json_path(pp), 'w') as fp:
            json.dump(obj, fp)

    def load_json(self, pp):
        with open(self.json_path(pp), 'r') as fp:
            return json.load(fp)

    def try_load_json(self, pp):
        if not os.path.exists(self.json_path(pp)):
            return None
        return self.load_json(pp)

    def remove_json(self, pp):
        os.remove(self.json_path(pp))

    def csv_path(self, pp):
        return self.ext_path(pp, 'csv.gz')

    def save_csv(self, df, pp):
        self._ensure_path(pp)
        df.to_csv(self.csv_path(pp), index=False, compression='gzip')

    def load_csv(self, pp, **kwargs):
        return pd.read_csv(self.csv_path(pp), encoding='utf-8', **kwargs)

    def try_load_csv(self, pp, **kwargs):
        if not os.path.exists(self.csv_path(pp)):
            return None
        return self.load_csv(pp, **kwargs)

    def remove_csv(self, pp):
        os.remove(self.csv_path(pp))

    def hdf_path(self, pp):
        return self.ext_path(pp, 'h5')

    def save_hdf(self, df, pp):
        self._ensure_path(pp)
        df.to_hdf(self.hdf_path(pp), 'data')

    def load_hdf(self, pp):
        pd.read_hdf(self.hdf_path(pp), key='data')

    def try_load_hdf(self, pp):
        if not os.path.exists(self.hdf_path(pp)):
            return None
        return self.load_hdf(pp)

    def remove_hdf(self, pp):
        os.remove(self.hdf_path(pp))

    def npy_path(self, pp):
        return self.ext_path(pp, 'npy')

    def npz_path(self, pp):
        return self.ext_path(pp, 'npz')

    def csr_exists(self, pp):
        return os.path.exists(self.npz_path(pp))

    def save_csr(self, arr, pp, prefix=''):
        self._ensure_path(pp)
        save_csr(self.npz_path(pp), arr, prefix=prefix)

    def load_csr(self, pp, prefix=''):
        return load_csr(self.npz_path(pp), prefix=prefix)

    def remove_csr(self, pp):
        os.remove(self.npz_path(pp))

    def try_load_csr(self, pp, prefix=''):
        if not os.path.exists(self.npz_path(pp)):
            return None
        return self.load_csr(pp, prefix=prefix)

    def npz_exists(self, pp):
        return os.path.exists(self.npz_path(pp))

    def save_npz(self, pp, **kwargs):
        self._ensure_path(pp)
        np.savez(self.npz_path(pp), **kwargs)

    def load_npz(self, pp):
        return np.load(self.npz_path(pp))

    def try_load_npz(self, pp):
        if not os.path.exists(self.npz_path(pp)):
            return None
        return self.load_npz(pp)

    def remove_npz(self, pp):
        os.remove(self.npz_path(pp))

    def npy_exists(self, pp):
        return os.path.exists(self.npy_path(pp))

    def save_npy(self, arr, pp):
        self._ensure_path(pp)
        np.save(self.npy_path(pp), arr)

    def load_npy(self, pp):
        return np.load(self.npy_path(pp))

    def try_load_npy(self, pp):
        if not os.path.exists(self.npy_path(pp)):
            return None
        return self.load_npy(pp)

    def remove_npy(self, pp):
        os.remove(self.npy_path(pp))

    def save_obj(self, obj, pp):
        self._ensure_path(pp)
        joblib.dump(obj, self.path(self.fix_pp(pp)))

    def load_obj(self, pp):
        return joblib.load(self.path(self.fix_pp(pp)))

    def try_load_obj(self, pp):
        if not os.path.exists(self.path(self.fix_pp(pp))):
            return None
        return self.load_obj(pp)

    def remove_obj(self, pp):
        os.remove(self.path(self.fix_pp(pp)))
