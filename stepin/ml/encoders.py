# coding=utf-8

import json
import os
import array
import itertools
import numpy as np
import pandas as pd
import scipy.sparse as sp
from collections import defaultdict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.feature_extraction import FeatureHasher

from sklearn.preprocessing.label import LabelBinarizer
from sklearn.utils.validation import check_is_fitted


def is_text(o):
    return isinstance(o, (str, bytes))


class LabelDecoder(object):
    def __init__(self, classes):
        self.classes = classes

    def __call__(self, index):
        return self.classes.index[index]


class NoFitTransformer(BaseEstimator, TransformerMixin):
    # noinspection PyUnusedLocal
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        raise NotImplementedError()


class RowSummator(NoFitTransformer):
    def __init__(self, column_indices=None):
        self.column_indices = column_indices

    def transform(self, data):
        if isinstance(data, pd.DataFrame):
            if self.column_indices is None:
                return data.sum(axis=1)
            return data[self.column_indices].sum(axis=1)
        if self.column_indices is None:
            return np.sum(data, axis=1)
        return np.sum(data[:, self.column_indices], axis=1)


class NoFitProcTransformer(NoFitTransformer):
    def __init__(self, proc, **kwargs):
        self.proc = proc
        self.kwargs = kwargs

    def transform(self, x):
        return self.proc(x, **self.kwargs)


def no_fit_proc_transformer(proc, **kwargs):
    return NoFitProcTransformer(proc, **kwargs)


def _fit_log(x):
    return np.log(x)


def log_transformer():
    return NoFitProcTransformer(_fit_log)


class NoFitByRowTransformer(NoFitTransformer):
    def __init__(self, proc):
        self.proc = proc

    def transform(self, data):
        if isinstance(data, pd.DataFrame):
            return data.apply(self.proc, axis=1)
        return np.apply_along_axis(self.proc, 1, data)


# noinspection PyAttributeOutsideInit
class MultiLabelBinarizer(BaseEstimator, TransformerMixin):
    """Transform between iterable of iterables and a multilabel format

    Although a list of sets or tuples is a very intuitive format for multilabel
    data, it is unwieldy to process. This transformer converts between this
    intuitive format and the supported multilabel format: a (samples x classes)
    binary matrix indicating the presence of a class label.

    Parameters
    ----------
    classes : array-like of shape [n_classes] (optional)
        Indicates an ordering for the class labels

    sparse_output : boolean (default: False),
        Set to true if output binary array is desired in CSR sparse format

    Attributes
    ----------
    classes_ : array of labels
        A copy of the `classes` parameter where provided,
        or otherwise, the sorted set of classes found when fitting.

    Examples
    --------
    >>> from sklearn.preprocessing import MultiLabelBinarizer
    >>> mlb = MultiLabelBinarizer()
    >>> mlb.fit_transform([(1, 2), (3,)])
    array([[1, 1, 0],
           [0, 0, 1]])
    >>> mlb.classes_
    array([1, 2, 3])

    >>> mlb.fit_transform([{'sci-fi', 'thriller'}, {'comedy'}])
    array([[0, 1, 1],
           [1, 0, 0]])
    >>> list(mlb.classes_)
    ['comedy', 'sci-fi', 'thriller']

    See also
    --------
    sklearn.preprocessing.OneHotEncoder : encode categorical integer features
        using a one-hot aka one-of-K scheme.
    """

    def __init__(self, classes=None, sparse_output=False):
        self.classes = classes
        self.sparse_output = sparse_output

    def fit(self, y):
        """Fit the label sets binarizer, storing `classes_`

        Parameters
        ----------
        y : iterable of iterables
            A set of labels (any orderable and hashable object) for each
            sample. If the `classes` parameter is set, `y` will not be
            iterated.

        Returns
        -------
        self : returns this MultiLabelBinarizer instance
        """
        if self.classes is None:
            classes = sorted(set(itertools.chain.from_iterable(y)))
        else:
            classes = self.classes
        dtype = np.int if all(isinstance(c, int) for c in classes) else object
        self.classes_ = np.empty(len(classes), dtype=dtype)
        self.classes_[:] = classes
        return self

    # noinspection PyMethodOverriding
    def fit_transform(self, y):
        """Fit the label sets binarizer and transform the given label sets

        Parameters
        ----------
        y : iterable of iterables
            A set of labels (any orderable and hashable object) for each
            sample. If the `classes` parameter is set, `y` will not be
            iterated.

        Returns
        -------
        y_indicator : array or CSR matrix, shape (n_samples, n_classes)
            A matrix such that `y_indicator[i, j] = 1` iff `classes_[j]` is in
            `y[i]`, and 0 otherwise.
        """
        if self.classes is not None:
            return self.fit(y).transform(y)

        # Automatically increment on new class
        class_mapping = defaultdict(int)
        class_mapping.default_factory = class_mapping.__len__
        yt = self._transform(y, class_mapping)

        # sort classes and reorder columns
        tmp = sorted(class_mapping, key=class_mapping.get)

        # (make safe for tuples)
        dtype = np.int if all(isinstance(c, int) for c in tmp) else object
        class_mapping = np.empty(len(tmp), dtype=dtype)
        class_mapping[:] = tmp
        self.classes_, inverse = np.unique(class_mapping, return_inverse=True)
        # ensure yt.indices keeps its current dtype
        yt.indices = np.array(inverse[yt.indices], dtype=yt.indices.dtype,
                              copy=False)

        if not self.sparse_output:
            yt = yt.toarray()

        return yt

    def transform(self, y):
        """Transform the given label sets

        Parameters
        ----------
        y : iterable of iterables
            A set of labels (any orderable and hashable object) for each
            sample. If the `classes` parameter is set, `y` will not be
            iterated.

        Returns
        -------
        y_indicator : array or CSR matrix, shape (n_samples, n_classes)
            A matrix such that `y_indicator[i, j] = 1` iff `classes_[j]` is in
            `y[i]`, and 0 otherwise.
        """
        check_is_fitted(self, 'classes_')

        class_to_index = dict(zip(self.classes_, range(len(self.classes_))))
        yt = self._transform(y, class_to_index)

        if not self.sparse_output:
            yt = yt.toarray()

        return yt

    # noinspection PyMethodMayBeStatic
    def _transform(self, y, class_mapping):
        """Transforms the label sets with a given mapping

        Parameters
        ----------
        y : iterable of iterables
        class_mapping : Mapping
            Maps from label to column index in label indicator matrix

        Returns
        -------
        y_indicator : sparse CSR matrix, shape (n_samples, n_classes)
            Label indicator matrix
        """
        indices = array.array('i')
        indptr = array.array('i', [0])
        for labels in y:
            indices.extend(set(class_mapping[label] for label in labels if label in class_mapping))
            indptr.append(len(indices))
        data = np.ones(len(indices), dtype=int)

        return sp.csr_matrix((data, indices, indptr),
                             shape=(len(indptr) - 1, len(class_mapping)))

    def inverse_transform(self, yt):
        """Transform the given indicator matrix into label sets

        Parameters
        ----------
        yt : array or sparse matrix of shape (n_samples, n_classes)
            A matrix containing only 1s ands 0s.

        Returns
        -------
        y : list of tuples
            The set of labels for each sample such that `y[i]` consists of
            `classes_[j]` for each `yt[i, j] == 1`.
        """
        check_is_fitted(self, 'classes_')

        if yt.shape[1] != len(self.classes_):
            raise ValueError('Expected indicator for {0} classes, but got {1}'
                             .format(len(self.classes_), yt.shape[1]))

        if sp.issparse(yt):
            yt = yt.tocsr()
            if len(yt.data) != 0 and len(np.setdiff1d(yt.data, [0, 1])) > 0:
                raise ValueError('Expected only 0s and 1s in label indicator.')
            return [tuple(self.classes_.take(yt.indices[start:end]))
                    for start, end in zip(yt.indptr[:-1], yt.indptr[1:])]
        else:
            unexpected = np.setdiff1d(yt, [0, 1])
            if len(unexpected) > 0:
                raise ValueError('Expected only 0s and 1s in label indicator. '
                                 'Also got {0}'.format(unexpected))
            return [tuple(self.classes_.compress(indicators)) for indicators
                    in yt]


# noinspection PyAttributeOutsideInit
class StrMultiLabelBinarizer(MultiLabelBinarizer):
    def __init__(self, classes=None, sparse_output=False, separator=','):
        super(StrMultiLabelBinarizer, self).__init__(classes=classes, sparse_output=sparse_output)
        self.separator = separator

    def _split(self, y):
        if isinstance(y, (pd.Series, pd.DataFrame)):
            # noinspection PyUnresolvedReferences
            return y.str.split(self.separator)
        else:
            return np.core.defchararray.split(y, sep=self.separator)

    def fit(self, y):
        return super(StrMultiLabelBinarizer, self).fit(self._split(y))

    def transform(self, y):
        return super(StrMultiLabelBinarizer, self).transform(self._split(y))

    def fit_transform(self, y):
        return super(StrMultiLabelBinarizer, self).fit_transform(self._split(y))


# noinspection PyAttributeOutsideInit
class MultiLabelToLabelTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, separator=','):
        self.separator = separator
        self.item0_selector = np.vectorize(lambda x: x[0])

    # noinspection PyUnusedLocal
    def fit(self, y):
        return self

    def transform(self, y):
        if isinstance(y, (pd.Series, pd.DataFrame)):
            # noinspection PyUnresolvedReferences
            return y.str.split(self.separator).apply(lambda x: x[0])
        else:
            return self.item0_selector(np.core.defchararray.split(y, sep=self.separator))


class HashingIntEncoder(NoFitTransformer):
    def __init__(self, n_features, dtype=None):
        self.n_features = n_features
        self.dtype = dtype

    def transform(self, x):
        ret = x % self.n_features
        if ret.dtype is not None and ret.dtype != self.dtype:
            ret = ret.astype(self.dtype)
        return ret


class HashingEncoder(NoFitTransformer):
    def __init__(self, n_features, dtype=np.uint8):
        self._hasher = FeatureHasher(n_features=n_features, input_type='string', alternate_sign=False, dtype=dtype)

    def transform(self, x):
        return self._hasher.transform(([item] for item in x)).indices


class HashingOneHotEncoder(NoFitTransformer):
    def __init__(self, n_features, dtype=np.uint8):
        self._hasher = FeatureHasher(n_features=n_features, input_type='string', alternate_sign=False, dtype=dtype)

    def transform(self, x):
        return self._hasher.transform(([item] for item in x))


class HashingBinarizer(NoFitTransformer):
    def __init__(self, n_features, separator, dtype=np.uint8):
        self.separator = separator
        self._hasher = FeatureHasher(n_features=n_features, input_type='string', alternate_sign=False, dtype=dtype)

    def _split(self, y):
        if isinstance(y, (pd.Series, pd.DataFrame)):
            # noinspection PyUnresolvedReferences
            return y.str.split(self.separator)
        else:
            return np.core.defchararray.split(y, sep=self.separator)

    def transform(self, x):
        return self._hasher.transform(self._split(x))


class CountEncoder(BaseEstimator, TransformerMixin):
    # noinspection PyAttributeOutsideInit
    def fit(self, y):
        if isinstance(y, pd.Series):
            s = y
        elif isinstance(y, pd.DataFrame):
            s = y[y.columns[0]]
        else:
            if y.ndim > 1:
                y = np.ravel(y)
            s = pd.Series(y)
        self.value_counts = s.value_counts()
        return self

    def transform(self, y):
        if isinstance(y, pd.DataFrame):
            y = y[y.columns[0]]
        if len(y.shape) > 1:
            y = np.ravel(y)
        res = self.value_counts.get(y, default=0)
        res.fillna(0, inplace=True)
        return res.values


# noinspection PyAttributeOutsideInit
class TargetAvgEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=None):
        self.alpha = alpha

    def fit(self, x, y):
        """Fit label encoder

        Parameters
        ----------
        x : {array-like}, shape [n_samples, n_features] or [features]
            Grouping features.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : returns an instance of self.
        """

        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=1)
        xw = x.shape[1]
        x_cols = {i: x[:, i] for i in range(xw)}
        x_cols[xw] = y
        df = pd.DataFrame(x_cols)
        grouped = df.groupby(list(range(xw)))
        size_series = grouped.size()
        if self.alpha is None:
            self.alpha = size_series.mean()
        self.global_mean_ = np.mean(y)
        self.mean_series_ = (
            grouped.sum()[xw] + self.alpha * self.global_mean_
        ) / (size_series + self.alpha)
        return self

    def transform(self, x):
        """Target average encoder.

        Parameters
        ----------
        x : {array-like}, shape [n_samples, n_features] or [features]
            Grouping features.

        Returns
        -------
        y : array-like of shape [n_samples]
        """
        if len(x.shape) == 1:
            # noinspection PyUnresolvedReferences
            averages = self.mean_series_.get(x)
        else:
            mi = pd.MultiIndex.from_arrays([x[:, i] for i in range(x.shape[1])])
            # noinspection PyUnresolvedReferences
            averages = self.mean_series_.loc[mi]

        averages.fillna(self.global_mean_, inplace=True)
        return averages.values


# noinspection PyAttributeOutsideInit
class KmeansTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, binarize_labels=True, return_distances=False, **kwargs):
        self.binarize_labels = binarize_labels
        self.return_distances = return_distances
        self.kmeans_params = kwargs

    def fit(self, y):
        self.kmeans = KMeans(**self.kmeans_params)
        self.kmeans.fit(y)
        if self.binarize_labels:
            self.binarizer = LabelBinarizer(sparse_output=True)
            self.binarizer.fit(self.kmeans.labels_)
        return self

    def transform(self, y):
        labels = self.kmeans.predict(y)
        if self.binarize_labels:
            ret_labels = self.binarizer.transform(labels)
        else:
            ret_labels = labels
        if self.return_distances:
            centroids = self.kmeans.cluster_centers_[labels]
            # noinspection PyTypeChecker
            dist = np.sum((y - centroids) ** 2, axis=1)
            if self.binarize_labels:
                dist = sp.csr_matrix(dist[:, None])
                return sp.hstack((ret_labels, dist))
            return np.hstack((
                np.expand_dims(ret_labels, axis=1),
                np.expand_dims(dist, axis=1)
            ))
        return ret_labels


class SelectTransformer(NoFitTransformer):
    def __init__(self, start_index=None, end_index=None):
        self.start_index = start_index
        self.end_index = end_index

    def transform(self, y):
        start_index = 0 if self.start_index is None else self.start_index
        end_index = y.shape[-1] if self.end_index is None else self.end_index
        return y[:, start_index: end_index]


class SelectByIndicesTransformer(NoFitTransformer):
    def __init__(self, column_indices):
        self.column_indices = column_indices

    def transform(self, data):
        if isinstance(data, pd.DataFrame):
            return data[self.column_indices]
        return data[:, self.column_indices]


class LabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, str_def_value=None, int_def_value=None):
        self.int_def_value = int_def_value
        self.str_def_value = '<UNKNOWN>' if str_def_value is None else str_def_value
        self.classes_ = None

    def fit_from_file(self, path, is_unique=True, compression='gzip', header=None, *args, **kvargs):
        classes = pd.read_csv(
            path, encoding='utf-8', compression=compression, header=header,
            *args, **kvargs
        )
        self.fit(classes, is_unique=is_unique)

    def save(self, path, compression='gzip'):
        df = pd.DataFrame(self.classes_.index, columns=['data'])
        if isinstance(df.iloc[0, 0], str):
            df.data = df.data.str.encode('utf-8')
        df.to_csv(path, compression=compression, header=False, index=False)
        if self.int_def_value is not None or self.str_def_value is not None:
            json.dump(
                dict(int_def_value=self.int_def_value, str_def_value=self.str_def_value),
                path + '-meta.json'
            )

    @staticmethod
    def load(path, *args, **kvargs):
        meta_path = path + '-meta.json'
        if os.path.exists(meta_path):
            meta = json.load(meta_path)
        else:
            meta = {}
        result = LabelEncoder(str_def_value=meta.get('str_def_value'), int_def_value=meta.get('int_def_value'))
        result.fit_from_file(*((path,) + args), **kvargs)
        return result

    def fit(self, y, is_unique=False):
        ii = pd.Index(y)
        if not is_unique:
            ii = ii.unique()
        self.classes_ = pd.Series(np.arange(len(ii)), index=ii)
        return self

    def transform(self, y):
        res = self.classes_.get(y)
        if res is None:
            res = self.classes_.get(list(y))
        res.fillna(len(self.classes_), inplace=True)
        return res.values.astype(np.uint64)

    def inverse_transform(self, y):
        values = self.classes_.index.values
        series = np.empty(len(values) + 1, dtype=values.dtype)
        series[0: len(values)] = values
        # noinspection PyUnresolvedReferences
        if np.issubdtype(values.dtype, np.integer):
            if self.int_def_value is None:
                raise Exception('No integer default value for inverse transform')
            series[-1] = self.int_def_value
        elif np.issubdtype(values.dtype, np.inexact):
            series[-1] = np.nan
        else:
            series[-1] = '<UNKNOWN>' if self.str_def_value is None else self.str_def_value
        return series[y]


class PredefinedIntervalBinner(NoFitTransformer):
    def __init__(self, origin=0.0, interval=1.0):
        self.origin = origin
        self.interval = interval

    def transform(self, y):
        # noinspection PyUnresolvedReferences
        return ((y - self.origin) / self.interval).astype(np.int64)


class IntervalBinner(BaseEstimator, TransformerMixin):
    def __init__(self, n_parts=10):
        self.n_parts = n_parts
        self.origin = None
        self.interval = None

    def fit(self, y):
        min_val = np.min(y)
        max_val = np.max(y)
        self.origin = min_val
        self.interval = (max_val - min_val) / self.n_parts
        return self

    def transform(self, y):
        return ((y - self.origin) / self.interval).astype(np.int64)


class QuantileBinner(BaseEstimator, TransformerMixin):
    def __init__(self, n_parts=10):
        self.n_parts = n_parts
        self.quantiles = None

    def fit(self, y):
        if len(y.shape) > 1:
            y = np.ravel(y)
        s = pd.Series(y)
        grid = np.linspace(0, 1, num=self.n_parts + 1)
        self.quantiles = s.quantile(grid).values
        return self

    def transform(self, y):
        if len(y.shape) > 1:
            y = np.ravel(y)
        return np.searchsorted(self.quantiles, y)


# class GroupMinMaxScaler(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         pass
#
#     def fit(self, y):
#         if not isinstance(y, pd.DataFrame):
#             y = pd.DataFrame(y)
#         g = y.groupby([0])
#         MinMaxScaler()
#         return self
#
#     def transform(self, y):
#         return
