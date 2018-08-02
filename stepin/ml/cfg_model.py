import json
import logging
# noinspection PyProtectedMember
from pydoc import locate

import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error, explained_variance_score, log_loss, roc_auc_score
from sklearn.model_selection import ParameterGrid

from stepin.ml.cv import cv_train


def save_model(model, config, custom_meta_proc=None):
    joblib.dump(model, config['model_path'])
    with open(config['model_meta_path'], 'w') as f:
        meta = dict()
        if custom_meta_proc is not None:
            custom_meta_proc(model, meta)
        json.dump(meta, f)


def load_model(config):
    model = joblib.load(config['model_path'])
    with open(config['model_meta_path'], 'r') as f:
        meta = json.load(f)
    return model, meta


class AbstractModelHelper(object):
    def __init__(self, loader, cmodel):
        self.loader = loader
        self.cmodel = cmodel

    def fit_params(self, params):
        pass

    def adjust_fit_params(self, params, x, y, val_x, val_y):
        pass

    def after_fit_proc(self, estimator, test_y, pred_y):
        pass

    def adjust_test_params(self, params, best_values):
        pass

    def custom_meta_proc(self, model, meta):
        pass


class ClassifierHelper(AbstractModelHelper):
    def after_fit_proc(self, estimator, test_y, pred_y):
        ll = np.sqrt(log_loss(test_y, pred_y))
        logging.debug('LogLoss: %s', ll)
        auc = roc_auc_score(
            test_y, pred_y
            # , average='weighted'
        )
        logging.debug('AUC: %s', auc)
        # f1 = f1_score(test_y, pred_y)
        # logging.debug('F1: %s', f1)
        # accuracy = accuracy_score(test_y, pred_y)
        # logging.debug('Acc: %s', accuracy)

        return self.regr_after_fit_proc(estimator, test_y, pred_y)

    def regr_after_fit_proc(self, estimator, test_y, pred_y):
        return None


class LbgmClfHelper(ClassifierHelper):
    def fit_params(self, params):
        # logging.debug('col_names: %s', self.cmodel.col_names)
        # logging.debug('cat_cols: %s', self.loader.cat_cols)
        cf = self.loader.cat_cols
        # logging.debug('categorical_feature: %s', cf)
        # cf = [self.cmodel.col_names.index(f) for f in cf]
        # logging.debug('categorical_feature indices: %s', cf)
        if len(cf) > 0:
            params['categorical_feature'] = cf

    def adjust_fit_params(self, params, x, y, val_x, val_y):
        if self.loader.sample_weight_proc is not None:
            params['sample_weight'] = self.loader.sample_weight_proc(x, y)
            # logging.debug('x:\n%s\nw:\n%s', y[:10], params['sample_weight'][:10])
        if val_x is not None:
            params['eval_set'] = [(val_x, val_y)]
            if self.loader.sample_weight_proc is not None:
                params['eval_sample_weight'] = [self.loader.sample_weight_proc(val_x, val_y)]

    def regr_after_fit_proc(self, estimator, test_y, pred_y):
        return estimator.n_estimators if estimator.best_iteration_ < 0 else estimator.best_iteration_

    def adjust_test_params(self, params, best_values):
        best_iteration = int(np.array(best_values).mean())
        logging.info('best_iteration = %s', best_iteration)
        params['n_estimators'] = best_iteration
        params['num_boost_round'] = best_iteration

    def custom_meta_proc(self, model, meta):
        meta['n_estimators'] = model.best_iteration_
        meta['num_boost_round'] = model.best_iteration_


class RegressionHelper(AbstractModelHelper):
    def after_fit_proc(self, estimator, test_y, pred_y):
        rmse = np.sqrt(mean_squared_error(test_y, pred_y))
        logging.debug('RMSE: %s', rmse)
        evs = explained_variance_score(test_y, pred_y)
        logging.debug('EVS: %s', evs)
        return self.regr_after_fit_proc(estimator, test_y, pred_y)

    def regr_after_fit_proc(self, estimator, test_y, pred_y):
        return None


class LbgmRegrHelper(RegressionHelper):
    def adjust_fit_params(self, params, x, y, val_x, val_y):
        params['eval_set'] = [(val_x, val_y)]
        params['categorical_feature'] = list(set(self.loader.cat_cols) & set(self.cmodel.col_names))

    def regr_after_fit_proc(self, estimator, test_y, pred_y):
        return estimator.n_estimators if estimator.best_iteration < 0 else estimator.best_iteration

    def adjust_test_params(self, params, best_values):
        best_iteration = int(np.array(best_values).mean())
        logging.info('best_iteration = %s', best_iteration)
        params['n_estimators'] = best_iteration

    def custom_meta_proc(self, model, meta):
        meta['n_estimators'] = model.best_iteration


class XgboostRegrHelper(RegressionHelper):
    def adjust_fit_params(self, params, x, y, val_x, val_y):
        params['eval_set'] = [(val_x, val_y)]

    def regr_after_fit_proc(self, estimator, test_y, pred_y):
        return estimator.n_estimators if estimator.best_iteration < 0 else estimator.best_iteration

    def adjust_test_params(self, params, best_values):
        best_iteration = int(np.array(best_values).mean())
        logging.info('best_iteration = %s', best_iteration)


class FfmRegrHelper(RegressionHelper):
    def adjust_fit_params(self, params, x, y, val_x, val_y):
        params['valid_set'] = (val_x, val_y)


class ConfiguredModel(object):
    def __init__(self, config, loader, col_names=None):
        self.col_names = loader.col_names if col_names is None else col_names
        self.loader = loader
        self.config = config
        self.klass = locate(config['klass'])
        self.klass_args = config['klass_args']
        self.helper = locate(config['helper_klass'])(loader, self)
        self.train_params = None
        self.tested_model = None

    def set_col_names(self, col_names):
        self.col_names = col_names

    def _def_fit_params(self, custom_params=None):
        params = self.config.get('default_fit', {}).copy()
        if custom_params is not None:
            params.update(custom_params)
        self.helper.fit_params(params)
        return params

    def _model_factory(self, custom_params=None):
        args = self.klass_args.copy()
        if custom_params is not None:
            args.update(custom_params)
        return self.klass(**args)

    def _train_model_factory(self):
        return self._model_factory(custom_params=self.train_params)

    def set_train_params(self, params):
        self.train_params = params

    def cv_train(self):
        return cv_train(
            lambda: self.loader.cv_iter(col_names=self.col_names), self._train_model_factory,
            lambda y_true, y_pred: - roc_auc_score(y_true, y_pred),
            fit_params=self._def_fit_params(custom_params=self.config.get('train_fit')),
            after_fit_proc=self.helper.after_fit_proc,
            fit_adjust_proc=self.helper.adjust_fit_params
        )

    def cv_train_test(self):
        score, values = self.cv_train()
        logging.info('Score = %s', score)
        test_params = {}
        self.helper.adjust_test_params(test_params, values)
        self.set_train_params(test_params)
        self.test()

    def grid_search_train(self, param_grid):
        best_params = None
        best_score = None
        best_cv_est_values = None
        for params in ParameterGrid(param_grid):
            logging.debug('Grid params: %s', params)
            self.set_train_params(params)
            score, cv_est_values = self.cv_train()
            if best_score is None or score < best_score:
                best_score = score
                best_params = params
                best_cv_est_values = cv_est_values
        self.set_train_params(None)
        logging.info('best_params = %s, best_score = %s', best_params, best_score)
        self.helper.adjust_test_params(best_params, best_cv_est_values)
        self.set_train_params(best_params)
        self.test()

    def _test_model_factory(self):
        params = self.config.get('create_test', {})
        if self.train_params is not None:
            params.update(self.train_params)
        return self._model_factory(custom_params=params)

    def test(self, need_save_model=True):
        train_x = self.loader.train_x[self.col_names]
        train_y = self.loader.train_y
        test_x = self.loader.test_x[self.col_names]
        test_y = self.loader.test_y
        fit_params = self._def_fit_params(custom_params=self.config.get('test_fit'))
        self.helper.adjust_fit_params(fit_params, train_x, train_y, None, None)
        logging.info('Training for all data ...')
        # logging.debug('param keys = %s', list(fit_params.keys()))
        model = self._test_model_factory()
        model.fit(train_x, train_y, **fit_params)
        # logging.info('Y mean: %s', test_y.mean())
        # logging.info('Y std: %s', test_y.std())
        pred_y = model.predict(test_x)
        # ll = log_loss(test_y, pred_y)
        # logging.info('LL: %s', ll)
        auc = roc_auc_score(test_y, pred_y)
        logging.debug('AUC: %s', auc)
        # f1 = f1_score(test_y, pred_y)
        # logging.info('F1: %s', f1)
        # accuracy = accuracy_score(test_y, pred_y)
        # logging.info('Acc: %s', accuracy)

        # fi = zip(col_names, model.feature_importances_)
        # fi = sorted(fi, key=lambda x: x[1], reverse=True)
        # logging.debug('Feature Importances: %s', fi)
        self.tested_model = model

        def _custom_meta_proc(model_, meta):
            meta['col_names'] = self.col_names
            self.helper.custom_meta_proc(model_, meta)

        if need_save_model:
            save_model(model, self.config, custom_meta_proc=_custom_meta_proc)


def std_grid_search_train(config, data_loader_factory, target_col):
    loader = data_loader_factory(config)
    loader.load()
    logging.debug('feature_names: %s', loader.get_feature_map().feature_names())  # todo remove
    mparams = config['model_params']
    cmodel = ConfiguredModel(mparams, loader, target_col)
    cmodel.grid_search_train(mparams['grid'])
    return loader, cmodel
