# coding=utf-8

import logging

import numpy as np
from sklearn.model_selection import ParameterGrid, train_test_split


def grid_search_train(
    param_grid, estimator_factory, cv_iter_factory, score_proc, fit_params=None,
    validation_rate=None, fit_adjust_proc=None,
    after_fit_proc=None
):
    if fit_params is None:
        fit_params = {}
    best_params = None
    best_score = None
    best_cv_est_values = None
    for params in ParameterGrid(param_grid):
        logging.debug('Grid params: %s', params)
        score, cv_est_values = cv_train(
            cv_iter_factory, estimator_factory, score_proc,
            model_params=params, fit_params=fit_params, validation_rate=validation_rate,
            after_fit_proc=after_fit_proc, fit_adjust_proc=fit_adjust_proc
        )
        if best_score is None or score < best_score:
            best_score = score
            best_params = params
            best_cv_est_values = cv_est_values
    return best_params, best_score, best_cv_est_values


def cv_train(
    cv_iter_factory, estimator_factory, score_proc, model_params=None, fit_params=None,
    validation_rate=None, after_fit_proc=None, fit_adjust_proc=None
):
    if fit_params is None:
        fit_params = {}
    scores = []
    cv_est_values = []
    for train_x, train_y, test_x, test_y in cv_iter_factory():
        fparams = fit_params.copy()
        if validation_rate is None:
            fit_adjust_proc(fparams, train_x, train_y, test_x, test_y)
        else:
            test_x, val_x, test_y, val_y = train_test_split(
                test_x, test_y, test_size=validation_rate, random_state=42
            )
            fit_adjust_proc(fparams, train_x, train_y, val_x, val_y)
        estimator = estimator_factory()
        if model_params is not None:
            estimator.set_params(**model_params)
        # logging.debug('Fit params: %s', list(fparams.keys()))
        estimator.fit(train_x, train_y, **fparams)
        pred_y = estimator.predict_proba(test_x)  # todo select predict method
        pred_y = pred_y[:, 1]  # todo remove
        score = score_proc(test_y, pred_y)
        logging.debug('Split Score: %s', score)
        if after_fit_proc is not None:
            est_values = after_fit_proc(estimator, test_y, pred_y)
            cv_est_values.append(est_values)
        scores.append(score)
    score = np.array(scores).mean()
    logging.debug('Score: %s', score)
    return score, cv_est_values

# def optimize(n_jobs, verbose=0, pre_dispatch='2*n_jobs'):
#     out = Parallel(
#         n_jobs=n_jobs, verbose=verbose,
#         pre_dispatch=pre_dispatch
#     )(delayed(_fit_and_score)(clone(base_estimator), X, y, self.scorer_,
#                               train, test, self.verbose, parameters,
#                               fit_params=self.fit_params,
#                               return_train_score=self.return_train_score,
#                               return_n_test_samples=True,
#                               return_times=True, return_parameters=True,
#                               error_score=self.error_score)
#       for parameters in parameter_iterable
#       for train, test in cv_iter)
#     gcv = GridSearchCV()
