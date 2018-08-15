# coding=utf-8

import logging

import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score

from stepin.ml.cfg_model import save_model


def lgbm_adjust_fit_params(params, val_x, val_y):
    params['eval_set'] = [(val_x, val_y)]


def _lbgm_after_fit_proc(estimator, test_y, pred_y):
    rmse = np.sqrt(mean_squared_error(test_y, pred_y))
    logging.debug('RMSE: %s', rmse)
    evs = explained_variance_score(test_y, pred_y)
    logging.debug('EVS: %s', evs)
    # r2 = r2_score(test_y, pred_y)
    # logging.debug('R2: %s', r2)
    return estimator.n_estimators if estimator.best_iteration < 0 else estimator.best_iteration


def _lbgm_custom_meta_proc(model, meta):
    meta['n_estimators'] = model.best_iteration


def _dict_val(d):
    return iter(d.itervalues()).next()


def train_lgbm(config, loader):
    df_x = loader.cs.get_columns(loader.col_names, format_='df')
    df_y = loader.cs.get_columns(['Quantity'])
    train_len = int(df_x.shape[0] * config['train_rate'])
    train_df_x = df_x.iloc[:train_len]
    train_df_y = df_y[:train_len]
    test_df_x = df_x.iloc[train_len:]
    test_df_y = df_y[train_len:]
    logging.info(
        '%s %s %s %s',
        train_df_x.shape, train_df_y.shape,
        test_df_x.shape, test_df_y.shape
    )
    model = lgb.LGBMRegressor(**config['create_params'])
    y_train = np.ravel(train_df_y)
    y_test = np.ravel(test_df_y)
    fit_params = config['fit_params'].copy()
    fit_params['categorical_feature'] = loader.cat_cols
    logging.debug('train_df_x.columns: %s', train_df_x.columns)
    model.fit(
        train_df_x, y_train,
        # eval_name='test',
        eval_set=[(test_df_x, y_test)],
        **fit_params
    )
    # a = iter(model.evals_result_.itervalues()).next()
    # print(a)
    score = _dict_val(_dict_val(model.evals_result_))[model.best_iteration]
    logging.debug('Score: %s', score)
    y_pred = model.predict(test_df_x, num_iteration=model.best_iteration)
    logging.info('Train Y mean: %s', y_train.mean())
    logging.info('Train Y std: %s', y_train.std())
    logging.info('Train Pred Y mean: %s', y_pred.mean())
    logging.info('Train Pred Y std: %s', y_pred.std())
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    logging.debug('Train RMSE: %s', rmse)
    mae = mean_absolute_error(y_test, y_pred)
    logging.debug('Train MAE: %s', mae)
    evs = explained_variance_score(y_test, y_pred)
    logging.debug('Train EVS: %s', evs)

    logging.info('Test Y mean: %s', y_test.mean())
    logging.info('Test Y std: %s', y_test.std())
    pred_y = model.predict(test_df_x)
    logging.info('Test Pred Y mean: %s', pred_y.mean())
    logging.info('Test Pred Y std: %s', pred_y.std())
    rmse = np.sqrt(mean_squared_error(y_test, pred_y))
    logging.debug('Test RMSE: %s', rmse)
    mae = mean_absolute_error(y_test, pred_y)
    logging.debug('Test MAE: %s', mae)
    evs = explained_variance_score(y_test, pred_y)
    logging.debug('Test EVS: %s', evs)
    # fi = zip(col_names, model.feature_importances_)
    # fi = sorted(fi, key=lambda x: x[1], reverse=True)
    # logging.debug('Feature Importances: %s', fi)
    save_model(model, config, custom_meta_proc=_lbgm_custom_meta_proc)


# def train_xgboost2(config, loader):
#     default_fit_params = config['default_fit'].copy()
#     fit_params = default_fit_params.copy()
#     fit_params.update(config['train_fit'])
#     model_factory = lambda: xgb.XGBRegressor(**config['default'])
#     best_params, best_score, best_values = grid_search_train(
#         config['grid'], model_factory,
#         loader.cv_iter, mean_squared_error, fit_params=fit_params,
#         # validation_rate=0.1,
#         validation_adjust_proc=lgbm_adjust_fit_params,
#         after_fit_proc=_lbgm_after_fit_proc
#     )
#     # logging.debug('best_values: %s', best_values)
#     best_iteration = int(np.array(best_values).mean())
#     logging.info('best_params = %s, best_score = %s, best_iteration = %s', best_params, best_score, best_iteration)
#     best_params['n_estimators'] = best_iteration
#
#     _test_model(config, loader, model_factory, best_params, default_fit_params)

    # importance = model.get_score(importance_type='weight')
    # fi_tuples = [(k, v) for k, v in importance.iteritems()]
    # fi_tuples = sorted(fi_tuples, key=lambda fi: fi[1], reverse=True)
    # con_factory = MsConFactory(config['db_con'])
    # con = con_factory()
    # fmap.set_desc_getters(dict(
    #     Categories=SqlDescGetter(con, 'select Name from ERP.Categories where Code = %s'),
    #     SubCategories=SqlDescGetter(con, 'select Name from ERP.SubCategories where Code = %s'),
    #     Groups=SqlDescGetter(con, 'select Name from ERP.Groups where Code = %s')
    # ))
    # logging.info('Saving importances ...')
    # with codecs.open(config['imp_path'], mode='w', encoding='utf-8') as f:
    #     for fit in fi_tuples:
    #         fidx = int(fit[0][1:])
    #         finfo = fmap.find_feature(fidx)
    #         f.write(
    #             unicode(fit[1]) + u', ' + unicode(finfo[0]) + u', ' +
    #             unicode(finfo[2]) + u', ' + unicode(finfo[3]) + u'\n'
    #         )
    # # plot_importance(model, height=1)
