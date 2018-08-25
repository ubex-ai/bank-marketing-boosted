dict(
  std_log_config = dict(
    path = rpath('logs', app_name + '.log'),
    screen_level = 'DEBUG'
  ),
  data_path = cpath('../../../data/bank-additional/bank-additional-full.csv'),
  train_path = rpath('train-rf.npz'),
  test_path = rpath('test-rf.npz'),
  cat_cols=[
    'job', 'marital', 'education', 'default', 'housing', 'loan',
    'contact', 'month', 'day_of_week',
    'poutcome',
  ],
  num_cols=[
    'age',
    'duration', 'campaign', 'pdays', 'previous',
    'emp.var.rate', 'cons.price.idx',
    'cons.conf.idx', 'euribor3m', 'nr.employed'
  ],
  cv_split = dict(
    path = rpath('cv_split'),
    main_class = 'sklearn.model_selection.StratifiedKFold',
    main_args = dict(
      n_splits = 3,
      shuffle = True,
    ),
  ),
  class_weights = dict(
    weights = [1.0, 9.0],
    classes = [0, 1],
  ),
  model_params = dict(
    klass = 'lightgbm.LGBMClassifier',
    klass_args = dict(
      objective = 'binary',
      # boosting = 'dart',
      # boosting = 'goss',
      # boosting = 'rf',
      # n_estimators = 300,
      n_jobs = -1,
    ),
    helper_klass='stepin.ml.cfg_model.LbgmClfHelper',
    grid=dict(
      # num_boost_round = [300],
      num_leaves=[31],
      learning_rate=[0.1],
      # reg_alpha=[1.0],
      reg_lambda=[1.0],
      # cat_l2=[10.0],
      # cat_smooth=[10.0],
      # max_cat_threshold=[32],
      # subsample=[1.0],
      # colsample_bytree=[1.0],
    ),
    default_fit=dict(
      verbose=True,
    ),
    train_fit=dict(
      # eval_metric='binary_logloss',
      eval_metric='auc',
      early_stopping_rounds=10,
    ),
    test_fit=dict(
      verbose=True,
    ),
    model_path=rpath('model.pkl'),
    model_meta_path=rpath('model_meta.json'),
    feature_importance_path=rpath('feature_imp.json'),
    # test_y_path=rpath('test_y.csv.gz'),
  ),
)
