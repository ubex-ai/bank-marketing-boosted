#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name="Predictor",
    version="0.1",
    author="Stepochkin Alexander",
    author_email="stepochkin@mail.ru",
    description="Predictor",
    url='file://./dist',
    install_requires=[
        'numpy', 'scipy', 'pandas', 'scikit-learn',
        # 'xgboost',
        # 'xgbfir',
        'lightgbm'
    ],
    package_dir={'': 'lib'},
    packages=find_packages(
        'lib',
        exclude=['*.test']
    ),
    scripts=[
        'pctr_train_lgbm.py',
    ],
    tests_require=["nose2", "mock"],
    test_suite='nose2.collector.collector',
    zip_safe=False,
)
