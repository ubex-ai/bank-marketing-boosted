# encoding=utf-8

import codecs
import gzip
import logging
import os
import shutil
import tempfile

from stepin.log_utils import safe_close, warn_exc


def atomic_file_gen(path, proc, gzipped=False, suffix="", prefix='tmp', folder=None):
    tfd, tf_path = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=folder)
    logging.debug('Build %s using temp file %s', path, tf_path)
    f = os.fdopen(tfd, 'w')
    if gzipped:
        f = gzip.GzipFile(fileobj=f, mode='w')
    try:
        proc(f)
    except:
        safe_close(f)
        try:
            if os.path.exists(tf_path):
                os.remove(tf_path)
        except:
            warn_exc()
        raise
    else:
        safe_close(f)
        if os.path.exists(path):
            os.remove(path)
        shutil.move(tf_path, path)


def check_remove(path):
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path)


def ensure_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def ensure_parent_path(path):
    path = os.path.dirname(path)
    if not os.path.exists(path):
        os.makedirs(path)


def file2str(f, encoding=None):
    if encoding is None:
        with open(f, 'rb') as of:
            return of.read()
    else:
        with codecs.open(f, 'rb', encoding=encoding) as of:
            return of.read()


def str2file(s, f, encoding=None):
    if encoding is None:
        with open(f, 'wb') as of:
            of.write(s)
            of.close()
    else:
        with codecs.open(f, 'wb', encoding=encoding) as of:
            of.write(s)
            of.close()
