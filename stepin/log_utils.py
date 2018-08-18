import sys
import logging


def log_exc(level, m=None, *args, **kwargs):
    if m is None:
        ei = sys.exc_info()
        # m = '%s: %s' % (ei[1].__class__.__name__, ei[1])
        m = str(ei[1])
    kwargs['exc_info'] = 1
    logging.log(level, m, *args, **kwargs)


def debug_exc(m=None, *args, **kwargs):
    log_exc(logging.DEBUG, m, *args, **kwargs)


def warn_exc(m=None, *args, **kwargs):
    log_exc(logging.WARNING, m, *args, **kwargs)


def error_exc(m=None, *args, **kwargs):
    log_exc(logging.ERROR, m, *args, **kwargs)


def safe_close_call(call):
    try:
        call()
    except:
        warn_exc()


def safe_close(obj):
    if obj is None:
        return
    safe_close_call(obj.close)


def safe_stop(obj):
    if obj is None:
        return
    safe_close_call(obj.stop)
