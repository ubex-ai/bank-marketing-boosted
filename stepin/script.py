import sys
import os
import logging.config
import argparse
import codecs

from stepin.file_utils import file2str
from stepin.log_utils import error_exc

__scriptPath = os.path.abspath(os.path.dirname(sys.argv[0]))


def spath(*pp):
    return os.path.join(__scriptPath, *pp)


def ensured_dir(path):
    dp = os.path.dirname(path)
    if not os.path.exists(dp):
        os.makedirs(dp)
    return path


def ensured_path(dp):
    if not os.path.exists(dp):
        os.makedirs(dp)
    return dp


def string_exp(text, globs=None, locs=None):
    return eval(
        compile(text, '<string>', 'eval'),
        globals() if globs is None else globs,
        locals() if locs is None else locs
    )


def file_exp(path, globs=None, locs=None):
    return string_exp(
        codecs.open(path, 'r', 'utf-8').read(),
        globals() if globs is None else globs,
        locals() if locs is None else locs
    )


def script_log_config(level='DEBUG'):
    logging.config.dictConfig(dict(
        version=1,
        disable_existing_loggers=True,
        formatters=dict(
            concise=dict(
                format='%(asctime)s [%(levelname)s] %(name)s %(message)s',
                datefmt='%H:%M:%S'
            ),
        ),
        handlers=dict(
            default={
                'level': level,
                'formatter': 'concise',
                'class': 'logging.StreamHandler',
            },
        ),
        loggers={
            '': dict(
                handlers=['default'],
                level=level,
                propagate=True
            ),
        }
    ))


def std_log_config(config, appname):
    std_log = config.get('std_log_config', None)
    if std_log is not None:
        handlers = dict(
            default={
                'level': std_log.get('screen_level', 'DEBUG'),
                'formatter': 'standard',
                'class': 'logging.StreamHandler',
            },
            file={
                'level': std_log.get('file_level', 'DEBUG'),
                'formatter': 'concise',
                'class': 'logging.handlers.TimedRotatingFileHandler',
                'filename': ensured_dir(std_log['path']),
                'when': "D",
                'interval': 1,
                'backupCount': 2
            },
        )
        if 'mail' in std_log:
            handlers['smtp'] = std_log['mail']

        logging.config.dictConfig(dict(
            version=1,
            disable_existing_loggers=True,
            formatters=dict(
                standard=dict(
                    format='%(asctime)s [%(levelname)s] %(message)s'
                ),
                concise=dict(
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S'
                ),
            ),
            handlers=handlers,
            loggers={
                '': dict(
                    handlers=['default', 'file'],
                    level='DEBUG',
                    propagate=True
                ),
            }
        ))
        return
    log_config = config.get('log_config', None)
    if log_config is None:
        log_config = spath(appname + '-log.conf')
        logging.config.fileConfig(log_config)
    elif isinstance(log_config, (str, unicode)):
        logging.config.fileConfig(log_config)
    elif isinstance(log_config, dict):
        logging.config.dictConfig(log_config)
    else:
        raise Exception('Invalid log_config type: ' + str(log_config))

__configPath = None


def cpath(*pp):
    return os.path.join(__configPath, *pp)


def std_parse_options(desc, app_name=None, customize_args=None, globs=None, locs=None):
    if app_name is None:
        app_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]

    arg_parser = argparse.ArgumentParser(description=app_name + ' - ' + desc)
    arg_parser.add_argument(
        '--config', default=spath(app_name + '.conf.py'),
        help='configuration file  [default: %(default)s]'
    )
    arg_parser.add_argument(
        '--config-statements', dest='config_statements', action='store_true',
        help=(
            'Configuration file will be parsed by execfile so it can contain statements.' +
            ' Resulting configuration must be defined in config variable.' +
            ' Please be carefull using this mode as it can ruin your application by redefining crutial variables.'
        )
    )
    arg_parser.add_argument(
        '--result-path', dest='result_path', default=spath('result'),
        help='Result folder where all file results should be. ' +
             'Enables rpath function for configuration file [default: %(default)s]'
    )
    if customize_args is not None:
        customize_args(arg_parser)
    args = arg_parser.parse_args()
    global __configPath
    __configPath = os.path.abspath(os.path.dirname(args.config))
    # noinspection PyUnresolvedReferences
    if globs is None:
        globs = globals()
    if locs is None:
        locs = locals()
    locs = locs.copy()

    def fill_req(name, value):
        if name not in locs and name not in globs:
            locs[name] = value

    fill_req('spath', spath)
    fill_req('cpath', cpath)
    fill_req('ensured_path', ensured_path)
    fill_req('ensured_dir', ensured_dir)
    fill_req('include', file_exp)
    fill_req('file_text', file2str)
    import datetime as dtm
    fill_req('dtm', dtm)

    if args.result_path:
        globs['rpath'] = lambda *pp: cpath(args.result_path, *pp)
    if args.config_statements:
        execfile(
            args.config,
            globals() if globs is None else globs,
            locs
        )
        if 'config' not in locs:
            raise Exception('Configuration variable "config" has not been defined in configuration file')
        config = locs['config']
    else:
        config = file_exp(
            args.config,
            globals() if globs is None else globs,
            locs
        )
    std_log_config(config, app_name)
    if customize_args is None:
        return config
    else:
        return config, args


def safe_main(title, proc):
    try:
        config = std_parse_options(title)
        proc(config)
    except KeyboardInterrupt:
        logging.info('KeyboardInterrupt')
    except SystemExit:
        logging.info('SystemExit')
        raise
    except:
        error_exc()
