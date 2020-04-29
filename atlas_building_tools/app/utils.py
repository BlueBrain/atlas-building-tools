'''app utils'''
import os
import logging
import inspect
from datetime import datetime
from collections import OrderedDict
from functools import wraps
import click


REQUIRED_PATH = click.Path(
    exists=True, readable=True, dir_okay=False, resolve_path=True
)
LOG_DIRECTORY = '.'


def set_verbose(logger, verbose):
    ''' Set the verbose level for the cli '''
    logger.setLevel((logging.WARNING, logging.INFO, logging.DEBUG)[min(verbose, 2)])


class ParameterContainer(OrderedDict):
    ''' A dict class used to contain and display the parameters '''

    def __repr__(self):
        ''' Better printing than the normal OrderedDict '''
        return ', '.join(str(key) + ':' + str(val) for key, val in self.items())

    __str__ = __repr__


def log_args(logger, handler_path=None):
    ''' A decorator used to redirect logger and log arguments '''

    def set_logger(file_, logger_path=handler_path):

        if handler_path is None:
            logger_path = os.path.join(LOG_DIRECTORY, file_.__name__ + '.log')

        @wraps(file_)
        def wrapper(*args, **kw):
            logger.addHandler(logging.FileHandler(logger_path))
            param = ParameterContainer(inspect.signature(file_).parameters)
            for name, arg in zip(inspect.signature(file_).parameters, args):
                param[name] = arg
            for key, value in kw.items():
                param[key] = value
            date_str = datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')
            logger.info(f'{date_str}:{file_.__name__} args:[{param}]')
            file_(*args, **kw)

        return wrapper

    return set_logger
