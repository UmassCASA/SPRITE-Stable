import logging
import time
import functools

from model_compare.entities.constants.Constants import Constants


def timing_decorator(attr_name: str = Constants.DECORATOR_TIMING_FILE_NAME_EXECUTIOON.value):
    def decorator(method):
        @functools.wraps(method)
        def wrapper(*args, **kwargs):
            logging.debug(">>> calling timing decorator <<<")
            logging.debug(f"method: {method.__name__}")
            logging.debug(f"attr_name: {attr_name}")
            start_time = time.time()
            result = method(*args, **kwargs)
            elapsed = time.time() - start_time

            # first param is self，corresponding to each children class
            cls = args[0].__class__
            # init time
            if not hasattr(cls, attr_name):
                setattr(cls, attr_name, 0.0)

            # set time
            setattr(cls, attr_name, elapsed)
            logging.debug(f"elapsed: {elapsed}")
            logging.debug(">>> timing decorator end <<<")
            return result

        return wrapper

    return decorator
