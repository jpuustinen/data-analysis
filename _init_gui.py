import matplotlib
import configparser
import logging
import datetime
import os

PATH_CONFIG = 'gui.cfg'
config = configparser.ConfigParser(interpolation=None)
config.read(PATH_CONFIG)
print('Using config: ' + PATH_CONFIG)

PATH_CONFIG_PATHS = 'paths.cfg'
config_paths = configparser.ConfigParser()
config_paths.read(PATH_CONFIG_PATHS)
print('Using config (paths): ' + PATH_CONFIG_PATHS)

logger_general = None

if not logging.getLogger('General').hasHandlers():
    logger_general = logging.getLogger('General')

    if config['DEBUG']['LOGGING_LEVEL'] == 'DEBUG':
        logger_general.setLevel(logging.DEBUG)
    else:
        logger_general.setLevel(logging.INFO)

    handler_logger_general = logging.StreamHandler()
    formatter_general = logging.Formatter('%(asctime)s.%(msecs)03d %(levelname)8s %(message)s',
                                          datefmt='%Y-%m-%d %H:%M:%S')
    handler_logger_general.setFormatter(formatter_general)
    logger_general.addHandler(handler_logger_general)

    if config.get('DEBUG', 'LOG_TO_FILE', fallback=None) == 'ON':
        dir_log = config_paths.get('LOGGING', 'DIR', fallback=None)
        if dir_log:
            filename_log = (
                    'dp_log_'
                    + datetime.datetime.now().isoformat()[0:19].replace(':', '-')
                    + '.txt')
            path_log = os.path.join(dir_log, filename_log)
            handler_logger_file = logging.FileHandler(path_log)
            handler_logger_file.setFormatter(formatter_general)
            logger_general.addHandler(handler_logger_file)
            logger_general.info(f'Logging to file: {path_log}')

# matplotlib backend has to be set before using plt
backend_mpl = config.get('PLOT', 'BACKEND', fallback=None)
if backend_mpl:
    matplotlib.use(backend_mpl)
    if logger_general:
        logger_general.info(f'Using matplotlib backend: {backend_mpl}')
