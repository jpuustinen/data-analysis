import configparser
import copy
import datetime
import importlib
import logging
import os
import re
import sys
from tkinter import filedialog
from tkinter import messagebox
import numpy as np
import pandas as pd

from _init_gui import config_paths, PATH_CONFIG_PATHS

reloading = False
if 'gui_methods' in sys.modules:
    reloading = True

import gradients
import calc_sample_params
import corrections
import value_recorder
import gui_methods
import calc_xrd
import conv_methods

MODS_TO_RELOAD = [gradients, calc_sample_params, corrections,
                  value_recorder, gui_methods, calc_xrd, conv_methods]

logger = logging.getLogger('General')


if reloading:
    for mod in MODS_TO_RELOAD:
        try:
            importlib.reload(mod)
            logger.info(f'Reloaded: {mod.__name__}')
        except NameError:
            logger.debug(f'Cannot reload module')
        try:
            converters.reload()
        except NameError:
            logger.debug(f'Cannot reload converters')
else:
    sample_list_dfs = {}

df_sample_data = pd.DataFrame()
df_sample_data_chips = pd.DataFrame()

global data_defaults
global data_defaults_analysis

FILE_CONFIG = 'converters.cfg'
config_main = conv_methods.get_config(FILE_CONFIG)
config_paths.read(PATH_CONFIG_PATHS)

paths_records_fit = config_paths.get(
    'VALUE_RECORDER', 'PATH_RECORDS_FIT', fallback='').split(';')

paths_records_conv = config_paths.get(
    'VALUE_RECORDER', 'PATH_RECORDS_CONV', fallback='').split(';')

start = datetime.datetime.now()
records_fit = {}
for path_records_fit in paths_records_fit:
    if path_records_fit:
        if not os.path.exists(path_records_fit):
            if messagebox.askyesno(
                    'Confirm',
                    f'{path_records_fit} does not exist. Create file?'):
                open(path_records_fit, 'a').close()

    if os.path.exists(path_records_fit):
        with open(path_records_fit, 'r') as f:
            conv_methods.get_records(f.read(), records_fit)
    else:
        logger.warning(f'Records path does not exist: {path_records_fit}')

records_conv = {}
for path_records_conv in paths_records_conv:
    if path_records_conv:
        if not os.path.exists(path_records_conv):
            if messagebox.askyesno(
                    'Confirm',
                    f'{path_records_conv} does not exist. Create file?'):
                open(path_records_conv, 'a').close()

    if os.path.exists(path_records_conv):
        with open(path_records_conv, 'r') as f:
            conv_methods.get_records(f.read(), records_conv)
    else:
        logger.warning(f'Records path does not exist: {path_records_conv}')

total = datetime.datetime.now() - start
logger.debug(f'Read records in: {total.microseconds} microseconds.')


def expand_envs_in_paths(path):

    if path:
        match = re.match('<(.*)>', path)
        if match:
            return str.replace(path, match[0], os.getenv(match[1]))
        else:
            return path
    else:
        return None


def main_init():

    get_sample_data()


def get_sample_data():

    global df_sample_data_chips
    global df_sample_data
    global data_defaults
    global data_defaults_analysis
    global sample_list_dfs


    # ------------ Get sample data from SAMPLE_LISTS defined in config -----

    path_strings_sample_list = []
    for i in range(1, 20):
        try:
            path_strings_sample_list.append(
                config_paths['CONVERTERS']['SAMPLE_LIST_' + str(i)])
        except KeyError:
            break

    dfs_sample_data = []
    df_part = pd.DataFrame()
    for path_string_sample_list in path_strings_sample_list:
        path_string_sample_list_list = path_string_sample_list.split(';')
        path_sample_list = path_string_sample_list_list[0]
        if len(path_string_sample_list_list) > 1:
            machine_string = path_string_sample_list_list[1]
        else:
            machine_string = 'N/A'

        # do not specify "#" as comment characters for csv saved from excel
        if path_sample_list != "" and os.path.exists(path_sample_list):

            if path_sample_list in sample_list_dfs and \
                sample_list_dfs[path_sample_list]['mdate'] == \
                os.path.getmtime(path_sample_list):
                logger.info(f'Data from {path_sample_list} already loaded.')
                df_part = sample_list_dfs[path_sample_list]['df_part']
            else:
                logger.debug(f'Loading data from {path_sample_list}')

                extension = os.path.splitext(path_sample_list)[1]
                if extension == '.xlsx':
                    # TODO check if Date column exists
                    try:
                        df_part = pd.read_excel(path_sample_list,
                                                parse_dates=['Date'])
                    except ValueError:
                        df_part = pd.read_excel(path_sample_list)
                elif extension == '.tab':
                    df_part = pd.read_csv(path_sample_list, sep='\t')
                elif extension == '.csv':
                    df_part = pd.read_csv(path_sample_list)


            if not df_part.empty:

                sample_list_dfs[path_sample_list] = {}
                sample_list_dfs[path_sample_list]['df_part'] = df_part
                sample_list_dfs[path_sample_list]['mdate'] = (
                    os.path.getmtime(path_sample_list))

                df_part['machine'] = machine_string
                dfs_sample_data.append(df_part)
                df_part.columns = df_part.columns.str.replace('sample', 'Sample')
            else:
                logger.warning(
                    f'Could not get sample data from {path_sample_list}')
        else:
            logger.warning(
                f'Sample list file not found: {path_sample_list}')

    list_headings = []
    for df_sample_data in dfs_sample_data:
        list_headings.extend(df_sample_data.columns)

    list_headings = [i for i in list_headings if i != 'Sample']

    if len(list_headings) > len(set(list_headings)):
        logger.warning('Defined SAMPLE_LISTS contain overlapping data')

    df_temp = pd.DataFrame()

    for i, df in enumerate(dfs_sample_data):

        df_previous = df.set_index('Sample')

        if i == 0:
            df_temp = df_previous
        else:
            df_temp = df_temp.combine_first(df_previous)

    df_temp = df_temp.reset_index()

    string_columns = config_main.get('GENERAL', 'STRING_COLUMNS', fallback=None)
    if string_columns:
        string_columns = string_columns.split(';')
    else:
        string_columns = []

    if not df_temp.empty:
        for col in df_temp.columns:
            if df_temp[col].dtype in (str, object):
                try:
                    if df_temp[col][0].endswith('%'):
                        df_temp[col] = df_temp[col].str.rstrip('%')
                        if col not in string_columns:
                            df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')
                            df_temp[col] = df_temp[col] / 100
                except AttributeError:
                    pass
                if col not in string_columns:
                    df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')

    df_sample_data = df_temp


def update_dict_from_config_section(_dict, name_section, config):
    try:
        for option in config.options(name_section):
            if len(str.split(option, '_')) > 1:
                subsection = str.split(option, '_')[0]
                key = str.split(option, '_')[1]
                if subsection not in _dict:
                    _dict[subsection] = {}
                _dict[subsection][key] = config[name_section][option]
            else:
                _dict[option] = config[name_section][option]
    except configparser.NoSectionError:
        logger.warning(f'Section {name_section} not in config.')
        pass


def _get_meas_mode_and_config(data):

    if 'modes' not in data:
        data['modes'] = {}

    for subsection in config_main.sections():
        match = re.match('MODE_' + '([^_]*)$', subsection)
        if match:
            if match[1] not in data['modes']:
                data['modes'][match[1]] = {}

            if 'defaults' not in data['modes'][match[1]]:
                data['modes'][match[1]]['defaults'] = {}

            update_dict_from_config_section(
                data['modes'][match[1]]['defaults'], subsection, data['config'])


def _get_meas_type_and_config(data):
    # TODO read config when plotting

    data['defaults'] = {}

    converters.get_meas_type(data)

    if not 'type' in data:
        logger.warning(f'Could not determine data type')
        return

    config = data['config']

    update_dict_from_config_section(data['defaults'], 'TYPE_DEFAULT',
                                    config_main)

    if 'path' in data['file']:
        update_dict_from_config_section(
            data['defaults'], 'GENERAL', config)

    if 'MODE_measurement' in config.sections():
        data['modes']['measurement']['defaults'] = {}
        update_dict_from_config_section(
            data['modes']['measurement']['defaults'],
            'MODE_measurement', config)

    if 'MODE_analysis' in config.sections():
        data['modes']['analysis']['defaults'] = {}
        update_dict_from_config_section(
            data['modes']['analysis']['defaults'],
            'MODE_analysis', config)

    _get_meas_mode_and_config(data)


def _get_sample_parameters_from_external(data):

    global df_sample_data

    if 'number' not in data['sample']:
        logger.warning('Sample number not found.')
        return

    for key in data['sample']:
        if not type(data['sample'][key]) == pd.DataFrame and not \
                data['sample'][key] is None:
            if not pd.isnull(data['sample'][key]):
                if type(data['sample'][key]) == dict:
                    for subkey in data['sample'][key]:
                        data['modes']['analysis']['df'][f'{key}_{subkey}'] = \
                            data['sample'][key][subkey]
                else:
                    data['modes']['analysis']['df'][key] = data['sample'][key]


class Converters():

    def reload(self):
        for converter in self.converters:
            importlib.reload(converter.mod)


    def get_meas_type(self, data):

        path = data['file']['path']
        matched_convs = []

        if 'type' in data and data['type'] == 'generic':
            for conv in self.converters:
                if conv.name == 'generic':
                    matched_convs.append(conv)
                    logger.debug(f'Using generic converter')
        else:

            for conv in self.converters:
                if conv.matches(path):
                    matched_convs.append(conv)

        if not matched_convs:
            logger.warning(f'No matching converter')
            return
        elif len(matched_convs) > 1:
            logger.warning(f'More than one matching converter')
            return
        else:
            data['config'] = matched_convs[0].config
            data['type'] = matched_convs[0].name
            return matched_convs[0]


    def get_data(self, data):

        matched_conv = self.get_meas_type(data)

        datas = matched_conv.get_data(data)

        return datas


    def __init__(self):
        self.converters = get_converters()


class Converter():

    def matches(self, path):

        for path_regexp in self.path_regexps:
            match = re.match(path_regexp, path)
            if match:
                return True
            else:
                return False
        else:
            logger.debug(f'PATH-STRING not defined in config.')
            return False


    def get_data(self, data):

        data['references']['converter'] = self
        datas = self.mod.get_data(data)

        return datas

    def __init__(self, mod_conv):
        self.mod = mod_conv
        self.name = self.mod.__name__.replace('converters.', '')
        path_config = os.path.join('./converters', self.name) + '.cfg'

        self.config = configparser.ConfigParser()
        self.config.read(path_config)

        path_string_raw = self.config.get('GENERAL', 'PATH-STRING',
                                           fallback=None)
        if path_string_raw:
            self.path_strings = path_string_raw.split(';')
        else:
            self.path_strings = []

        self.path_regexps = []

        for path_string in self.path_strings:
                self.path_regexps.append(path_string.replace(
                    '*', '.*').replace('?', '.'))


def get_converters():

    dir_base = 'converters'
    files = os.listdir(dir_base)
    conv_names = []
    for file in files:
        if (os.path.isfile(os.path.join(dir_base, file))
                and file.endswith('.py')) and not '__init__' in file:
            conv_names.append(f'{dir_base}.{file}'.replace('.py', ''))

    converters = []

    for name in conv_names:
        mod_conv = importlib.import_module(name)

        logger.debug(f'Adding converter: {name}')
        converter = Converter(mod_conv)
        converters.append(converter)

    return converters


def get_data_from_source(
        path_source, data=None, path_newconfig=None, generic=False) -> list:

    if path_newconfig:
        config_main.read(path_newconfig)

    global df_sample_data

    path_source_original = path_source

    if not data:
        data = {'modes': {}}
        data['modes']['analysis'] = {}
        data['modes']['analysis']['df'] = pd.DataFrame()
        data['modes']['measurement'] = {}

        data['file'] = {}
        data['sample'] = {}
    else:
        data = copy.deepcopy(data)

    if generic:
        data['type'] = 'generic'

    if path_source:
        path_source = str.replace(path_source, '\\', '/')

        data['file']['path'] = path_source
        data['file']['name'] = os.path.basename(path_source)
        data['file']['directory'] = os.path.dirname(path_source)
        data['file']['modtime'] = datetime.datetime.fromtimestamp(
                os.path.getmtime(path_source_original)).isoformat()

    if 'measurement' not in data['modes']:
        data['modes']['measurement'] = {}
    if 'analysis' not in data['modes']:
        data['modes']['analysis'] = {}
    if 'params' not in data:
        data['params'] = {}

    if 'df' not in data['modes']['measurement']:
        data['modes']['measurement']['df'] = pd.DataFrame()
    if 'df' not in data['modes']['analysis']:
        data['modes']['analysis']['df'] = pd.DataFrame()

    calc_sample_params._get_sample_parameters_from_filename(data, refs)

    _get_meas_type_and_config(data)
    if not 'type' in data:
        return []

    conv_methods._calculate_derived_parameters(data, df_sample_data)

    df_sample_data_current = pd.DataFrame()
    try:
        df_sample_data_current = df_sample_data[
            df_sample_data['Sample'] == int(data['sample']['number'])]
    except KeyError:
        logger.warning(
            f'Could not find sample number in sample data '
            f'for {data["file"]["name"]}')
    except IndexError:
        logger.warning(
            f'Could not find sample {data["sample"]["number"]} in sample data')

    if df_sample_data_current.empty:
        df_sample_data_current = pd.concat(
            [df_sample_data_current, pd.Series(dtype=np.float64)],
            axis=0, join='outer', ignore_index=True)

    if 'SD' not in data:
       data['SD'] = {}

    try:
        for col in df_sample_data_current.columns:
            data['SD'][col] = df_sample_data_current[col].iloc[0]
            data['modes']['analysis']['df'][col] = (
                df_sample_data_current[col].iloc[0])
    except (KeyError, IndexError):
        pass

    logger.info('Getting data from: ' + data['file']['name'])
    logger.info(f'Data type detected: {data["type"]}')

    data['config_paths'] = config_paths
    data['references'] = {}
    data['references']['df_sample_data'] = df_sample_data

    add_records(data, records_fit)

    datas = converters.get_data(data)

    if datas:
        if type(datas) == list:
            datas = datas
        else:
            datas = [datas]
    else:
        datas = []


    for i, data_ in enumerate(datas):

        _get_sample_parameters_from_external(data_)

        conv_methods._calculate_derived_parameters(data_, df_sample_data)

        if not data_['type'] == 'records':

            add_records(data_, records_fit)

            record_string = conv_methods.get_record_string(data_)
            if record_string:
                value_recorder.save_record_conv(record_string)

        try:
            for col in data['modes']['analysis']['df'].columns:
                data['SL'][col] = data['modes']['analysis']['df'][col].iloc[0]
        except (KeyError, IndexError):
            pass

        order_columns([data_])

        conv_methods.calc_extra(data_)

    return datas


def add_records(data, records):
    if 'path' not in data['file']:
        logger.warning(f'Records not added (no file path)')
        return
    path_string = data['file']['path']
    if 'num_subresult' in data['file']:
        path_string += '_#' + str(data['file']['num_subresult'])
    df = data['modes']['analysis']['df']
    if path_string in records:

        record_id = list(records[path_string].keys())[-1]
        for property in records[path_string][record_id].keys():
            df[property] = records[path_string][record_id][property]
            df['record_id'] = record_id


def order_columns(datas):
    pass


def add_to_df(df_orig: pd.DataFrame, df_new: pd.DataFrame):

    return df_orig.reset_index(drop=True).combine_first(
        df_new.reset_index(drop=True))


class Refs:
    def __init__(self, pars_list):
        (self.df_sample_data,
         self.df_sample_data_chips,
         self.config,
         self.config_paths) = pars_list


main_init()

pars_list = [df_sample_data, df_sample_data_chips, config_main, config_paths]

refs = Refs(pars_list)

converters = Converters()
