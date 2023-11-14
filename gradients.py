import numpy as np
import pandas as pd
import configparser
import logging

FILE_CONFIG = 'gradients.cfg'
config = configparser.ConfigParser()
config.read(FILE_CONFIG)
print('Using config file: ' + FILE_CONFIG)

logger = logging.getLogger('General')

config_materials = configparser.ConfigParser()
config_materials.read('materials.cfg')

SUBST = float(config_materials.get('LATTICE_CONSTANTS', 'SUBST'))
COMP = float(config_materials.get('LATTICE_CONSTANTS', 'COMPRESSIVE'))
TENS = float(config_materials.get('LATTICE_CONSTANTS', 'TENSILE'))


def get_stoi(sample, df_sample_data):
    try:
        _stoi = pd.to_numeric(
            df_sample_data[df_sample_data['Sample'] == float(
                sample)]['Stoichiometric'].iloc[0], errors='coerce')
        if pd.isna(_stoi):
            raise IndexError
    except (IndexError, KeyError):
        _stoi = np.nan

    if not _stoi or np.isnan(_stoi):
        logger.debug(f'c_xdep: Could not find stoich. for {sample}')

    if not np.isnan(_stoi):
        stoi = float(_stoi) - 25.2  # TODO from config
    else:
        stoi = None

    return stoi


def get_gradients(version=None):

    try:
        g_1 = float(config.get('GRADIENTS', '1'))
    except KeyError:
        logger.warning(f'g_1 not found, set to 0')
        g_1 = 0

    try:
        g_2 = float(config.get('GRADIENTS', '2'))
    except KeyError:
        logger.warning(f'g_2 not found, set to 0')
        g_2 = 0

    return g_1, g_2


def fr_from_stoi(x, stoi, gradients_version=None, sample=None):
    df = pd.DataFrame({'x': [x]})

    extra_cols_to_df(
        df, stoi, gradients_version=gradients_version, sample=sample)
    
    fr = df['FR'].iloc[0]
    return fr


def extra_cols_to_df(df, stoi, gradients_version=None, sample=None,
                     thickness_0=None, time=1,
                     df_sample_data=pd.DataFrame()):

    try:
        df_sample_data_sample = df_sample_data[
            df_sample_data['Sample'] == int(sample)]
    except KeyError:
        df_sample_data_sample = pd.DataFrame()

    if not df_sample_data.empty:
        if 'Thickness' in df_sample_data_sample:
            thickness_0 = df_sample_data_sample['Thickness'].iloc[0]

    if not df_sample_data.empty and 'Thickness(fit)' in df_sample_data_sample:
        thickness_0_fit = df_sample_data_sample['Thickness(fit)'].iloc[0]
    else:
        thickness_0_fit = 0


    g_1, g_2 = get_gradients(version=gradients_version)


    f_1_0 = 1
    f_1_stoi = f_1_0 * (1 + g_1 * stoi)

    f_2_0 = f_1_stoi / (1 + g_2 * stoi)

    flux_per_gr_1 = None
    flux_per_gr_1_fit = None
    if thickness_0:
        flux_per_gr_1 = f_1_0 / (thickness_0 * time)
    if thickness_0_fit:
        flux_per_gr_1_fit = f_1_0 / (thickness_0_fit * time)

    df['F_1'] = f_1_0 * (1 + g_1 * df['x'])
    df['F_2'] = f_2_0 * (1 + g_2 * df['x'])

    df['FR'] = df['F_2'] / df['F_1']

    if thickness_0:
        df['thickness_N'] = np.where(
            df['x'] > stoi, (df['F_1'] / flux_per_gr_1) * time,
            (df['F_2'] / flux_per_gr_1) * time)
    else:
        df['thickness_N'] = 0

    if thickness_0_fit:
        df['thickness_N_fit'] = np.where(
            df['x'] > stoi, (df['F_1'] / flux_per_gr_1_fit) * time,
            (df['F_2'] / flux_per_gr_1_fit) * time)

    if 'thickness_fit' in df:
        df['thickness_fit/thickness_N'] = float(df['thickness_fit'].iloc[0]) / \
                                        float(df['thickness_N'].iloc[0])
