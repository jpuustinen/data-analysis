import configparser
import numpy as np
import logging
from conv_methods import calc_lattice_mismatch
from scipy.signal import savgol_filter

WINDOW_LENGTH = 11
POLYORDER = 2

logger = logging.getLogger('General')

config_materials = configparser.ConfigParser()
config_materials.read('materials.cfg')

TENSILE_PER_ARCSEC = float(config_materials.get(
    'XRD', 'TENSILE_PER_ARCSEC'))
COMPRESSIVE_PER_ARCSEC = float(config_materials.get(
    'XRD', 'COMPRESSIVE_PER_ARCSEC'))


def smooth_data(int_meas):

    if len(int_meas) > WINDOW_LENGTH:
        int_meas_smoothed = savgol_filter(int_meas, WINDOW_LENGTH, POLYORDER)
    else:
        int_meas_smoothed = None

    return int_meas_smoothed


def calc_analysis(data):

    # sp=side peak (either side of the most intense peak)

    df_meas = data['modes']['measurement']['df']
    analysis_dict = data['modes']['analysis']['df']
    config = data['config']

    df_meas['int_lin'] = np.log10(df_meas['int'].replace(0, np.nan))
    df_meas['int_smoothed'] = smooth_data(df_meas['int'])
    df_meas['int_lin_smoothed'] = smooth_data(df_meas['int_lin'])

    peak_separation = float(
        config['GENERAL']['PEAK_SEPARATION'])

    peak_slope_arcsec = float(
        config['GENERAL']['PEAK_SLOPE'])
    peak_slope_index = int(peak_slope_arcsec / (
            df_meas['x_arcsec'].iloc[1] - df_meas['x_arcsec'].iloc[0]))

    # get arcsec value with max intensity
    analysis_dict['x_arcsec_max_int'] = df_meas['x_arcsec'][df_meas['int_smoothed'].idxmax()]

    # get arcsec value with max intensity outside PEAK_SEPARATION
    if df_meas['x_arcsec'].iloc[0] < -peak_separation \
            and df_meas['x_arcsec'].iloc[-1] > peak_separation:
        df_outside = df_meas[(abs(df_meas['x_arcsec']) > peak_separation)]

        x_arcsec_max_int_sp_raw = \
            df_outside['x_arcsec'][df_outside['int_smoothed'].idxmax()]
        if x_arcsec_max_int_sp_raw < 0:
            sign_arcsec = -1
        else:
            sign_arcsec = 1

        try:
            if df_meas['int_smoothed'][df_outside['int_smoothed'].idxmax() -
                              sign_arcsec * peak_slope_index] < \
                    df_meas['int_smoothed'][df_outside['int_smoothed'].idxmax()]:

                analysis_dict['x_arcsec_max_int_sp'] = \
                    x_arcsec_max_int_sp_raw

                analysis_dict['index_max_int_sp'] = df_outside['int_smoothed'].idxmax()
            else:
                analysis_dict['x_arcsec_max_int_sp'] = np.nan
        except KeyError:
            analysis_dict['x_arcsec_max_int_sp'] = np.nan
        else:
            pass
    else:
        print('Warning: xrd data does not cover peak separation.')
        analysis_dict['x_arcsec_max_int_sp'] = np.nan

    if analysis_dict['x_arcsec_max_int_sp'].iloc[0] < 0:
        analysis_dict['comp-%_max_int_sp'] = (
                analysis_dict['x_arcsec_max_int_sp'] *
                COMPRESSIVE_PER_ARCSEC * 0.01 * (-1))
    else:
        analysis_dict['tens-%_max_int_sp'] = (
                analysis_dict['x_arcsec_max_int_sp'] *
                TENSILE_PER_ARCSEC * 0.01)


    analysis_dict['integ'] = df_meas['int'].sum() / len(df_meas['int'])
    analysis_dict['integ_lin'] = df_meas['int_lin'].sum() / len(df_meas['int_lin'])

    analysis_dict['max_int'] = df_meas['int'].max()

    analysis_dict['max_int_lin'] = df_meas['int_lin'].max()

    if 'index_max_int_sp' in analysis_dict:

        analysis_dict['max_int_sp'] = df_outside.iloc[analysis_dict['index_max_int_sp']]['int'].iloc[0]
        analysis_dict['max_int_lin_sp'] = df_outside.iloc[analysis_dict['index_max_int_sp']]['int_lin'].iloc[0]

        if analysis_dict['max_int_lin'].iloc[0] > 0 and \
                analysis_dict['max_int_lin_sp'].iloc[0] > 0:
            analysis_dict['max_int_lin_sp/max_int'] = \
                analysis_dict['max_int_lin_sp'].iloc[0] / \
                analysis_dict['max_int_lin'].iloc[0]

        analysis_dict['max_int_sp/max_int'] = \
            analysis_dict['max_int_sp'] / analysis_dict['max_int']

    analysis_dict['lattice_mismatch'] = calc_lattice_mismatch(
        analysis_dict['x_arcsec_max_int_sp'])

    return