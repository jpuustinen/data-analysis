import configparser
import numpy as np
import gradients
import pandas as pd

try:
    import fit_xrd
except ModuleNotFoundError:
    pass

config_materials = configparser.ConfigParser()
config_materials.read('materials.cfg')

TENSILE_PER_ARCSEC = float(config_materials.get(
    'XRD', 'TENSILE_PER_ARCSEC'))
COMPRESSIVE_PER_ARCSEC = float(config_materials.get(
    'XRD', 'COMPRESSIVE_PER_ARCSEC'))


SUBST = float(config_materials.get('LATTICE_CONSTANTS', 'SUBST'))
COMP = float(config_materials.get('LATTICE_CONSTANTS', 'COMPRESSIVE'))
TENS = float(config_materials.get('LATTICE_CONSTANTS', 'TENSILE'))


def get_record_string(data):
    record = {}

    if 'path' not in data['file']:
        return

    path_string = data['file']['path']
    if 'num_subresult' in data['file']:
        path_string += '_#' + str(data['file']['num_subresult'])
    record['path_string'] = path_string

    line = []
    df = data['modes']['analysis']['df']
    for col in df.columns:
        record[col] = df[col].iloc[0]
    for key in record.keys():
        line.append(key)
        line.append(f'{record[key]}')

    return '\t'.join(line)


def calc_extra(data):
    df = data['modes']['analysis']['df']

    if data['type'] == 'xrdml':
        if 'concentration_fit' in data['modes']['analysis']['df']:

            df = data['modes']['analysis']['df']
            pars = {}
            pars['concentration'] = float(df['concentration_fit'].iloc[0])
            pars['thickness'] = float(df['thickness_fit'].iloc[0])
            pars['reswidth_as'] = float(df['reswidth_as_fit'].iloc[0])
            pars['cmax_log'] = float(df['cmax_log_fit'].iloc[0])
            if 'gradient_fit' in df:
                pars['gradient'] = float(df['gradient_fit'].iloc[0])
            else:
                pars['gradient'] = 0.0

            cancelled = False
            for par in pars.keys():
                if pars[par] < 0:
                    print(f'{par} is less than zero, cancelling fit simulation...')
                    cancelled = True

            if not cancelled:
                pars['material'] = df['material_fit'].iloc[0]
                try:
                    int_fit = fit_xrd.sim_xrd(
                        pars=pars, x_arcsec_array=np.array(
                            data['modes']['measurement']['df']['x_arcsec']))
                except NameError:
                    pass
                else:
                    int_fit = pd.Series(int_fit)
                    data['modes']['measurement']['df']['int_fit'] = int_fit
                    data['modes']['measurement']['df']['int_fit_lin'] = np.log10(
                        int_fit.replace(0, np.nan))


def get_records(records_raw, records):

    record_lines = records_raw.split('\n')
    for line in record_lines:
        if str.strip(line, ' '):

            temp = {}

            parts = line.split('     INFO ')
            dt = parts[0]
            line_record = parts[1]
            items = line_record.split('\t')

            for key, value in zip(items[0::2], items[1::2]):
                temp[key] = value

            if temp['path_string'] not in records:
                records[temp['path_string']] = {}

            if 'record_id' in temp:
                record_id = temp['record_id']
            else:
                record_id = '0000'

            records[temp['path_string']][record_id] = {}

            records[temp['path_string']][record_id]['record_dt'] = dt

            for key in temp.keys():
                records[temp['path_string']][record_id][key] = temp[key]

    return records


def df_from_dict(data_dict):
    filtered_dict = {}
    for key in data_dict:
        if type(data_dict[key]) != pd.DataFrame:
            filtered_dict[key] = data_dict[key]

    df = pd.DataFrame(filtered_dict, index=[0])
    return df


def get_blank_data():
    data_blank = {'sample': {}, 'file': {}, 'modes': {}}
    data_blank['modes'] = {'measurement': {}, 'analysis': {}, 'params': {}}
    return data_blank


def get_config(file_config='converters.cfg'):
    config = configparser.ConfigParser(interpolation=None)
    config.read(file_config)
    print('Using config file: ' + file_config)
    return config


def calc_lattice_mismatch(x_arcsec: pd.Series, material=None):
    conc_compressive = - COMPRESSIVE_PER_ARCSEC * x_arcsec
    conc_tensile = TENSILE_PER_ARCSEC * x_arcsec
    new_lattice_constant_comp = SUBST + (COMP - SUBST) * conc_compressive
    lattice_mismatch_comp = (new_lattice_constant_comp - SUBST) / SUBST
    new_lattice_constant_tens = SUBST + (TENS - SUBST) * conc_tensile
    lattice_mismatch_tens = (new_lattice_constant_tens - SUBST) / SUBST


    if x_arcsec.iloc[0] < 0:
        lattice_mismatch = lattice_mismatch_comp
    else:
        lattice_mismatch = lattice_mismatch_tens

    return lattice_mismatch


def _calculate_derived_parameters(data, df_sample_data):
    stoi = None

    try:
        x = data['sample']['x']
    except KeyError:
        try:
            x = data['params']['measurement']['x']
        except KeyError:
            x = None

    if x is not None and 'number' in data['sample']:
        stoi = gradients.get_stoi(data['sample']['number'], df_sample_data)

    if stoi:
        data['sample']['FR'] = gradients.fr_from_stoi(
            x, stoi, sample=data['sample']['number'])

        gradients.extra_cols_to_df(data['modes']['analysis']['df'], stoi)

    if 'number' in data['sample']:
        try:
            data['sample']['Stoi'] = float(
                df_sample_data[df_sample_data['Sample'].iloc[0] == int(
                    data['sample']['number'])]['Stoi'].iloc[0])
        except (TypeError, ValueError, KeyError):
            pass
        else:
            data['sample']['Stoi'] = data['sample']['Stoi'] - float(
                gradients.config['GENERAL']['WAFER_CENTER'])
