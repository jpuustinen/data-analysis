import pandas as pd
import re
import math
import copy
import logging
import calc_xrd

logger = logging.getLogger('General')


def get_data(data):

    datas = []

    with open(data['file']['path']) as f:
        data_raw_all = f.read()

    # get all measurements (xy map files have multiple)
    xrdmeasurements = data_raw_all.split('xrdMeasurement ')

    for i, data_raw in enumerate(xrdmeasurements[1:]):

        data['references'].pop('converter')
        return_dict = copy.deepcopy(data)

        logger.info(f'Importing subdata '
                    f'{i + 1}/{len(xrdmeasurements[1:])} '
                    f'from {return_dict["file"]["name"]}')

        return_dict['file']['num_subresult'] = i + 1

        return_dict['file']['path_string'] = \
            return_dict['file']['path'] + '_#'\
            + str(return_dict['file']['num_subresult'])

        if len(xrdmeasurements) > 2:
            return_dict['file']['path_parent'] = return_dict['file']['path']

        if 'params' not in return_dict:
            return_dict['params'] = {}
        return_dict['params']['measurement'] = {}

        match = re.search('Diffractometer system=(.*)<', data_raw)
        if match:
            name_system = match[1]
        else:
            name_system = 'N/A'

        match = re.search('positions.*\n.*<commonCountingTime.*>(.*)<',
                          data_raw)
        param_steptime = float(match[1])

        match = re.search('intensities.*>(.*)<', data_raw)
        data_intensities_raw = match[1]
        list_intensities_string = data_intensities_raw.split(' ')
        list_intensities_int = [int(i) for i in list_intensities_string]
        list_intensities_int = [
            i / param_steptime for i in list_intensities_int]
        points_amount = len(list_intensities_int)

        list_intensities_int_lin = [
            math.log10(i) if i > 0 else float('nan')
            for i in list_intensities_int]

        match = re.search('axis="X".*\n.*.*<commonPosition>(.*)<', data_raw)
        position_x = float(match[1])
        match = re.search('axis="Y".*\n.*.*<commonPosition>(.*)<', data_raw)
        position_y = float(match[1])

        match = re.search('Omega.*\n.*<startPosition>(.*)<', data_raw)
        start_omega_str = match[1]
        start_omega = float(start_omega_str)
        match = re.search('Omega.*\n.*\n.*<endPosition>(.*)<', data_raw)
        end_omega_str = match[1]
        end_omega = float(end_omega_str)

        size_step = (end_omega - start_omega) / (points_amount - 1)

        list_positions_deg = [start_omega + i * size_step for i in range(
            0, points_amount)]
        max_position_index = list_intensities_int.index(
            max(list_intensities_int))
        max_position_deg = list_positions_deg[max_position_index]
        list_positions_deg_norm = [
            i - max_position_deg for i in list_positions_deg]
        list_positions_deg_norm_rounded = [
            round(i, 3) for i in list_positions_deg_norm]
        list_positions_arcsec_norm = [
            i * 3600 for i in list_positions_deg_norm]
        list_positions_arcsec_norm_rounded = [
            round(i, 3) for i in list_positions_arcsec_norm]

        data_new = {'x_arcsec': list_positions_arcsec_norm_rounded,
                    'x_deg_orig': list_positions_deg,
                    'x_deg': list_positions_deg_norm_rounded,
                    'int': list_intensities_int}
        df_meas = pd.DataFrame(data_new)


        return_dict['modes']['measurement']['df'] = df_meas
        return_dict['params']['measurement']['system'] = name_system
        return_dict['params']['measurement']['x'] = position_x
        return_dict['params']['measurement']['y'] = position_y
        return_dict['params']['measurement']['steptime'] = param_steptime
        return_dict['params']['measurement']['stepsize'] = round(size_step, 6)

        analysis_dict = {}
        analysis_dict['x'] = return_dict['params']['measurement']['x']
        analysis_dict['y'] = return_dict['params']['measurement']['y']

        calc_xrd.calc_analysis(return_dict)

        df_analysis = pd.DataFrame(analysis_dict, index=[0])

        if 'df' not in return_dict['modes']['analysis']:
            return_dict['modes']['analysis']['df'] = pd.DataFrame()
        try:
            return_dict['modes']['analysis']['df'] =\
                return_dict['modes']['analysis'][
                    'df'].reset_index(drop=True).combine_first(
                    df_analysis.reset_index(drop=True))
        except ValueError as err:
            logger.warning(err)

        datas.append(return_dict)

    return datas
