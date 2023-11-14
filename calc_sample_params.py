import re
import pandas as pd
import conv_methods
import logging

logger = logging.getLogger('General')


def _get_sample_parameters_from_filename(data, refs):

    df_sample_data = refs.df_sample_data

    name_file = data['file']['name']

    if not 'sample' in data:
        data['sample'] = {}


    # --------------------------------------------------
    # PARSE SAMPLE PARAMETERS IN THIS SECTION

    match = re.search(r'.*SAMPLE(\d{1,4})_*', name_file.upper())
    if match:
        data['sample']['number'] = match[1]
        data['sample']['name'] = 'sample' + match[1]




    # --------------------------------------------------

    conv_methods._calculate_derived_parameters(data, df_sample_data)

    df_analysis_dict = {}
    for key in data['sample']:
        if not type(data['sample'][key]) == pd.DataFrame:
            df_analysis_dict[key] = data['sample'][key]

    df_analysis = pd.DataFrame(df_analysis_dict, index=[0])

    if 'df' not in data['modes']['analysis']:
        data['modes']['analysis']['df'] = pd.DataFrame()
    data['modes']['analysis']['df'] = data['modes']['analysis'][
        'df'].reset_index(drop=True).combine_first(
        df_analysis.reset_index(drop=True))
