""" Example converter
"""

import logging
import pandas as pd

# include this to enable logging
logger = logging.getLogger('General')


# all converters are called by the get_data method. It should accept only one
# argument (dictionary type), and return the modified dictionary or a list of
# dictionaries. Add DataFrames to data['modes']['measurement']['df'] and
# to data['modes']['analysis']['df'].
def get_data(data):

    path_source = data['file']['path']

    df = pd.read_csv(path_source, sep='\t')

    # add measurement data
    data['modes']['measurement']['df'] = df

    # calculate properties from measurement
    return_dict = {}
    return_dict['sum_x'] = df['x'].sum()
    return_dict['sum_y'] = df['y'].sum()

    df_analysis = pd.DataFrame(return_dict, index=[0])

    # add calculated properties to analysis
    data['modes']['analysis']['df'] = df_analysis

    # in case of multiple measurements from one file, return a list of datas
    return data
