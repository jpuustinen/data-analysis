import logging
import copy
import pandas as pd
import conv_methods

logger_general = logging.getLogger('General')


def get_data(data):

    records = {}

    with open(data['file']['path'], 'r') as f:
        conv_methods.get_records(f.read(), records)

    datas = []

    data['references'].pop('converter')

    for key in records.keys():

        record = records[key]
        return_data = copy.deepcopy(data)
        df = pd.DataFrame(record[list(record.keys())[0]], index=[0])
        return_data['modes']['analysis']['df'] = df
        datas.append(return_data)

    return datas