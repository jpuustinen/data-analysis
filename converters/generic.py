from tkinter import simpledialog
import logging
import pandas as pd

logger = logging.getLogger('General')

generic_separator = None
generic_header = None
generic_skiprows = None


def init():

    global generic_header
    global generic_separator
    global generic_skiprows

    generic_separator = None
    generic_header = None
    generic_skiprows = None


def get_data(data):

    global generic_header
    global generic_separator
    global generic_skiprows

    path_source = data['file']['path']

    df = pd.DataFrame()

    if path_source.endswith('.csv'):
        generic_separator = ','
    elif not path_source.endswith('.xlsx'):
        if not generic_separator:
            generic_separator = simpledialog.askstring(
                'Data import:', 'Separator: (write "tab" for tab)')
        if generic_separator == 'tab':
            generic_separator = '\t'

    if not path_source.endswith('.xlsx'):
        if generic_separator == '':
            logger.warning(f'Invalid separator')
            return

        if not generic_skiprows:
            generic_skiprows = simpledialog.askinteger(
                'Data import', 'Rows to skip:')

        if not generic_header:
            generic_header = simpledialog.askinteger(
                'Data import', 'Header line')
            generic_header = generic_header - generic_skiprows - 1

        df = pd.read_csv(path_source, sep=generic_separator,
                         header=generic_header, skiprows=generic_skiprows)

    else:
        df = pd.read_excel(path_source)
        df = df.dropna(axis='rows', how='all')
        df = df.dropna(axis='columns', how='all')

    if df.empty:
        logger.warning(f'Not a valid data file')
        return

    data['modes']['analysis']['df'] = df
    data['modes']['measurement']['df'] = df

    return data