import matplotlib.image as mpimg
from PIL import Image


def get_data(data):

    data['modes']['image'] = {}

    if data['config'].get('GENERAL', 'LOAD_IMAGES', fallback='OFF') == 'ON':

        img = mpimg.imread(data['file']['path'])
        data['modes']['image']['nparray'] = img

        img = Image.open(data['file']['path'])
        data['modes']['image']['PIL-image'] = img

    return data
