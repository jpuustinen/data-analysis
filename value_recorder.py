import tkinter as tk
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
import gui_plotting as plotting
import logging
import configparser
import os

PATH_CONFIG = 'value_recorder.cfg'
PATH_CONFIG_PATHS = 'paths.cfg'

logger = logging.getLogger('General')
config = configparser.ConfigParser()
config.read(PATH_CONFIG)

config_paths = configparser.ConfigParser()
config_paths.read(PATH_CONFIG_PATHS)


def get_logger_record(path, name):

    path_records = path
    if os.path.exists(path):
        logger.info(f'Creating records logger at: {path_records}')
        logger_record = logging.getLogger(name)
        logger_record.setLevel('DEBUG')
        handler_file = logging.FileHandler(path_records)
        handler_stream = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(levelname)8s %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        handler_stream.setFormatter(formatter)
        # logger_record.addHandler(handler_stream)
        handler_file.setFormatter(formatter)
        logger_record.addHandler(handler_file)

        return logger_record
    else:
        logger.warning(f'Records path does not exist: {path}')


def save_record_conv(text):
    try:
        if logger_record_conv:
            logger_record_conv.info(text)
    except NameError:
        pass


def save_record(text):
    try:
        if logger_record:
            logger_record.info(text)
    except NameError:
        pass


class Recorder(tk.Frame):

    def customize_fig(self):
        range_y_raw = config.get('PLOT', 'RANGE_Y', fallback=None)

        range_x_raw = config.get('PLOT', 'RANGE_X', fallback=None)
        if range_x_raw:
            range_x = range_x_raw.split(';')
            try:
                range_x_min = float(range_x[0])
            except ValueError:
                pass
            else:
                self.fig.ax_p.set_xlim(left=range_x_min)
            try:
                range_x_max = float(range_x[1])
            except ValueError:
                pass
            else:
                self.fig.ax_p.set_xlim(right=range_x_max)

        if range_y_raw:
            range_y = range_y_raw.split(';')
            try:
                range_y_min = float(range_y[0])
            except ValueError:
                pass
            else:
                self.fig.ax_p.set_ylim(bottom=range_y_min)
            try:
                range_y_max = float(range_y[1])
            except ValueError:
                pass
            else:
                self.fig.ax_p.set_ylim(top=range_y_max)
        else:
            data = self.fig.ax_p.lines[0].get_xydata()
            y_max = data[(data[:, 0]< 1.2) & (data[:,0]>0.8)][:,1].max()
            self.fig.ax_p.set_ylim(top=y_max, bottom=0)



    def test(self):

        self.num_meas += 1
        if self.num_meas == len(self.meas_toplot) + 1:
            self.num_meas = 1
            self.e_record.insert(0, 'Wrapped')

        meas = self.meas_toplot[self.num_meas - 1]
        for ax in self.fig.axs_p + self.fig.axs_s:
            ax.clear()
        plotting.plot_data([meas], 'measurement', self.figures, self.fig)
        self.fig.ax_p.minorticks_on()
        self.fig.ax_p.yaxis.grid(which='major')
        self.fig.ax_p.yaxis.grid(which='minor', linestyle='--')
        self.fig.ax_p.xaxis.grid(which='major')
        self.fig.ax_p.xaxis.grid(which='minor', linestyle='--')
        self.fig.axs_p[0].set_title(
            meas.data['file']['name'], fontsize=9)
        self.customize_fig()

        self.master.lift()


    def show_line(self, _event=None):

        meas = self.meas_toplot[self.num_meas - 1]

        try:
            OFFSET = float(self.e_offset.get())
        except ValueError:
            OFFSET = 0

        try:
            result_raw = self.e_record.get()
            result = [float(i) for i in result_raw.split(';')]
        except ValueError:
            return

        if self.test_line:
            try:
                self.test_line.remove()
            except ValueError:
                pass

        df = meas.data['modes']['measurement']['df']
        ser_x = meas.sers_x[0]
        ser_y = meas.sers_yp[0]

        self.test_line, = self.fig.ax_p.plot(result[0], result[1], '-')

        self.fig.fig.canvas.draw_idle()

    def accept(self, _event=None):

        result = self.e_record.get()
        property_ = self.e_property.get()
        note = self.e_note.get()
        offset = self.e_offset.get()
        data_ = self.meas_toplot[self.num_meas - 1].data

        self.e_record.delete(0, tk.END)
        path_string = data_['file']['path']
        if 'num_subresult' in data_['file']:
            path_string += '>_' + data_['file']['num_subresult']

        text = f'path_string\t{path_string}\tproperty\t{property_}\tnote\t{note}\tresult\t{result}\toffset\t{offset}'
        save_record(text)

        self.test_line.remove()

        self.test()


    def __init__(self, _master, meas_toplot, path_records, fig, figures):

        super().__init__(_master)
        self.master = _master
        self.grid()
        self.fig = fig
        self.figures = figures
        self.path_records = path_records
        self.canvas = FigureCanvasTkAgg(self.fig.fig, master=self.master)
        self.canvas.draw()
        self.num_meas = 0
        self.canvas.get_tk_widget().grid(row=1, column=1)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.master,
                                            pack_toolbar=False)

        self.test_line = None

        self.toolbar.update()
        self.toolbar.grid(row=2, column=1)

        self.button_test = tk.Button(self, text='test', command=self.test)
        self.meas_toplot = meas_toplot
        self.button_test.grid(row=3, column=3)

        self.l_property = tk.Label(self, text='Property')
        self.l_property.grid(row=2, column=2)

        self.e_property = tk.Entry(self)
        self.e_property.grid(row=3, column=2)

        self.l_note = tk.Label(self, text='Note')
        self.l_note.grid(row=4, column=2)

        self.e_note = tk.Entry(self)
        self.e_note.grid(row=5, column=2)

        self.l_record = tk.Label(self, text='Record')
        self.l_record.grid(row=6, column=2)

        self.e_record = tk.Entry(self)
        self.e_record.grid(row=7, column=2)

        self.l_offset = tk.Label(self, text='Offset')
        self.l_offset.grid(row=8, column=2)

        self.e_offset = tk.Entry(self)
        self.e_offset.grid(row=9, column=2)

        self.button_accept = tk.Button(
            self, text='Accept', command=self.accept)
        self.button_accept.grid(row=3, column=4)
        self.master.bind('<Control-Return>', self.accept)

        self.button_show = tk.Button(
            self, text='Show', command=self.show_line)
        self.button_show.grid(row=3, column=5)
        self.master.bind('<Return>', self.show_line)


def record_values(meas_toplot, path_records, fig, figures):

    window_recorder_top = tk.Toplevel()
    window_recorder_top.geometry('1800x1000+5+5')
    window_recorder_top.lift()

    fig_size_inches_raw = config.get('PLOT', 'SIZE_INCHES', fallback=None)
    if fig_size_inches_raw:
        fig_size_inches = [
            float(i) for i in fig_size_inches_raw.split(';')]
        fig.fig.set_size_inches(
            fig_size_inches[0], fig_size_inches[1], forward=True)
    window_recorder = Recorder(
        window_recorder_top, meas_toplot, path_records, fig, figures)

    return window_recorder


if not logging.getLogger('Records').hasHandlers():
    path = config_paths.get(
        'VALUE_RECORDER', 'PATH_RECORDS_FIT', fallback=None)
    if path:
        path = path.split(';')[0]
        logger_record = get_logger_record(path, 'Records')
    else:
        logger.warning(f'Records (fit) path not specified.')

if not logging.getLogger('Records_conv').hasHandlers():
    path = config_paths.get(
        'VALUE_RECORDER', 'PATH_RECORDS_CONV', fallback=None)
    if path:
        path = path.split(';')[0]
        logger_record_conv = get_logger_record(path, 'Records_conv')
    else:
        logger.warning(f'Records (conv) path not specified.')
