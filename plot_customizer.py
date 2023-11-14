import configparser
import logging
import tkinter as tk
from tkinter import simpledialog
import gui_methods
import matplotlib.pyplot as plt


PATH_CONFIG = 'plot_customizer.cfg'
config = configparser.ConfigParser(interpolation=None)
config.read(PATH_CONFIG)

logger = logging.getLogger('General')

WIDTH_ENTRIES = 40

# ALL OPTIONS HAVE TO BE DEFINED HERE
OPTIONS = ['TITLE',
           'X_RANGE',
           'Y_RANGE',
           'COLOR_RANGE',
           'X_RANGE_SEC',
           'Y_RANGE_SEC',
           'X_LABEL',
           'Y_LABEL',
           'Y_LABEL_SEC',
           'LEGEND',
           'TIGHT_LAYOUT',
           'COLORED_SEC_AXES',
           'COLOR_CURVES',
           'STYLE_CURVES',
           'MARKER_CURVES',
           'MARKERSIZE',
           'MINOR_TICKS',
           'PICKING',
           'SIZE',
           'NEXT_TO_CURVES_TITLE',
           'LEGEND_TITLE',
           'Y_SCALE_TYPES']

OPTIONS = sorted(OPTIONS)

class WindowCustomizer(tk.Frame):


    def populate(self):

        options_all = OPTIONS

        for key in self.optionvalues:
            if key not in options_all:
                options_all.append(key)

        for option in OPTIONS:
            var = tk.StringVar(self.master)
            self.vars.append(var)
            lb = tk.Label(self, text=option)
            self.lbs.append(lb)
            lb.grid(row=len(self.lbs) + self.row_start, column=1)
            en = tk.Entry(self, width=WIDTH_ENTRIES)
            en.grid(row=len(self.lbs) + self.row_start, column=2)
            self.ens.append(en)
            i = len(self.lbs) - 1
            om = tk.OptionMenu(self, var, *self.optionvalues[option],
                               command=lambda _, i=i: self.change(_, i))
            om.grid(row=len(self.lbs) + self.row_start, column=3)
            self.oms.append(om)

    def get_values(self):
        for section in config.sections():

            for option in config[section].keys():

                option_upper = str.upper(option)
                if option_upper not in self.optionvalues:
                    self.optionvalues[option_upper] = []
                value = config[section][option_upper]
                if value not in self.optionvalues[option_upper]:
                    self.optionvalues[option_upper].append(value)
        for option in OPTIONS:
            if option not in self.optionvalues:
                self.optionvalues[option] = ['N/A']


    def exit_window(self):
        self.master.destroy()

    def change(self, _, i):
        value = self.vars[i].get()
        self.ens[i].delete(0, tk.END)
        self.ens[i].insert(0, value)


    def select_section(self, _, section=None):
        if not section:
            section = self.var_main.get()

        for en, option in zip(self.ens, OPTIONS):
            en.delete(0, tk.END)
            if option in config[section]:

                en.insert(0, config[section][option])

    def save_section(self, name_section=None):

        if not name_section:
            name_section = simpledialog.askstring('Config query',
                                                  'Section name:')


        if name_section:

            config.remove_section(name_section)

            config.add_section(name_section)

            for en, option in zip(self.ens, OPTIONS):
                value = en.get()
                if value:
                    config.set(name_section, option, value)

            with open(PATH_CONFIG, 'w') as f:
                config.write(f)

    def customize_plot(self):
        self.save_section(name_section='PLOT')

        gui_methods.customize_figure(self.app.figure)


    def save_fig(self):
        gui_methods.save_fig(
            self.app.figure)

    def close_figs(self):
        for img in self.list_image_windows:
            img.close()

    def __init__(self, _master, config=None, app=None, figure=None):
        super().__init__(_master)
        self.master = _master
        self.grid()
        self.app = app
        self.figure = figure
        self.lbs = []
        self.oms = []
        self.ens = []
        self.bts = []
        self.vars = []
        self.values = []
        self.config = config
        self.optionvalues = {}
        self.list_image_windows = []

        self.bt_exit = tk.Button(self, text='Exit', command=self.exit_window)
        self.bt_exit.grid(row=1, column=1)

        self.get_values()
        self.var_main = tk.StringVar(self)
        self.om_main = tk.OptionMenu(self, self.var_main, *config.sections(),
                                     command=self.select_section)
        self.om_main.grid(row=2, column=1)

        self.bt_save_section = tk.Button(
            self, text='Save options', command=self.save_section)
        self.bt_save_section.grid(row=3, column=4)

        self.bt_customize = tk.Button(
            self, text='Customize', command=self.customize_plot)
        self.bt_customize.grid(row=4, column=4)

        self.bt_save_fig = tk.Button(
            self, text='Save fig.', command=self.save_fig)
        self.bt_save_fig.grid(row=5, column=4)

        self.row_start = 3

        self.populate()

        self.select_section(None, 'PLOT')

        if self.figure:
            plt.show()



def customize(master, config_=None, app=None, figure=None):

    if not config_:
        config_ = config

    window_customizer_top = tk.Toplevel()
    window_customizer_top.geometry('700x1100+0+0')
    window_customizer_top.title('temp')

    window_customizer = WindowCustomizer(window_customizer_top, config_, app, figure=figure)
    return window_customizer


if __name__ == '__main__':
    master = tk.Tk()
    window_customizer = customize(master, config)

    master.mainloop()