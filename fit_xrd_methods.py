from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
import tkinter as tk
import datetime

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from fit_xrd import FitData

logger = logging.getLogger('General')


def normalize_to_range(y, min_new, max_new):
    max_y = np.max(y)
    min_y = np.min(y)
    range_y = max_y - min_y
    y_norm_01 = (y - min_y) / range_y
    range_new = max_new - min_new
    y_norm = min_new + y_norm_01 * range_new

    return y_norm


def update_meas(fitdata: FitData):

    for key in fitdata.fit_results.keys():
        if 'int' not in key:
            fitdata.meas.data['modes']['analysis']['df'][key] = \
                fitdata.fit_results[key]


def get_record_string(fitdata):
    record = {'path_string': fitdata.path_string,
              'record_id': fitdata.record_id,
              'fit_desc': fitdata.fit_desc,
    'fit_id': fitdata.fit_id, 'fit_time': fitdata.time_fit}

    if fitdata.num_subresult:
        record['num_subresult'] = fitdata.num_subresult

    for key in fitdata.fit_results.keys():
        if 'int' not in key:
            record[key + '_fit'] = fitdata.fit_results[key]

    record['fit_errors_steps'] = fitdata.errors_steps
    record['fit_mismatch'] = fitdata.fit_mismatch

    record['fit_pars'] = '--'.join(fitdata.fit_pars).replace('\n', '<>')

    if fitdata.fitconfig.fit_smooth:
        record['fit_smoothing'] = f'{fitdata.fitconfig.smooth_window_length};' \
                              f'{fitdata.fitconfig.smooth_polyorder}'
    else:
        record['fit_smoothing'] = 'OFF'

    if fitdata.fitconfig.interpolate_width:
        record['fit_interpolation'] = f'{fitdata.fitconfig.interpolate_width}'
    else:
        record['fit_interpolation'] = 'OFF'

    record['fit_exponent_base'] = f'{fitdata.fitconfig.exponent_base}'
    record['fit_main_peak_weight'] = f'{fitdata.fitconfig.main_peak_weigth}'
    record['fit_main_peak_width'] = f'{fitdata.fitconfig.main_peak_width}'

    record['residual_function'] = f'{fitdata.fitconfig.residual_function}'

    text = ''
    for key in record:
        text += f'{key}\t{record[key]}\t'
    text = text[0:-1]

    return text


def save_figure(fitdata):

    fitconfig = fitdata.fitconfig

    if fitconfig.fit_int_linear:
        fit_int_toplot = fitdata.fit_results['int_lin']
        meas_int_toplot = fitdata.fit_meas['int_lin']
    else:
        fit_int_toplot = fitdata.fit_results['int_log']
        meas_int_toplot = fitdata.fit_meas['int_log']

    fig = fitdata.fitapp.result_figure
    ax = fitdata.fitapp.result_ax

    ax.clear()
    ax.plot(fitdata.fit_meas['x_arcsec'], meas_int_toplot, '-',
            color='black', linewidth=1)
    ax.plot(fitdata.fit_meas['x_arcsec'], fit_int_toplot, '-', color='red',
            linewidth=1)

    if fitconfig.dir_figs:
        filename = f'{os.path.basename(fitdata.path_string)}_XRD-FIT_' \
                   f'{fitdata.record_id}_{fitdata.fit_id}'
        path_fig = os.path.join(fitconfig.dir_figs, filename) + '.png'
        logger.debug(f'Saving XRD fit result to: {path_fig}.')
        fig.savefig(path_fig)
    else:
        logger.warning(f'DIR_FIGS not defined in config, '
                       f'cancelling save.')


def update_window_result(fitdata):

    ax = fitdata.fitapp.window_result.ax
    if not fitdata.fitapp.window_result.title:
        fitdata.fitapp.window_result.record_id = fitdata.record_id
        currtime = datetime.datetime.now().isoformat()[11:19]
        fitdata.fitapp.window_result.frame.title(
            f'XRD fit - comb ({fitdata.fitapp.window_result.record_id}_'
            f'{fitdata.meas.parent.label.replace("|", "_")}) '
            f'[{currtime}]')

    offset = (fitdata.fitapp.number_meas - 1) * 2.5
    ax.plot(fitdata.fit_meas['x_arcsec'],
            fitdata.fit_meas['int_lin'] - offset, color='black',
            linewidth=1)
    ax.plot(fitdata.fit_meas['x_arcsec'],
            fitdata.fit_results['int_lin'] - offset, color='red',
            linewidth=1)

    ax.get_figure().canvas.draw_idle()
    ax.get_figure().canvas.flush_events()


def update_plot_steps(fitdata):

    fitconfig = fitdata.fitconfig
    fitapp = fitdata.fitapp

    if fitconfig.fit_int_linear:
        fit_int_toplot = fitdata.fit_results['int_lin']
    else:
        fit_int_toplot = fitdata.fit_results['int_log']

    ax = fitapp.ax_steps
    line, = ax.plot(fitdata.fit_meas['x_arcsec'], fit_int_toplot, '-',
            color=fitconfig.fit_curve_colors[fitdata.fit_step], linewidth=1)

    fitapp.ax_steps_lines.append(line)

    for line in fitapp.ax_steps_lines[0:-1]:
        line.set_linestyle(':')

    ax.set_title(f'Step: {fitdata.fit_step}, ' +
                 'errors: ' + ';'.join([f'{i:.3g}' for i in fitdata.errors_steps]),
                 fontsize=9, wrap=True)

    ax.get_figure().canvas.draw_idle()
    ax.get_figure().canvas.flush_events()


def update_plot_iterations(fitdata):

    fitapp = fitdata.fitapp
    fitconfig = fitdata.fitconfig

    if fitapp.line_iters:
        fitapp.line_iters.remove()

    if fitconfig.fit_int_linear:
        fit_int_toplot = fitdata.fit_results['int_lin']
    else:
        fit_int_toplot = fitdata.fit_results['int_log']

    fitapp.line_iters, = fitapp.ax_iters.plot(
        fitdata.fit_meas['x_arcsec'], fit_int_toplot, color='red',
        linewidth=1)
    fitapp.ax_iters.set_title(f'Error: {fitdata.error_iter:.3g}')

    fig_realtime = fitapp.ax_iters.get_figure()
    fig_realtime.canvas.draw_idle()
    fig_realtime.canvas.flush_events()


class ResultWindow(tk.Frame):

    def __init__(self, _master, figsize, yscale):
        super().__init__(_master)
        self.grid()
        self.frame = _master
        self.title = None
        self.grid(row=1, column=1)
        self.fig: plt.Figure = plt.Figure(figsize=figsize)
        self.ax: plt.Axes = self.fig.add_subplot()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().grid(row=1, column=1)
        self.canvas.draw()
        self.toolbar = NavigationToolbar2Tk(
            self.canvas, _master, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.grid(row=2, column=1)
        self.sb_y = tk.Scrollbar(
            self.frame, orient="vertical",
            command=self.canvas.get_tk_widget().yview)
        self.sb_y.grid(row=1, column=2, sticky=tk.N)
        self.canvas.get_tk_widget().config(yscrollcommand=self.sb_y.set)
        self.record_id = None
        self.ax.set_ylim(yscale)
