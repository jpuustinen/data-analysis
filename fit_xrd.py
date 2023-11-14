import copy
import datetime
import os
import random
import tkinter as tk
import matplotlib.pyplot as plt
import pandas as pd
import xrayutilities as xu
import monkey_patches
import numpy as np
import sys
import logging
import importlib
import value_recorder
import configparser
from gui import config_paths
from scipy.signal import savgol_filter
from matplotlib.widgets import Button
from tkinter import simpledialog
from multiprocessing import Pool

reloading = False
if 'fit_xrd_methods' in sys.modules:
    reloading = True

import conv_methods
import fit_xrd_methods

if reloading:
    importlib.reload(conv_methods)
    importlib.reload(fit_xrd_methods)

logger = logging.getLogger('General')

FIT_CURVE_COLORS = ['black', 'red', 'blue', 'green', 'teal', 'orange', 'magenta', 'brown',
                    'cyan', 'purple', 'gold']



def sim_xrd(fitdata=None, pars=None, x_arcsec_array=None):

    if fitdata:
        logger.debug(f'Simulating {fitapp.number_meas}/{fitapp.number_meas_all} '
                     f'(step: {fitdata.fit_step} '
                     f'(iter: {fitdata.fit_iteration}/'
                     f'{fitdata.fit_iterations_total}) '
                     f'with c: {pars["concentration"]:.4f}, '
                     f't: {pars["thickness"]:.2f}, '
                     f'reswidth: {pars["reswidth_as"]:.2f}, '
                     f'cmax: {pars["cmax_log"]:.1f} '
                     f'gradient: {pars["gradient"]:.2f} '
                     f'material: {pars["material"]}, '
                     f'function: {fitconfig.residual_function}')

        x_arcsec_array = np.array(fitdata.fit_meas['x_arcsec'])

    x_deg = x_arcsec_array / 3600 + fitconfig.subst_angle


    def sim_single():

        sub = xu.simpack.Layer(xu.materials.GaAs, np.inf)
        if pars['material'] == 'Compressive':
            lay1 = xu.simpack.Layer(
                xu.materials.GaAsBi(pars['concentration']),
                pars['thickness'] * 10, relaxation=0.0)
        elif pars['material'] == 'Tensile':
            lay1 = xu.simpack.Layer(
                xu.materials.GaAsN(pars['concentration']),
                pars['thickness']*10, relaxation=0.0)
        pls = xu.simpack.PseudomorphicStack001('pseudo', sub+lay1)
        md = xu.simpack.DynamicalModel(
            pls, resolution_width=pars['reswidth_as']/3600)
        Idyn = md.simulate(x_deg, hkl=(0, 0, 4))

        Idyn_norm = fit_xrd_methods.normalize_to_range(Idyn, 1, pars['cmax_log'])
        return Idyn_norm


    if 'gradient' not in pars or not pars['gradient']:
        Idyn_norm = sim_single()
    else:
        Idyns = []
        conc_orig = pars['concentration']
        step = conc_orig * pars['gradient']
        concs = np.linspace(conc_orig - step, conc_orig + step, 25)
        for conc in concs:
            if conc < 0:
                conc = 0
            pars['concentration'] = conc
            Idyns.append(sim_single())
        Idyn_norm = sum(Idyns) / len(Idyns)

        pars['concentration'] = conc_orig


    if fitdata:
        fitdata.fit_results['int_lin'] = np.log10(Idyn_norm)
        fitdata.fit_results['int_log'] = Idyn_norm
    else:
        return Idyn_norm


def get_range_string(par):
    try:
        step1 = par[1] - par[0]
        step2 = par[-1] - par[-2]
        if np.isclose(step1, step2):
            step = step1  # constant range
        else:
            step = 9999  # variable range
        result = f'{par[0]:7.3g}<->{par[-1]:7.3g} ({step:7.3g})'
    except IndexError:
        result = f'{par[0]:7.3g}'

    return result


def residual_1(int_meas, int_res):
    residuals = np.abs(fitconfig.exponent_base ** int_meas -
                              fitconfig.exponent_base ** int_res)

    return residuals

def residual_2(int_meas, int_res):
    residuals = np.abs(int_meas ** fitconfig.exponent_base -
                              int_res ** fitconfig.exponent_base)

    return residuals



def fit_partial(fitargs):

    fitdata = fitargs[2]


    concs = fitdata.par_ranges['concentrations']
    thicks = fitdata.par_ranges['thicknesses']
    reswidths_as = fitdata.par_ranges['reswidths_as']
    cmax_logs = fitdata.par_ranges['cmax_logs']
    gradients = fitdata.par_ranges['gradients']

    concs_partial = fitargs[0]
    results = fitargs[1]

    if fitconfig.fit_int_linear:
        int_meas = fitdata.fit_meas['int_lin']
    else:
        int_meas = fitdata.fit_meas['int_log']

    pars = {}
    pars['material'] = fitdata.pars['material']

    for conc in concs_partial:
        pars['concentration'] = conc
        if conc < 0:
            continue
        for thick in thicks:
            pars['thickness'] = thick
            if thick < 0:
                continue
            for reswidth_as in reswidths_as:
                if reswidth_as < 0:
                    continue
                pars['reswidth_as'] = reswidth_as
                for cmax_log in cmax_logs:
                    if cmax_log < 0:
                        continue
                    pars['cmax_log'] = cmax_log
                    for gradient in gradients:
                        if gradient < 0:
                            continue
                        pars['gradient'] = gradient

                        fitdata.fit_iteration += 1
                        if fitdata.fit_iteration % 20 == 0:
                            if query_cancel_button(fitdata):
                                return

                        sim_xrd(fitdata, pars)

                        if fitconfig.fit_int_linear:
                            int_res = fitdata.fit_results['int_lin']
                        else:
                            int_res = fitdata.fit_results['int_log']

                        if fitconfig.residual_function == 1:
                            residuals_direct = residual_1(int_meas, int_res)
                        elif fitconfig.residual_function == 2:
                            residuals_direct = residual_2(int_meas, int_res)
                        else:
                            logger.warning(f'Residual function not defined in config.')

                        residuals_orig = residuals_direct

                        if fitconfig.main_peak_width:
                            indexes = np.where(
                                (fitdata.fit_meas['x_arcsec'] >
                                (- fitconfig.main_peak_width / 2)) &
                            (fitdata.fit_meas['x_arcsec'] <
                             (fitconfig.main_peak_width / 2)))[0]
                            residuals = residuals_orig
                            residuals[indexes] = residuals_orig[indexes] * \
                                                 fitconfig.main_peak_weigth
                        else:
                            residuals = residuals_orig

                        error_res = residuals.sum()
                        results.append([conc, thick, reswidth_as, cmax_log, error_res, gradient])

                        fitdata.error_iter = error_res

                        if fitconfig.plot_realtime_iterations:
                            fit_xrd_methods.update_plot_iterations(fitdata)


def fit_brute(fitdata):

    results = []
    fitdata.fit_iteration = 0
    fitdata.fit_step += 1

    product = 1
    for i in [len(fitdata.par_ranges[i]) for i in fitdata.par_ranges.keys()]:
        product *= i
    fitdata.fit_iterations_total = product

    concs = fitdata.par_ranges['concentrations']
    thicks = fitdata.par_ranges['thicknesses']
    reswidths_as = fitdata.par_ranges['reswidths_as']
    cmax_logs = fitdata.par_ranges['cmax_logs']
    gradients = fitdata.par_ranges['gradients']

    fit_pars_cur = [get_range_string(i) for i in [
        concs, thicks, reswidths_as, cmax_logs, gradients]]

    fitdata.fit_pars.append(f'---- {fitdata.fit_step} -----' + '\n' +
                            '\n'.join(fit_pars_cur))

    midpoint = int(len(concs) / 2)

    fitargs_1 = [concs[0:midpoint], results, fitdata]
    fitargs_2 = [concs[midpoint:], results, fitdata]

    fit_partial(fitargs_1)
    fit_partial(fitargs_2)

    results_sorted = sorted(results, key=lambda x: x[4])
    for i, desc in enumerate(
            ['concentration', 'thickness', 'reswidth_as', 'cmax_log', 'error', 'gradient']):
        fitdata.fit_results[desc] = results_sorted[0][i]

    fitdata.errors_steps.append(results_sorted[0][-1])

    for key in fitdata.fit_results.keys():
        fitdata.pars[key] = fitdata.fit_results[key]

    sim_xrd(fitdata, fitdata.fit_results)

    if fitconfig.plot_realtime_steps:
        fit_xrd_methods.update_plot_steps(fitdata)

    pairs = [('concentration', 'concentrations'), ('thickness', 'thicknesses'),
             ('reswidth_as', 'reswidths_as'), ('cmax_log', 'cmax_logs'),
             ('gradient', 'gradients')]
    for pair in pairs:
        par, range = pair
        fitdata.par_ranges[range] = [fitdata.fit_results[par]]


def smooth_data(fitdata):
    logger.info(f'Smoothing data')

    if len(fitdata.fit_meas['x_arcsec']) > fitconfig.smooth_window_length:
        fitdata.fit_meas['int_lin'] = savgol_filter(
            fitdata.fit_meas['int_lin'], fitconfig.smooth_window_length,
            fitconfig.smooth_polyorder)
        fitdata.fit_meas['int_log'] = savgol_filter(
            fitdata.fit_meas['int_log'], fitconfig.smooth_window_length,
            fitconfig.smooth_polyorder)
    else:
        logger.warning(f'Cannot perform smoothing.')


def fit_xrd_test(fitdata):


    # --------------- Initialize -------------------
    pars = fitdata.par_ranges
    fitdata.fit_results['material'] = fitdata.pars['material']
    # ------------------------------------------------

    pars['concentrations'] = [0.015]
    pars['thicknesses'] = [fitdata.thick_nom]
    pars['reswidths_as'] = [0]
    pars['cmax_logs'] = [fitdata.cmax_log]
    pars['gradients'] = np.linspace(0, 0.2, 20)

    fit_brute(fitdata)
    if query_cancel_button(fitdata):
        return


def fit_xrd(fitdata):

    pars = fitdata.par_ranges
    res = fitdata.fit_results
    thick_nom = fitdata.thick_nom

    fitdata.fit_results['material'] = fitdata.pars['material']


    # ------------------------ FIT STEP ----------------------------


    pars['concentrations'] = np.linspace(0, 0.08, 16)
    pars['thicknesses'] = [fitdata.thick_nom]
    pars['reswidths_as'] = [0]
    pars['cmax_logs'] = np.geomspace(
        fitdata.cmax_log / 50, fitdata.cmax_log * 5, 10)
    pars['gradients'] = [0]

    if fitconfig.testing:
        pars['concentrations'] = np.linspace(0, 0.08, 2)

    step_concentrations = pars['concentrations'][1] - \
                          pars['concentrations'][0]

    fit_brute(fitdata)
    if query_cancel_button(fitdata):
        return


    # ------------------------ FIT STEP ----------------------------


    conc = res['concentration']
    pars['concentrations'] = np.linspace(
        conc - (step_concentrations * 0.65),
        conc + (step_concentrations * 0.65), 5)
    pars['thicknesses'] = np.linspace(
        thick_nom - (thick_nom * 0.4),
        thick_nom + (thick_nom * 0.4), 10)

    fit_brute(fitdata)
    if query_cancel_button(fitdata):
        return


    # ------------------------ FIT STEP ----------------------------


    pars['cmax_logs'] = np.geomspace(
        fitdata.cmax_log / 50, fitdata.cmax_log * 5, 50)

    if fitconfig.testing:
        pars['cmax_logs'] = np.geomspace(
            fitdata.cmax_log / 50, fitdata.cmax_log * 5, 5)

    fit_brute(fitdata)
    if query_cancel_button(fitdata):
        return

    if fitconfig.testing:
        logger.info(f'Testing on, return early...')
        return

    # ------------------------ FIT STEP ----------------------------

    conc = res['concentration']
    pars['concentrations'] = np.linspace(
        conc - (step_concentrations * 0.65),
        conc + (step_concentrations * 0.65), 5)
    pars['thicknesses'] = np.linspace(
        thick_nom - (thick_nom * 0.4),
        thick_nom + (thick_nom * 0.4), 3)
    pars['cmax_logs'] = np.geomspace(
        res['cmax_log'] / 5, res['cmax_log'] * 2, 7)

    step_concentrations = pars['concentrations'][1] - \
                                 pars['concentrations'][0]

    fit_brute(fitdata)
    if query_cancel_button(fitdata):
        return

    # ------------------------ FIT STEP ----------------------------

    pars['thicknesses'] = np.linspace(
        thick_nom - (thick_nom * 0.4),
        thick_nom + (thick_nom * 0.4), 50)

    step_thicknesses = pars['thicknesses'][1] - \
                                 pars['thicknesses'][0]

    fit_brute(fitdata)
    if query_cancel_button(fitdata):
        return

    # ------------------------ FIT STEP ----------------------------


    conc = res['concentration']
    pars['concentrations'] = np.linspace(
        conc - step_concentrations / 2,
        conc + step_concentrations / 2, 5)
    thick = res['thickness']
    pars['thicknesses'] = np.linspace(
        thick - step_thicknesses / 2,
        thick + step_thicknesses / 2, 6)
    pars['cmax_logs'] = np.geomspace(
        res['cmax_log'] / 4, res['cmax_log'] * 2, 7)

    fit_brute(fitdata)
    if query_cancel_button(fitdata):
        return


    # ------------------------ FIT STEP ----------------------------

    pars['thicknesses'] = np.linspace(
        thick_nom - (thick_nom * 0.4),
        thick_nom + (thick_nom * 0.4), 50)

    fit_brute(fitdata)
    if query_cancel_button(fitdata):
        return


    # ------------------------ FIT STEP ----------------------------

    pars['thicknesses'] = np.linspace(
        res['thickness'] - (res['thickness'] * 0.07),
        res['thickness'] + (res['thickness'] * 0.07), 75)

    fit_brute(fitdata)
    if query_cancel_button(fitdata):
        return


    # ------------------------ FIT STEP ----------------------------

    fit_brute(fitdata)
    if query_cancel_button(fitdata):
        return


    return


def query_cancel_button(fitdata):
    app = fitdata.fitapp.app_main
    app.window_messages.update_idletasks()
    app.window_buttons.update()
    if app.window_buttons.cancel_process or fitdata.cancelled:
        fitdata.cancelled = True
        app.window_buttons.cancel_process = False
        logger.info(
            f'Cancelled fitting')
        return True
    else:
        return False


def fit_and_save(meass, app, testing=False):

    record_id = random.randint(100000, 999999)

    if testing:
        fit_desc = 'TEST'
    else:
        fit_desc = simpledialog.askstring(
            title='Fit XRD', prompt='Fit description\n '
                                    '(leave empty to update default records)')

    fitapp.number_meas_all = len(meass)
    fitapp.app_main = app
    fitconfig.update()

    fitapp.reset()

    label = 'NA'

    if testing:
        fitconfig.testing = True
        meass = meass[0:2]

    for i, meas in enumerate(meass):

        fitdata = FitData(meas)
        if i == 0:
            label = fitdata.meas.parent.label.replace('|', '_')

        fitapp.number_meas = i + 1
        fitdata.record_id = record_id
        fitdata.app = app
        fitdata.fit_desc = fit_desc

        if not fitdata.thick_nom:
            continue

        logger.info(f'XRD fitting ({i}: {meas.data["file"]["path"]}')
        app.window_messages.update_idletasks()

        time_start = datetime.datetime.now()

        # ------------------------- Run fit ------------------------------
        fitdata.fit(testing=testing)
        # ---------------------------------------------------------------

        time_end = datetime.datetime.now()
        fitdata.time_fit = (time_end - time_start).total_seconds()

        if not fitdata.cancelled:

            fit_xrd_methods.update_window_result(fitdata)

            fit_xrd_methods.update_meas(fitdata)

            if fitconfig.save_records:
                record_string = fit_xrd_methods.get_record_string(fitdata)
                value_recorder.save_record(record_string)

            if fitconfig.save_figures:
                fit_xrd_methods.save_figure(fitdata)

        if query_cancel_button(fitdata):
            break

    if fitconfig.save_figures:

        if fitconfig.dir_figs:
            filename = f'XRD-FIT_{fitapp.window_result.record_id}' \
                       f'_{label}'
            path_fig = os.path.join(fitconfig.dir_figs, filename) + '.png'
            logger.debug(f'Saving XRD fit comb result to: {path_fig}.')
            fitapp.window_result.fig.savefig(path_fig)
        else:
            logger.warning(f'DIR_FIGS not defined in config, '
                           f'cancelling save.')

    logger.info('XRD fitting finished.')


def toggle_realtime_plot(fitapp):

    conf = fitapp.fitconfig

    if conf.plot_realtime_iterations:
        conf.plot_realtime_iterations = False
    else:
        conf.plot_realtime_iterations = True


class FitApp:

    def reset(self):
        self.__window_result = None

    def __init__(self):

        self.__fit_figure = None
        self.__ax_steps = None
        self.__ax_iters = None
        self.__result_figure = None
        self.__result_ax = None
        self.line_iters = None
        self.meass = []
        self.meas = []
        self.number_meas = 0
        self.number_meas_all = 0
        self.fitconfig = fitconfig
        self.__result_figure_comb = None
        # self.__ax_result_comb = None
        self.__window_result = None
        self.app_main = None
        self.ax_steps_lines = []
    @property
    def window_result(self):
        if self.__window_result == None:
            top = tk.Toplevel()
            currtime = datetime.datetime.now().isoformat()[11:19]
            figsize = (4.5, fitapp.number_meas_all * 0.7 + 1)
            yscale = [- fitapp.number_meas_all * 2.5, 7]
            self.__window_result = fit_xrd_methods.ResultWindow(
                top, figsize, yscale)
            top.geometry = ('180x700+0+0')
            self.app_main.figs.append(self.__window_result)
        return self.__window_result
    @property
    def result_figure(self) -> plt.Figure:
        if self.__result_figure == None:
            self.__result_figure = plt.figure()
            self.__result_figure.canvas.manager.set_window_title('XRD fit result')
            self.app_main.figs.append(self.__result_figure)
        return self.__result_figure
    @property
    def fit_figure(self) -> plt.Figure:
        if self.__fit_figure == None:
            self.__fit_figure = plt.figure(figsize=(10, 5))
            self.__fit_figure.canvas.manager.set_window_title('XRD fitting')

            self.ax_button_pause = self.__fit_figure.add_axes(
                [0.01, 0.01, 0.12, 0.05])
            self.button_pause = Button(self.ax_button_pause, 'Toggle iter.')
            self.button_pause.on_clicked(
                lambda event: toggle_realtime_plot(self))

            self.__fit_figure.show()
            self.app_main.figs.append(self.__fit_figure)
        return self.__fit_figure
    @property
    def ax_steps(self) -> plt.Axes:
        if self.__ax_steps is None:
            self.__ax_steps = self.fit_figure.add_subplot(1, 2, 1)
            self.__ax_steps.set_title('XRD fit (steps)')
        return self.__ax_steps
    @property
    def ax_iters(self) -> plt.Axes:
        if self.__ax_iters is None:
            self.__ax_iters = self.fit_figure.add_subplot(1, 2, 2)
            self.__ax_iters.set_title('XRD fit (iters)')
        return self.__ax_iters
    @property
    def result_ax(self) -> plt.Axes:
        if self.__result_ax is None:
            self.__result_ax = self.result_figure.add_subplot()
            self.__result_ax.set_title('XRD fit result')
        return self.__result_ax


class FitData:

    def sim(self):
        sim_xrd(self)

    def calc_mismatch(self):
        fit_concentration = self.fit_results['concentration']
        lc_subst = conv_methods.SUBST
        lc_comp = conv_methods.COMP
        lc_tens = conv_methods.TENS

        if self.pars['material'] == 'Compressive':
            fit_lc = lc_subst + (fit_concentration * (lc_comp - lc_subst))
            fit_mismatch = (fit_lc - lc_subst) / lc_subst
        elif self.pars['material'] == 'Tensile':
            fit_lc = lc_subst + (fit_concentration * (lc_tens - lc_subst))
            fit_mismatch = (fit_lc - lc_subst) / lc_subst
        else:
            fit_mismatch = None

        self.fit_mismatch = fit_mismatch

    def fit(self, testing=False):
        if testing:
            fit_xrd_test(self)
        else:
            fit_xrd(self)
        if query_cancel_button(self):
            return
        self.calc_mismatch()

    def get_thick_nom(self):

        options = ['Thickness']

        for option in options:
            if option in self.meas.data['SD']:
                self.thick_nom = float(self.meas.data['SD'][option])

                if not np.isnan(self.thick_nom):
                    logger.info(f'Nominal thickness: {self.thick_nom:.3g}')
                    break

        if not self.thick_nom:
            logger.warning(f'Could not get thickness, cancelling...')
            self.cancelled = True


    def init_data(self):
        df = self.df

        self.get_thick_nom()

        self.fit_meas['int_log'] = self.df['int'].fillna(0)
        self.fit_meas['int_lin'] = np.nan_to_num(
            np.log10(self.df['int']), neginf=0)
        self.fit_meas['x_arcsec'] = self.df['x_arcsec']

        self.fit_meas = pd.DataFrame(self.fit_meas)

        self.path_string = self.meas.data['file']['path']
        if 'num_subresult' in self.meas.data['file']:
            self.num_subresult = self.meas.data["file"]["num_subresult"]
            self.path_string += f'_#{self.meas.data["file"]["num_subresult"]}'

        if fitconfig.fit_smooth:
            smooth_data(self)

        if fitconfig.interpolate_width:
            new_x = np.arange(self.fit_meas['x_arcsec'].min(),
                              self.fit_meas['x_arcsec'].max(),
                              fitconfig.interpolate_width)

            self.fit_meas['int_lin'] = pd.Series(
                np.interp(new_x, self.fit_meas['x_arcsec'],
                          self.fit_meas['int_lin']))
            self.fit_meas['int_log'] = pd.Series(
                np.interp(new_x, self.fit_meas['x_arcsec'],
                          self.fit_meas['int_log']))

            self.fit_meas['x_arcsec'] = pd.Series(new_x)
            self.fit_meas = self.fit_meas.dropna()

        integ_left = self.fit_meas[
            (self.fit_meas['x_arcsec'] < -50) &
            (self.fit_meas['x_arcsec'] > -1500)]['int_lin'].sum()
        integ_right = self.fit_meas[
            (self.fit_meas['x_arcsec'] > 50) &
            (self.fit_meas['x_arcsec'] < 1500)]['int_lin'].sum()
        if integ_left > integ_right:
            self.pars['material'] = 'Compressive'
        else:
            self.pars['material'] = 'Tensile'

        self.cmax_log = np.max(self.fit_meas['int_log'].max())
        self.cmax_lin = np.max(self.fit_meas['int_lin'].max())

        if fitconfig.fit_int_linear:
            int_meas_toplot = self.fit_meas['int_lin']
        else:
            int_meas_toplot = self.fit_meas['int_log']

        if fitconfig.plot_realtime_steps:
            fitapp.ax_steps.clear()

            fitapp.ax_steps_lines = []
            fitapp.ax_steps.plot(
                self.fit_meas['x_arcsec'], int_meas_toplot, '-',
                color = fitconfig.fit_curve_colors[0])

        if fitconfig.plot_realtime_iterations:
            fitapp.ax_iters.clear()
            fitapp.ax_iters.plot(
                self.fit_meas['x_arcsec'], int_meas_toplot, '-',
                color=fitconfig.fit_curve_colors[0])

    def __init__(self, meas):
        self.meas = meas
        self.df = copy.deepcopy(self.meas.data['modes']['measurement']['df'])
        self.par_ranges = {}
        self.pars = {}
        self.fit_meas = {}
        self.fit_results = {'int_log': None, 'int_lin': None}
        self.fit_iteration = 0
        self.fit_step = 0
        self.thick_nom = None
        self.num_subresult = None
        self.fitapp = fitapp
        self.fitconfig = fitconfig
        self.fit_iterations_total = 0
        self.record_id = 0
        self.fit_id = random.randint(100000, 999999)
        self.errors_steps = []
        self.error_iter = None
        self.app = None
        self.cmax_lin = None
        self.cmax_log = None
        self.cancelled = False
        self.fit_pars = []
        self.pars_current = {}
        self.fit_desc = ''

        self.init_data()

class FitConfig:

    def __init__(self):
        PATH_CONFIG = 'fit_xrd.cfg'
        config = configparser.ConfigParser()
        config.read(PATH_CONFIG)

        if config.get('GENERAL', 'SAVE_RECORDS', fallback='OFF') == 'ON':
            self.save_records = True
        else:
            self.save_records = False

        if config.get('GENERAL', 'SAVE_FIGURES', fallback='OFF') == 'ON':
            self.save_figures = True
        else:
            self.save_figures = False

        self.subst_angle = float(config.get(
            'GENERAL', 'SUBST_ANGLE'))

        if config.get('FIT', 'LINEAR', fallback='OFF') == 'ON':
            self.fit_int_linear = True
        else:
            self.fit_int_linear = False

        if config.get('FIT', 'SMOOTH', fallback='OFF') == 'ON':
            self.fit_smooth = True
        else:
            self.fit_smooth = False

        interpolate_width = config.get('FIT', 'INTERPOLATE_WIDTH',
                                      fallback=None)
        if interpolate_width:
            self.interpolate_width = int(interpolate_width)
        else:
            self.interpolate_width = None

        smooth_window_length = config.get('FIT', 'SMOOTH_WINDOW_LENGTH',
                                      fallback=None)
        if smooth_window_length:
            self.smooth_window_length = int(smooth_window_length)
        else:
            self.smooth_window_length = None

        smooth_polyorder = config.get('FIT', 'SMOOTH_POLYORDER',
                                      fallback=None)
        if smooth_polyorder:
            self.smooth_polyorder = int(smooth_polyorder)
        else:
            self.smooth_polyorder = None

        exponent_base = config.get('FIT', 'EXPONENT_BASE',
                                      fallback=None)
        if exponent_base:
            self.exponent_base = float(exponent_base)
        else:
            self.exponent_base = None

        main_peak_weigth = config.get('FIT', 'MAIN_PEAK_WEIGHT',
                                      fallback=None)
        if main_peak_weigth:
            self.main_peak_weigth = float(main_peak_weigth)
        else:
            self.main_peak_weigth = None

        main_peak_width = config.get('FIT', 'MAIN_PEAK_WIDTH',
                                     fallback=None)
        if main_peak_width:
            self.main_peak_width = float(main_peak_width)
        else:
            self.main_peak_width = None

        if config.get('PLOT', 'REALTIME_STEPS', fallback='OFF') == 'ON':
            self.plot_realtime_steps = True
        else:
            self.plot_realtime_steps = False

        if config.get('DEBUG', 'TESTING', fallback='OFF') == 'ON':
            self.testing = True
        else:
            self.testing = False

        if config.get('PLOT', 'REALTIME_ITERATIONS',
                      fallback='OFF') == 'ON':
            self.plot_realtime_iterations = True
        else:
            self.plot_realtime_iterations = False

        self.dir_figs = config_paths.get('FIT_XRD', 'DIR_FIGS', fallback=None)

        self.fit_curve_colors = FIT_CURVE_COLORS

        residual_function = config.get('FIT', 'RESIDUAL_FUNCTION',
                                                 fallback=1)
        self.residual_function = int(residual_function)

    def update(self):
        self.__init__()


fitconfig = FitConfig()
fitapp = FitApp()
