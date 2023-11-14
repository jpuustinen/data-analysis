import _tkinter
import configparser
import matplotlib.pyplot as plt
import tkinter as tk
import sys
import os
import copy
import datetime
import json
import math
import subprocess
import pandas as pd
import numpy as np
from tkinter import filedialog
from tkinter import simpledialog
from tkinter import messagebox
from typing import List
from typing import Optional
import conv_main
import gradients
import image_viewer
import importlib
import logging
import tkinter.scrolledtext as scrolledtext
from matplotlib.widgets import Button
from PIL import Image
import _init_gui
from _init_gui import config
from _init_gui import config_paths
import gui_plotting as plotting
import gui_methods as methods
import matplotlib.image as mpimg
import plot_customizer
import df_viewer
import value_recorder

import converters.generic

try:
    import fit_xrd
except (ModuleNotFoundError, SyntaxError):
    fit_xrd = None
try:
    import extra as extra
except (ModuleNotFoundError, SyntaxError):
    extra = None

MODS_TO_RELOAD = []

MODS_TO_RELOAD.extend([plotting, methods, extra, df_viewer,
                       fit_xrd, image_viewer, plot_customizer])

logger_general = logging.getLogger('General')


def reload_modules():
    update_config()
    reload_plotting()
    reload_converters()


def reload_converters():
    importlib.reload(conv_main)
    logger_general.info('Reloaded c_converters.')


def reload_plotting():

    for mod in MODS_TO_RELOAD:
        if mod:
            try:
                importlib.reload(mod)
                logger_general.info(f'Reloaded: {mod.__name__}')
            except NameError:
                logger_general.debug(f'Cannot reload module: {mod.__name__}')


def exit_program(close_figures=False):

    global figures

    if close_figures:
        for figure_ in figures.list:
            figure_.fig.clf()

    master.destroy()


def add_measurements():
    global measurements
    measurements = Measurements()

    if window_main:
        update_windows()

    return measurements


def reload_files(selected=False):

    if selected:
        meass_toplot, meass_orig = get_selections('measurement')
        paths = []
        for meas in meass_toplot:
            path = meas.data['file']['path']
            if path not in paths:
                paths.append(path)
    else:
        paths = app.last_added_paths


    add_measurements()
    update_config()
    reload_converters()
    reload_plotting()

    add_files(paths_selected_arg=paths)


def add_files(paths_selected_arg=None, appending=False, generic=False):
    """
    Select files for plotting with dialog.
    """

    global measurements
    global dataset
    global last_open_directory

    converters.generic.init()

    if paths_selected_arg and type(paths_selected_arg) is not list:
        paths_selected = [paths_selected_arg]
    else:
        paths_selected = paths_selected_arg

    filetypes_raw = config.get('FILES', 'FILETYPES', fallback=None)
    if filetypes_raw:
        filetypes = [tuple(i.split(';')) for i in
                     config['FILES']['FILETYPES'].split(',')]
    else:
        filetypes = ()

    if not measurements:
        add_measurements()

    if not paths_selected:
        paths_selected = list(filedialog.askopenfilenames(
            title='Select files for plotting:', filetypes=filetypes,
            initialdir=last_open_directory))

    if not appending:
        app.last_added_paths = []

    if paths_selected:

        color_normal = master['background']
        master.update_idletasks()
        if not paths_selected_arg:
            last_open_directory = os.path.split(paths_selected[0])[0]

        num_loaded = 0

        time_start_process = datetime.datetime.now()

        for i, path in enumerate(paths_selected):

            window_messages.l_message['text'] = (
                f'Adding file'
                f'({i + 1}/{len(paths_selected)}: {os.path.basename(path)}')
            logger_general.info(
                f'Adding path ({i}/{len(paths_selected)}: {path}')

            if path.endswith('.djson'):
                load_dataset_files_select(path)
            elif path.endswith('.plist'):
                with open(path, 'r') as fplist:
                    paths_fromlist = fplist.read().split('\n')
                    paths_fromlist = [i for i in paths_fromlist if i != ""]
                    add_files(paths_fromlist, appending=True)
            else:
                app.last_added_paths.append(path)
                measurements.add_file(
                    path, index=len(measurements.list), generic=generic)

            window_messages.update_idletasks()
            window_buttons.update()

            if window_buttons.cancel_process:
                window_buttons.cancel_process = False
                num_loaded = i + 1
                logger_general.info(
                    f'Cancelled further loading of files')
                break

            num_loaded = i + 1

        time_end_process = datetime.datetime.now()
        time_process = time_end_process - time_start_process
        time_average = time_process / len(paths_selected)
        message_time = (f'{num_loaded}/{len(paths_selected)} '
                        f'files processed, '
                        f'total {time_process.total_seconds():.2f} s,'
                        f'average {time_average.total_seconds():.2f} s.')
        window_messages.l_message['text'] = message_time
        logger_general.info(message_time)

        measurements.update_label()
        master.configure(background=color_normal)

    dataset = None
    measurements.sort(confirm=False)

    update_windows()  # TODO redundant?

    _save_meas_dfs = config.get('GENERAL', 'SAVE_MEAS_DFS', fallback=None)
    if _save_meas_dfs == 'ON':
        save_meas_dfs()

    window_main.lift()


def update_windows(_event=False):

    global measurements
    window_main.update_lb_list_measurementsets()
    window_main.update_lb_files()
    window_main.update_lb_list_modes()
    window_main.update_lb_list_pars()
    window_main.update_lb_list_groupby()
    window_main.update_lb_list_meas()
    window_main.update_lb_datasets()
    window_desc_selector.update_lb_list_descs()
    measurements.sort(confirm=False)


def show_selected(_figure=None, twod=False, separate=False, inset=False):

    try:
        mode = window_main.lb_list_modes.get(
            window_main.lb_list_modes.curselection()[0])
    except IndexError:
        logger_general.warning('No mode selected')
        return False

    if mode == 'image':
        if config.get('IMAGES', 'MODE', fallback=None) == 'MATPLOTLIB':
            show_image_selected_mpl(mode)
        else:
            show_images_selected_tk(mode)
    else:
        if separate:
            plot_selected(mode, _figure=_figure, twod=twod, separate=True)
        elif inset:
            plot_selected(mode, _figure=_figure, twod=twod, inset=True)
        else:
            plot_selected(mode, _figure=_figure, twod=twod, separate=False)


def show_images_selected_tk(_mode):

    global window_imageviewer

    window_imageviewer = get_window_imageviewer()

    meas_toshow = []
    for i in window_main.lb_list_measurements.curselection():
        meas_toshow.append(measurements.list[i])

    window_imageviewer.clear()

    labels = []
    images_raw = []

    logger_general.debug(f'Getting image files...')

    for i, meas in enumerate(meas_toshow):

        labels.append(get_label_string(meas, _mode))

        if 'PIL-image' not in meas.data['modes']['image']:
            logger_general.info(f'Loading image file: {meas.data["file"]["path"]}')
            img = Image.open(meas.data['file']['path'])
            meas.data['modes']['image']['PIL-image'] = img

        images_raw.append(meas.data['modes']['image']['PIL-image'])

    logger_general.debug('Populating window images...')

    window_imageviewer.populate(
        images_raw=images_raw, labels=labels, meass_toplot=meas_toshow)


def show_image_selected_mpl(_mode, _figure=None):

    global measurements
    global config

    meas_toshow = []
    for i in window_main.lb_list_measurements.curselection():
        meas_toshow.append(copy.copy(measurements.list[i]))

    if _figure is None:
        if len(meas_toshow) > 1:
            if len(meas_toshow) < 4:
                rows = 1
            elif len(meas_toshow) < 7:
                rows = 2
            else:
                rows = 3
            _figure = figures.add(
                rows=rows, cols=math.ceil(len(meas_toshow) / rows),
                connect_picking=False)
        else:
            _figure = figures.add(connect_picking=False)

    for i, meas in enumerate(meas_toshow):

        label = get_label_string(meas, _mode)

        # TODO move to custimization
        plot = _figure.axs_p[i].imshow(
            meas_toshow[0].data['modes']['image']['nparray'],
            cmap=config.get('IMAGES', 'COLORMAP', fallback=None),
            vmin=config.get('IMAGES', 'VMIN', fallback=None),
            vmax=config.get('IMAGES', 'VMAX', fallback=None))
        _figure.fig.colorbar(plot)

        if not label:
            label = 'N/A'
        _figure.axs_p[i].set_title(label, fontsize=10)

    plt.show()


def get_meas_info(data):

    params = config['INFO']['PARAMETERS'].split(';')
    lines = []
    lines.append(data['file']['name'])

    for param in params:
        try:
            value = data['modes']['analysis']['df'][param].iloc[0]
        except KeyError:
            value = 'N/A'

        lines.append(f'{param} = {value}')

    return '\n'.join(lines)


def move_picked_point(event, fig, direction=None):
    if not direction:
        direction = event.key
    show_picked_point(None, fig, direction=direction)


def show_picked_point(event, fig, direction=None):

    logger_general.debug('Start picking')

    mode = fig.mode

    text_label = ''
    meas_picked = None
    picked_x = None
    picked_y = None
    ax_picked = None

    if event or direction:

        df_singlecols = []

        logger_general.debug('Creating dfs')
        for i, col_name in enumerate(fig.df_plotted.columns):

            if i > 0:
                df_single = pd.DataFrame()
                col_x_name = fig.df_plotted.columns[0]
                df_single['x'] = fig.df_plotted[col_x_name]
                df_single['y'] = fig.df_plotted[col_name]

                df_singlecols.append(df_single)

        df_long = pd.concat(df_singlecols).reset_index(drop=True)
        xy_datas = df_long.to_numpy()

        if event:
            clicked_x = event.x
            clicked_y = event.y
        else:
            clicked_x = fig.picked_x
            clicked_y = fig.picked_y

        logger_general.debug('Clicked: {}, {}'.format(clicked_x, clicked_y))

        axs_plotted = fig.axs_p + fig.axs_s

        dfs = []

        logger_general.debug('Iterate axes, transforms')
        for i, ax in enumerate(axs_plotted):

            df_ax = pd.DataFrame(xy_datas, columns=['x', 'y'])
            df_ax['ind_axis'] = i

            xy_transformed = ax.transData.transform(
                df_ax[['x', 'y']])
            df_ax['x_pix'] = xy_transformed[:, 0]
            df_ax['y_pix'] = xy_transformed[:, 1]

            def get_distance(x1, x2, y1, y2):
                try:
                    return math.sqrt((x1-x2)**2 + (y1-y2)**2)
                except TypeError:
                    return

            def apply_get_distance(_df):
                return get_distance(_df['x_pix'],
                                    clicked_x, _df['y_pix'],
                                    clicked_y)

            df_ax['distance'] = df_ax.apply(apply_get_distance, axis=1)
            if 'distance' not in df_ax:
                logger_general.warning(f'Picking error, cancelling...')
                return

            dfs.append(df_ax)

        df_all = pd.concat(dfs)
        df_all = df_all.reset_index(drop=True)
        df_limited = None

        if not direction:
            df_limited = df_all
        else:
            if direction == 'right':
                df_limited = df_all[
                    (df_all['x_pix'] > clicked_x)
                    & ~(np.isclose(df_all['x_pix'], clicked_x))]
            if direction == 'left':
                df_limited = df_all[
                    (df_all['x_pix'] < clicked_x)
                    & ~(np.isclose(df_all['x_pix'], clicked_x))]
            if direction == 'up':
                df_limited = df_all[
                    (df_all['y_pix'] > clicked_y)
                    & ~(np.isclose(df_all['y_pix'], clicked_y))]
            if direction == 'down':
                df_limited = df_all[
                    (df_all['y_pix'] < clicked_y)
                    & ~(np.isclose(df_all['y_pix'], clicked_y))]

        picked_x = df_limited['x'][df_limited['distance'].idxmin()]
        picked_y = df_limited['y'][df_limited['distance'].idxmin()]

        logger_general.debug(f'Picked point: {picked_x}, {picked_y}.')

        picked_x_final = df_limited['x_pix'][df_limited['distance'].idxmin()]
        picked_y_final = df_limited['y_pix'][df_limited['distance'].idxmin()]

        if fig.picked_x and (np.isclose(picked_x_final, fig.picked_x)
                and np.isclose(picked_y_final, fig.picked_y) and fig.pickmark):
            logger_general.info(f'Repicked the same point, removing mark')
            fig.pickmark.remove()
            fig.pickmark = None
            fig.fig.canvas.draw()
            return
        else:
            fig.picked_x = picked_x_final
            fig.picked_y = picked_y_final

        ax_picked = axs_plotted[
            df_all['ind_axis'][df_all['distance'].idxmin()]]

        fig.picked_ax = ax_picked

        meas_picked: Optional[Measurement] = None
        meas_picked = methods.find_picked(fig, picked_x, picked_y)

    if not meas_picked:
        logger_general.warning('Picked meas. not found.')
        return

    if 'image' in meas_picked.data['modes']:

        fig_image = figures.get_pick()

        if 'nparray' not in meas_picked.data['modes']['image']:
            logger_general.info(f'Loading image file: {meas_picked.data["file"]["path"]}')
            img = mpimg.imread(meas_picked.data['file']['path'])
            meas_picked.data['modes']['image']['nparray'] = img

        fig_image.ax_p.imshow(meas_picked.data['modes']['image']['nparray'])
        fig_image.fig.show()
    else:
        fig_image = None

    if fig.pickmark and fig.pickmark.axes != ax_picked:
        fig.pickmark.remove()
        fig.pickmark = None

    if not fig.pickmark:
        fig.pickmark, = ax_picked.plot(
            picked_x, picked_y, 'o', color='red', markersize=10,
            fillstyle='none')
    else:
        fig.pickmark.set_xdata(picked_x)
        fig.pickmark.set_ydata(picked_y)

    fig.fig.canvas.draw_idle()

    defaults_pick = meas_picked.get_defaults('measurement')   # mode is irrelevant
    modes_pick = str.split(defaults_pick['pick']['modes'], ';')

    if (mode != 'measurement' and meas_picked.data['modes']['measurement'] and
            not fig_image):

        logger_general.debug('Getting pick figure')

        fig_pick = figures.get_pick(clearax=True, rows=len(modes_pick))
        fig_pick.init_color()

        for i, mode_pick in enumerate(modes_pick):

            defaults_pick = meas_picked.get_defaults(mode_pick)

            meas_picked.label_string = get_meas_info(meas_picked.data)

            if 'series-x' in defaults_pick:
                meas_picked.sers_x = [
                    meas_picked.data['modes'][mode_pick]['df'][
                        defaults_pick['series-x']]]
            else:
                meas_picked.sers_x = []
            if 'series-yp' in defaults_pick:
                meas_picked.sers_yp = [
                    meas_picked.data['modes'][mode_pick][
                        'df'][i] for i in defaults_pick[
                        'series-yp'].split(';')]
            else:
                meas_picked.sers_yp = []

            if 'series-ys' in defaults_pick:
                try:
                    meas_picked.sers_ys = [
                        meas_picked.data['modes'][mode_pick][
                            'df'][i] for i in defaults_pick[
                            'series-ys'].split(';')]
                except KeyError:
                    pass
            else:
                meas_picked.sers_ys = []

            plotting.plot_data([meas_picked], 'measurement',
                               figures, _figure=fig_pick, meass_orig=None,
                               ind_ax=i)

        if fig_pick.ax_p.get_legend():
            fig_pick.ax_p.get_legend().remove()  # TODO fix

        # overrides earlier
        if meas_picked.text_info:
            text_label = meas_picked.text_info
        elif meas_picked.label_string:
            text_label = meas_picked.label_string
        else:
            text_label = 'N/A'

    text_label += '\nx={:.3f}, y={:.3f}'.format(picked_x, picked_y)
    window_info.l_info['text'] = text_label
    window_info_top.lift()

    logger_general.debug('Finished picking')

    return


def get_label_string(meas, mode, groupby=None, onlyvalue=False):
    label_list = []
    values_list = []
    if groupby:
        lb_list = window_main.lb_list_groupby
    else:
        lb_list = window_main.lb_list_pars

    value_digits_raw = config.get('PLOT', 'DIGITS', fallback=3)
    value_digits = int(value_digits_raw)

    for index2, keystring in enumerate(
            lb_list.get(0, tk.END)):
        if index2 in lb_list.curselection():
            list_value = methods.get_value_from_dict(
                keystring, meas.data['modes'][mode])
            if not list_value:
                list_value = methods.get_value_from_dict(keystring, meas.data)
            try:
                list_value = float(list_value)
                list_value = '{:.{value_digits}f}'.format(
                    list_value, value_digits=value_digits)
            except ValueError:
                list_value = str(list_value)
            except TypeError:
                list_value = 'N/A'

            values_list.append(list_value)

            label_list.append(keystring.split('>')[-1] + '=' + str(list_value))
    label_string = ' ; '.join(label_list)
    if onlyvalue:
        label_string = ';'.join(values_list)

    return label_string


def run_fit_xrd_test():
    meass_toplot, meas_orig = get_selections(
        'measurement', force_deepcopy=False)
    fit_xrd.fit_and_save(meass_toplot, app, testing=True)


def run_fit_xrd():

    meass_toplot, meas_orig = get_selections(
        'measurement', force_deepcopy=False)
    fit_xrd.fit_and_save(meass_toplot, app)


def run_extra():

    global window_extra

    logger_general.info('Starting extra...')
    try:
        mode = window_main.lb_list_modes.get(
            window_main.lb_list_modes.curselection()[0])
    except IndexError:
        logger_general.warning('No mode selected')
        return False

    meas_toplot, meas_orig = get_selections(mode, force_deepcopy=False)
    logger_general.info(f'Sending {len(meas_toplot)} datas to extra.')
    window_extra = extra.run_extra(meas_toplot, window_buttons,
                                   config_paths=config_paths)


def record_values():

    global window_recorder

    try:
        mode = window_main.lb_list_modes.get(
            window_main.lb_list_modes.curselection()[0])
    except IndexError:
        logger_general.warning('No mode selected')
        return False

    meas_toplot, meas_orig = get_selections(mode, force_deepcopy=False)

    # TODO move this to records module
    plt.ioff()
    fig_record = figures.add(mode=mode)

    path_records = config_paths.get(
        'VALUE_RECORDER', 'PATH_RECORDS_FIT', fallback=None)

    if path_records:
        window_recorder = value_recorder.record_values(
            meas_toplot, path_records, fig_record, figures)
    else:
        logger_general.warning(f'No records path specified')


def plot_selected(mode, _figure=None, twod=False, title=None, separate=False,
                  inset=False):

    logger_general.debug('Plotting...')

    meas_toplot, meas_orig = get_selections(mode, twod)

    if separate:
        for meas_toplot_single, meas_orig_single in zip(meas_toplot, meas_orig):
            plotting.plot_data([meas_toplot_single], mode, figures,
                               _figure=_figure, meass_orig=[meas_orig_single],
                               twod=twod, keep_ref=False, savefig=True)

    else:
        plotting.plot_data(meas_toplot, mode, figures, _figure=_figure,
                           meass_orig=meas_orig, twod=twod, inset=inset)

        _save_fig_dfs = config.get('GENERAL', 'SAVE_FIG_DFS', fallback=None)
        if _save_fig_dfs == 'ON':
            save_fig_dfs()

        _save_figs = config.get('GENERAL', 'SAVE_FIGS', fallback=None)
        if _save_figs == 'ON':
            plotting.save_fig(_figure, config_paths)


def get_selections(mode, twod=False, force_deepcopy=None):

    logger_general.debug(f'Getting user selections')

    meas_toplot: List[Measurement] = []
    meas_orig = []

    if mode == 'measurement':
        offset_y_raw = float(window_buttons.e_offset_y.get())
        offset_x = float(window_buttons.e_offset_x.get())
        multiply_p = float(window_buttons.e_multiply_p.get())
        multiply_s = float(window_buttons.e_multiply_s.get())
    else:
        offset_y_raw = 0
        offset_x = 0
        multiply_p = 1
        multiply_s = 1

    if force_deepcopy is True:
        deepcopy_default = True
    else:
        deepcopy_default = False

    for i in window_main.lb_list_measurements.curselection():
        meas_orig.append(measurements.list[i])
        if deepcopy_default:
            # deepcopy can lead to high memory use
            logger_general.warning(f'Using deepcopy (get selections)')
            meas_toplot.append(copy.deepcopy(measurements.list[i]))
        else:
            meas_toplot.append(copy.copy(measurements.list[i]))

        meas_toplot[-1].sers_x = []
        meas_toplot[-1].sers_yp = []
        meas_toplot[-1].sers_ys = []
        meas_toplot[-1].sers_c = []
        meas_toplot[-1].sers_shape = []
        meas_toplot[-1].sers_size = []

    logger_general.debug(f'Finished creating copies')

    if 'df' not in meas_toplot[0].data['modes'][mode]:
        logger_general.debug('No df found, cancelling...')
        return

    sernames_x = []
    sernames_yp = []
    sernames_ys = []
    sernames_c = []
    sernames_size = []
    sernames_shape = []

    logger_general.debug('Getting series selections')
    for sel in window_desc_selector.lb_list_desc_x.curselection():
        sernames_x.append(window_desc_selector.lb_list_desc_x.get(sel))
    for sel in window_desc_selector.lb_list_desc_yp.curselection():
        sernames_yp.append(window_desc_selector.lb_list_desc_yp.get(sel))
    for sel in window_desc_selector.lb_list_desc_ys.curselection():
        sernames_ys.append(window_desc_selector.lb_list_desc_ys.get(sel))
    for sel in window_desc_selector.lb_list_desc_c.curselection():
        sernames_c.append(window_desc_selector.lb_list_desc_c.get(sel))
    for sel in window_desc_selector.lb_list_desc_size.curselection():
        sernames_size.append(window_desc_selector.lb_list_desc_size.get(sel))
    for sel in window_desc_selector.lb_list_desc_shape.curselection():
        sernames_shape.append(window_desc_selector.lb_list_desc_shape.get(sel))


    def concat_all(measurements_toplot_):

        dfs_toplot = [
            j.data['modes'][mode]['df'] for j in measurements_toplot_]

        # need to use deepcopy to avoid changing original measurements
        measurement_toplot = copy.deepcopy(measurements_toplot_[0])
        df_toplot_ = pd.concat(dfs_toplot)
        df_toplot_ = df_toplot_.sort_values(by=sernames_x[0])

        df_toplot_ = df_toplot_.reset_index(drop=True)

        # for data with placeholder y value (separate points in y axis)
        if 'point' in df_toplot_:
            for value in df_toplot_['x'].unique():
                for counter, index3 in enumerate(
                        df_toplot_[df_toplot_['x'] == value].index):
                    df_toplot_['point'][index3] += 0.1 * counter

        measurement_toplot.data['modes'][mode]['df'] = df_toplot_

        return [measurement_toplot]


    def group_meas(meass, _keystring):

        group_dict = {}

        for meas in meass:
            groupby = methods.get_value_from_dict(_keystring, meas.data)
            if groupby not in group_dict:
                group_dict[groupby] = []
            group_dict[groupby].append(meas)

        meass_conc_all = []

        group_dict = dict(sorted(group_dict.items()))

        for key in group_dict.keys():
            if len(group_dict[key]) > 1:
                meass_conc = concat_if_single(group_dict[key])[0]
            else:
                meass_conc = group_dict[key][0]
            meass_conc_all.append(meass_conc)

        return meass_conc_all

    def concat_if_single(measurements_toplot_):

        dfs_toplot = [
            j.data['modes'][mode]['df'].dropna(axis=1) for j in measurements_toplot_]

        # if length of each df is one, concatenate to one df
        if len(dfs_toplot[0]) == 1 and len(
                set([len(j) for j in dfs_toplot])) == 1:

            new = Measurement()
            measurement_toplot = methods.copy_meas(measurements_toplot_[0], new)
            df_toplot_ = pd.concat(dfs_toplot)
            df_toplot_ = df_toplot_.sort_values(by=sernames_x[0])
            df_toplot_ = df_toplot_.reset_index(drop=True)

            # for data with placeholder y value (separate points in y)
            if 'point' in df_toplot_:
                for value in df_toplot_['x'].unique():
                    for counter, index3 in enumerate(
                            df_toplot_[df_toplot_['x'] == value].index):
                        df_toplot_['point'][index3] += 0.1 * counter

            measurement_toplot.data['modes'][mode]['df'] = df_toplot_
            measurement_toplot.isdeepcopy = True

            return [measurement_toplot]

        else:
            return measurements_toplot_

    if window_main.lb_list_groupby.curselection():
        groupby = True
        keystring = window_main.lb_list_groupby.get(
            window_main.lb_list_groupby.curselection())
        meas_toplot = group_meas(meas_toplot, keystring)

    else:
        meas_toplot = concat_if_single(meas_toplot)

        groupby = False

    substrings = None

    if config.get('PLOT', 'LEGEND_ONLY-VALUES', fallback='OFF') == 'ON':
        common_string = ''
        substrings = [get_label_string(
            meas, mode, groupby=groupby, onlyvalue=True) for meas in meas_toplot]
    elif config.get('PLOT', 'LEGEND_SHORTENER', fallback='OFF') == 'ON'\
            and mode == 'measurement':
            common_string, substrings = methods.get_substrings(
                [get_label_string(meas, mode) for meas in meas_toplot])
    else:
        common_string = ''
        substrings = [get_label_string(
            meas, mode, groupby=groupby) for meas in meas_toplot]

    for index, meas in enumerate(meas_toplot):
        label_string = substrings[index]

        meas.label_string = label_string.replace('$', '\$')
        meas.common_string = common_string


    # ------------ Apply modifiers to selected measurements ----------

    for index, meas in enumerate(meas_toplot):

        logger_general.debug(f'Applying modifiers to meas '
                             f'{index}/{len(meas_toplot)}')

        df_toplot: pd.DataFrame = meas.data['modes'][mode]['df']
        meas.data['modes'][mode]['df_toplot'] = df_toplot

        # apply filter
        filter_string = window_buttons.e_filter.get()
        # TODO
        if mode == 'analysis' and filter_string:
            logger_general.debug(f'Filtering with: {filter_string}')
            df_toplot = df_toplot.query(str(filter_string))

        # ----------- get selected modifiers from entry boxes -----------
        counter_meas = 0

        meas.offset_y_raw = offset_y_raw
        meas.offset_x = offset_x
        meas.multiply_p = multiply_p
        meas.multiply_s = multiply_s
        meas.offset_y_meas = ((counter_meas - 1) * meas.offset_y_raw)
        meas.index = index

        for sername_yp in sernames_yp:
            if sername_yp in df_toplot.columns:
                df_toplot[sername_yp] = (
                        pd.to_numeric(
                            df_toplot[sername_yp], errors='coerce'))
                meas.sers_yp.append(df_toplot[sername_yp])

        for sername_ys in sernames_ys:
            if sername_ys in df_toplot.columns:
                df_toplot[sername_ys] = (
                        pd.to_numeric(
                            df_toplot[sername_ys], errors='coerce'))
                meas.sers_ys.append(df_toplot[sername_ys])

        for sername_x in sernames_x:
            if sername_x in df_toplot.columns:

                if not pd.api.types.is_datetime64_any_dtype(
                    df_toplot[sername_x]):
                    df_toplot[sername_x] = (
                            pd.to_numeric(df_toplot[sername_x],
                                          errors='coerce'))
                meas.sers_x.append(df_toplot[sername_x])

        for sername_c in sernames_c:
            df_toplot[sername_c] = (
                    pd.to_numeric(df_toplot[sername_c],
                                  errors='coerce'))
            meas.sers_c.append(df_toplot[sername_c])

        for sername_size in sernames_size:
            df_toplot[sername_size] = (
                    pd.to_numeric(df_toplot[sername_size],
                                  errors='coerce'))
            meas.sers_size.append(df_toplot[sername_size])

        for sername_shape in sernames_shape:
            df_toplot[sername_shape] = (
                    pd.to_numeric(df_toplot[sername_shape],
                                  errors='coerce'))
            meas.sers_shape.append(df_toplot[sername_shape])

    if twod:
        meas_toplot = concat_all(meas_toplot)

    logger_general.debug(f'Got user selections')
    return meas_toplot, meas_orig


def view_df():

    if measurements:
        dfs = []
        for meas in measurements.list:
            dfs.append(get_data_from_measurement(meas))

        df_toview = pd.concat(dfs)
        df_viewer.create_dfviewer(df_toview)

    else:
        logger_general.warning('No measurements to view.')


def view_df_fig():
    df_toview: pd.DataFrame() = pd.concat(figure.dfs)
    topwindow_dfviewer = tk.Toplevel(width=300, height=100)
    _window_dfviewer = df_viewer.WindowDfViewer(topwindow_dfviewer,
                                                 df_toview, 20)


def copy_fig_dfs_to_clipboard():

    if app.figure:
        df_combined = pd.concat(app.figure.dfs)
        df_combined.to_clipboard()
        logger_general.info(
            'Copied {} dfs to clipboard.'.format(len(app.figure.dfs)))


def save_fig_dfs(path_tosave=None):

    if app.figure:
        df_combined: pd.DataFrame() = pd.concat(app.figure.dfs)

        filename = (app.figure.name
                    + "_" + datetime.datetime.now().isoformat()[0:19]
                    + '.csv').replace(':', '-').replace('|', '_')

        if not path_tosave:
            result_dfs_dir = config_paths.get('GUI', 'RESULT_DFS',
                                              fallback=None)
            path_tosave = os.path.join(result_dfs_dir, filename)

        if path_tosave:
            df_combined.to_csv(path_tosave)
            logger_general.info(
                f'Saved dfs to {path_tosave}.')


def save_meas_dfs(path_tosave=None):

    global measurements

    dfs = []

    if measurements:
        for meas in measurements.list:
            dfs.append(get_data_from_measurement(meas))

        if dfs:

            df_comb = pd.concat(dfs)

            filename = (measurements.label
                        + "_" + datetime.datetime.now().isoformat()[0:19]
                        + '.csv').replace(':', '-').replace('|', '_').replace(
                '/', '-')

            if not path_tosave:
                result_dfs_dir = config_paths.get('GUI', 'RESULT_DFS',
                                                  fallback=None)
                path_tosave = os.path.join(result_dfs_dir, filename)

            if path_tosave:
                df_comb.to_csv(path_tosave)
                logger_general.info(
                    f'Saved dfs to {path_tosave}.')
        else:
            logger_general.warning('No dfs to save')


def copy_meas_dfs_to_clipboard():

    global measurements

    dfs = []

    if measurements:
        for meas in measurements.list:
            dfs.append(get_data_from_measurement(meas))

        df_comb = pd.concat(dfs)

        df_comb.to_clipboard()
        logger_general.info(
            f'Copied {len(dfs)} dfs to clipboard.')


def view_df_default_app():

    global temp_directory

    dfs = []
    if measurements:
        for meas in measurements.list:
            dfs.append(get_data_from_measurement(meas))

        df_toview = pd.concat(dfs)

        if temp_directory:
            path_tempfile = (conv_main.expand_envs_in_paths(temp_directory)
                             + '/TEMP_DF_CSV.csv')
            df_toview.to_csv(path_tempfile)
            if sys.platform == 'win32':
                os.startfile(path_tempfile, 'open')
            elif sys.platform == 'linux':
                subprocess.Popen(['xdg-open', path_tempfile])
            else:
                logger_general.warning(
                    f'OS not recognized: {sys.platform}')

        else:
            logger_general.warning('TEMP_DIRECTORY not defined in cfg')
    else:
        logger_general.warning('No measurements to view.')


def sort_measurements():
    global measurements
    measurements.sort()


class Figures:

    global figure

    def __init__(self):
        self.list: List[Figure] = []
        self.counter = 0
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        self.pick = None

    def clear(self):
        self.list = []

    def add(self, rows=1, cols=1, mode=None, connect_picking=None, keep_ref=True):

        if keep_ref:
            global figure

        self.counter += 1
        if connect_picking == None and config.get(
                'PLOT', 'PICKING', fallback='OFF') == 'ON':
            if mode == 'measurement':
                connect_picking = False
            else:
                connect_picking = True
        elif connect_picking is None:
            connect_picking = False

        figure = Figure(self.counter, rows=rows, cols=cols,
                        connect_picking=connect_picking)

        if keep_ref:
            app.figure = figure
        self.list.append(figure)
        window_buttons.update_l_figure()
        window_main.lb_list_figures.insert(tk.END,  figure.name)

        return figure

    def set_figure(self, name: str):

        global figure
        for fig in self.list:
            if fig.name == name:
                figure = fig
                app.figure = fig
                window_buttons.update_l_figure()

    def get_pick(self, clearax=False, connect_click=False, rows=1, cols=1):
        try:
            self.pick.fig.canvas.manager.set_window_title('Picked curve')
        except (_tkinter.TclError, AttributeError):
            self.pick = None
        if not self.pick:
            self.pick = Figure('pick', rows=rows, cols=cols)
            if not self.pick.ax_p:
                self.pick.ax_p = self.pick.axs_p[0]
            if connect_click:
                self.pick.fig.canvas.mpl_connect(
                    'button_press_event',
                    lambda event: plotting.click_dist(event, self.pick))

            self.counter += 1
            self.list.append(figure)

        if clearax:
            for ax in self.pick.axs_p:
                ax.clear()
            for ax in self.pick.axs_s:
                ax.clear()

        return self.pick


class Figure:
    def __init__(self, counter, rows=1, cols=1, connect_picking=False):
        self.name = 'Fig. ' + str(counter)

        if config.get('PLOT', 'TIGHT_LAYOUT_DEFAULT') == 'ON':
            tight_layout = True
        else:
            tight_layout = False

        size_raw=config.get('PLOT', 'SIZE', fallback=None)
        if size_raw:
            size = [float(i) for i in size_raw.split(';')]
        else:
            size=None

        self.ax_s: Optional[plt.Axes] = None
        self.axs_p: List[plt.Axes] = []
        self.axs_s: List[plt.Axes] = []
        kwargs = {}
        dpi_raw = config.get('PLOT', 'FIG_DPI', fallback=None)
        if dpi_raw:
            dpi = float(dpi_raw)
            kwargs['dpi'] = dpi
        # TODO fix tight layout
        self.fig: plt.Figure = plt.figure(figsize=size, **kwargs)

        for ind in range(rows * cols):
            self.axs_p.append(
                self.fig.add_subplot(rows, cols, ind+1))

        self.ax_p: Optional[plt.Axes] = self.axs_p[0]
        self.ax_inset: Optional[plt.Axes] = None
        self.title = None
        self.number = 0
        self.color_current = -1
        self.dfs = []
        self.lines_p = []
        self.lines_s = []
        self.meass_orig = []
        self.meass_toplot = []
        self.pickmark = None
        self.picked_ind = None
        self.picked_line = None
        self.picked_x = None
        self.picked_y = None
        self.picked_ax = None
        self.next_to_curves_title = None
        self.dist_point_1 = None
        self.dist_point_2 = None
        self.dist_coords_1 = None
        self.dist_coords_2 = None
        self.dist_lines = []
        self.dist_points = []
        self.dfs_plotted: List[pd.DataFrame] = []
        self.df_plotted: Optional[pd.DataFrame] = None
        self.sers_x = []
        self.sers_yp = []
        self.sers_ys = []
        self.sers_c = []
        self.sers_size = []
        self.sers_shape = []
        self.line_gb = None
        self.line_lm = None
        self.vlines = []
        self.line_h_zero = None
        self.vlines_extra = []
        self.hlines_extra = []
        self.labels_text = []
        self.colorbars = []
        self.legends = []
        self.cid_picking_key = None
        self.cid_picking_button = None
        self.curves = []

        self.df_orig = pd.DataFrame()

        self.mode = window_main.lb_list_modes.get(
            window_main.lb_list_modes.curselection()[0])

        window_buttons.update_l_figure()

        if connect_picking:
            self.connect_picking()

        curr_time = datetime.datetime.now().time()
        self.window_title = f'{self.name} [{str(curr_time)[0:9]}]'

        self.fig.canvas.manager.set_window_title(self.window_title)

        self.fig.canvas.mpl_connect(
            'close_event', window_main.lb_list_figures_delete_closed)


    def connect_picking(self):
        self.cid_picking_button = self.fig.canvas.mpl_connect(
            'button_press_event',
            lambda event: show_picked_point(
                event, self))
        self.cid_picking_key = self.fig.canvas.mpl_connect(
            'key_press_event',
            lambda event: move_picked_point(
                event, self))
        picking_buttons_raw = config.get(
            'PLOT', 'PICKING_BUTTONS', fallback=None)
        if picking_buttons_raw == 'ON':
            self.ax_button_left = self.fig.add_axes([0.01, 0.01, 0.12, 0.05])
            self.button_left = Button(self.ax_button_left, 'Left')
            self.button_left.on_clicked(
                lambda event: move_picked_point(event, self, direction='left'))
            self.ax_button_right = self.fig.add_axes([0.15, 0.01, 0.12, 0.05])
            self.button_right = Button(self.ax_button_right, 'Right')
            self.button_right.on_clicked(
                lambda event: move_picked_point(event, self, direction='right'))


    def disconnect_picking(self):
        self.fig.canvas.mpl_disconnect(self.cid_picking_button)
        self.fig.canvas.mpl_disconnect(self.cid_picking_key)

    def clear_dist(self):
        self.dist_point_1 = None
        self.dist_point_2 = None
        self.dist_coords_1 = None
        self.dist_coords_2 = None
        for line in self.dist_lines:
            line.remove()
        for point in self.dist_points:
            point.remove()
        self.dist_lines = []
        self.dist_points = []

    def init_color(self):
        self.color_current = -1

    def get_color_same(self):
        if self.color_current == -1:

            return figures.colors[0]
        return figures.colors[self.color_current]

    def get_color_next(self, secondary=False, force=True):

        if (len(self.meass_toplot) == 1 and len(
                self.sers_yp) == 1 and len(self.sers_ys) <= 1) and not force:
            if secondary:
                color = self.meass_toplot[0].get_defaults()[
                    'plot']['color-secondary']
            else:
                color = self.meass_toplot[0].get_defaults()[
                    'plot']['color-primary']
            return color

        self.color_current = self.color_current + 1
        try:
            color = figures.colors[self.color_current]
        except IndexError:
            self.color_current = 0
            color = figures.colors[self.color_current]

        return color


class Measurements:

    def sort(self, confirm=True):
        if not self.list or (not self.list[0].sort_values):
            return

        message_order = ','.join(
            [str(i+1) for i in range(len(self.list[0].sort_values))])
        order_new_raw = message_order
        if confirm:
            order_new_raw = simpledialog.askstring(
                title='Query', prompt='New sort order:',
                initialvalue=message_order)

        if order_new_raw.startswith('-'):
            order_reverse = True
            order_new_raw = order_new_raw[1:]
        else:
            order_reverse = False

        order_new = [int(i)-1 for i in str.split(order_new_raw, ',')]

        if len(order_new) == len(self.list[0].sort_values):

            try:
                list_sorted = sorted(
                    self.list, key=lambda x: tuple(
                        [x.sort_values[i] for i in order_new]),
                    reverse=order_reverse)
            except IndexError:
                logger_general.warning('List sorting failed (IndexError)')
                list_sorted = None

            if list_sorted:
                self.list = list_sorted
                window_main.update_lb_list_meas()
        else:
            logger_general.warning('Invalid sort order.')

    def clear(self):
        self.list = []

    def add_file(self, path_source, index=0, generic=False):
        """ Get new Measurement object(s) to current Measurements from file
        using converters.
        """

        datas = []
        datas_new = conv_main.get_data_from_source(
            path_source, generic=generic)
        if datas_new:
            datas.extend(datas_new)
            for data in datas:
                self.list.append(Measurement(data, self))
            logger_general.info('Added ' + str(len(datas_new)) +
                                ' measurements from ' + path_source)
        else:
            logger_general.warning('No new data from: ' + path_source)


    def update_label(self):

        self.label = methods.get_label(self.list)


    def __init__(self, append_to_all=True):
        self.list = []
        if append_to_all:
            measurements_all.append(self)
        self.number = len(measurements_all)
        self.label = 'N/A'
        self.selected = []
        self.combined = []


class Measurement:

    def get_defaults(self, mode=None):

        if not mode:
            mode = window_main.lb_list_modes.get(
                window_main.lb_list_modes.curselection()[0])
        defaults = copy.deepcopy(self.data['defaults'])
        if 'defaults' in self.data['modes'][mode]:
            methods.update_nested_dict(defaults, self.data['modes'][mode]['defaults'])
        return defaults

    def __init__(self, data=None, parent=None):
        if not data:
            self.data = {}
            self.data['modes'] = {'measurement': {}, 'analysis': {}}
        else:
            self.data = data
        self.parent = parent
        self.sort_value = None
        self.sort_values = []

        self.sers_ys = []
        self.sers_yp = []
        self.sers_x = []
        self.sers_c = []
        self.sers_size = []
        self.sers_shape = []
        self.text_info = ''
        self.label_string = ''
        self.common_string = ''
        self.offset_y_raw = None
        self.offset_x = 0
        self.multiply_p = 1
        self.multiply_s = 1
        self.offset_y_meas = 0
        self.isdeepcopy = False


class WindowListFiles(tk.Frame):

    def __init__(self, _master):

        super().__init__(_master)
        self.master = _master
        self.grid(row=1, column=1)
        self.l_files = tk.Label(self, text='Select files to load:')
        self.l_files.grid(row=1, column=1)
        self.sb_files = tk.Scrollbar(self)
        self.sb_files.grid(row=2, column=2, sticky=tk.N + tk.S)
        self.lb_files = tk.Listbox(self, height=40,
                                   width=100,
                                   font=('Consolas', 10),
                                   yscrollcommand=self.sb_files.set,
                                   selectmode=tk.EXTENDED,
                                   exportselection=False)
        self.lb_files.grid(row=2, column=1)
        self.sb_files.config(command=self.lb_files.yview)

        self.bt_load = tk.Button(self, text='Load', command=self.load_files)
        self.bt_load.grid(row=3, column=1)

        self.bt_cancel = tk.Button(self, text='Cancel', command=self.exit)
        self.bt_cancel.grid(row=4, column=1)

    def load_files(self):

        paths = []

        for sel in self.lb_files.curselection():
            path = self.lb_files.get(sel)
            paths.append(path)

        add_files(paths)

        self.master.destroy()

    def exit(self):
        self.master.destroy()


class WindowMain(tk.Frame):

    def select_figure(self, _event=False):

        name_fig = window_main.lb_list_figures.get(
            window_main.lb_list_figures.curselection()[0])
        figures.set_figure(name_fig)
        app.figure.fig.show()


    def lb_modes_update_others(self, _event=False):
        if window_desc_selector:
            window_desc_selector.update_lb_list_descs()

    def unused_select_dataset(self, _event=False):
        dataset_toload = datasets[int(self.lb_datasets.get(
            self.lb_datasets.curselection()[0]))-1]
        dataset_toload.load()

    def select_measurements(self, _event=False):

        global measurements_all
        global measurements

        if len(self.lb_list_measurementsets.curselection()) == 1:
            measurements = measurements_all[
                self.lb_list_measurementsets.curselection()[0]]
        else:
            measurements = Measurements(append_to_all=False)
            measurements.label = ''

            for selection in self.lb_list_measurementsets.curselection():
                measurements.list.extend(measurements_all[selection].list)
                measurements.label += '+' + measurements_all[selection].label

                measurements.combined.append(selection)

        logger_general.info(
            f'Selected {len(measurements.list)} measurements'
            f' from {len(self.lb_list_measurementsets.curselection())}'
            f'sets')
        update_windows()

    def update_lb_datasets(self):
        global datasets
        global dataset
        self.lb_datasets.delete(0, tk.END)
        for i, dataset_temp in enumerate(datasets):
            self.lb_datasets.insert(tk.END, dataset_temp.data['name'])
            if dataset_temp == dataset:
                self.lb_datasets.select_set(i)

    def update_lb_list_measurementsets(self):

        self.lb_list_measurementsets.delete(0, tk.END)
        for i, measr in enumerate(measurements_all):
            self.lb_list_measurementsets.insert(tk.END, str(measr.label).replace('EXAMPLE-',''))
            if measr == measurements or i in measurements.combined:
                self.lb_list_measurementsets.select_set(i)

    def update_lb_list_meas(self, _event=False):

        self.lb_list_measurements.delete(0, tk.END)
        list_label_parts_strings = []  # TODO addition

        for i, meas in enumerate(measurements.list):
            label_parts = []
            for sel_index in self.lb_list_pars.curselection():
                keystring = self.lb_list_pars.get(sel_index)
                label_parts.append(str(methods.get_value_from_dict(keystring,
                                                           meas.data)))

            # clean up the label parts for sorting (float/string)
            meas.sort_values = []
            for part in label_parts:
                try:
                    value = float(part)
                except ValueError:
                    value = np.sum([int(ord(char)) for char in part])
                if np.isnan(value):
                    value = 0

                meas.sort_values.append(value)

            label_parts_strings = []
            for value in label_parts:

                try:
                    value = float(value)
                except ValueError:
                    pass

                if type(value) == float:
                    label_parts_strings.append('{:.3f}'.format(float(value)))
                else:
                    label_parts_strings.append(str(value))

            meas.sort_value = ' ; '.join(label_parts)

            if label_parts_strings:
                list_label_parts_strings.append(label_parts_strings)

        # get max widths
        if list_label_parts_strings:
            max_widths = [0 for _i in range(len(list_label_parts_strings[0]))]
            for label_parts_strings in list_label_parts_strings:
                for i, part in enumerate(label_parts_strings):
                    if len(part) > max_widths[i]:
                        max_widths[i] = len(part)

            for i, label_parts_strings in enumerate(list_label_parts_strings):

                parts_filled = []
                for j, part in enumerate(label_parts_strings):
                    parts_filled.append(str.ljust(part, max_widths[j]))

                self.lb_list_measurements.insert(
                    tk.END, ' ; '.join(parts_filled))
                self.lb_list_measurements.select_set(i)

    def update_lb_list_modes(self):
        self.lb_list_modes.delete(0, tk.END)
        if measurements.list:
            for i, measurement_mode in enumerate(
                    measurements.list[0].data['modes'].keys()):
                self.lb_list_modes.insert(tk.END, measurement_mode)
                if i == 0:
                    self.lb_list_modes.select_set(0)
                if 'mode' in measurements.list[0].data['defaults']:
                    if measurement_mode == measurements.list[0].data[
                            'defaults']['mode']:
                        self.lb_list_modes.selection_clear(0, 'end')
                        self.lb_list_modes.select_set(i)

    def update_lb_files(self):
        self.lb_files.delete(0, tk.END)
        for meas in measurements.list:
            if 'name_parent' in meas.data['file']:
                if not meas.data['file'][
                           'name_parent'] in self.lb_files.get(0, tk.END):
                    self.lb_files.insert(
                        tk.END, meas.data['file']['name_parent'])
            else:
                self.lb_files.insert(tk.END, meas.data['file']['name'])

    def update_lb_list_groupby(self, _event=False):

        defaults_descriptions = None

        self.lb_list_groupby.delete(0, tk.END)

        if measurements.list:

            list_data_all = []

            for meas in measurements.list:
                dict_pars = {}

                for key in meas.data:
                    if key != 'modes' and key != 'defaults':
                        dict_pars[key] = meas.data[key]

                list_data = methods.dict_to_list(dict_pars, list_data=[])
                list_data_all.extend(list_data)

            list_data_all = sorted(set(list_data_all))

            defaults = measurements.list[0].get_defaults()
            if 'selection' in defaults and 'groupby' in defaults[
                    'selection']:
                defaults_descriptions = defaults[
                    'selection']['groupby'].split(';')

            counter_list = 0

            for keystring in list_data_all:

                if not 'df_' in keystring and not 'sd' in keystring:

                    self.lb_list_groupby.insert(tk.END, keystring)

                    if defaults_descriptions and (
                            keystring in defaults_descriptions):
                        self.lb_list_groupby.select_set(counter_list)

                    counter_list += 1

    def update_lb_list_pars(self):

        defaults_descriptions = None

        self.lb_list_pars.delete(0, tk.END)

        if measurements.list:

            list_data_all = []

            for meas in measurements.list:
                dict_pars = {}
                for key in meas.data:
                    if key != 'modes' and key != 'defaults':
                        dict_pars[key] = meas.data[key]

                list_data = methods.dict_to_list(dict_pars, list_data=[])
                list_data_all.extend(list_data)

            list_data_all = sorted(set(list_data_all))

            defaults = measurements.list[0].get_defaults()
            if 'selection' in defaults and 'descriptions' in defaults[
                    'selection']:
                defaults_descriptions = defaults[
                    'selection']['descriptions'].split(';')

            counter_list = 0

            for keystring in list_data_all:
                if (not str.endswith(keystring, '>sd')
                        and not 'df_' in keystring):
                    self.lb_list_pars.insert(tk.END, keystring)

                    if defaults_descriptions and (
                            keystring in defaults_descriptions):
                        self.lb_list_pars.select_set(counter_list)

                    counter_list += 1

    def lb_list_figures_delete_closed(self, event):

        fig_closed = event.canvas.figure
        for i, fig in enumerate(figures.list[::-1]):
            if fig.fig is fig_closed:
                del(figures.list[::-1][i])
                logger_general.debug(
                    f'Closed figure number {len(figures.list)-i} '
                    f'(from window)')
                for j, item in enumerate(self.lb_list_figures.get(0, tk.END)):
                    if item == fig.name:
                        self.lb_list_figures.delete(j)

    def lb_list_figures_delete_selected(self, _event=False):

        global figures

        names_fig_to_delete = []
        # need to delete listbox items in reverse order, since index
        # will update each time
        for selection in self.lb_list_figures.curselection()[::-1]:
            names_fig_to_delete.append(self.lb_list_figures.get(selection))
            self.lb_list_figures.delete(selection)

        fig: Figure
        for i, fig in enumerate(figures.list[::-1]):
            if fig.name in names_fig_to_delete:
                plt.close(fig.fig)
                del(figures.list[::-1][i])
                logger_general.debug(
                    f'Closed figure number {len(figures.list) - i} ')



    def __init__(self, _master):

        super().__init__(_master)
        self.master = _master

        master.geometry('+50+50')

        self.l_datasets = tk.Label(self, text='Datasets')
        self.l_datasets.grid(row=4, column=1)
        self.sb_datasets = tk.Scrollbar(self)
        self.sb_datasets.grid(row=5, column=2, sticky=tk.N + tk.S)
        self.lb_datasets = tk.Listbox(self, height=5,
                                      width=config['INTERFACE'][
                                          'WIDTH_WINDOWS'],
                                      font=('Consolas', 10),
                                      yscrollcommand=self.sb_datasets.set,
                                      selectmode=tk.EXTENDED,
                                      exportselection=False)
        self.lb_datasets.grid(row=5, column=1)
        self.sb_datasets.config(command=self.lb_datasets.yview)

        self.l_files = tk.Label(self, text='Selected files')
        self.l_files.grid(row=4, column=3)
        self.sb_files = tk.Scrollbar(self)

        self.sb_files.grid(row=5, column=4, sticky=tk.N + tk.S)
        self.lb_files = tk.Listbox(self, height=5,
                                   width=config['INTERFACE'][
                                       'WIDTH_WINDOWS'],
                                   font=('Consolas', 10),
                                   yscrollcommand=self.
                                   sb_files.set,
                                   selectmode=tk.EXTENDED,
                                   exportselection=False)

        self.lb_files.grid(row=5, column=3)
        self.sb_files.config(command=self.lb_files.yview)

        self.l_list_pars = tk.Label(self, text='List parameters')
        self.l_list_pars.grid(row=6, column=3)
        self.sb_list_pars = tk.Scrollbar(self)

        self.sb_list_pars.grid(row=7, column=4, sticky=tk.N + tk.S)
        self.lb_list_pars = tk.Listbox(self, height=11,
                                       width=config['INTERFACE'][
                                           'WIDTH_WINDOWS'],
                                       font=('Consolas', 10),
                                       yscrollcommand=self.sb_list_pars.set,
                                       selectmode=tk.EXTENDED,
                                       exportselection=False)
        self.lb_list_pars.grid(row=7, column=3)

        self.sb_list_pars.config(command=self.lb_list_pars.yview)

        self.lb_list_pars.bind('<<ListboxSelect>>', self.update_lb_list_meas)

        self.l_list_measurements = tk.Label(self, text='Measurement list')
        self.l_list_measurements.grid(row=6, column=5)
        self.sb_list_measurements = tk.Scrollbar(self)

        self.sb_list_measurements.grid(row=7, column=6, sticky=tk.N + tk.S)
        self.lb_list_measurements = tk.Listbox(self, height=11,
                                               width=config['INTERFACE'][
                                                   'WIDTH_WINDOWS'],
                                               font=('Consolas', 10),
                                               yscrollcommand=self.
                                               sb_list_measurements.set,
                                               selectmode=tk.EXTENDED,
                                               exportselection=False)
        self.lb_list_measurements.grid(row=7, column=5)

        self.sb_list_measurements.config(
            command=self.lb_list_measurements.yview)

        self.l_list_measurementsets = tk.Label(self, text='Measurement sets')
        self.l_list_measurementsets.grid(row=4, column=5)
        self.sb_list_measurementsets = tk.Scrollbar(self)

        self.sb_list_measurementsets.grid(row=5, column=6, sticky=tk.N + tk.S)
        self.lb_list_measurementsets = tk.Listbox(self, height=5,
                                                  width=config['INTERFACE'][
                                                      'WIDTH_WINDOWS'],
                                                  font=('Consolas', 10),
                                                  yscrollcommand=self.
                                                  sb_list_measurementsets.set,
                                                  selectmode=tk.EXTENDED,
                                                  exportselection=False)
        self.lb_list_measurementsets.grid(row=5, column=5)

        self.sb_list_measurementsets.config(command=self.
                                            lb_list_measurementsets.yview)

        self.lb_list_measurementsets.bind('<<ListboxSelect>>',
                                          self.select_measurements)


        self.l_list_figures = tk.Label(self, text='Figures')
        self.l_list_figures.grid(row=4, column=7)
        self.sb_list_figures = tk.Scrollbar(self)

        self.sb_list_figures.grid(row=5, column=8, sticky=tk.N + tk.S)
        self.lb_list_figures = tk.Listbox(self, height=5,
                                                  width=config['INTERFACE'][
                                                      'WIDTH_WINDOWS'],
                                                  font=('Consolas', 10),
                                                  yscrollcommand=self.
                                                  sb_list_figures.set,
                                                  selectmode=tk.EXTENDED,
                                                  exportselection=False)
        self.lb_list_figures.grid(row=5, column=7)

        self.sb_list_figures.config(command=self.
                                            lb_list_figures.yview)

        self.lb_list_figures.bind('<<ListboxSelect>>',
                                  self.select_figure)
        self.lb_list_figures.bind(
            '<Delete>', self.lb_list_figures_delete_selected)

        self.l_list_modes = tk.Label(self, text='Modes')
        self.l_list_modes.grid(row=6, column=1)
        self.sb_list_modes = tk.Scrollbar(self)

        self.sb_list_modes.grid(row=7, column=2, sticky=tk.N + tk.S)
        self.lb_list_modes = tk.Listbox(self, height=11,
                                        width=config['INTERFACE'][
                                            'WIDTH_WINDOWS'],
                                        font=('Consolas', 10),
                                        yscrollcommand=self.
                                        sb_list_modes.set,
                                        selectmode=tk.EXTENDED,
                                        exportselection=False)
        self.lb_list_modes.grid(row=7, column=1)

        self.sb_list_modes.config(command=self.lb_list_modes.yview)

        self.lb_list_modes.bind('<<ListboxSelect>>',
                                self.lb_modes_update_others)

        self.l_list_groupby = tk.Label(self, text='Group by')
        self.l_list_groupby.grid(row=6, column=7)
        self.sb_list_groupby = tk.Scrollbar(self)

        self.sb_list_groupby.grid(row=7, column=8, sticky=tk.N + tk.S)
        self.lb_list_groupby = tk.Listbox(
            self, height=11,
            width=config['INTERFACE'][
                'WIDTH_WINDOWS'],
            font=('Consolas', 10),
            yscrollcommand=self.sb_list_groupby.set,
            selectmode=tk.EXTENDED,
            exportselection=False)
        self.lb_list_groupby.grid(row=7, column=7)

        self.sb_list_groupby.config(command=self.lb_list_groupby.yview)

        self.grid()

        self.update_lb_list_measurementsets()


class WindowDescSelector(tk.Frame):

    def update_lb_list_descs(self, _event=False):

        list_headings = []

        # clear all windows
        for lb in [self.lb_list_desc_x, self.lb_list_desc_yp,
                   self.lb_list_desc_ys, self.lb_list_desc_c,
                   self.lb_list_desc_size, self.lb_list_desc_shape]:
            lb.delete(0, tk.END)

        if window_main.lb_list_modes.curselection():
            mode = window_main.lb_list_modes.get(
                window_main.lb_list_modes.curselection()[0])
        else:
            mode = window_main.lb_list_modes.get(0)

        if window_main.lb_list_measurements.curselection():
            meas_selected = [measurements.list[i] for i in
                             window_main.lb_list_measurements.curselection()]
        else:
            meas_selected = measurements.list

        if len(meas_selected) > 0:
            defaults = meas_selected[0].get_defaults()

            if 'df' in meas_selected[0].data['modes'][mode]:

                for meas in meas_selected:
                    if 'df' in meas.data['modes'][mode]:
                        for heading in meas.data['modes'][mode]['df'].columns:
                            if heading not in list_headings:
                                list_headings.append(heading)

                if config.get(
                        'LISTS', 'SORT_SERIES_LISTS', fallback='OFF') == 'ON':
                    try:
                        list_headings = sorted(list_headings)
                    except TypeError:
                        logger_general.warning(
                            f'Could not sort list headings due to string values.')
                        pass

                for lb in [self.lb_list_desc_x, self.lb_list_desc_yp,
                           self.lb_list_desc_ys, self.lb_list_desc_c,
                           self.lb_list_desc_size, self.lb_list_desc_shape]:

                    for heading in list_headings:
                        lb.insert(tk.END, heading)

                if 'series-x' in defaults:
                    for i, ser in enumerate(
                            self.lb_list_desc_x.get(0, tk.END)):
                        if defaults['series-x'] == ser:
                            self.lb_list_desc_x.select_set(i)

                if 'series-yp' in defaults:
                    sers_yp = defaults['series-yp'].split(';')
                    for i, ser in enumerate(
                            self.lb_list_desc_yp.get(0, tk.END)):
                        for ser_yp in sers_yp:
                            if ser_yp == ser:
                                self.lb_list_desc_yp.select_set(i)

                if 'series-ys' in defaults:
                    sers_ys = defaults['series-ys'].split(';')
                    for i, ser in enumerate(
                            self.lb_list_desc_ys.get(0, tk.END)):
                        for ser_ys in sers_ys:
                            if ser_ys == ser:
                                self.lb_list_desc_ys.select_set(i)

                if 'series-color' in defaults:
                    for i, ser in enumerate(
                            self.lb_list_desc_c.get(0, tk.END)):
                        if defaults['series-color'] == ser:
                            self.lb_list_desc_c.select_set(i)

                if 'series-size' in defaults:
                    for i, ser in enumerate(
                            self.lb_list_desc_size.get(0, tk.END)):
                        if defaults['series-size'] == ser:
                            self.lb_list_desc_size.select_set(i)

                if 'series-shape' in defaults:
                    for i, ser in enumerate(
                            self.lb_list_desc_shape.get(0, tk.END)):
                        if defaults['series-shape'] == ser:
                            self.lb_list_desc_shape.select_set(i)

    def __init__(self, _master):

        super().__init__(_master)
        self.master = _master
        self.grid()

        global config

        WINDOWS_DESC_HEIGHT = config.get('INTERFACE', 'WINDOWS_DESC_HEIGHT',
                                         fallback=20)

        WINDOWS_DESC_WIDTH = config.get('INTERFACE', 'WINDOWS_DESC_WIDTH',
                                         fallback=30)

        self.l_list_desc_x = tk.Label(self, text='x series')
        self.l_list_desc_x.grid(row=1, column=1)
        self.sb_list_desc_x = tk.Scrollbar(self)
        self.sb_list_desc_x.grid(row=2, column=2, sticky=tk.N + tk.S)
        self.lb_list_desc_x = tk.Listbox(self, height=WINDOWS_DESC_HEIGHT,
                                         width=WINDOWS_DESC_WIDTH,
                                         font=('Consolas', 10),
                                         yscrollcommand=self.
                                         sb_list_desc_x.set,
                                         selectmode=tk.EXTENDED,
                                         exportselection=False)
        self.lb_list_desc_x.grid(row=2, column=1)
        self.sb_list_desc_x.config(command=self.lb_list_desc_x.yview)

        self.l_list_desc_yp = tk.Label(self, text='y series (primary)')
        self.l_list_desc_yp.grid(row=1, column=3)
        self.sb_list_desc_yp = tk.Scrollbar(self)
        self.sb_list_desc_yp.grid(row=2, column=4, sticky=tk.N + tk.S)
        self.lb_list_desc_yp = tk.Listbox(self, height=WINDOWS_DESC_HEIGHT,
                                          width=WINDOWS_DESC_WIDTH,
                                          font=('Consolas', 10),
                                          yscrollcommand=self.
                                          sb_list_desc_yp.set,
                                          selectmode=tk.EXTENDED,
                                          exportselection=False)
        self.lb_list_desc_yp.grid(row=2, column=3)
        self.sb_list_desc_yp.config(command=self.lb_list_desc_yp.yview)

        self.l_list_desc_ys = tk.Label(self, text='y series (secondary)')
        self.l_list_desc_ys.grid(row=1, column=5)
        self.sb_list_desc_ys = tk.Scrollbar(self)
        self.sb_list_desc_ys.grid(row=2, column=6, sticky=tk.N + tk.S)
        self.lb_list_desc_ys = tk.Listbox(self, height=WINDOWS_DESC_HEIGHT,
                                          width=WINDOWS_DESC_WIDTH,
                                          font=('Consolas', 10),
                                          yscrollcommand=self.
                                          sb_list_desc_ys.set,
                                          selectmode=tk.EXTENDED,
                                          exportselection=False)
        self.lb_list_desc_ys.grid(row=2, column=5)
        self.sb_list_desc_ys.config(command=self.lb_list_desc_ys.yview)

        self.l_list_desc_c = tk.Label(self, text='Color')
        self.l_list_desc_c.grid(row=1, column=7)
        self.sb_list_desc_c = tk.Scrollbar(self)
        self.sb_list_desc_c.grid(row=2, column=8, sticky=tk.N + tk.S)
        self.lb_list_desc_c = tk.Listbox(self, height=WINDOWS_DESC_HEIGHT,
                                         width=WINDOWS_DESC_WIDTH,
                                         font=('Consolas', 10),
                                         yscrollcommand=self.
                                         sb_list_desc_c.set,
                                         selectmode=tk.EXTENDED,
                                         exportselection=False)
        self.lb_list_desc_c.grid(row=2, column=7)
        self.sb_list_desc_c.config(command=self.lb_list_desc_c.yview)

        self.l_list_desc_size = tk.Label(self, text='Size')
        self.l_list_desc_size.grid(row=1, column=9)
        self.sb_list_desc_size = tk.Scrollbar(self)
        self.sb_list_desc_size.grid(row=2, column=10, sticky=tk.N + tk.S)
        self.lb_list_desc_size = tk.Listbox(self, height=WINDOWS_DESC_HEIGHT,
                                            width=WINDOWS_DESC_WIDTH,
                                            font=('Consolas', 10),
                                            yscrollcommand=self.
                                            sb_list_desc_size.set,
                                            selectmode=tk.EXTENDED,
                                            exportselection=False)
        self.lb_list_desc_size.grid(row=2, column=9)
        self.sb_list_desc_size.config(command=self.lb_list_desc_size.yview)

        self.l_list_desc_shape = tk.Label(self, text='Shape')
        self.l_list_desc_shape.grid(row=1, column=11)
        self.sb_list_desc_shape = tk.Scrollbar(self)
        self.sb_list_desc_shape.grid(row=2, column=12, sticky=tk.N + tk.S)
        self.lb_list_desc_shape = tk.Listbox(self, height=WINDOWS_DESC_HEIGHT,
                                             width=WINDOWS_DESC_WIDTH,
                                             font=('Consolas', 10),
                                             yscrollcommand=self.
                                             sb_list_desc_shape.set,
                                             selectmode=tk.EXTENDED,
                                             exportselection=False)
        self.lb_list_desc_shape.grid(row=2, column=11)
        self.sb_list_desc_shape.config(command=self.lb_list_desc_shape.yview)


class WindowInfo(tk.Frame):

    def __init__(self, _master):

        super().__init__(_master)
        self.master = _master
        self.grid()

        self.l_info = tk.Label(self, text='BLANK', font=('Consolas', 13))
        self.l_info.grid(row=1, column=1)


class WindowMessages(tk.Frame):

    def __init__(self, _master):

        super().__init__(_master)
        self.master = _master
        self.grid(sticky=tk.W)

        self.l_message = tk.Label(self, text='Messages appear here')
        self.l_message.grid(row=1, column=1)


def show_customizer():

    if not app.window_customizer:
        app.window_customizer = plot_customizer.customize(
            master, app=app, figure=figure)

    try:
        app.window_customizer.master.lift()
    except _tkinter.TclError:
        app.window_customizer = plot_customizer.customize(
            master, app=app, figure=figure)
        app.window_customizer.master.lift()

    app.figure.fig.show()

class WindowButtons(tk.Frame):

    global figure

    @staticmethod
    def debug_general():
        logger_general.info('Debug started')
        logger_general.info('Debug pass')
        pass

    def update_l_figure(self):
        if isinstance(app.figure, Figure):
            self.l_figure['text'] = app.figure.name


    def get_free_button(self, newcol=False):

        row, col = self.free_button

        if row < 8 and not newcol:
            row += 1
        else:
            row = 1
            col += 1

        self.free_button = [row, col]

        return {'row': row, 'column': col}


    def __init__(self, _master):

        super().__init__(_master)
        self.master = _master
        self.grid()

        self.free_button = [0, 1]

        self.bt_add_measurements = tk.Button(self, text='Add measurements',
                                             command=add_measurements)

        self.bt_add_measurements.grid(**self.get_free_button())

        self.bt_add_files = tk.Button(self, text='Add files',
                                      command=add_files)
        self.bt_add_files.grid(**self.get_free_button())

        self.bt_add_files = tk.Button(self, text='Add files (gen.)',
                                      command=lambda: add_files(generic=True))
        self.bt_add_files.grid(**self.get_free_button())

        self.bt_load_dataset_selected = tk.Button(
            self, text='Load dataset', command=load_dataset_selected)

        self.bt_save_datasets_to_file = tk.Button(
            self, text='Save dataset',
            command=lambda: save_datasets_to_file(selected_only=True))
        self.bt_save_datasets_to_file.grid(**self.get_free_button())

        self.bt_load_dataset_file = tk.Button(
            self, text='Load dataset file', command=load_dataset_files_select)

        self.bt_copy_to_clipboard = tk.Button(
            self, text='Copy fig dfs to clipboard',
            command=copy_fig_dfs_to_clipboard)

        self.bt_exit = tk.Button(self, text='Exit',
                                 command=lambda: exit_program(
                                     close_figures=True))
        self.bt_exit.grid(**self.get_free_button())

        self.bt_plot = tk.Button(self, text='Plot',
                                 command=show_selected)
        self.bt_plot.grid(**self.get_free_button(newcol=True))

        self.bt_plot_add = tk.Button(self, text='Plot (add)',
                                     command=lambda: show_selected(
                                             _figure=figure))
        self.bt_plot_add.grid(**self.get_free_button())

        self.bt_plot_2d = tk.Button(self, text='Plot (2d)',
                                    command=lambda: show_selected(
                                        twod=True))
        self.bt_plot_2d.grid(**self.get_free_button())

        self.bt_plot_add_inset = tk.Button(
            self, text='Plot(ins.)', command=lambda: show_selected(
                _figure=figure, inset=True))
        self.bt_plot_add_inset.grid(**self.get_free_button())

        self.bt_customizer = tk.Button(self, text='Customize fig.',
                                 command=show_customizer)
        self.bt_customizer.grid(**self.get_free_button())

        self.bt_copy_to_clipboard = tk.Button(
            self, text='Copy to clipboard',
            command=copy_meas_dfs_to_clipboard)
        self.bt_copy_to_clipboard.grid(**self.get_free_button(newcol=True))

        self.bt_viewdf = tk.Button(self, text='View data (fig.)',
                                   command=view_df_fig)
        self.bt_viewdf.grid(**self.get_free_button())

        self.bt_viewdf = tk.Button(self, text='View data',
                                   command=view_df)
        self.bt_viewdf.grid(**self.get_free_button())

        self.bt_viewdf_app = tk.Button(self, text='View data (app)',
                                       command=view_df_default_app)
        self.bt_viewdf_app.grid(**self.get_free_button())

        self.bt_sort = tk.Button(self, text='Sort', command=sort_measurements)
        self.bt_sort.grid(**self.get_free_button())

        self.e_mode = tk.Entry(self)
        self.e_mode.insert(0, 'normal')

        self.bt_cust_fig = tk.Button(self, text='Cust. fig.',
                                  command=lambda: methods.customize_figure(figure))

        self.bt_records = tk.Button(
            self, text='Manual entry', command=record_values)
        self.bt_records.grid(**self.get_free_button(newcol=True))

        self.bt_fit_xrd = tk.Button(
            self, text='Fit xrd', command=run_fit_xrd)
        self.bt_fit_xrd.grid(**self.get_free_button())

        self.bt_cancel_process = tk.Button(
            self, text='Cancel process', command=self.cancel_process_do)
        self.bt_cancel_process.grid(**self.get_free_button())

        self.bt_extra = tk.Button(
            self, text='Extra', command=run_extra)
        self.bt_extra.grid(**self.get_free_button())

        self.bt_reload_modules = tk.Button(self, text='Reload modules',
                                            command=reload_modules)
        self.bt_reload_modules.grid(**self.get_free_button(newcol=True))

        self.bt_reload = tk.Button(
            self, text='Reload', command=reload_files)
        self.bt_reload.grid(**self.get_free_button())

        self.bt_update_config = tk.Button(self, text='Update config',
                                          command=update_config)
        self.bt_update_config.grid(**self.get_free_button())

        self.bt_debug = tk.Button(self, text='DEBUG',
                                  command=self.debug_general)
        self.bt_debug.grid(**self.get_free_button())

        self.bt_run_tests = tk.Button(self, text='Run tests',
                                      command=run_tests)
        self.bt_run_tests.grid(**self.get_free_button())

        self.l_figure = tk.Label(self, text='Figure:')

        self.l_offset_y = tk.Label(self, text='Offset (y):')
        self.l_offset_y.grid(**self.get_free_button(newcol=True))

        self.e_offset_y = tk.Entry(self)
        self.e_offset_y.grid(**self.get_free_button())
        self.e_offset_y.insert(0, '0')

        self.l_offset_x = tk.Label(self, text='Offset (x):')
        self.l_offset_x.grid(**self.get_free_button())

        self.e_offset_x = tk.Entry(self)
        self.e_offset_x.grid(**self.get_free_button())
        self.e_offset_x.insert(0, '0')

        self.l_filter = tk.Label(self, text='Filter:')
        self.l_filter.grid(**self.get_free_button())

        self.e_filter = tk.Entry(self)
        self.e_filter.grid(**self.get_free_button())

        self.l_multiply_p = tk.Label(self, text='Multiply (prim.):')
        self.l_multiply_p.grid(**self.get_free_button())

        self.e_multiply_p = tk.Entry(self)
        self.e_multiply_p.grid(**self.get_free_button())
        self.e_multiply_p.insert(0, '1')

        self.l_multiply_s = tk.Label(self, text='Multiply (sec.):')
        self.l_multiply_s.grid(**self.get_free_button())

        self.e_multiply_s = tk.Entry(self)
        self.e_multiply_s.grid(**self.get_free_button())
        self.e_multiply_s.insert(0, '1')

        self.l_groupby = tk.Label(self, text='Group by')
        self.l_groupby.grid(**self.get_free_button())

        self.e_groupby = tk.Entry(self)
        self.e_groupby.grid(**self.get_free_button())

        self.l_mode = tk.Label(self, text='Mode')

        self.cancel_process = False

    def cancel_process_do(self):
        self.cancel_process = True


def get_table_pars():
    pass


def update_config():

    global config

    _init_gui.config = configparser.ConfigParser(interpolation=None)
    _init_gui.config.read(_init_gui.PATH_CONFIG)
    _init_gui.config_path = configparser.ConfigParser(interpolation=None)
    _init_gui.config_paths.read(_init_gui.PATH_CONFIG_PATHS)
    gradients.config = configparser.ConfigParser(interpolation=None)
    gradients.config.read(gradients.FILE_CONFIG)
    conv_main.config = configparser.ConfigParser(interpolation=None)
    conv_main.config.read(conv_main.FILE_CONFIG)
    config = configparser.ConfigParser(interpolation=None)
    config.read(_init_gui.PATH_CONFIG)
    logger_general.info('Configs updated.')


def add_dataset():
    global dataset
    dataset = Dataset()
    dataset.add_files()


def load_dataset_files_select(paths_source_dataset=None, confirm=False):

    datasets_toload = []
    if not paths_source_dataset:
        paths_source_dataset = filedialog.askopenfilenames(
            title='Dataset file:',
            initialdir=conv_main.expand_envs_in_paths(
                config_paths.get('GUI', 'DATASETS', fallback=None)),
            filetypes=[('json files', '*.djson'), ('d lists', '*.dlist')])
    else:
        if type(paths_source_dataset) is not list:
            paths_source_dataset = [paths_source_dataset]

    for path_source_dataset in paths_source_dataset:
        if os.path.exists(path_source_dataset):
            with open(path_source_dataset, 'r') as f:
                if path_source_dataset.endswith('djson'):
                    datasets_toload.append(json.load(f))
                elif path_source_dataset.endswith('list'):
                    pass  # TODO
        else:
            logger_general.warning(
                f'Dataset file not found: {path_source_dataset}')

    paths_source = []

    for dataset_toload in datasets_toload:
        for key in dataset_toload.keys():
            paths_source.extend(
                [path for path in dataset_toload[key]['paths']])

    if confirm:
        window_files_top = tk.Toplevel()
        window_files = WindowListFiles(window_files_top)

        window_files.lb_files.delete(0, tk.END)

        for i, path_source in enumerate(paths_source):
            window_files.lb_files.insert(tk.END, path_source)
            window_files.lb_files.select_set(i)
    else:
        add_files(paths_source)


def load_dataset_selected(datasets_=None):

    global dataset

    if not datasets_:
        try:
            sel = window_main.lb_datasets.curselection()[0]
        except IndexError:
            logger_general.warning('No dataset available.')
            return
        else:
            datasets_ = [datasets_[sel]]

    for dataset in datasets_:
        paths = dataset.data['paths']
        if messagebox.askokcancel(title='Load files:', message=paths):
            dataset.load()
        else:
            logger_general.info('Load dataset cancelled.')


def save_datasets_to_file(selected_only=False):
    global datasets
    global dataset

    data_datasets = {}

    if selected_only:
        add_dataset()
        _dataset = datasets[-1]
        data_datasets[_dataset.data['name']] = _dataset.data
    else:
        for _dataset in datasets:
            data_datasets[_dataset.data['name']] = _dataset.data

    path_tosave = filedialog.asksaveasfilename(
        title='Save dataset in: ',
        defaultextension='.djson',
        initialfile=_dataset.label.replace('|', '-'),
        initialdir=config_paths.get('GUI', 'DATASET_DIRECTORY',
                                    fallback=None), filetypes=[('json files',
                                                                '*.djson')])

    if path_tosave:
        with open(path_tosave, 'w') as f:
            json.dump(data_datasets, f, indent=4)
            logger_general.info('Saved: ' + path_tosave)


class Dataset:

    # TODO save also current selections to file

    def add_files(self):
        """
        Add all files in the currently active measurements instances
        to this dataset.
        """

        for meas in self.measurements_tosave:
            if 'path_parent' in meas.data['file']:
                # this may be redundant
                if not meas.data['file']['path_parent'] in self.data['paths']:
                    self.data['paths'].append(meas.data['file']['path_parent'])
            else:
                if not meas.data['file']['path'] in self.data['paths']:
                    self.data['paths'].append(meas.data['file']['path'])
                else:
                    logger_general.warning(
                        f'Path already in dataset: '
                        f'{meas.data["file"]["path"]}')

    def load(self):
        paths = [path for path in self.data['paths']]
        add_files(paths)

    def __init__(self, data=None):
        global dataset
        global datasets

        dataset = self
        datasets.append(self)

        self.measurements_tosave, self.measurements_orig = get_selections('measurement')

        self.label = methods.get_label(self.measurements_tosave)

        if not data:
            self.data = {'paths': []}
            dtime = datetime.datetime.now().isoformat()[0:19]
            description = simpledialog.askstring(title='Query',
                                                 prompt='Dataset description?',
                                                 initialvalue=self.label)  # TODO fix
            if not description:
                description = 'TEMP'

            self.data['name'] = description

        else:
            self.data = data

        window_main.update_lb_datasets()


class LogWindowHandler(logging.Handler):

    def __init__(self, text):
        logging.Handler.__init__(self)
        self.text = text
        self.text.tag_config('INFO', foreground='black')
        self.text.tag_config('DEBUG', foreground='grey')
        self.text.tag_config('WARNING', foreground='orange')
        self.text.tag_config('ERROR', foreground='red')
        self.text.tag_config('CRITICAL', foreground='red', underline=1)

    def emit(self, record):
        msg = self.format(record)

        def append():
            self.text.configure(state='normal')
            self.text.insert(tk.END, msg + '\n\n', record.levelname)
            window_messages.l_message['text'] = msg
            self.text.configure(state='disabled')
            self.text.yview(tk.END)
        self.text.after(0, append)


class WindowLogging(tk.Frame):

    def __init__(self, parent, logger, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, *kwargs)
        self.root = parent
        self.build_gui(logger)

    def build_gui(self, logger):
        self.root.title('LOG')
        self.grid(column=0, row=0, sticky='ew')

        st = scrolledtext.ScrolledText(self, state='disabled')
        st.configure(font=('Consolas', 10))
        st.grid()

        handler_logger_window = LogWindowHandler(st)
        formatter_general = logging.Formatter(
            '%(asctime)s %(levelname)8s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')
        handler_logger_window.setFormatter(formatter_general)

        logger.addHandler(handler_logger_window)
        self.lift()


def run_tests():
    test_modes = config['DEBUG']['TEST_MODES'].split(',')
    global measurements
    for test_mode in test_modes:
        test_path = config_paths[
            'GUI']['TEST_PATH_' + test_mode]
        add_measurements()
        add_files(test_path)


def get_data_from_measurement(meas: Measurement):
    dfs_list = []
    df = pd.json_normalize(meas.data)
    df.drop(df.filter(regex='(^defaults.*|^modes.*)').columns, axis=1, inplace=True)
    dfs_list.append(df)

    dfs_list.append(meas.data['modes']['analysis']['df'].add_prefix('an_'))
    dfs_comb = pd.concat(dfs_list, axis=1)
    for col in dfs_comb.columns:
        if type(dfs_comb[col].iloc[0]) == pd.DataFrame:
            dfs_comb.drop(col, axis=1, inplace=True)

    return dfs_comb


def create_logging_window():
    window_logging_top = tk.Toplevel()
    window_logging_top.geometry('700x700+1150+0')
    window_logging_top.lift()
    window_logging = WindowLogging(window_logging_top, logger_general)
    return window_logging

def get_window_imageviewer():

    global window_imageviewer
    window_imageviewer_top = tk.Toplevel()
    window_imageviewer_top.title('ImageViewer')
    window_imageviewer_top.geometry('1050x1000+0+0')
    window_imageviewer = image_viewer.WindowImages(
        window_imageviewer_top,
        int(config['IMAGES']['SCALED_WIDTH']),
        int(config['IMAGES']['WIDTH_COUNT']), app)
    window_imageviewer.pack(side='top', fill='both', expand=True)

    return window_imageviewer


def update_windowlogging_loop():
    master.after(3000, update_windowlogging_loop)
    window_main.l_datasets['text'] = datetime.datetime.now().isoformat()
    window_messages.update_idletasks()


class App:
    def __init__(self, pars_list):
        (self.master,
        self.window_main,
        self.window_desc_selector,
        self.window_buttons,
        self.window_messages,
        self.window_logging,
        self.window_info,
        self.window_imageviewer,
        self.temp_directory,
        self.measurements,
        self.measurements_all,
        self.datasets,
        self.last_open_directory,
        self.last_added_paths,
        self.figs,
        self.figures,
        self.window_customizer,
        self.figure,
        window_data) = pars_list


def update_task_loop():
    pass



if __name__ == '__main__':
    """
    Initiates the main interface.
    """

    # TODO get rid of global variables

    global window_main
    global window_desc_selector
    global window_buttons
    global temp_directory
    global window_messages

    last_added_paths = []

    logger_general.info('Starting: ' + os.path.basename(__file__))

    dataset_file = None

    dataset = None
    figure: Optional[Figure] = None

    measurements_all = []
    datasets = []

    window_main = None
    window_desc_selector = None
    window_buttons = None
    window_info = None
    window_messages = None
    window_customizer = None

    # Add empty instance of Measurements (needed to initialize the windows).
    measurements = add_measurements()

    figures = Figures()

    master = tk.Tk()
    master.geometry('+0+0')

    if config.get('DEBUG', 'WINDOW_LOGGING', fallback='OFF') == 'ON':
        window_logging = create_logging_window()
    else:
        window_logging = None

    window_main = WindowMain(master)

    window_desc_selector = WindowDescSelector(master)
    window_buttons = WindowButtons(master)
    window_messages = WindowMessages(master)

    window_info_top = tk.Toplevel()
    window_info_top.geometry('500x150+1150+0')
    window_info_top.lift()

    window_info = WindowInfo(window_info_top)

    master.title('Data analysis')
    window_info_top.title('Info')

    window_data = None

    window_imageviewer: image_viewer.WindowImages = None

    if len(sys.argv) > 1:
        DEFAULT_DIRECTORY = sys.argv[1]
    elif config_paths['GUI']['DATA']:
        DEFAULT_DIRECTORY = config_paths['GUI']['DATA']
    else:
        DEFAULT_DIRECTORY = None

    last_open_directory = DEFAULT_DIRECTORY

    temp_directory = None

    figs = []

    app_pars = (
        master,
        window_main,
        window_desc_selector,
        window_buttons,
        window_messages,
        window_logging,
        window_info,
        window_imageviewer,
        temp_directory,
        measurements,
        measurements_all,
        datasets,
        last_open_directory,
        last_added_paths,
        figs,
        figures,
        window_customizer,
        figure,
        window_data)

    app = App(app_pars)

    if config['DEBUG']['TESTING'] == 'ON':
        run_tests()

    master.lift()

    update_task_loop()
    master.mainloop()

    df_sample_data = conv_main.df_sample_data

    logger_general.info('Finished running: ' + os.path.basename(__file__))
