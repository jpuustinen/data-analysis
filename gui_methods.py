import collections.abc
import logging
import re
from functools import reduce
import numpy as np
import pandas as pd
import copy
import configparser
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from typing import TYPE_CHECKING
from gui import config as config_main
from gui import config_paths
import os
from datetime import datetime

if TYPE_CHECKING:
    from gui import Figure

PATH_CONFIG_PLOT_CUSTOMIZATION = 'plot_customizer.cfg'
logger = logging.getLogger('General')


def copy_meas(meas, new):

    to_deepcopy =  ['modes']

    for item in meas.data:
        if item in to_deepcopy:
            new.data[item] = copy.deepcopy(meas.data[item])
        else:
            new.data[item] = copy.copy(meas.data[item])

    return new


def get_label(meass):
    samples = []
    types = []

    for meas in meass:
        if 'number' in meas.data['sample']:
            samples.append(meas.data['sample']['number'])
        else:
            samples.append('N/A')
        if 'type' in meas.data:
            types.append(meas.data['type'])
        else:
            types.append('N/A')

    samples = set(samples)
    types = set(types)

    label = '{0}|{1}'.format(','.join(types),
                                   ','.join(samples))

    if len(label) > 34:
        label = label[0:31] + "..."
    label += f' ({len(meass)})'

    return label


def close_figs(figs):
    for fig in figs:
        if type(fig) == plt.Figure:
            plt.close(fig)
        else:
            fig.destroy()
    figs = []


def find_picked(fig, picked_x, picked_y):
    logger.debug('Finding picked meas.')
    df = fig.df_orig
    meas_picked = None
    for ser in (fig.sers_yp + fig.sers_ys):
        if fig.sers_x[0].name in df:
            df_x = df[np.abs(df[fig.sers_x[0].name].astype(float) - picked_x) < 0.000001]
            if df_x.empty:
                df_x = df[df[fig.sers_x[0].name] == str(int(picked_x))]
            if not df_x.empty:
                df_y = df[np.abs(df[ser.name].astype(float) - picked_y) < 0.000001]
                if not df_y.empty:
                    df_comb = pd.merge(df_x, df_y, how='inner', on=['number'])
                    df_selected = df_x[df_x['number'] == df_comb['number'].iloc[0]]
                    meas_picked = copy.copy(
                        fig.meass_orig[df_selected.index[0]])
                    logger.debug('Found picked meas.')
                    break

    return meas_picked


def customize_figure(fig, inset=False):

    fig: Figure

    if inset:
        temp_ax_p = fig.ax_p
        fig.ax_p = fig.ax_inset


    logger.info(f'Customizing figure {fig.name} from: {PATH_CONFIG_PLOT_CUSTOMIZATION}')
    config = configparser.ConfigParser(interpolation=None)
    config.read(PATH_CONFIG_PLOT_CUSTOMIZATION)

    tight_layout_raw = config.get('PLOT', 'TIGHT_LAYOUT', fallback='OFF')
    if tight_layout_raw == 'ON':
        fig.fig.tight_layout()

    x_range_raw = config.get('PLOT', 'X_RANGE', fallback='')
    if x_range_raw:
        x_range = [float(i) for i in x_range_raw.split(';')]
        fig.ax_p.set_xlim(x_range[0], x_range[1])

        if fig.labels_text:
            for label in fig.labels_text:
                pos = label.get_position()
                label.set_position(
                    (x_range[-1] + ((x_range[1] - x_range[0]) * 0.06), pos[1]))
        if fig.next_to_curves_title:
            pos = fig.next_to_curves_title.get_position()
            fig.next_to_curves_title.set_position(
                (x_range[-1] + ((x_range[1] - x_range[0]) * 0.06), pos[1]))

    color_range_raw = config.get('PLOT', 'COLOR_RANGE', fallback='')
    if color_range_raw:
        color_range = [float(i) for i in color_range_raw.split(';')]
        cbar = fig.colorbars[0]
        cbar.set_clim(color_range[0], color_range[1])

    y_range_raw = config.get('PLOT', 'Y_RANGE', fallback='')
    if y_range_raw:
        y_range = [float(i) for i in y_range_raw.split(';')]
        fig.ax_p.set_ylim(y_range[0], y_range[1])

    y_ranges_sec_raw = config.get('PLOT', 'Y_RANGE_SEC', fallback='')
    if y_ranges_sec_raw:
        y_ranges_sec_raw = y_ranges_sec_raw.split(',')
        for ax, y_range_sec_raw in zip(fig.axs_s, y_ranges_sec_raw):
            y_range_sec = [float(i) for i in y_range_sec_raw.split(';')]
            ax.set_ylim(y_range_sec[0], y_range_sec[1])

    title = config.get('PLOT', 'TITLE', fallback=None)
    if title:
        fig.ax_p.set_title(title)
    else:
        fig.ax_p.set_title('')

    x_label = config.get('PLOT', 'X_LABEL', fallback=None)
    if x_label:
        fig.ax_p.set_xlabel(x_label)

    y_label = config.get('PLOT', 'Y_LABEL', fallback=None)
    if y_label:
        fig.ax_p.set_ylabel(y_label)

    y_label_sec_raw = config.get('PLOT', 'Y_LABEL_SEC', fallback=None)
    if y_label_sec_raw:
        y_labels_sec = y_label_sec_raw.split(';')
        for y_label_sec, ax_s in zip(y_labels_sec, fig.axs_s):
            ax_s.set_ylabel(y_label_sec)

    colorbar_labels_raw = config.get('PLOT', 'COLORBAR_LABELS', fallback='')
    colorbar_labels = colorbar_labels_raw.split(';')
    for colorbar_label, cbar in  zip(colorbar_labels, fig.colorbars):
        cbar.set_label(colorbar_label)

    axs = [fig.ax_p] + fig.axs_s
    if len(axs) > 1:
        lineaxs = []
        for line in fig.lines_p:
            lineaxs.append([line, fig.axs_p])
        for line in fig.lines_s:
            lineaxs.append([line, fig.axs_s])
    else:
        lineaxs = []
        for line in fig.lines_p:
            lineaxs.append([line, fig.axs_p])
        for line in fig.lines_s:
            lineaxs.append([line, fig.axs_s])

    colors_curves_raw = config.get('PLOT', 'COLOR_CURVES', fallback=None)
    linestyles_curves_raw = config.get('PLOT', 'STYLE_CURVES', fallback=None)
    markers_curves_raw = config.get('PLOT', 'MARKER_CURVES', fallback=None)
    markersize_raw = config.get('PLOT', 'MARKERSIZE', fallback=None)
    if colors_curves_raw or linestyles_curves_raw or markers_curves_raw or markersize_raw:

        if colors_curves_raw:
            colors_curves = colors_curves_raw.split(';')
            colors_curves = colors_curves * len(lineaxs)
        elif colors_curves_raw == 'OFF':
            colors_curves = [None] * len(lineaxs)
        else:
            colors_curves = [False] * len(lineaxs)
        if linestyles_curves_raw:
            linestyles_curves = linestyles_curves_raw.split(';')
        elif linestyles_curves_raw == 'OFF':
            linestyles_curves = [None] * len(lineaxs)
        else:
            linestyles_curves = [False] * len(lineaxs)
        if markers_curves_raw == 'OFF':
            markers_curves = [None] * len(lineaxs)
        elif markers_curves_raw:
            markers_curves = markers_curves_raw.split(';')
        else:
            markers_curves = [False] * len(lineaxs)
        if markersize_raw:
            markersize = float(markersize_raw)
        else:
            markersize = None


        axnum = -1
        for lineax, color, linestyle, marker in zip(
                lineaxs, colors_curves, linestyles_curves, markers_curves):
            lines, ax = lineax

            if type(lines) != list:
                lines = [lines]

            for line_ in lines:
                try:
                    line = line_[0]
                except TypeError:
                    line = line_
                if color is not False:
                    line.set_color(color)
                if linestyle is not False:
                    line.set_linestyle(linestyle)
                if marker == 'OFF':
                    line.set_marker('None')
                elif marker is not False:
                    line.set_marker(marker)
                if markersize:
                    line.set_markersize(markersize)
                if config.get('PLOT', 'COLORED_SEC_AXES', fallback='OFF') == 'ON':
                    if axnum > -1:
                        ax.yaxis.label.set_color(color)
                        ax.tick_params(axis='y', colors=color)
                        ax.title.set_color(color)
                        ax.spines['right'].set_color(color)
                        if axnum > 0:
                            ax.spines['right'].set_position(('axes', 1 + axnum * 0.13))
            axnum += 1


    if config.get('PLOT', 'MINOR_TICKS', fallback='OFF') == 'ON':
        fig.ax_p.xaxis.set_minor_locator(AutoMinorLocator())

    ticks_left = config.get('PLOT', 'TICKS_LEFT', fallback='')
    if ticks_left == 'OFF':
        fig.ax_p.set_yticks([], [])

    picking = config.get('PLOT', 'PICKING', fallback=None)
    if picking:
        if picking == 'ON':
            fig.connect_picking()
        elif picking == 'OFF':
            fig.disconnect_picking()
            try:
                fig.ax_button_left.set_visible(False)
                fig.ax_button_right.set_visible(False)
            except AttributeError:
                pass
            if fig.pickmark:
                fig.pickmark.remove()
                fig.pickmark = None

    y_scale_types_raw = config.get('PLOT', 'Y_SCALE_TYPES', fallback=None)
    if y_scale_types_raw:
        for y_scale_type, ax in zip(
                y_scale_types_raw.split(';'), fig.axs_p + fig.axs_s):
            ax.set_yscale(y_scale_type)

    legend_title = config.get('PLOT', 'LEGEND_TITLE', fallback=None)
    legends_raw = config.get('PLOT', 'LEGEND', fallback=None)
    if legends_raw:
        if legends_raw != 'OFF':
            legends = legends_raw.split(';')
            fig.ax_p.legend(legends, title=legend_title, frameon=False)
        elif legends_raw == 'OFF':
            if fig.ax_p.get_legend():
                fig.ax_p.get_legend().remove()
            if fig.axs_s:
                for ax_s in fig.axs_s:
                    if ax_s.get_legend():
                        ax_s.get_legend().remove()
            if fig.labels_text:
                for label in fig.labels_text:
                    label.remove()
                fig.labels_text = []


    next_to_curves_title = config.get('PLOT', 'NEXT_TO_CURVES_TITLE', fallback=None)
    if next_to_curves_title:
        if next_to_curves_title == 'OFF':
            fig.next_to_curves_title.set_text('')
        elif fig.next_to_curves_title:
            fig.next_to_curves_title.set_text(next_to_curves_title)
        else:
            pos = fig.labels_text[0].get_position()
            fig.next_to_curves_title = fig.ax_p.text(pos[0], pos[1] + 2, next_to_curves_title)

    legend_pos_raw = config.get('PLOT', 'LEGEND_POSITION', fallback=None)
    if legend_pos_raw:
        fig.ax_p.legend(loc=legend_pos_raw)
        fig.ax_p.get_legend().get_frame().set_linewidth(0.0)

    if legend_title:
        fig.ax_p.get_legend().set_title(title=legend_title)

    # TODO finish, use a function
    grid_major_vertical_raw = config.get('PLOT', 'GRID_VERTICAL', fallback='')
    if grid_major_vertical_raw == 'ON':
        fig.ax_p.grid(axis='x')
    elif grid_major_vertical_raw == 'OFF':
        fig.ax_p.grid(False)

    grid_major_vertical_raw = config.get('PLOT', 'GRID_HORIZONTAL', fallback='')
    if grid_major_vertical_raw == 'ON':
        fig.ax_p.grid(axis='y')
    elif grid_major_vertical_raw == 'OFF':
        fig.ax_p.grid(False)

    size_raw = config.get('PLOT', 'SIZE', fallback='')
    if size_raw:
        size = [float(i) for i in size_raw.split(';')]
        fig.fig.set_size_inches(size[0], size[1], forward=True)

    save_customized_auto = config_main.get(
        'GENERAL', 'SAVE_CUSTOMIZED_FIG', fallback='OFF')
    if save_customized_auto == 'ON':
        save_fig(fig)

    if inset:
        fig.ax_p = temp_ax_p

    fig.fig.canvas.draw_idle()
    plt.show()


def save_fig(fig, use_date=True):

    types_raw = config_paths.get('GUI', 'SAVED_FIGS_TYPES', fallback=None)

    if use_date and types_raw:
        types = types_raw.split(';')

        dt = datetime.now().isoformat()[0:19]

        filenames = [f'Fig_customized_[{dt}].{type}' for type in types]
    else:
        filenames = None

    if filenames:
        dir_customized = config_paths.get('GUI', 'SAVED_FIGS_DIR',
                                          fallback=None)
        if dir_customized:
            paths_customized = [
                os.path.join(dir_customized, filename) for filename
                in filenames]
        else:
            logger.warning(f'Cust. figure directory not specified.')
            return
    else:
        paths_customized = None


    if paths_customized:
        kwargs = {}
        dpi_raw = config_main.get('PLOT', 'SAVEFIG_DPI', fallback=None)
        if dpi_raw:
            dpi = float(dpi_raw)
            kwargs['dpi'] = dpi

        for path_customized in paths_customized:
            logger.info(f'Saving cust. fig. to: {path_customized} with {kwargs}')

            fig.fig.savefig(path_customized, transparent=False, **kwargs)


def get_longest_substring(data):

    def is_substr(find, data_):
        if len(data_) < 1 and len(find) < 1:
            return False
        for i_ in range(len(data_)):
            if find not in data_[i_]:
                return False
        return True

    substr = ''
    if len(data) > 1 and len(data[0]) > 0:
        for i in range(len(data[0])):
            for j in range(len(data[0]) - i + 1):
                if j > len(substr) and is_substr(data[0][i:i+j], data):
                    substr = data[0][i:i+j]
    return substr


def get_substrings(strings_orig):

    strings = strings_orig
    sstrings = []
    counter = 1

    for i in range(10):

        sstring = get_longest_substring(strings)

        try:
            _value = float(sstring)
        except ValueError:
            pass
        else:
            sstring = None

        if sstring and '[' not in sstring:
            sstrings.append(sstring)
            strings = [
                s.replace(sstring,
                          '[{}]'.format(counter)) for s in strings]
            counter += 1

    return sstrings, strings


def update_nested_dict(data_to, data_from):
    for key, value in data_from.items():
        if isinstance(value, collections.abc.Mapping):
            data_to[key] = update_nested_dict(data_to.get(key, {}), value)
        else:
            data_to[key] = value
    return data_to


def get_value_from_dict(dict_string, dict_data, sep='>'):

    list_keys = str.split(dict_string, sep)

    last = dict_data
    for key in list_keys[0:-1]:
        if key not in last:
            return None
        last = last[key]
    try:
        value = last[list_keys[-1]]
    except KeyError:
        value = 'N/A'

    return value


def dict_to_list(data, keys_upper=None, list_data=None, include_values=False):
    if keys_upper is None:
        keys_upper = []
    for key in list(data.keys()):
        if type(data[key]) == dict:
            keys_upper.append(key)
            dict_to_list(data[key], keys_upper, list_data)
        else:
            if include_values:
                keystring = re.sub('^>', '', ('>'.join(
                        keys_upper) + '>' + key + ':' + data[key]))
                list_data.append(keystring)
            else:
                keystring = re.sub('^>', '', ('>'.join(
                        keys_upper) + '>' + key))
                list_data.append(keystring)
    if len(keys_upper) > 0:
        keys_upper.pop(-1)

    return list_data


def concat_by_column(dfs, by):
    df = reduce(lambda left, right: pd.merge(left, right, on=by, how='outer'),
                dfs)
    return df


def normalize_to_range(ser, orig_min, orig_max, final_min, final_max):
    normalized = [((float(i) - orig_min) / (orig_max - orig_min))
                  * (final_max - final_min) + final_min for i in ser]

    return np.array(normalized)
