import datetime
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import logging
import matplotlib.markers as markers
import matplotlib.lines as mlines
import os
import matplotlib

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import gui
from _init_gui import config
from gui_methods import normalize_to_range, concat_by_column
from _init_gui import config_paths

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from gui import Figure, Figures

logger = logging.getLogger('General')

font_size = config.get('PLOT', 'FONT_SIZE', fallback=10)
matplotlib.rcParams.update({'font.size': font_size})


def click_dist(event, fig):

    if event.xdata:
        if not fig.dist_point_1:
            fig.dist_point_1, = fig.ax_p.plot(
                event.xdata, event.ydata, 'o', color='red', fillstyle='none')
            fig.dist_points.append(fig.dist_point_1)
            fig.dist_coords_1 = [event.xdata, event.ydata]
        elif fig.dist_point_1 and not fig.dist_point_2:
            fig.dist_point_2, = fig.ax_p.plot(
                event.xdata, event.ydata, 'o', color='red', fillstyle='none')
            fig.dist_points.append(fig.dist_point_2)
            fig.dist_coords_2 = [event.xdata, event.ydata]
            line_dist, = fig.ax_p.plot(
                [fig.dist_coords_1[0], fig.dist_coords_2[0]],
                [fig.dist_coords_1[1], fig.dist_coords_2[1]], '-', color='red')
            fig.dist_lines.append(line_dist)
            dist = math.sqrt(
                (abs(fig.dist_coords_1[0] - fig.dist_coords_2[0]) ** 2
                 + abs(fig.dist_coords_1[1] - fig.dist_coords_2[1]) ** 2))
            fig.ax_p.text(fig.dist_coords_1[0] + 0.5
                          * (fig.dist_coords_2[0] - fig.dist_coords_1[0]),
                          fig.dist_coords_1[1] + 0.5
                          * (fig.dist_coords_2[1] - fig.dist_coords_1[1]),
                          '{:.3e}'.format(dist),
                          color='blue',
                          fontsize=11)

            fig.ax_p.set_title('{}\nLast dist: {}'.format(fig.title, dist))

            fig.dist_point_1 = None
            fig.dist_point_2 = None
            fig.dist_coords_1 = None
            fig.dist_coords_2 = None


        fig.fig.canvas.draw()


def set_marker_shapes(pc, markers_list):
    paths = []
    for marker_string in markers_list:
        marker_obj = markers.MarkerStyle(marker_string)
        paths.append(
            marker_obj.get_path().transformed(marker_obj.get_transform()))
    pc.set_paths(paths)


def add_new_legend(fig, ax, handles, labels, title):

    # TODO adjust
    if fig.legends:
        location = (1.4, 0.2)
    else:
        location = (1.4, 0.8)

    legend = plt.legend(handles, labels, loc='center left', bbox_to_anchor=location)
    fig.legends.append(legend)
    legend.set_title(title)

    ax.add_artist(legend)


def create_marker_legend(fig, ax: plt.Axes, markers_list, markerranges_shape, title):
    handles = []
    labels = []
    for markershape, markerrange in zip(markers_list, markerranges_shape):
        handles.append(
            mlines.Line2D([], [], marker=markershape, linestyle='none',
                          label=f'{markerrange[0]:.1f}-{markerrange[1]:.1f}',
                          fillstyle='none', markersize=8))
        labels.append(f'{markerrange[0]:.1f}-{markerrange[1]:.1f}')

    add_new_legend(fig, ax, handles, labels, title)


def create_markersize_legend(fig, ax, markersizes_legend, markerranges_size, title):
    handles = []
    labels = []
    for markersize, markerrange in zip(markersizes_legend, markerranges_size):
        handles.append(
            mlines.Line2D([], [], marker='o', markersize=markersize,
                          linestyle='none', fillstyle='none',
                          label=f'{markerrange[0]:.3}-{markerrange[1]:.3}'))
        labels.append(f'{markerrange[0]:.3}-{markerrange[1]:.3}')

    add_new_legend(fig, ax, handles, labels, title)


def save_fig(figure, config_paths, path_tosave=None):

    if figure:
        filename = (figure.ax_p.get_title()
                    + "_" + datetime.datetime.now().isoformat()[0:22]
                    + '.png').replace(':', '-').replace('|', '_')

        if not path_tosave:
            result_dfs_dir = config_paths.get('GUI', 'RESULT_FIGS',
                                              fallback=None)
            path_tosave = os.path.join(result_dfs_dir, filename)

        if path_tosave:
            figure.fig.savefig(path_tosave)
            logger.info(
                f'Saved fig to {path_tosave}.')


def plot_data(meass_toplot,
              mode, _figures: gui.Figures,
              _figure: gui.Figure=None,
              meass_orig=None, ind_ax=0,
              twod=False, show_plot=True, keep_ref=True, savefig=False,
              inset=False):

    global figure

    logger.debug('Start plotting')

    if not meass_orig:
        meass_orig = meass_toplot

    if not _figure:
        _figure = _figures.add(mode=mode, keep_ref=keep_ref)

    _figure.color_current = -1  # TEMP

    if inset:
        _figure.ax_inset = inset_axes(_figure.axs_p[0],
                                      width="42%", height="42%", loc='upper right',
                                      borderpad=2)
        ax_toplot = _figure.ax_inset
    else:
        ax_toplot = _figure.axs_p[0]

    ax_toplot.set_prop_cycle(None)

    ax_toplot.tick_params(axis='x', direction='in')
    ax_toplot.tick_params(axis='y', direction='in')

    if keep_ref:
        figure = _figure

    _figure.meass_toplot.extend(meass_toplot)
    _figure.meass_orig.extend(meass_orig)

    if (config.get('PLOT', 'PICKING', fallback='OFF') == 'ON'
            and mode == 'analysis'):
        dfs = [meas.data['modes']['analysis']['df']
               for meas in _figure.meass_orig]
        _figure.df_orig = pd.concat(dfs).reset_index(drop=True)

    _figure.dfs_plotted = []
    _figure.ax_p = _figure.axs_p[ind_ax]

    ser_x, ser_yp, ser_ys, ser_c, ser_size,\
    ser_shape = None, None, None, None, None, None


    max_ser_c = None
    min_ser_c = None
    max_ser_size = None
    min_ser_size = None
    min_ser_shape = None
    max_ser_shape = None

    for index, meas in enumerate(meass_toplot):

        if meas.sers_c:
            ser_c = meas.sers_c[0]
            if max_ser_c is None or np.nanmax(ser_c) > max_ser_c:
                max_ser_c = np.nanmax(ser_c)
            if min_ser_c is None or np.nanmin(ser_c) < min_ser_c:
                min_ser_c = np.nanmin(ser_c)

        if meas.sers_size:
            ser_size = np.nan_to_num(meas.sers_size[0])  # TODO needed?
            if max_ser_size is None or np.nanmax(ser_size) > max_ser_size:
                max_ser_size = np.nanmax(ser_size)
            if min_ser_size is None or np.nanmin(ser_size) < min_ser_size:
                min_ser_size = np.nanmin(ser_size)

        if meas.sers_shape:
            ser_shape = meas.sers_shape[0]
            if max_ser_shape is None or np.nanmax(ser_shape) > max_ser_shape:
                max_ser_shape = np.nanmax(ser_shape)
            if min_ser_shape is None or np.nanmin(ser_shape) < min_ser_shape:
                min_ser_shape = np.nanmin(ser_shape)


    if meass_toplot[-1].offset_y_meas:
        legend_mode = 'next-to-curves'
    elif len(meass_toplot) > 3 and mode == 'measurement':
        if meass_toplot[0].get_defaults(mode)['legend-mode']\
                == 'NEXT-TO-CURVES':
            legend_mode = 'next-to-curves'
        else:
            legend_mode = 'normal'
    else:
        legend_mode = 'normal'

    for index, meas in enumerate(meass_toplot):

        logger.debug(f'Plotting {index}/{len(meass_toplot)}')

        defaults = meas.get_defaults(mode)

        if meas.sers_x:
            ser_x = meas.sers_x[0]  # only the first x series is used
        else:
            logger.warning('No x series')
            return
        df_plotted_single = pd.DataFrame({ser_x.name: ser_x})
        if meas.sers_c:
            ser_c = meas.sers_c[0]

        _figure.sers_x.extend(meas.sers_x)
        _figure.sers_yp.extend(meas.sers_yp)
        _figure.sers_ys.extend(meas.sers_ys)
        _figure.sers_c.extend(meas.sers_c)
        _figure.sers_size.extend(meas.sers_size)
        _figure.sers_shape.extend(meas.sers_shape)

        if meas.sers_size:
            ser_size = np.nan_to_num(meas.sers_size[0])
            lower = float(defaults['plot']['markersize']) * float(
                defaults['plot']['markersize-ratio-lower'])
            upper = float(defaults['plot']['markersize']) * float(
                defaults['plot']['markersize-ratio-upper'])

            markersizes = np.array(
                normalize_to_range(
                    ser_size, min_ser_size, max_ser_size, lower, upper))

        else:
            markersizes = int(defaults['plot']['markersize'])

        markershapes_ser_shape = []
        markerranges = []
        markershapes = []
        if meas.sers_shape:
            markers_raw = config.get('PLOT', 'SHAPE_MARKERS', fallback=None)
            if markers_raw:
                markershapes = markers_raw.split(',')
                range_vals = max_ser_shape - min_ser_shape
                step = range_vals / len(markershapes)
                for i, markerhape in enumerate(markershapes):
                    markerranges.append(
                        [min_ser_shape + i * step, min_ser_shape
                         + (i + 1) * step])

                for val in meas.sers_shape[0]:
                    try:
                        pos_in_range = int((val - min_ser_shape) / step) - 1
                    except ValueError:
                        pos_in_range = 0
                    if pos_in_range == -1:
                        pos_in_range = 0
                    if pos_in_range > len(markershapes):
                        pos_in_range = len(markershapes) - 1
                    try:
                        markershapes_ser_shape.append(markershapes[pos_in_range])
                    except IndexError:
                        pass

        if twod:

            kwargs = {
                'levels': config.getint('PLOT', '2D_LEVELS', fallback=None),
                'twod_cmap': config.get('PLOT', '2D_COLORMAP', fallback=None)}
            kwargs = {k: v for k, v in kwargs.items() if v is not None}

            df_toplot = pd.concat([meas.sers_x[0], meas.sers_yp[0],
                                      meas.sers_c[0]], axis=1)

            for col in df_toplot.columns:
                df_toplot = df_toplot[df_toplot[col].notna()]

            df_toplot_2d = df_toplot.replace(np.inf, np.nan).dropna()

            cont = _figure.ax_p.tricontourf(df_toplot_2d[meas.sers_x[0].name],
                                            df_toplot_2d[meas.sers_yp[0].name],
                                            df_toplot_2d[meas.sers_c[0].name],
                                            **kwargs)

            cbar = _figure.fig.colorbar(cont, ax=_figure.ax_p)
            cbar.set_label(ser_c.name)
            _figure.colorbars.append(cbar)

            if config.get('PLOT', '2D_LABELS', fallback=False) == 'ON':
                for i in range(len(meas.sers_x[0])):
                    _figure.ax_p.text(meas.sers_x[0][i], meas.sers_yp[0][i],
                                      meas.sers_c[0][i])

            df_plotted_single[meas.label_string + '_ax-p'] = meas.sers_yp[0]
            _figure.dfs_plotted.append(df_plotted_single)

        else:

            for i, ser_yp in enumerate(meas.sers_yp):

                mask_yp = np.isfinite(ser_yp)

                style = defaults['plot']['style-primary']

                if ('color-primary' in defaults['plot'] and
                      len(_figure.lines_p) == 0) and len(meas.sers_yp) == 1:
                    color = defaults['plot']['color-primary']
                elif i == 0 or (i >= 1 and 'fit' not in ser_yp.name):
                    color = _figure.get_color_next()
                else:
                    color = _figure.get_color_same()

                ser_yp: pd.Series

                if ser_c is None and ser_size is None and ser_shape is None:
                    logger.debug(f'Color={color}')
                    line, = ax_toplot.plot(
                        ser_x.add(meas.offset_x)[mask_yp],
                        ser_yp.add(meas.offset_y_meas * index)[mask_yp]
                               * meas.multiply_p,
                        style,
                        color=color,
                        linewidth=defaults['plot']['linewidth'],
                        markersize=markersizes,
                        fillstyle=defaults['plot']['fillstyle-primary'],
                        label=meas.label_string)
                    line.set_pickradius(
                        int(config['PLOT']['PICKER_TOLERANCE']))
                else:
                    if ser_c is None:
                        ser_c_norm = [
                            _figure.get_color_same() for
                            _ in range(len(ser_x))]
                    else:
                        ser_c_norm = [(i-min_ser_c) /
                                      (max_ser_c - min_ser_c) for i in ser_c]
                        ser_c_norm = plt.cm.viridis(ser_c_norm)

                    if type(markersizes) == list:
                        markersizes_plot = [i * 12 for i in markersizes]
                    else:
                        markersizes_plot = markersizes * 12
                    logger.debug(f'Color={color}')
                    line = _figure.ax_p.scatter(
                        ser_x, ser_yp,
                        s=markersizes_plot,
                        color=color,
                        facecolors='None',
                        edgecolors=ser_c_norm,
                        linewidth=2,
                        label=meas.label_string,
                        picker=int(config['PLOT']['PICKER_TOLERANCE']))

                    if not ser_shape is None:
                        set_marker_shapes(line, markershapes_ser_shape)

                _figure.lines_p.append(line)

                df_plotted_single[meas.label_string + '_ax-p'] = ser_yp

            if meas.sers_ys:

                for i, ser_ys in enumerate(meas.sers_ys):

                    mask_ys = np.isfinite(ser_ys)

                    style = defaults['plot']['style-secondary']

                    color=_figure.get_color_next(force=True)

                    if config.get('PLOT', 'SEPARATE-AXES_Y-SEC',
                                  fallback='OFF') == 'ON':

                        if i == 0 and index == 0 and _figure.axs_s:
                            _figure.ax_s = _figure.ax_p.twinx()
                            _figure.axs_s.append(_figure.ax_s)

                            _figure.ax_s.spines.right.set_color(color)
                            _figure.ax_s.tick_params(axis='y', colors=color)
                            _figure.ax_s.yaxis.label.set_color(color)
                            if len(_figure.axs_s) > 1:
                                _figure.ax_s.spines.right.set_position(
                                    ('axes', 1 + len(_figure.axs_s) * 0.15))
                                _figure.fig.subplots_adjust(
                                    right= (1 - len(_figure.axs_s) * 0.15))
                        else:
                            if not _figure.ax_s:
                                _figure.ax_s = _figure.ax_p.twinx()
                                _figure.axs_s.append(_figure.ax_s)
                                _figure.ax_s.spines.right.set_color(color)
                                _figure.ax_s.tick_params(axis='y', colors=color)
                                _figure.ax_s.yaxis.label.set_color(color)
                    else:
                        if not _figure.ax_s:
                            _figure.ax_s = _figure.ax_p.twinx()
                            _figure.axs_s.append(_figure.ax_s)
                            _figure.ax_s.spines.right.set_color(color)
                            _figure.ax_s.tick_params(axis='y', colors=color)
                            _figure.ax_s.yaxis.label.set_color(color)

                    line, = _figure.ax_s.plot(
                        ser_x.add(meas.offset_x)[mask_ys], ser_ys.add(meas.offset_y_meas * meas.index)[mask_ys]
                               * meas.multiply_s,
                        style,
                        color=color,
                        linewidth=defaults['plot']['linewidth'],
                        markersize=defaults['plot']['markersize'],
                        fillstyle=defaults['plot']['fillstyle-secondary'],
                        label=meas.label_string,
                        picker=int(config['PLOT']['PICKER_TOLERANCE']))

                    _figure.lines_s.append(line)

                    df_plotted_single[meas.label_string + 'ax_s'] = ser_ys

            _figure.dfs_plotted.append(df_plotted_single)
            _figure.dfs.append(meas.data['modes'][mode]['df'])

            if legend_mode == 'next-to-curves':
                x_pos = ser_x.iloc[-1] + meas.offset_x
                y_orig = 0
                if pd.isna(y_orig):
                    y_orig = 0
                y_pos = (y_orig + (meas.offset_y_meas * meas.index)) * meas.multiply_s
                xscale = _figure.ax_p.get_xlim()
                xrange = abs(xscale[-1] - xscale[0])

                _figure.labels_text.append(
                    _figure.ax_p.text(x_pos + xrange * 0.06,
                                      y_pos,
                                      meas.label_string))

    if max_ser_c is not None and not twod:
        cbar = _figure.fig.colorbar(
            ax=_figure.ax_p, mappable=_figure.lines_p[0], ticks=[0, 1],
            cmap='plasma')
        cbar.ax.set_yticklabels([
            '{0:.2f}'.format(min_ser_c),
            '{0:.2f}'.format(max_ser_c)])
        label_c = ser_c.name
        cbar.set_label(label_c)
        _figure.colorbars.append(cbar)


    if max_ser_size:
        markersizes_legend = []
        markersizes_range = []
        step = (max(markersizes) - min(markersizes)) / 5
        for i in range(0,5):
            markersizes_legend.append(min(markersizes) + step * (i + 0.5))
            markersizes_range.append((min(markersizes) + step * i, (min(markersizes) + step * (i+1))))

        label_size = _figure.sers_size[0].name
        create_markersize_legend(_figure, _figure.ax_p, markersizes_legend, markersizes_range,
                                 label_size)

    if max_ser_size and False:
        legend_size = _figure.ax_p.legend(
            *_figure.lines_p[0].legend_elements(
                num=7, prop='sizes',
                func=lambda ser: normalize_to_range(
                    ser, np.nanmin(markersizes),
                    np.nanmax(markersizes), min_ser_size, max_ser_size)),
            title='TEMP')
        _figure.ax_p.add_artist(legend_size)

    _figure.df_plotted = concat_by_column(_figure.dfs_plotted, ser_x.name)

    defaults_common = meass_toplot[0].get_defaults(mode)

    if not ser_shape is None:
        label_shape = _figure.sers_shape[0].name

        create_marker_legend(_figure,
                             _figure.ax_p, markershapes, markerranges,
                             label_shape)

    if legend_mode == 'normal' and config.get(
            'PLOT', 'LEGEND', fallback='OFF') == 'ON' and not ax_toplot.get_legend():

        if defaults_common['legend-position'] == 'next':
            if len(_figure.lines_p) > 1:
                ax_toplot.legend(bbox_to_anchor=(1, 1), frameon=False)
            if _figure.ax_s:
                if len(_figure.lines_s) > 1:
                    _figure.ax_s.legend(bbox_to_anchor=(1, 1))
        else:
            if len(_figure.lines_p) > 1:
                ax_toplot.legend(loc='best', frameon=False)
            if _figure.ax_s:
                if len(_figure.lines_s) > 1:
                    _figure.ax_s.legend(loc='best')


    if meass_toplot[0].sers_yp and\
            '_lin' not in meass_toplot[0].sers_yp[0].name:
        _figure.ax_p.set_yscale(defaults_common['plot']['scaletype-y-pri'])
    if _figure.ax_s and '_lin' not in meass_toplot[0].sers_ys[0].name:
        _figure.ax_s.set_yscale(defaults_common['plot']['scaletype-y-sec'])

    if defaults_common['plot']['grid-vertical'] == 'ON':
        _figure.ax_p.grid(axis='x', color='black')
    if defaults_common['plot']['grid-horizontal'] == 'ON':
        _figure.ax_p.grid(axis='y', color='black')


    if config.get('PLOT', 'MINORGRID', fallback='OFF') == 'ON':
        _figure.ax_p.minorticks_on()
        _figure.ax_p.yaxis.grid(which='minor', linestyle='--')
        _figure.ax_p.xaxis.grid(which='minor', linestyle='--')


    xlabel = ', '.join(set([ser.name for ser in meass_toplot[0].sers_x]))

    ax_toplot.set_xlabel(xlabel)

    ylabel = ', '.join(set([ser.name for ser in meass_toplot[0].sers_yp]))
    ax_toplot.set_ylabel(ylabel)

    if _figure.ax_s:
        ylabel_s = ', '.join((set([ser.name for ser in meass_toplot[0].sers_ys])))
        _figure.ax_s.set_ylabel(ylabel_s)

    if ser_yp is not None and ser_yp.name == 'point':
        _figure.ax_p.set_ylim(1, ser_yp.max() * 4)
    if ser_ys is not None and ser_ys.name == 'point':
        _figure.ax_s.set_ylim(1, ser_ys.max() * 2)

    title_common = []
    for i, string in enumerate(meass_toplot[0].common_string):
        title_common.append('[{}]={}'.format(i + 1, string))

    _figure.fig.number += 1  # TODO fix

    if title_common:
        title_common_string = '{}'.format(', '.join(title_common))
        if len(title_common) > 15:
            title_common_string = title_common_string[:13] + '...'

        _figure.ax_p.set_title(title_common_string, fontsize=9)
    else:
        if len(meass_toplot) > 1:
            _figure.ax_p.set_title(
                meass_toplot[0].label_string.split('\n')[0],
                fontsize=10)
        else:
            pass

    if pd.api.types.is_datetime64_any_dtype(
            meass_toplot[0].sers_x[0]):
        format_dates = mdates.DateFormatter('%Y-%m-%d %H:%M:%S')
        _figure.ax_p.xaxis.set_major_formatter(format_dates)
        plt.setp(_figure.ax_p.xaxis.get_majorticklabels(), rotation=70)

    if savefig:
        save_fig(_figure, config_paths)

    if show_plot:
        logger.debug(f'Showing plot')
        plt.show(block=False)
        _figure.fig.canvas.draw_idle()
        logger.debug(f'Plotting finished')
    else:
        logger.debug(f'Will not show plot')

    plt.show()
