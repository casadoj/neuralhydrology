from typing import Tuple, Dict, Optional, Union, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cf
import numpy as np
import pandas as pd
from pathlib import Path

from neuralhydrology.utils.config import Config


def percentile_plot(y: np.ndarray,
                    y_hat: np.ndarray,
                    title: str = '') -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """Plot the time series of observed values with 3 specific prediction intervals (i.e.: 25 to 75, 10 to 90, 5 to 95).

    Parameters
    ----------
    y : np.ndarray
        Array of observed values.
    y_hat : np.ndarray
        Array of simulated values, where the last dimension contains the samples for each time step.
    title : str, optional
        Title of the plot.

    Returns
    -------
    Tuple[mpl.figure.Figure, mpl.axes.Axis]
        The percentile plot.
    """
    fig, ax = plt.subplots()

    y_median = np.median(y_hat, axis=-1).flatten()
    y_25 = np.percentile(y_hat, 25, axis=-1).flatten()
    y_75 = np.percentile(y_hat, 75, axis=-1).flatten()
    y_10 = np.percentile(y_hat, 10, axis=-1).flatten()
    y_90 = np.percentile(y_hat, 90, axis=-1).flatten()
    y_05 = np.percentile(y_hat, 5, axis=-1).flatten()
    y_95 = np.percentile(y_hat, 95, axis=-1).flatten()

    x = np.arange(len(y_05))

    ax.fill_between(x, y_05, y_95, color='#35B779', label='05-95 PI')
    ax.fill_between(x, y_10, y_90, color='#31688E', label='10-90 PI')
    ax.fill_between(x, y_25, y_75, color="#440154", label='25-75 PI')
    ax.plot(y_median, '-', color='red', label="median")
    ax.plot(y.flatten(), '--', color='black', label="observed")
    ax.legend()
    ax.set_title(title)

    return fig, ax


def regression_plot(y: np.ndarray,
                    y_hat: np.ndarray,
                    title: str = '') -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """Plot the time series of observed and simulated values.

    Parameters
    ----------
    y : np.ndarray
        Array of observed values.
    y_hat : np.ndarray
        Array of simulated values.
    title : str, optional
        Title of the plot.

    Returns
    -------
    Tuple[mpl.figure.Figure, mpl.axes.Axis]
        The regression plot.
    """

    fig, ax = plt.subplots()

    ax.plot(y.flatten(), label="observed", lw=1)
    ax.plot(y_hat.flatten(), label="simulated", alpha=.8, lw=1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
    ax.set_title(title)

    return fig, ax


def uncertainty_plot(y: np.ndarray, y_hat: np.ndarray, title: str = '') -> Tuple[mpl.figure.Figure, np.ndarray]:
    """Plots probability plot alongside a hydrograph with simulation percentiles.
    
    The probability plot itself is analogous to the calibration plot for classification tasks. The plot compares the 
    theoretical percentiles of the estimated conditional distributions (over time) with the respective relative 
    empirical counts. 
    The probability plot is often also referred to as probability integral transform diagram, Q-Q plot, or predictive 
    Q-Q plot. 
    

    Parameters
    ----------
    y : np.ndarray
        Array of observed values.
    y_hat : np.ndarray
        Array of simulated values.
    title : str, optional
        Title of the plot, by default empty.

    Returns
    -------
    Tuple[mpl.figure.Figure, np.ndarray]
        The uncertainty plot.
    """

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6.5, 3), gridspec_kw={'width_ratios': [4, 5]})

    # only take part of y to have a better zoom-in
    y_long = y[:, -1].flatten()
    y_hat_long = y_hat[:, -1, :].reshape(y_long.shape[0], -1)
    x_bnd = np.arange(0, 400)
    y_bnd_len = len(x_bnd)

    # hydrograph:
    y_r = [0, 0, 0, 0, 0, 0]  # used later for probability-plot
    quantiles = [0.9, 0.80, 0.50, 0.20, 0.1]
    labels_and_colors = {
        'labels': ['05-95 PI', '10-90 PI', '25-75 PI', '40-60 PI', '45-55 PI'],
        'colors': ['#FDE725', '#8FD744', '#21908C', '#31688E', '#443A83']
    }
    for idx in range(len(quantiles)):
        lb = round(50 - (quantiles[idx] * 100) / 2)
        ub = round(50 + (quantiles[idx] * 100) / 2)
        y_lb = np.percentile(y_hat_long[x_bnd, :], lb, axis=-1).flatten()
        y_ub = np.percentile(y_hat_long[x_bnd, :], ub, axis=-1).flatten()
        y_r[idx] = np.sum(((y_long[x_bnd] > y_lb) * (y_long[x_bnd] < y_ub))) / y_bnd_len
        if idx <= 3:
            axs[1].fill_between(x_bnd,
                                y_lb,
                                y_ub,
                                color=labels_and_colors['colors'][idx],
                                label=labels_and_colors['labels'][idx])

    y_median = np.median(y_hat_long, axis=-1).flatten()
    axs[1].plot(x_bnd, y_median[x_bnd], '-', color='red', label="median")
    axs[1].plot(x_bnd, y_long[x_bnd], '--', color='black', label="observed")
    axs[1].legend(prop={'size': 5})
    axs[1].set_ylabel("value")
    axs[1].set_xlabel("time index")
    # probability-plot:
    quantiles = np.arange(0, 101, 5)
    y_r = quantiles * 0.0
    for idx in range(len(y_r)):
        ub = quantiles[idx]
        y_ub = np.percentile(y_hat_long[x_bnd, :], ub, axis=-1).flatten()
        y_r[idx] = np.sum(y_long[x_bnd] < y_ub) / y_bnd_len

    axs[0].plot([0, 1], [0, 1], 'k--')
    axs[0].plot(quantiles / 100, y_r, 'ro', ms=3.0)
    axs[0].set_axisbelow(True)
    axs[0].yaxis.grid(color='#ECECEC', linestyle='dashed')
    axs[0].xaxis.grid(color='#ECECEC', linestyle='dashed')
    axs[0].xaxis.set_ticks(np.arange(0, 1, 0.2))
    axs[0].yaxis.set_ticks(np.arange(0, 1, 0.2))
    axs[0].set_xlabel("theoretical quantile frequency")
    axs[0].set_ylabel("count")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0.1, 1, 0.95])

    return fig, axs


def boxplot_performance(
    cfg: Config,
    performance: Dict[str, pd.DataFrame],
    save: Optional[Union[str, Path]] = None,
    **kwargs
):
    """
    Generates a boxplot comparing the performance of the different sample sets at all epochs.

    Parameters:
    -----------
    cfg: Config
        An instance of the Config class containing the configuration of the experiment, including the number of epochs.
    performance: dictionary of pandas.DataFrame
        A dictionary where keys are sample sets (e.g. 'train', 'validation') and values are pandas.DataFrame with the performance for every sample and epoch
    save: string or pathlib.Path (optional)
        An optional path where the plot will be saved. If None, the plot is not saved.
        
    Keyword arguments:
    ------------------
    width: float
        Boxplot width
    alpha: float
        Boxplot transparency
    ylim: Tuple
        Y-axis limits.

    Returns:
    --------
        None. Displays a boxplot and optionally saves it to the specified path.
    """
    
    w = kwargs.get('width', .4)
    alpha = kwargs.get('alpha', .333)
    ylim = kwargs.get('ylim', (-1, 1))
    
    metric = cfg.metrics[0]
    epochs = np.arange(start=1, stop=cfg.epochs + 1)
    fig, ax = plt.subplots(figsize=(len(epochs) * .8, 4))
    patches = []
    
    for j, (label, df) in enumerate(performance.items()):
        c = f'C{j}'
        for i, epoch in enumerate(epochs, start=1):
            ax.boxplot(df[epoch].dropna(), positions=[i + (j - .5) * w], widths=w,
                       patch_artist=True, showfliers=False, showcaps=False,
                       boxprops={'facecolor': c, 'edgecolor': 'none', 'alpha': alpha},
                       medianprops={'color': c})
            if i == 1:
                patches.append(mpatches.Patch(color=c, alpha=alpha, label=label))
    ax.set_xticks(epochs)
    ax.set_xticklabels(epochs)
    ax.set(
        xlabel='epoch',
        ylabel=f'{metric} (-)',
        ylim=ylim,
    )
    ax.spines[['top', 'bottom', 'right']].set_visible(False)
    ax.tick_params(axis='x', length=0)

    fig.legend(handles=patches, frameon=False, bbox_to_anchor=[.9, .4, .1, .2])
    
    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches='tight')
        
        
def map_results(
    x: pd.Series,
    y: pd.Series,
    color: pd.Series,
    samples: pd.Series,
    save: Optional[Union[str, Path]] = None,
    **kwargs
):
    """
    Plot a map showing the performance of different samples based on a specified metric.

    This function creates a scatter plot on a map with points representing the performance 
    metric values at their respective latitude and longitude coordinates. The points are 
    color-coded based on the metric value and use different markers for each sample.

    Parameters:
    -----------
    x: pd.Series
        Coordinate X or longitude
    y: pd.Series
        Coordinate Y or latitude
    color: pd.Series
        Values used to define the colour of the points
    samples: pd.Series
        Defines if the points belong to the train, validation or test samples
    save: string or pathlib.Path (optional)
        The file path or name where the plot should be saved. If None, the plot will not be saved. Defaults to None.
    **kwargs: Arbitrary keyword arguments. Recognized arguments:
        - figsize (tuple): The figure size in inches (width, height). Defaults to (12, 4).
        - clim (tuple): The limits of the colorbar
        - cmap (str): The colormap name used to color the points. Defaults to 'coolwarm_r'.

    Returns:
    --------
        None
    """
    
    clim = kwargs.get('clim', (-1 , 1))
    figsize = kwargs.get('figsize', (12, 4))
    cmap = kwargs.get('cmap', 'coolwarm_r')

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection=ccrs.PlateCarree()))
    ax.add_feature(cf.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='lightgray'), alpha=.5, zorder=0)
    ax.axis('off')

    markers = ['o', '^', 'x']
    legend_handles = []
    for sample, marker in zip(samples.unique(), markers):
        mask = samples == sample
        sct = ax.scatter(x[mask], y[mask], c=color[mask], marker=marker, vmin=clim[0], vmax=clim[1], cmap=cmap, label=sample)
        legend_handle = mlines.Line2D([], [], color='gray', marker=marker, linestyle='None',
                                      markersize=6, label=sample)
        legend_handles.append(legend_handle)

    ax.legend(handles=legend_handles, frameon=False)
    fig.colorbar(sct, shrink=.666, label=color.name);
    
    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches='tight')

        
def lineplot_timeseries(
    cfg: Config,
    sim: pd.DataFrame,
    obs: pd.DataFrame,
    parameter_ranges: Dict[str, Tuple],
    save: Optional[Union[str, Path]] = None,
    **kwargs
):
    """
    Plot line charts for time series data comparing simulated and observed values.

    This function creates a multi-panel line plot (typically three subplots) where each panel
    can represent a different aspect of the time series data, such as parameter values, 
    outflows, and storage. The simulated and observed data are plotted for comparison.

    Parameters:
    -----------
    cfg (Config): A configuration object that contains settings and parameters used
        for the simulation and plotting.
    sim (pd.DataFrame): A DataFrame containing the simulated time series data with
        columns corresponding to time ('T'), outflows ('outflow'), and storage ('storage').
    obs (pd.DataFrame): A DataFrame containing the observed time series data with
        columns corresponding to normalized outflows ('outflow_norm'), storage 
        ('storage_norm'), and any dynamic inputs specified in `cfg.dynamic_inputs`.
    parameter_ranges: dictionary
        Defines the search range of each model parameter
    save (Optional[Union[str, Path]], optional): The file path or name where the plot
        should be saved. If None, the plot will be displayed without saving. Defaults to None.
    **kwargs: Arbitrary keyword arguments. Recognized arguments:
        - figsize (tuple): The figure size in inches (width, height). Defaults to (15, 9).
        - lw (float): The line width for the plot lines. Defaults to 0.8.
        - xlim (tuple): The limit for the x-axis. If not provided, it defaults to automatic scaling.
        - ylim (tuple): The limit for the y-axis of the first subplot. If not provided,
          it defaults to automatic scaling.
        - title (str): The title for the first subplot. If not provided, no title is set.

    Returns:
    --------
    None
    """

    figsize = kwargs.get('figsize', (15, 9))
    lw = kwargs.get('lw', .8)
    xlim = kwargs.get('xlim', None)
    title = kwargs.get('title', None)
    
    fig, ax = plt.subplots(nrows=3, figsize=figsize, sharex=True)
    
    # parameter value
    for i, par in enumerate(list(parameter_ranges)):
        ax[0].plot(sim[par], lw=lw, c=f'C{i}', label=par)
    lows, highs = zip(*parameter_ranges.values())
    ylim = (min(lows), max(highs))
    ax[0].set(xlim=xlim,
              ylim=ylim,
              ylabel='parameters')
    ax[0].legend(frameon=False)
    if title is not None:
        ax[0].set_title(title)
    ax2 = ax[0].twinx()
    ax2.plot(obs[cfg.dynamic_inputs], c='gray', alpha=.5, lw=lw, zorder=0)
    ax2.set_ylabel('dynamic inputs')
    
    # outflow
    ax[1].plot(obs['outflow_norm'], lw=lw, c='gray', label='obs')
    if 'outflow' in sim.columns:
        ax[1].plot(sim['outflow'], lw=lw, c='C0', label='sim')
    ax[1].set_ylabel('outflow (-)')
    ax[1].legend(frameon=False)
    
    # storage
    ax[2].plot(obs['storage_norm'], lw=lw, c='gray', label='obs')
    ax[2].plot(sim['storage'], lw=lw, c='C0', label='sim')
    ax[2].set(ylim=(-.02, 1.02),
              ylabel='storage (-)')
    ax[2].legend(frameon=False)
    
    if save is not None:
        plt.savefig(save, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
        
        
# def boxplot_parameters(
#     results: Dict[str, pd.DataFrame],
#     parameter: str = 'T',
#     reference: Optional[pd.Series] = None,
#     save: Optional[Union[str, Path]] = None,
#     **kwargs
# ):
#     """
#     Create a boxplot for a specific parameter across multiple datasets.

#     This function generates a boxplot for each dataset provided in the `results` dictionary,
#     displaying the distribution of values for a specified parameter. Additionally, a reference
#     value can be overlaid as a scatter plot on the boxplot for comparison.

#     Parameters:
#     -----------
#     results (Dict[str, pd.DataFrame]): A dictionary where each key is an identifier (ID)
#         for a dataset and each value is a DataFrame containing the data for that dataset.
#     parameter (str, optional): The name of the parameter to be plotted. Defaults to 'T'.
#     reference (Optional[pd.Series], optional): A Series containing reference values for the
#         parameter, indexed by the dataset ID. If provided, these reference values will be
#         plotted as individual points on the boxplots. Defaults to None.
#     save (Optional[Union[str, Path]], optional): The file path or name where the plot should
#         be saved. If None, the plot will not be saved. Defaults to None.
#     **kwargs: Arbitrary keyword arguments. Recognized arguments:
#         - figsize (tuple): The figure size in inches (width, height). Defaults to (16, 4).
#         - ylabel (str): The label for the y-axis. Defaults to the name of the parameter.
#         - ylim (tuple): The limit for the y-axis. If not provided, it defaults to automatic scaling.

#     Returns:
#     --------
#     None
#     """
    
#     figsize = kwargs.get('figsize', (16, 4))
#     ylabel = kwargs.get('ylabel', parameter)
#     ylim = kwargs.get('ylim', None)
    
#     fig, ax = plt.subplots(figsize=figsize)
#     positions = []
#     labels = []
#     for x, (ID, df) in enumerate(results.items()):
#         positions.append(x + 1)
#         labels.append(ID)
#         ax.boxplot(df[parameter], positions=[x + 1], widths=0.6, patch_artist=True, showfliers=False, showcaps=False, boxprops={'facecolor': 'lightgray', 'edgecolor': 'none'})
#         if reference is not None:
#             ax.scatter(x + 1, reference.loc[int(ID)], color='k', s=5, zorder=10)
#     ax.set_xticks(positions, labels, rotation=90)
#     ax.set(
#         xlabel='ID',
#         ylim=ylim,
#         ylabel=ylabel
#     );
#     ax.spines[['top', 'bottom', 'right']].set_visible(False)
#     ax.tick_params(axis='x', length=0)
#     if ylim is not None:
#         yticks = ax.get_yticks()
#         yticks[0] = ylim[0]
#         yticks[-1] = ylim[-1]
#         ax.set_yticks(yticks)
        
#     if save is not None:
#         plt.savefig(save, bbox_inches='tight');

        

def boxplot_parameters(
    results: Dict[str, Dict[str, pd.DataFrame]],
    parameter_ranges: Dict[str, Tuple],
    save: Optional[Union[str, Path]] = None,
    **kwargs
):
    """
    Create a boxplot for a specific parameter across multiple datasets.

    This function generates a boxplot for each dataset provided in the `results` dictionary,
    displaying the distribution of values for a specified parameter. Additionally, a reference
    value can be overlaid as a scatter plot on the boxplot for comparison.

    Parameters:
    -----------
    results (Dict[str, pd.DataFrame]): A dictionary where each key is an identifier (ID)
        for a dataset and each value is a DataFrame containing the data for that dataset.
    parameter (str, optional): The name of the parameter to be plotted. Defaults to 'T'.
    reference (Optional[pd.Series], optional): A Series containing reference values for the
        parameter, indexed by the dataset ID. If provided, these reference values will be
        plotted as individual points on the boxplots. Defaults to None.
    save (Optional[Union[str, Path]], optional): The file path or name where the plot should
        be saved. If None, the plot will not be saved. Defaults to None.
    **kwargs: Arbitrary keyword arguments. Recognized arguments:
        - figsize (tuple): The figure size in inches (width, height). Defaults to (16, 4).
        - ylabel (str): The label for the y-axis. Defaults to the name of the parameter.
        - ylim (tuple): The limit for the y-axis. If not provided, it defaults to automatic scaling.

    Returns:
    --------
    None
    """
    

    figsize = kwargs.get('figsize', (20, 4))
    
    nrows = len(parameter_ranges)
    fig, axes = plt.subplots(nrows=nrows, figsize=(figsize[0], figsize[1] * nrows), sharex=True)
    positions = []
    labels = []
    legend_lines = []
    for ax, (par, ylim) in zip(axes, parameter_ranges.items()):
        x = 1
        for i, (sample, dct) in enumerate(results.items()):
            for ID, df in dct.items():
                positions.append(x)
                labels.append(ID)
                ax.boxplot(df[par], positions=[x], widths=0.6, patch_artist=True, showfliers=False, showcaps=False,
                           boxprops={'facecolor': 'lightgray', 'edgecolor': 'none'},
                           medianprops={'color': f'C{i}'}
                          )
                x += 1
            if ax == axes[0]:
                legend_lines.append(mlines.Line2D([0], [0], color=f'C{i}', label=sample))
            
        ax.set_xticks(positions, labels, rotation=90)
        ax.tick_params(axis='x', length=0)
        if ax == axes[-1]:
            ax.set_xlabel('ID')
        ax.set(
            ylim=ylim,
            ylabel=par
        );
        ax.spines[['top', 'bottom', 'right']].set_visible(False)
        yticks = ax.get_yticks()
        yticks[0] = ylim[0]
        yticks[-1] = ylim[-1]
        ax.set_yticks(yticks)
        
    fig.legend(handles=legend_lines, loc='center left', bbox_to_anchor=(.925, .5), frameon=False)
    
    if save is not None:
        plt.savefig(save, bbox_inches='tight');