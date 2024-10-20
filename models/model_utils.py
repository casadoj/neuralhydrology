from typing import List, Dict, Union, Optional, Literal
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import random
import matplotlib.pyplot as plt
# import pickle

from neuralhydrology.evaluation import metrics



def split_samples(ids: List[Union[str, int]], cal: float = .7, val: float = .3, path: Optional[Path] = None, seed: int = 0) -> Optional[Dict]:
    """Given a sample, it generates two (three) subsamples: training, validation (and test). The results can be put out as a dictionary or saved in TXT files.

    Parameters:
    -----------
    ids: list
        List of station, reservoir (or other) ID
    cal: float
        Proportion of samples to be included in the training subsample
    val: float
        Proportion of samples to be included in the validation subsample. If the sum of "cal" and "val" is smaller than 1, the remaining samples will be included in the test subsample
    path: optionalor pathlib.Path
        If provided, the directory where the TXT files with the subsamples will be saved. By default is None and the output is a dictionary
    seed: int
        Seed used to randomly select subsambples

    Devuelve:
    ---------
    optional or dictionary
        If "path" is provided, the output are 2 (or 3) TXT files. Else, the output is a dictionary
    """

    n_stns = len(ids)
    random.seed(seed)
    
    assert 0 < cal <= 1, '"cal" debe de ser un valor entre 0 y 1.'
    assert 0 <= val <= 1, '"val" debe de ser un valor entre 0 y 1.'

    if path is not None:
        # export the complete sample
        with open(path / 'sample_complete.txt', 'w') as file:
            file.writelines(f"{id}\n" for id in ids)

    # set the validation size
    if cal + val > 1:
        val = 1 - cal
        print(f'"val" fue truncado a {val:0.2f}')
        
    # select the evaluation subsample
    if cal + val < 1.:
        test = 1 - cal - val
        n_test = int(n_stns * test)
        ids_test = random.sample(ids, n_test)
        ids_test.sort()
        ids = [id for id in ids if id not in ids_test]
    else:
        test = None
        ids_test = []
        
    # select the validation subsample
    n_val = int(n_stns * val)
    if n_val > 0:
        ids_val = random.sample(ids, n_val)
        ids_val.sort()
        ids = [id for id in ids if id not in ids_val]
    else:
        val = None
        ids_val = []

    # select the training subsample
    ids_cal = ids
    ids_cal.sort()
    
    assert (len(ids_cal) + len(ids_val) + len(ids_test)) == n_stns, 'The union of training, validation and testing subsamples is smaller than the original sample "ids"'

    # reorganize as dictionary
    samples = {'train': ids_cal}
    if val is not None:
        samples.update({'validation': ids_val})
    if test is not None:
        samples.update({'test': ids_test})
            
    # export or put out
    if path is not None:
        for key, ls in samples.items():
            with open(path / f'sample_{key}.txt', 'w') as file:
                file.writelines(f"{id}\n" for id in ls)
    else:
        return samples



def split_periods(serie: pd.Series, cal: float = .6, val: float = .2) -> xr.DataArray:
    """Given a time series, it defines the start and end of the training, validation (and testing) periods.

    Parameters:
    -----------
    serie: pd.Series
        Time series to be split
    cal: float
        Proportion of the original time series to be included in the training period. This data will be taken from the most recent period
    val: float
        Proportion of the original time series to be included as validation period. This data will be taken from the period immediately preceeding the training period. If the sum of "cal" and "val" is smaller than 1, the remaining data will be used as testing period.

    Devuelve:
    ---------
    xr.DataArray
        For each period, it defines the start and end date.
    """

    assert 0 <= cal <= 1, '"cal" debe de tener un valor entre 0 y 1'
    assert 0 <= val <= 1, '"val" debe de tener un valor entre 0 y 1'
    assert cal + val <= 1, 'La suma de "entrenamiento" y "validación no puede ser mayor de 1."'

    if cal + val < 1:
        test = 1 - cal - val
    else:
        test = 0

    # complete period
    start = serie.first_valid_index()
    end = serie.last_valid_index()
    n_days = (end - start).days

    # testing period
    if test > 0:
        start_test = start
        n_test = round(test * n_days)
        end_test = start_test + pd.Timedelta(days=n_test)
    else:
        start_test, end_test = np.datetime64('NaT', 'ns'), np.datetime64('NaT', 'ns')

    # validation period
    if val > 0:
        if test > 0:
            start_val = end_test + pd.Timedelta(days=1)
        else:
            start_val = start
        n_val = round(val * n_days)
        end_val = start_val +  pd.Timedelta(days=n_val)
    else:
        start_val, end_val = np.datetime64('NaT', 'ns'), np.datetime64('NaT', 'ns')

    # training period
    if cal > 0:
        if val > 0:
            start_cal = end_val + pd.Timedelta(days=1)
        else:
            if test > 0:
                start_cal = end_test + pd.Timedelta(days=1)
            else:
                start_cal = start
        end_cal = end
    else:
        start_cal, end_cal = np.datetime64('NaT', 'ns'), np.datetime64('NaT', 'ns')

    # xarray.DataArray cwith the start and end dates for each period
    da = xr.DataArray([[start_cal, start_val, start_test], [end_cal, end_val, end_test]],
                          coords={'date': ['start', 'end'],
                                  'period': ['train', 'validation', 'test']},
                          dims=['date', 'period'])
        
    return da



# def plot_results(results: Dict, period: str, target: str, save: Union[str, Path] = None, **kwargs):
#     """
#     """
    
#     figsize = kwargs.get('figsize', (16, 4))
#     ylim = kwargs.get('ylim', (-1, None))
#     ylabel = kwargs.get('ylabel', None)
    
#     for id, dct in results.items():

#         fig, ax = plt.subplots(figsize=figsize)#, sharex=True, sharey=True)

#         # extraer series
#         qobs = dct['1D']['xr'][f'{target}_obs'].isel(time_step=0).to_pandas()
#         qsim = dct['1D']['xr'][f'{target}_sim'].isel(time_step=0).to_pandas()
        
#         # plot series
#         qobs.plot(c='darkgray', lw=1, label='obs', ax=ax)
#         qsim.plot(c='steelblue', lw=1.2, label='sim', ax=ax)

#         # config
#         ax.set(xlabel='', ylabel=ylabel, ylim=ylim)
#         ax.set_title(f'{id} - {period}')
#         ax.text(0.01, 0.925, 'KGE = {0:.3f}'.format(dct['1D']['KGE']), transform=ax.transAxes, fontsize=11)
#         ax.legend(loc=0, frameon=False);

#         if save is not None:
#             plt.savefig(f'{save}/{id:04}.jpg', dpi=300, bbox_inches='tight')
            
            
            
def plot_timeseries(results: Dict, target: str, save: Optional[Union[str, Path]] = None, **kwargs):
    """
    Plot the observed and simulated time series data from a dictionary of results.

    Parameters
    ----------
    results : Dict
        A dictionary containing the time series data to plot. The dictionary should be organized with
        periods as keys and another dictionary as values, which then contain IDs as keys and time series
        data as values.
    target : str
        The target variable name that will be used to extract observed and simulated data from the time
        series data. The expected keys in the time series data are formatted as '{target}_obs' and
        '{target}_sim'.
    save : Union[str, Path], optional
        The directory path where the plots should be saved. If not specified, plots will not be saved.
        Default is None.
    **kwargs
        Arbitrary keyword arguments. Currently supported are:
        - figsize: tuple of int or float, optional
            The figure size of each plot. Default is (16, 4).
        - ylim: tuple of int, float or None, optional
            The y-axis limits for the plots. Default is (-1, None).
        - ylabel: str or None, optional
            The label for the y-axis. Default is None.
    """
    
    figsize = kwargs.get('figsize', (16, 4))
    ylim = kwargs.get('ylim', (-.05, None))
    # ylabel = kwargs.get('ylabel', None)
    
    for period, dct in results.items():
        
        if save is not None:
            path = save / 'plots' / period
            path.mkdir(parents=True, exist_ok=True)
    
        for ID, ts in dct.items():

            fig, ax = plt.subplots(figsize=figsize)#, sharex=True, sharey=True)

            # extraer series
            obs = ts[f'{target}_obs'].isel(time_step=0).to_pandas()
            sim = ts[f'{target}_sim'].isel(time_step=0).to_pandas()

            # plot series
            obs.plot(c='darkgray', lw=1, label='obs', ax=ax)
            sim.plot(c='steelblue', lw=1.2, label='sim', ax=ax)

            # config
            ax.set(xlabel='',
                   ylabel=target,
                   ylim=ylim)
            ax.set_title(f'{ID} - {period}')
            ax.text(0.01, 0.925, 'NSE = {0:.3f}'.format(metrics.nse(obs, sim)), transform=ax.transAxes, fontsize=11)
            ax.legend(loc=0, frameon=False);

            if save is not None:
                plt.savefig(path / f'{ID:04}.jpg', dpi=300, bbox_inches='tight')
                
    plt.close()
        
        
        
def get_results(run_dir: Union[str, Path],
                 sample: Literal['train', 'validation', 'test'],
                 epoch: int,
                 freq: str = '1D'
               ) -> (dict, pd.DataFrame):
    """Reads the results for a given subsample and epoch
    
    Parameters:
    -----------
    run_dir: string or pathlib.Path
        directory of the run to be analysed. It is defined in the `neuralhydrology` configuration file, otherwise it takes the default value './runs'
    sample: string
        the subsample to be analysed. Either 'train', 'validation' or 'test'
    epoch: integer
        the training epoch for which results will be read
    freq: string
        temporal resolution of the time series
        
    Returns:
    --------
    results: dictionary
        it contains for each basin in the sample an xarray.Dataset with simulated and observed target variables and the performance metrics
    metrics: pd.DataFrame
        a table with the performance for each basin in the sample
    """
    
    if isinstance(run_dir, str):
        run_dir = Path(run_dir)
    path = run_dir / sample / f'model_epoch{epoch:03}'
    
    # time series
    results = pd.read_pickle(path / f'{sample}_results.p')
        
    # list basin IDs
    IDs = [int(ID) for ID in results.keys()]
    IDs.sort()

    # extract time series
    timeseries = {ID: results[str(ID)][freq].pop('xr', None) for ID in IDs}

    # extract performance
    performance = {ID: results[str(ID)][freq] for ID in IDs}
    performance = pd.DataFrame.from_dict(performance).transpose()
    
    return timeseries, performance