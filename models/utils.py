from typing import List, Dict, Union, Optional
from pathlib import Path
import pandas as pd
import xarray as xr
import random



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
    assert 0 < val <= 1, '"val" debe de ser un valor entre 0 y 1.'

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
    ids_val = random.sample(ids, n_val)
    ids_val.sort()
    
    # select the training subsample
    ids_cal = [id for id in ids if id not in ids_val]
    ids_cal.sort()
    
    assert (len(ids_cal) + len(ids_val) + len(ids_test)) == n_stns, 'The union of training, validation and testing subsamples is smaller than the original sample "ids"'

    # reorganize as dictionary
    samples = {'train': ids_cal, 'validation': ids_val}
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