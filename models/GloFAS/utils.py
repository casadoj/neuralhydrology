import numpy as np
import pandas as pd
from pathlib import Path
import xml.etree.ElementTree as ET
from dateutil import parser
from typing import Optional, Union, List, Tuple


def xml_timeinfo(xml: Union[str, Path]):
    """It extracts the temporal information from the settings XML file.
    
    Input:
    ------
    xml:               str. A XML settings file (path, filename and extension)
    
    Output:
    -------
    CalendarDayStart:  datetime. Origin of time considered in the simulation
    DtSec:             int. Temporal resolution of the simulation in seconds
    StepStart:         datetime. First timestep of the simulation
    StepEnd:           datetime. Last timestep of the simulation
    """
    
    # extract temporal info from the XML
    tree = ET.parse(xml)
    root = tree.getroot()
    CalendarDayStart = root.find('.//textvar[@name="CalendarDayStart"]').attrib['value']
    CalendarDayStart = parser.parse(CalendarDayStart, dayfirst=True)
    DtSec = int(root.find('.//textvar[@name="DtSec"]').attrib['value'])
    StepStart = root.find('.//textvar[@name="StepStart"]').attrib['value']
    try:
        StepStart = CalendarDayStart + int(StepStart) * timedelta(seconds=DtSec)
    except:
        StepStart = parser.parse(StepStart, dayfirst=True)
    StepEnd = root.find('.//textvar[@name="StepEnd"]').attrib['value']
    try:
        StepEnd = CalendarDayStart + int(StepEnd) * timedelta(seconds=DtSec)
    except:
        StepEnd = parser.parse(StepEnd, dayfirst=True)
        
    return CalendarDayStart, DtSec, StepStart, StepEnd



def xml_parameters(xml: Union[str, Path]) -> pd.Series:
    """It extracts the model parameters from the settings XML file.
    
    Input:
    ------
    xml: string or pathlib.Path
        A XML settings file (path, filename and extension)
    
    Output:
    -------
    pars:  pandas.Series
        Values of the calibrated model parameters in the xml file
    """
    
    parameter_names = {
        'UZTC': 'UpperZoneTimeConstant',
        'LZTC': 'LowerZoneTimeConstant',
        'GwPercValue': 'GwPercValue',
        'LZthreshold': "LZThreshold",
        'b_Xinanjiang': 'b_Xinanjiang',
        'PowerPrefFlow': 'PowerPrefFlow',
        'SnowMeltCoef': "SnowMeltCoef",
        'CalChanMan1': "CalChanMan",
        'CalChanMan2': "CalChanMan2",
        'LakeMultiplier': "LakeMultiplier",
        'adjustNormalFlood': "adjust_Normal_Flood",
        'ReservoirRnormqMult': "ReservoirRnormqMult",
        'QSplitMult': "QSplitMult",
        'GwLoss': "GwLoss",
    }

    # extract calibrated model parameters from the XML
    tree = ET.parse(xml)
    root = tree.getroot()
    pars = pd.Series({name: float(root.find(f'.//textvar[@name="{par}"]').attrib['value']) for name, par in parameter_names.items()})
    
    return pars



def read_tss(tss: Union[str, Path],
             xml: Optional[Union[str, Path]] = None,
             squeeze: bool = True
            ) -> Union[pd.DataFrame, pd.Series]:
    """It generates a Pandas DataFrame or Series from a TSS file. The settings XML file is required to add the temporal information to the time series; if not provided, the index will contain integers indicating the timestep since the considered origin of time.
    
    Inputs:
    -------
    tss:     str
        The TSS file (path, filename and extension) to be read.
    xml:     str. The XML settings file (path, filename and extension) for the simulation that created the TSS file.
    squeeze: boolean. If the TSS file has only one timeseries, if converts the pandas.DataFrame into a pandas.Series
    
    Output:
    -------
    df:      pandas.DataFrame or pandas.Series. 
    """
 
    # extract info from the header
    with open(tss, mode='r') as f:
        header = f.readline()
        n_cols = int(f.readline().strip())
        cols = [f.readline().strip() for i in range(n_cols)]        
    
    # extract timeseries
    df = pd.read_csv(tss, skiprows=2 + n_cols, delim_whitespace=True, header=None)
    df.columns = cols
    
    # define timesteps
    if xml is None:
        df.set_index(cols[0], drop=True, inplace=True)
    else:
        df.drop(cols[0], axis=1, inplace=True)
        # extract temporal info from the XML
        CalendarDayStart, DtSec, StepStart, StepEnd = xml_timeinfo(xml)
        # generate timesteps
        df.index = pd.date_range(start=StepStart, end=StepEnd, freq=f'{DtSec}s')
        df.index.name = 'Timestamp'

    if squeeze & (df.shape[1] == 1):
        df = df.iloc[:,0]
        
    return df



def KGEmod(obs: pd.Series, sim: pd.Series, sa: float = 1, sb: float = 1, sr: float = 1) -> List[float]:
    """It computes the modified Kling-Gupta efficiency coefficient.
    
    Parameters:
    -----------
    obs:   pd.Series.
        Observed time series
    sim:   pd.Series
        Simulated time series
    sa, sb, sr: float
        Scale factors of the three terms of the modified KGE: ratio of the coefficient of variation (alpha), bias (beta), and coefficient of correlation (r), respectively
    
    Returns:
    -------
    KGE: float
        Modified KGE
    alpha: float
        Ratio of the coefficient of variation
    beta: float
        Bias, i.e., ratio of the mean values
    r: float
        Coefficient of correlation
    """
    
    # Eliminar pasos sin dato
    data = pd.concat((obs, sim), axis=1)
    data.columns = ['obs', 'sim']
    data.dropna(axis=0, how='any', inplace=True)
    # Para la función si no hay datos
    assert data.shape[0] > 0, "ERROR. No indices match between the observed and the simulated series."

    # calcular cada uno de los términos del KGE
    alpha = (data.sim.std() / data.sim.mean()) / (data.obs.std() / data.obs.mean())
    beta = data.sim.mean() / data.obs.mean()
    r = np.corrcoef(data.obs, data.sim)[0, 1]
    
    # Cacular KGE
    ED = np.sqrt((sr * (r - 1))**2 + (sa * (alpha - 1))**2 + (sb * (beta - 1))**2)
    KGEmod = 1 - ED
    
    return KGEmod, alpha, beta, r