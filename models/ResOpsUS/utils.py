import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional
import torch

from neuralhydrology.utils.config import Config

def KGE(obs: pd.Series, sim: pd.Series, sa: float = 1, sb: float = 1, sr: float = 1) -> List[float]:
    """It computes the Kling-Gupta efficiency coefficient.
    
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
        Ratio of the standard deviations
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
    alpha = data.sim.std() / data.obs.std()
    beta = data.sim.mean() / data.obs.mean()
    r = np.corrcoef(data.obs, data.sim)[0, 1]
    
    # Cacular KGE
    ED = np.sqrt((sr * (r - 1))**2 + (sa * (alpha - 1))**2 + (sb * (beta - 1))**2)
    KGE = 1 - ED
    
    return KGE, alpha, beta, r



def ECDF(series: pd.Series, plot: bool = True, **kwargs) -> pd.Series:
    """it computes the empirical cumulative distribution function (ECDF) of the input data. If specified, the ECDF is plotted.

    Parametres:
    -----------
    series:     pd.Series. 
        Input data
    plot:      bool
        Whether to plot or not the ECMWF

    Output:
    -------
    ecdf:      pd.Series
        The ECDF, where the index represents the non-exceedance probability and the values are those of the input data in ascending order
        
    Keyword arguments:
    ------------------
    lw:        float
        Line width for the line plot
    c:         str
        Line colour for the line plot
    xlim:      Tuple (2,)
        Limits of the X axis in the plot
    ylim:      Tuple (2,)
        Limits of the Y axis in the plot
    ylabel:    str
        Lable of the Y axis in the plot
    title:     str
        Text to be added as the plot title.
    """
    
    if series.isnull().sum() > 0:
        print('WARNING: there are NaN in the input time series')
        data = series.dropna()
    else:
        data = series
        
    ecdf = pd.Series(data=data.sort_values().values,
                     index=np.arange(1, data.shape[0] + 1) / data.count(),
                     name='ECDF')

    if plot:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(ecdf.index, ecdf, lw=kwargs.get('lw', 1), c=kwargs.get('c', 'steelblue'))
        ax.set(ylim=kwargs.get('ylim', (-1.02, 1.02)), ylabel=kwargs.get('ylabel', series.name),
               xlim=(-.02, 1.02), xlabel='ECDF (-)',
               title=kwargs.get('title', ''));

    return ecdf


def ECDF_AUC(series: pd.Series) -> float:
    """
    This function computes the area under the ECDF curve.

    Parameters:
    -----------
    series: pd.Series
        Input data

    Output:
    -------
    auc : float
        The area under the ECDF curve.
    """
    
    ecdf = ECDF(series, plot=False)
    auc = np.trapz(ecdf.values, ecdf.index)
    
    return auc



def evaluate(cfg, model, dataloader):
    """It runs the hybrid model step by step for a single basin and puts out the input of the conceptual model, the model parameters, internal states and target variable
    
    Taken from `HybridModel.forward()`
    
    Parameters:
    -----------
    cfg: Config
    model: HybridModel
    dataloader: 
    
    Returns:
    --------
    pred: Dict[str, torch.Tensor]
    """
    
    model.eval()
    predictions = []
    inflow = []
    # obs = []
    parnames = list(model.conceptual_model.parameter_ranges)
    with torch.inference_mode():
        for data in dataloader:

            # run the LSTM
            # concatenate dynamic and static variables
            x_s_expanded = data['x_s'].unsqueeze(1).expand(-1, data['x_d'].size(1), -1)
            x_concatenated = torch.cat((data['x_d'], x_s_expanded), dim=2).to(cfg.device)
            lstm_output, _ = model.lstm(x_concatenated)

            # run the FC
            linear_output = model.linear(lstm_output[:, cfg.warmup_period:, :])

            # run conceptual model
            x_conceptual = data['x_d_c'][:, cfg.warmup_period:, :]
            pred = model.conceptual_model(x_conceptual=x_conceptual.to(cfg.device),
                                          lstm_out=linear_output)
            inflow.append(x_conceptual)
            predictions.append(pred)
    
    pred = {
        'inflow': torch.cat(inflow, dim=0),
        'storage': torch.cat([pred['internal_states']['ff'] for pred in predictions], dim=0).detach().cpu(),
        'parameters': {par: torch.cat([pred['parameters'][par] for pred in predictions], dim=0).detach().cpu() for par in parnames}
    }
    if cfg.target_variables[0].split('_')[0] == 'outflow':
        pred['outflow'] = torch.cat([pred['y_hat'] for pred in predictions], dim=0).detach().cpu()
    
    return pred