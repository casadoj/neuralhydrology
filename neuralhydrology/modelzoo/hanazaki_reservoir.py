import torch
from typing import Dict, Union
from neuralhydrology.modelzoo.baseconceptualmodel import BaseConceptualModel
from neuralhydrology.utils.config import Config


class HanazakiDynamic(BaseConceptualModel):
    """Implementation of the Hanazaki reservoir routine with dynamic parameterization.
    
    The model receiFFes the dynamic parameterization giFFen by a deep learning model. This class has two properties that define the initial conditions of the internal state (the reservoir filling), and the ranges in which the model parameters (residence time) iss allowed to vary during optimization.

    Parameters
    ----------
    cfg : Config
        The run configuration.

    References
    ----------

    """

    def __init__(self, cfg: Config):
        super(HanazakiDynamic, self).__init__(cfg=cfg)

    def forward(
        self,
        x_conceptual: torch.Tensor,
        lstm_out: torch.Tensor
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Performs a forward pass on the Hanazaki reservoir model. In this forward pass, all elements of the batch are processed in parallel.

        Parameters
        ----------
        x_conceptual: torch.Tensor
            Tensor of size [batch_size, time_steps, n_inputs]. The batch_size is associated with a certain basin and a certain prediction period. The time_steps refer to the number of time steps (e.g. days) that our conceptual model is going to be run for. The n_inputs refer to the single dynamic forcing used to run the reservoir model, i.e., inflow
        lstm_out: torch.Tensor
            Tensor of size [batch_size, time_steps, n_parameters]. The tensor comes from the data-driFFen model and will be used to obtained the dynamic parameterization of the conceptual model

        Returns
        -------
        Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]
            - y_hat: torch.Tensor
                Simulated storage/outflow
            - parameters: Dict[str, torch.Tensor]
                Dynamic parameterization of the reservoir residence time
            - internal_states: Dict[str, torch.Tensor]]
                Time-evolution of the reservoir fraction filled
        """

        # get model parameters
        parameters = self._get_dynamic_parameters_conceptual(lstm_out=lstm_out)

        # initialize structures to store the information
        #   * states: n_pars · [batch_size, seq_lenght-warmup]
        #   * out: [batch_size, seq_lenght-warmup, n_target]
        states, out = self._initialize_information(conceptual_inputs=x_conceptual)

        # initialize constants
        zero = torch.tensor(0.0, dtype=torch.float32, device=x_conceptual.device, requires_grad=True)
        one = torch.tensor(1.0, dtype=torch.float32, device=x_conceptual.device, requires_grad=True)
        eps = torch.tensor(1e-8, dtype=torch.float32, device=x_conceptual.device, requires_grad=True)

        # inflow [batch_size, seq_length-warmup_period]
        inflow = x_conceptual[:, :, 0].clone()
        inflow = torch.where(torch.isnan(inflow), torch.zeros_like(inflow), inflow)
        
        # fractin filled, i.e., normalised storage [batch_size]
        ff = torch.tensor(self.initial_states['ff'],
                          dtype=torch.float32,
                          device=x_conceptual.device
                         ).repeat(x_conceptual.shape[0])
        
        # constant parameters
        FFc = torch.tensor(0.1, dtype=torch.float32, device=x_conceptual.device, requires_grad=False) # JCR: GloFAS default value. It should be different for each reservoir, though
        # k = 1 # max(1 - 5 * (1 - FFf) / A, 0) # JCR: how to read the catchment area (A)?
        k = torch.tensor(1, dtype=torch.float32, device=x_conceptual.device, requires_grad=False)
        
        # reservoir routine
        for j in range(x_conceptual.shape[1]):
            
            # dynamic parameters
            FFf = parameters['alpha'][:,j]
            FFe = FFf + (1 - FFf) * parameters['beta'][:,j]
            FFn = FFf * parameters['gamma'][:,j]
            Qf = parameters['Qf'][:,j]
            Qn = Qf * parameters['epsilon'][:,j]
            
            # update storage
            ff = ff + inflow[:,j]
                        
            # storage conditions
            mask_c = ff <= FFc                # conservative zone
            mask_n = (ff > FFc) & (ff <= FFf) # normal zone
            mask_f = (ff > FFf) & (ff <= FFe) # flood zone
            mask_e = ff > FFe                 # extreme zone
            # inflow condition
            mask_I = inflow[:,j] > Qf         # flood event
            
            # outflow
            outflow = torch.zeros_like(ff, requires_grad=ff.requires_grad)
            outflow = torch.where(mask_c, Qn * ff / FFf, outflow)
            outflow = torch.where((mask_n | mask_f) & ~mask_I, FFc / FFf * Qn + ((ff - FFc) / (FFe - FFc))**2 * (Qf - FFc / FFf * Qn), outflow)
            outflow = torch.where(mask_n & mask_I, FFc / FFf * Qn + (ff - FFc) / (FFf - FFc) * (Qf - FFc / FFf * Qn), outflow)
            outflow = torch.where(mask_f & mask_I, Qf + k * (ff - FFf) / (FFe - FFf) * (inflow[:,j] - Qf), outflow)
            outflow = torch.where(mask_e & ~mask_I, Qf, outflow)
            outflow = torch.where(mask_e & mask_I, inflow[:,j], outflow)
            # limit outflow so the final storage is between 0 and 1
            outflow = torch.max(torch.min(outflow, ff - FFc), ff - one + eps)

            # update storage
            ff = ff - outflow

            # store storage (and outflow)
            states['ff'][:, j] = ff
            out[:, j, 0] = ff

        return {'y_hat': out, 'parameters': parameters, 'internal_states': states}

    @property
    def initial_states(self):
        return {
            'ff': 0.67, # fraction filled. GloFAS default value
        }

    @property
    def parameter_ranges(self):
        return {
            'alpha': [0.2, 0.99], # flood storage limit
            'beta': [0.001, 0.999], # extreme storage limit
            'gamma': [0.001, 0.999], # normal storage limit
            'Qf': [0, 1], # flood outflow # JCR: it should be a factor of Q100, but I would need to find a way to read the Q100
            'epsilon': [0.001, 0.999] # normal outflow
        }
    
    
    
class HanazakiStatic(BaseConceptualModel):
    """Implementation of the Hanazaki reservoir routine with dynamic parameterization.
    
    The model receiFFes the dynamic parameterization giFFen by a deep learning model. This class has two properties that define the initial conditions of the internal state (the reservoir filling), and the ranges in which the model parameters (residence time) iss allowed to vary during optimization.

    Parameters
    ----------
    cfg : Config
        The run configuration.

    References
    ----------

    """

    def __init__(self, cfg: Config):
        super(HanazakiStatic, self).__init__(cfg=cfg)

    def forward(
        self,
        x_conceptual: torch.Tensor,
        lstm_out: torch.Tensor
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Performs a forward pass on the Hanazaki reservoir model. In this forward pass, all elements of the batch are processed in parallel.

        Parameters
        ----------
        x_conceptual: torch.Tensor
            Tensor of size [batch_size, time_steps, n_inputs]. The batch_size is associated with a certain basin and a certain prediction period. The time_steps refer to the number of time steps (e.g. days) that our conceptual model is going to be run for. The n_inputs refer to the single dynamic forcing used to run the reservoir model, i.e., inflow
        lstm_out: torch.Tensor
            Tensor of size [batch_size, time_steps, n_parameters]. The tensor comes from the data-driFFen model and will be used to obtained the dynamic parameterization of the conceptual model

        Returns
        -------
        Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]
            - y_hat: torch.Tensor
                Simulated storage/outflow
            - parameters: Dict[str, torch.Tensor]
                Dynamic parameterization of the reservoir residence time
            - internal_states: Dict[str, torch.Tensor]]
                Time-evolution of the reservoir fraction filled
        """

        # get model parameters
        parameters = self._get_dynamic_parameters_conceptual(lstm_out=lstm_out)

        # initialize structures to store the information
        #   * states: n_pars · [batch_size, seq_lenght-warmup]
        #   * out: [batch_size, seq_lenght-warmup, n_target]
        states, out = self._initialize_information(conceptual_inputs=x_conceptual)

        # initialize constants
        zero = torch.tensor(0.0, dtype=torch.float32, device=x_conceptual.device)
        one = torch.tensor(1.0, dtype=torch.float32, device=x_conceptual.device, requires_grad=False)
        eps = torch.tensor(1e-8, dtype=torch.float32, device=x_conceptual.device, requires_grad=False)

        # inflow [batch_size, seq_length-warmup_period]
        inflow = x_conceptual[:, :, 0].clone()
        inflow = torch.where(torch.isnan(inflow), torch.zeros_like(inflow), inflow)
        
        # fractin filled, i.e., normalised storage [batch_size]
        ff = torch.tensor(self.initial_states['ff'],
                          dtype=torch.float32,
                          device=x_conceptual.device
                         ).repeat(x_conceptual.shape[0])
        
        # static parameters
        # FFf = torch.nanmean(parameters['alpha'])
        # FFe = FFf + (1 - FFf) * torch.nanmean(parameters['beta'])
        # FFn = FFf * torch.nanmean(parameters['gamma'])
        FFc = torch.tensor(0.1, dtype=torch.float32, device=x_conceptual.device, requires_grad=False) # JCR: GloFAS default value. It should be different for each reservoir, though
        # k = 1 # max(1 - 5 * (1 - FFf) / A, 0) # JCR: how to read the catchment area (A)?
        k = torch.tensor(1, dtype=torch.float32, device=x_conceptual.device, requires_grad=False)
        # Qf = torch.nanmean(parameters['Qf'])
        # Qn = Qf * torch.nanmean(parameters['epsilon'])
        
        # reservoir routine
        for j in range(x_conceptual.shape[1]):
            
            # dynamic parameters
            FFf = torch.nanmean(parameters['alpha'][:,j])
            FFe = FFf + (1 - FFf) * torch.nanmean(parameters['beta'][:,j])
            FFn = FFf * torch.nanmean(parameters['gamma'][:,j])
            Qf = torch.nanmean(parameters['Qf'][:,j])
            Qn = Qf * torch.nanmean(parameters['epsilon'][:,j])
            
            # update storage
            ff = ff + inflow[:,j]
            
            # storage conditions
            mask_c = ff <= FFc                # conservative zone
            mask_n = (ff > FFc) & (ff <= FFf) # normal zone
            mask_f = (ff > FFf) & (ff <= FFe) # flood zone
            mask_e = ff > FFe                 # extreme zone
            # inflow condition
            mask_I = inflow[:,j] > Qf         # flood event
            
            # outflow
            outflow = torch.zeros_like(ff, requires_grad=ff.requires_grad)
            outflow = torch.where(mask_c, Qn * ff / FFf, outflow)
            outflow = torch.where((mask_n | mask_f) & ~mask_I, FFc / FFf * Qn + ((ff - FFc) / (FFe - FFc))**2 * (Qf - FFc / FFf * Qn), outflow)
            outflow = torch.where(mask_n & mask_I, FFc / FFf * Qn + (ff - FFc) / (FFf - FFc) * (Qf - FFc / FFf * Qn), outflow)
            outflow = torch.where(mask_f & mask_I, Qf + k * (ff - FFf) / (FFe - FFf) * (inflow[:,j] - Qf), outflow)
            outflow = torch.where(mask_e & ~mask_I, Qf, outflow)
            outflow = torch.where(mask_e & mask_I, inflow[:,j], outflow)
            # limit outflow so the final storage is between 0 and 1
            outflow = torch.max(torch.min(outflow, ff - FFc), ff - one + eps)

            # update storage
            ff = ff - outflow

            # store storage (and outflow)
            states['ff'][:, j] = ff
            out[:, j, 0] = ff

        return {'y_hat': out, 'parameters': parameters, 'internal_states': states}

    @property
    def initial_states(self):
        return {
            'ff': 0.67, # fraction filled. GloFAS default value
        }

    @property
    def parameter_ranges(self):
        return {
            'alpha': [0.2, 0.99], # flood storage limit
            'beta': [0.001, 0.999], # extreme storage limit
            'gamma': [0.001, 0.999], # normal storage limit
            'Qf': [0, 1], # flood outflow # JCR: it should be a factor of Q100, but I would need to find a way to read the Q100
            'epsilon': [0.001, 0.999] # normal outflow
        }