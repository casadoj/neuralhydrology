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
        self.target_variables = cfg.target_variables

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
        # zero = torch.tensor(0.0, dtype=torch.float32, device=x_conceptual.device, requires_grad=False)
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
        
        # constant parameters
        # k = 1 # max(1 - 5 * (1 - FFf) / A, 0) # JCR: how to read the catchment area (A)?
        k = torch.tensor(1, dtype=torch.float32, device=x_conceptual.device, requires_grad=False)
        Qn = torch.nanmean(inflow[:,-1])
        
        # reservoir routine
        for j in range(x_conceptual.shape[1]):
            
            # dynamic parameters
            FFf = parameters['Vf'][:,j]
            FFe = 0.8 + 0.2 * FFf
            FFc = 0.5 * FFf
            Qf = parameters['Qf'][:,j]
            Qc = Qn * FFc / FFf
            
            # inflow
            Qin = inflow[:,j]
            # inflow condition
            mask_I = Qin > Qf
            
            # update storage
            ff = ff + Qin
            
            # outflow
            outflow = torch.zeros_like(ff)
            # conservative zone
            outflow = torch.where(
                ff <= FFc,
                Qn * ff / FFf,
                outflow
            )
            # normal zone and NO flood inflow
            outflow = torch.where(
                (ff > FFc) & (ff <= FFe) & ~mask_I,
                Qc + ((ff - FFc) / (FFe - FFc))**2 * (Qf - Qc),
                outflow
            )
            # normal zone and flood inflow
            outflow = torch.where(
                (ff > FFc) & (ff <= FFf) & mask_I,
                FFc / FFf * Qn + (ff - FFc) / (FFf - FFc) * (Qf - FFc / FFf * Qn),
                outflow
            )
            # flood zone and flood inflow
            outflow = torch.where(
                (ff > FFf) & (ff <= FFe) & mask_I,
                Qf + k * (ff - FFf) / (FFe - FFf) * (Qin - Qf),
                outflow
            )
            # emergency zone and NO flood inflow
            outflow = torch.where(
                (ff > FFe) & ~mask_I,
                Qf,
                outflow
            )
            # emergency zone and flood inflow
            outflow = torch.where(
                (ff > FFe) & mask_I,
                Qin,
                outflow
            )
            # limit outflow so the final storage is between 0 and 1
            outflow = torch.max(torch.min(outflow, ff - FFc), ff - one + eps)

            # update storage
            ff = ff - outflow

            # store storage (and outflow)
            states['ff'][:, j] = ff
            if len(self.target_variables) == 1:
                if 'outflow' in self.target_variables[0]:
                    out[:, j, 0] = outflow
                elif 'storage' in self.target_variables[0]:
                    out[:, j, 0] = ff
            elif len(self.target_variables) == 2:
                out[:, j, 0] = ff
                out[:, j, 1] = outflow
            

        return {'y_hat': out, 'parameters': parameters, 'internal_states': states}

    @property
    def initial_states(self):
        return {
            'ff': 0.67, # fraction filled. GloFAS default value
        }

    @property
    def parameter_ranges(self):
        return {
            'Vf': [0.5, 0.99], # flood storage limit
            'Qf': [0.0001, 0.5], # flood outflow # JCR: it should be a factor of Q100, but I would need to find a way to read the Q100
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
        self.target_variables = cfg.target_variables

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
        # zero = torch.tensor(0.0, dtype=torch.float32, device=x_conceptual.device, requires_grad=False)
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
        
        # constant parameters
        # FFf = torch.nanmedian(parameters['Vf'])
        # FFe = 0.8 + 0.2 * FFf
        # FFc = 0.5 * FFf
        # Qf = torch.nanmedian(parameters['Qf'])
        
        # k = 1 # max(1 - 5 * (1 - FFf) / A, 0) # JCR: how to read the catchment area (A)?
        k = torch.tensor(1, dtype=torch.float32, device=x_conceptual.device, requires_grad=False)
        Qn = torch.nanmean(inflow[:,-1])
        
        # reservoir routine
        for j in range(x_conceptual.shape[1]):
            
            # dynamic parameters
            FFf = torch.mean(parameters['Vf'][:,j])
            FFe = 0.8 + 0.2 * FFf
            FFc = 0.5 * FFf
            Qf = torch.mean(parameters['Qf'][:,j])
            Qc = Qn * FFc / FFf
            
            # inflow
            Qin = inflow[:,j]
            # inflow condition
            mask_I = Qin > Qf
            
            # update storage
            ff = ff + Qin
            
            # outflow
            outflow = torch.zeros_like(ff)
            # conservative zone
            outflow = torch.where(
                ff <= FFc,
                Qn * ff / FFf,
                outflow
            )
            # normal zone and NO flood inflow
            outflow = torch.where(
                (ff > FFc) & (ff <= FFe) & ~mask_I,
                Qc + ((ff - FFc) / (FFe - FFc))**2 * (Qf - Qc),
                outflow
            )
            # normal zone and flood inflow
            outflow = torch.where(
                (ff > FFc) & (ff <= FFf) & mask_I,
                FFc / FFf * Qn + (ff - FFc) / (FFf - FFc) * (Qf - FFc / FFf * Qn),
                outflow
            )
            # flood zone and flood inflow
            outflow = torch.where(
                (ff > FFf) & (ff <= FFe) & mask_I,
                Qf + k * (ff - FFf) / (FFe - FFf) * (Qin - Qf),
                outflow
            )
            # emergency zone and NO flood inflow
            outflow = torch.where(
                (ff > FFe) & ~mask_I,
                Qf,
                outflow
            )
            # emergency zone and flood inflow
            outflow = torch.where(
                (ff > FFe) & mask_I,
                Qin,
                outflow
            )
            # limit outflow so the final storage is between 0 and 1
            outflow = torch.max(torch.min(outflow, ff - FFc), ff - one + eps)

            # store storage (and outflow)
            states['ff'][:, j] = ff
            if len(self.target_variables) == 1:
                if 'outflow' in self.target_variables[0]:
                    out[:, j, 0] = outflow
                elif 'storage' in self.target_variables[0]:
                    out[:, j, 0] = ff
            elif len(self.target_variables) == 2:
                out[:, j, 0] = ff
                out[:, j, 1] = outflow

        return {'y_hat': out, 'parameters': parameters, 'internal_states': states}

    @property
    def initial_states(self):
        return {
            'ff': 0.67, # fraction filled. GloFAS default value
        }

    @property
    def parameter_ranges(self):
        return {
            'Vf': [0.5, 0.99], # flood storage limit
            'Qf': [0.0001, 0.5], # flood outflow # JCR: it should be a factor of Q100, but I would need to find a way to read the Q100
        }