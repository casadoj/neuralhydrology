import torch
from typing import Dict, Union
from neuralhydrology.modelzoo.baseconceptualmodel import BaseConceptualModel
from neuralhydrology.utils.config import Config

class LisfloodReservoir(BaseConceptualModel):
    """Modified version of the reservoir routine int he LISFLOOD [#] OS hydrological model with dynamic parameterization.
    
    The LisfloodReservoir receives the dynamic parameterization given by a deep learning model. This class has two properties which define the initial condition of the reservoir storage and the ranges in which the model parameters are allowed to vary during optimization.

    Parameters
    ----------
    cfg : Config
        The run configuration.

    References
    ----------
    .. [#] Joint Research Centre - European Commission. https://github.com/ec-jrc/lisflood-code
    """
    
    def __init__(self, cfg: Config):
        super(LisfloodReservoir, self).__init__(cfg=cfg)
        
    def forward(self, x_conceptual: torch.Tensor, lstm_out: torch.Tensor) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Perform a forward pass on the reservoir model. In this forward pass, all elements of the batch are processed in  parallel

        Parameters
        ----------
        x_conceptual: torch.Tensor
            Tensor of size [batch_size, time_steps, n_inputs]. The batch_size is associated with a certain basin and a certain prediction period. The time_steps refer to the number of time steps (e.g. days) that our conceptual model is going to be run for. The n_inputs refer to the dynamic forcings used to run the conceptual model (e.g. Precipitation, Temperature...)

        lstm_out: torch.Tensor
            Tensor of size [batch_size, time_steps, n_parameters]. The tensor comes from the data-driven model  and will be used to obtained the dynamic parameterization of the conceptual model

        Returns
        -------
        Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]
            - y_hat: torch.Tensor
                Simulated outflow
            - parameters: Dict[str, torch.Tensor]
                Dynamic parameterization of the conceptual model
            - internal_states: Dict[str, torch.Tensor]]
                Time-evolution of the internal states of the conceptual model

        """
        # get model parameters
        parameters = self._get_dynamic_parameters_conceptual(lstm_out=lstm_out)

        # initialize structures to store the information
        states, out = self._initialize_information(conceptual_inputs=x_conceptual)

        # initialize constants
        zero = torch.tensor(0.0, dtype=torch.float32, device=x_conceptual.device)
        one = torch.tensor(1.0, dtype=torch.float32, device=x_conceptual.device)
        #klu = torch.tensor(0.90, requires_grad=False, dtype=torch.float32, device=x_conceptual.device)  # land use correction factor [-]
        At = torch.tensor(3600 * 24, requires_grad=False, dtype=torch.float32, device=x_conceptual.device) # time step [s]

        # initial storage
        ff = torch.tensor(self.initial_states['ff'], dtype=torch.float32, device=x_conceptual.device).repeat(x_conceptual.shape[0])
        # V = ff * self.Vtot
        
        # reservoir zones (constant in time)
        # Vf = parameters['FFf'][:, 0] * self.Vtot
        # Vn = self.Vc + parameters['alpha'][:, 0] * (Vf - self.Vc)
        # Vn_adj = Vn + parameters['beta'][:, 0] * (Vf - Vn)
        # Qf = torch.quantile(x_conceptual[:, :, :], parameters['FFf'][:, 0])
        # Qn = self.Qc + parameters['gamma'][:, 0] * (Qf - self.Qc)

        # input time series
        inflow = x_conceptual[:, :, 0]
        evaporation = x_conceptual[:, :, 1]
        
        # run hydrological model for each time step
        for j in range(x_conceptual.shape[1]):            

            # reservoir zones (in case they would be time dependent)
            FFf = parameters['FFf'][:, j]
            FFc = parameters['FFc'][:, j]
            FFn = FFc + parameters['alpha'][:, j] * (FFf - FFc)
            FFn_adj = FFn + parameters['beta'][:, j] * (FFf - FFn)
            Qf = torch.quantile(inflow, parameters['QQf'][:, j])
            Qc = torch.quantile(inflow, parameters['QQc'][:, j])
            Qn = Qc + parameters['gamma'][:, j] * (Qf - Qc)
            
            # update storage
            FF += I * At # 1st (and unique) input is inflow
            
            # ouflow depending on the storage level
            if FF < 2 * FFc:
                Q = torch.min([Qc, (FF - FFc) / At])
            elif FF < FFn:
                Q = Qc + (Qn - Qc) * (FF - 2 * FFc) / (FFn - 2 * FFc)
            elif FF < FFn_adj:
                Q = Qn
            elif FF < FFf:
                Q = Qn + (Qf - Qn) * (FF - FFn_adj) / (FFf - FFn_adj)
                if Q > parameters['k'][:, j] * inflow[:, j]:
                        Q = torch.max([parameters['k'][:, j] * inflow[:, j], Qn])
            elif FF > FFf:
                Q = torch.max([Qf, inflow[:, j]])

            # limit outflow so the final storage is between FFc and 1
            Q = torch.max([torch.min([Q, FF - FFc]), FF - 1])

            # update reservoir storage with the outflow volume
            FF -= Q

            assert 0 <= FF, 'The fraction filled at the end of the timestep is negative.'
            assert FF <= 1, 'The fraction filled at the end of the timestep is larger than the total reservoir capacity (1).'

            # Store time evolution of the internal states
            states['ff'][:, j] = FF

            # total outflow
            out[:, j, 0] = Q

        return {'y_hat': out, 'parameters': parameters, 'internal_states': states}

    @property
    def initial_states(self):
        return {'ff': 0.9}

    @property
    def parameter_ranges(self):
        return {'FFf': [0.2, 0.99], # [-] fraction filled for flood protection
                'FFc': [0.01, 0.199], # [-] minimum fraction filled
                'alpha': [0.001, 0.999], # [-] limit between the normal and flood zones
                'beta': [0.001, 0.999], # [-] proportion of the normal zone with constant release
                'QQf': [0.1, 0.99], # [-] quantile associated with the non-damaging flow (Qf)
                'QQc': [0.001, 0.099], # [-] quantile associated with the environmental flow (Qc)
                'gamma': [0.001, 0.999], # [-] proportion between Qf and Qc that defines the constant release at the normal zone 
                'k': [1.0, 5.0] # [-] release coefficient
                }