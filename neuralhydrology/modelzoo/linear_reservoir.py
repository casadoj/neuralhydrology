import torch
from typing import Dict, Union
from neuralhydrology.modelzoo.baseconceptualmodel import BaseConceptualModel
from neuralhydrology.utils.config import Config


class LinearReservoir(BaseConceptualModel):
    """Implementation of the linear reservoir model with dynamic parameterization.
    
    The model receives the dynamic parameterization given by a deep learning model. This class has two properties that define the initial conditions of the internal state (the reservoir filling), and the ranges in which the model parameter (residence time) iss allowed to vary during optimization.

    Parameters
    ----------
    cfg : Config
        The run configuration.

    References
    ----------

    """

    def __init__(self, cfg: Config):
        super(LinearReservoir, self).__init__(cfg=cfg)

    def forward(
        self,
        x_conceptual: torch.Tensor,
        lstm_out: torch.Tensor
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Performs a forward pass on the linear reservoir model. In this forward pass, all elements of the batch are processed in parallel.

        Parameters
        ----------
        x_conceptual: torch.Tensor
            Tensor of size [batch_size, time_steps, n_inputs]. The batch_size is associated with a certain basin and a certain prediction period. The time_steps refer to the number of time steps (e.g. days) that our conceptual model is going to be run for. The n_inputs refer to the single dynamic forcing used to run the reservoir model, i.e., inflow
        lstm_out: torch.Tensor
            Tensor of size [batch_size, time_steps, n_parameters]. The tensor comes from the data-driven model and will be used to obtained the dynamic parameterization of the conceptual model

        Returns
        -------
        Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]
            - y_hat: torch.Tensor
                Simulated outflow
            - parameters: Dict[str, torch.Tensor]
                Dynamic parameterization of the reservoir residence time
            - internal_states: Dict[str, torch.Tensor]]
                Time-evolution of the reservoir fraction filled
        """

        # get model parameters
        parameters = self._get_dynamic_parameters_conceptual(lstm_out=lstm_out)

        # initialize structures to store the information
        states, out = self._initialize_information(conceptual_inputs=x_conceptual)

        # initialize constants
        # zero = torch.tensor(0.0, dtype=torch.float32, device=x_conceptual.device)
        one = torch.tensor(1.0, dtype=torch.float32, device=x_conceptual.device)
        eps = torch.tensor(1e-8, requires_grad=True, dtype=torch.float32, device=x_conceptual.device)

        # inflow
        inflow = x_conceptual[:, :, 0].clone()

        # fractin filled, i.e., normalised storage
        ff = torch.tensor(self.initial_states['ff'],
                          dtype=torch.float32,
                          device=x_conceptual.device
                         ).repeat(x_conceptual.shape[0])
        
        #  conservative storage
        # JCR: I use the GloFAS default value. It should be different for each reservoir, though
        FFc = torch.tensor(0.1, dtype=torch.float32, device=x_conceptual.device) # zero

        # run model for each time step
        for j in range(x_conceptual.shape[1]):

            # reservoir routine
            # -----------------
            # update storage
            ff = ff + inflow[:, j]
            # linear outflow
            outflow = ff / parameters['T'][:, j]
            # limit outflow so the final storage is between 0 and 1
            outflow = torch.max(torch.min(outflow, ff - FFc), ff - one + eps)
            # update storage
            ff = ff - outflow

            # store storage and outflow
            states['ff'][:, j] = ff
            out[:, j, 0] = outflow

        return {'y_hat': out, 'parameters': parameters, 'internal_states': states}

    @property
    def initial_states(self):
        return {
            'ff': 0.67, # fraction filled. I use the GloFAS default value
        }

    @property
    def parameter_ranges(self):
        return {
            'T': [7, 2190]
        }