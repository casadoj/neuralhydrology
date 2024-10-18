from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
import argparse
from typing import Literal, Optional, Union, List

from neuralhydrology.utils.config import Config
from neuralhydrology.nh_run import eval_run

# set device type
if torch.cuda.is_available():
    gpu = 0
    print('running on GPU')
    print('no. GPU available:\t{0}'.format(torch.cuda.device_count()))
else:
    gpu = -1
    print('running on CPU')

def _get_args() -> dict:
    """Parse arguments specifying the configuration file"""
    parser = argparse.ArgumentParser(
        description="""
        Evaluate the model for the specified epoch and period.
        """
    )
    parser.add_argument('-d', '--run-dir', type=lambda p: Path(p), required=True,
                        help='The directory where the model and results are stored')
    parser.add_argument('-e', '--epoch', type=int, default=None,
                        help='The epoch to evaluate the model on. Default is None and all epochs will be evaluated.')
    parser.add_argument('-p', '--periods', type=str, nargs='+', default=['test'],
                        help='The periods of evaluation. Default is ["test"].')
    args = vars(parser.parse_args())
    
    return args
    
def evaluate(
    run_dir: Path,
    periods: List[str] = ['test'],
    epoch: Optional[int] = None
):  

    # load configuration file
    cfg = Config(run_dir / 'config.yml')

    # evaluate
    for period in periods:
        if epoch is None:
            for epoch in tqdm(np.arange(1, cfg.epochs + 1), desc='epoch'):
                try:
                    eval_run(run_dir=run_dir, period=period, epoch=epoch, gpu=gpu)
                except Exception as e:
                    print(e)
                    continue
        else:
            eval_run(run_dir=run_dir, period=period, epoch=epoch, gpu=gpu)

def _main():
    args = _get_args()
    print('run directory:', args['run_dir'], sep='\t')
    evaluate(
        run_dir=args['run_dir'],
        periods=args['periods'],
        epoch=args['epoch']
    )

if __name__ == "__main__":
    _main()