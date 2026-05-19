import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_fast(nnUNetTrainer):
    """
    Speed-optimized trainer for A800 80GB (or similar high-VRAM GPUs).
    ePAI variant: no unpack_dataset parameter (ePAI trainer signature).

    Changes vs default:
    - torch.compile enabled unconditionally (10-20% speedup after warmup)
    - cudnn.benchmark=True (5-15% speedup for fixed patch sizes)
    - DA worker processes raised to 24 (112 CPU cores available)
    - num_epochs reduced to 500 (half the default wall-clock time)
    - save_every raised to 100 (less checkpoint I/O)
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 500
        self.save_every = 100

    def initialize(self):
        super().initialize()
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True

    def _do_i_compile(self):
        return self.device.type == 'cuda'

    def get_dataloaders(self):
        import os
        os.environ.setdefault('nnUNet_n_proc_DA', '24')
        return super().get_dataloaders()


class nnUNetTrainer_fast_300epochs(nnUNetTrainer_fast):
    """Same as nnUNetTrainer_fast but only 300 epochs."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 300


class nnUNetTrainer_fast_1000epochs(nnUNetTrainer_fast):
    """Full 1000 epochs with all speed optimizations."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1000


class nnUNetTrainer_fast_50epochs(nnUNetTrainer_fast):
    """50 epochs — quick validation run."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 50
