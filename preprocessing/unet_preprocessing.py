import torch
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from dataset import *
from dist_utils import get_world_size, get_rank
import numpy as np

class UNetPreprocessor:
    """Handles data preprocessing and dataset creation for UNet models."""
    def __init__(self, conf):
        """
        Initialize the preprocessor.
        
        Args:
            conf: Configuration object
        """
        self.conf = conf
        self.train_data = None
        self.val_data = None
    
    def setup(self, seed=None, global_rank=0):
        """
        Set up datasets with proper seeding.
        
        Args:
            seed: Random seed
            global_rank: Current process rank
        """
        # Set seed for each worker separately
        if seed is not None:
            seed_worker = seed * get_world_size() + global_rank
            np.random.seed(seed_worker)
            torch.manual_seed(seed_worker)
            torch.cuda.manual_seed(seed_worker)
            print('local seed:', seed_worker)
        
        # Create datasets
        self.train_data = self.conf.make_dataset()
        print('train data:', len(self.train_data))
        self.val_data = self.train_data
        print('val data:', len(self.val_data))
    
    def create_train_dataloader(self, batch_size, drop_last=True, shuffle=True):
        """
        Create training dataloader.
        
        Args:
            batch_size: Batch size
            drop_last: Whether to drop the last incomplete batch
            shuffle: Whether to shuffle the data
            
        Returns:
            DataLoader for training
        """
        if not hasattr(self, "train_data") or self.train_data is None:
            self.setup()
        
        # Create a DataLoader directly
        dataloader = torch.utils.data.DataLoader(
            self.train_data,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=0,  # Use 0 to avoid pickling issues
            persistent_workers=False
        )
        return SizedIterableWrapper(dataloader, len(self.train_data))
    
    def create_val_dataloader(self, batch_size, drop_last=False):
        """
        Create validation dataloader.
        
        Args:
            batch_size: Batch size
            drop_last: Whether to drop the last incomplete batch
            
        Returns:
            DataLoader for validation
        """
        if not hasattr(self, "val_data") or self.val_data is None:
            self.setup()
        
        dataloader = torch.utils.data.DataLoader(
            self.val_data,
            batch_size=batch_size,
            shuffle=False,
            drop_last=drop_last,
            num_workers=0,
            persistent_workers=False
        )
        return dataloader
    
    def create_latent_dataset(self, conds):
        """
        Create a dataset from latent conditions.
        
        Args:
            conds: Latent conditions tensor
            
        Returns:
            TensorDataset containing the conditions
        """
        return TensorDataset(conds)


class SizedIterableWrapper:
    """Wrapper for iterables that provides a __len__ method."""
    def __init__(self, dataloader, length):
        self.dataloader = dataloader
        self._length = length

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return self._length
