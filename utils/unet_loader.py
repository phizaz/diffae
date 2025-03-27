import os
import torch
import json
import numpy as np

class UNetLoader:
    """Handles model loading, saving and checkpoint management."""
    def __init__(self, conf, logdir=None):
        """
        Initialize the loader.
        
        Args:
            conf: Configuration object
            logdir: Directory for logs and checkpoints
        """
        self.conf = conf
        self.logdir = logdir or conf.logdir
        
        # Create log directory if it doesn't exist
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
    
    def save_checkpoint(self, model, optimizer, scheduler=None, global_step=0, filename=None):
        """
        Save a checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state to save
            scheduler: Learning rate scheduler to save
            global_step: Current training step
            filename: Filename for the checkpoint
        """
        if filename is None:
            filename = f'{self.logdir}/checkpoint_{global_step}.ckpt'
        
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        torch.save(checkpoint, filename)
        print(f"Saved checkpoint to {filename}")
    
    def save_last_checkpoint(self, model, optimizer, scheduler=None, global_step=0):
        """
        Save the latest checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state to save
            scheduler: Learning rate scheduler to save
            global_step: Current training step
        """
        self.save_checkpoint(model, optimizer, scheduler, global_step, f'{self.logdir}/last.ckpt')
    
    def load_checkpoint(self, model, optimizer=None, scheduler=None, filename=None, map_location='cpu'):
        """
        Load a checkpoint.
        
        Args:
            model: Model to load weights into
            optimizer: Optimizer to load state into
            scheduler: Learning rate scheduler to load state into
            filename: Checkpoint filename
            map_location: Device to load tensors onto
            
        Returns:
            global_step from the checkpoint
        """
        if filename is None:
            filename = f'{self.logdir}/last.ckpt'
        
        if not os.path.exists(filename):
            print(f"No checkpoint found at {filename}")
            return 0
        
        print(f"Loading checkpoint from {filename}")
        checkpoint = torch.load(filename, map_location=map_location)
        
        model.load_state_dict(checkpoint['state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        global_step = checkpoint.get('global_step', 0)
        print(f"Loaded checkpoint from step {global_step}")
        
        return global_step
    
    def load_pretrained(self, model, pretrain_path, map_location='cpu'):
        """
        Load pretrained weights.
        
        Args:
            model: Model to load weights into
            pretrain_path: Path to pretrained weights
            map_location: Device to load tensors onto
        """
        if pretrain_path is None:
            return
        
        print(f'Loading pretrained model from {pretrain_path}')
        state = torch.load(pretrain_path, map_location=map_location, weights_only=False)
        print('step:', state['global_step'])
        model.load_state_dict(state['state_dict'], strict=False)
    
    def save_latent_stats(self, conds, conds_mean, conds_std, path=None):
        """
        Save latent statistics.
        
        Args:
            conds: Latent conditions
            conds_mean: Mean of conditions
            conds_std: Standard deviation of conditions
            path: Save path
        """
        if path is None:
            path = f'checkpoints/{self.conf.name}/latent.pkl'
        
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        
        torch.save({
            'conds': conds,
            'conds_mean': conds_mean,
            'conds_std': conds_std,
        }, path)
        print(f"Saved latent stats to {path}")
    
    def load_latent_stats(self, path=None, map_location='cpu'):
        """
        Load latent statistics.
        
        Args:
            path: Load path
            map_location: Device to load tensors onto
            
        Returns:
            Dictionary containing conds, conds_mean, and conds_std
        """
        if path is None:
            path = f'checkpoints/{self.conf.name}/latent.pkl'
        
        if not os.path.exists(path):
            print(f"No latent stats found at {path}")
            return None
        
        print(f"Loading latent stats from {path}")
        stats = torch.load(path, map_location=map_location)
        return stats
