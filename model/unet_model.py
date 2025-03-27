import copy
import torch
from torch import nn
from torch.cuda import amp

class UNetModel:
    """Core model architecture implementation for diffusion models."""
    def __init__(self, conf):
        """
        Initialize the UNet model.
        
        Args:
            conf: Configuration object containing model parameters
        """
        self.conf = conf
        self.model = conf.make_model_conf().make_model()
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.requires_grad_(False)
        self.ema_model.eval()
        
        # Calculate model size
        model_size = 0
        for param in self.model.parameters():
            model_size += param.data.nelement()
        print('Model params: %.2f M' % (model_size / 1024 / 1024))
        
        # Initialize samplers
        self.sampler = conf.make_diffusion_conf().make_sampler()
        self.eval_sampler = conf.make_eval_diffusion_conf().make_sampler()
        self.T_sampler = conf.make_T_sampler()
        
        # Initialize latent samplers if needed
        if conf.train_mode.use_latent_net():
            self.latent_sampler = conf.make_latent_diffusion_conf().make_sampler()
            self.eval_latent_sampler = conf.make_latent_eval_diffusion_conf().make_sampler()
        else:
            self.latent_sampler = None
            self.eval_latent_sampler = None
    
    def update_ema(self, decay):
        """
        Update the exponential moving average model.
        
        Args:
            decay: EMA decay rate
        """
        self._ema(self.model, self.ema_model, decay)
    
    def _ema(self, source, target, decay):
        """
        Apply exponential moving average update.
        
        Args:
            source: Source model
            target: Target model (EMA)
            decay: EMA decay rate
        """
        source_dict = source.state_dict()
        target_dict = target.state_dict()
        for key in source_dict.keys():
            target_dict[key].data.copy_(target_dict[key].data * decay +
                                        source_dict[key].data * (1 - decay))
    
    def encode(self, x):
        """
        Encode input using the model's encoder.
        
        Args:
            x: Input tensor
            
        Returns:
            Encoded representation
        """
        assert self.conf.model_type.has_autoenc()
        cond = self.ema_model.encoder.forward(x)
        return cond
    
    def encode_stochastic(self, x, cond, T=None):
        """
        Stochastically encode input.
        
        Args:
            x: Input tensor
            cond: Conditioning tensor
            T: Number of diffusion steps
            
        Returns:
            Stochastically encoded sample
        """
        if T is None:
            sampler = self.eval_sampler
        else:
            sampler = self.conf._make_diffusion_conf(T).make_sampler()
        out = sampler.ddim_reverse_sample_loop(self.ema_model,
                                               x,
                                               model_kwargs={'cond': cond})
        return out['sample']
    
    def forward(self, noise=None, x_start=None, use_ema=False):
        """
        Forward pass through the model.
        
        Args:
            noise: Input noise
            x_start: Starting point for diffusion
            use_ema: Whether to use EMA model
            
        Returns:
            Generated sample
        """
        with amp.autocast(False):
            model = self.ema_model if use_ema else self.model
            gen = self.eval_sampler.sample(model=model,
                                           noise=noise,
                                           x_start=x_start)
            return gen
