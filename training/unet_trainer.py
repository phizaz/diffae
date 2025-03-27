import os
import json
import numpy as np
import torch
from torch import nn
from torch.cuda import amp
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from dist_utils import get_world_size, get_rank, all_gather
from metrics import evaluate_fid, evaluate_lpips
from renderer import render_uncondition, render_condition

class UNetTrainer:
    """Handles the training process for UNet diffusion models."""
    def __init__(self, model, preprocessor, loader, conf, device='cuda'):
        """
        Initialize the trainer.
        
        Args:
            model: UNetModel instance
            preprocessor: UNetPreprocessor instance
            loader: UNetLoader instance
            conf: Configuration object
            device: Device to use for training
        """
        self.model = model
        self.preprocessor = preprocessor
        self.loader = loader
        self.conf = conf
        self.device = device
        
        # Initialize training state
        self.global_step = 0
        self.num_samples = 0
        self.global_rank = get_rank()
        
        # Register buffer for consistent sampling
        self.x_T = torch.randn(conf.sample_size, 3, conf.img_size, conf.img_size, device=device)
        
        # Initialize latent normalization stats
        self.conds = None
        self.conds_mean = None
        self.conds_std = None
        
        # Load latent stats if path is provided
        if conf.latent_infer_path is not None:
            print('Loading latent stats...')
            stats = self.loader.load_latent_stats(conf.latent_infer_path)
            if stats:
                self.conds = stats['conds']
                self.conds_mean = stats['conds_mean'][None, :].to(device)
                self.conds_std = stats['conds_std'][None, :].to(device)
    
    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.
        
        Returns:
            Tuple of (optimizer, scheduler)
        """
        if self.conf.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                self.model.model.parameters(),
                lr=self.conf.lr,
                weight_decay=self.conf.weight_decay
            )
        elif self.conf.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(
                self.model.model.parameters(),
                lr=self.conf.lr,
                weight_decay=self.conf.weight_decay
            )
        else:
            raise NotImplementedError(f"Optimizer {self.conf.optimizer} not implemented")
        
        scheduler = None
        if self.conf.warmup > 0:
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=self._warmup_lr(self.conf.warmup)
            )
        
        return optimizer, scheduler
    
    def _warmup_lr(self, warmup):
        """
        Create a warmup learning rate function.
        
        Args:
            warmup: Number of warmup steps
            
        Returns:
            Learning rate lambda function
        """
        def lr_lambda(step):
            return min(step, warmup) / warmup
        return lr_lambda
    
    def normalize(self, cond):
        """
        Normalize latent conditions.
        
        Args:
            cond: Conditions to normalize
            
        Returns:
            Normalized conditions
        """
        if self.conds_mean is None or self.conds_std is None:
            return cond
        return (cond - self.conds_mean) / self.conds_std
    
    def denormalize(self, cond):
        """
        Denormalize latent conditions.
        
        Args:
            cond: Normalized conditions
            
        Returns:
            Denormalized conditions
        """
        if self.conds_mean is None or self.conds_std is None:
            return cond
        return (cond * self.conds_std) + self.conds_mean
    
    def is_last_accum(self, batch_idx):
        """
        Check if this is the last gradient accumulation step.
        
        Args:
            batch_idx: Current batch index
            
        Returns:
            Boolean indicating if this is the last accumulation step
        """
        return (batch_idx + 1) % self.conf.accum_batches == 0
    
    def train_step(self, batch, batch_idx):
        """
        Perform a single training step.
        
        Args:
            batch: Batch of data
            batch_idx: Index of the current batch
            
        Returns:
            Loss value
        """
        with amp.autocast(False):
            # Handle different training modes
            if self.conf.train_mode.require_dataset_infer():
                # This mode has pre-calculated cond
                cond = batch[0]
                if self.conf.latent_znormalize:
                    cond = self.normalize(cond)
                x_start = None
            else:
                imgs, idxs = batch['img'], batch['index']
                x_start = imgs
                cond = None

            # Different training modes
            if self.conf.train_mode == 'diffusion':
                # Main training mode
                t, weight = self.model.T_sampler.sample(len(x_start), x_start.device)
                losses = self.model.sampler.training_losses(model=self.model.model,
                                                           x_start=x_start,
                                                           t=t)
            elif self.conf.train_mode.is_latent_diffusion():
                # Training the latent variables
                t, weight = self.model.T_sampler.sample(len(cond), cond.device)
                latent_losses = self.model.latent_sampler.training_losses(
                    model=self.model.model.latent_net, x_start=cond, t=t)
                # Train only do the latent diffusion
                losses = {
                    'latent': latent_losses['loss'],
                    'loss': latent_losses['loss']
                }
            else:
                raise NotImplementedError(f"Training mode {self.conf.train_mode} not implemented")

            loss = losses['loss'].mean()
            
            # Gather losses from all processes
            gathered_losses = {}
            for key in ['loss', 'vae', 'latent', 'mmd', 'chamfer', 'arg_cnt']:
                if key in losses:
                    gathered_losses[key] = all_gather(losses[key]).mean()

        return loss, gathered_losses    
    
    def train_epoch(self, dataloader, optimizer, scheduler=None, scaler=None, log_interval=10):
        """
        Train for one epoch.
        
        Args:
            dataloader: DataLoader for training data
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            scaler: Gradient scaler for mixed precision
            log_interval: How often to log
            
        Returns:
            Average loss for the epoch
        """
        self.model.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
            # Move batch to device
            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            elif isinstance(batch, list) or isinstance(batch, tuple):
                batch = [b.to(self.device) if torch.is_tensor(b) else b for b in batch]
            
            # Forward and backward pass
            if scaler is not None:
                with amp.autocast(True):
                    loss, gathered_losses = self.train_step(batch, batch_idx)
                    loss = loss / self.conf.accum_batches  # Normalize for gradient accumulation
                
                scaler.scale(loss).backward()
                
                if self.is_last_accum(batch_idx):
                    # Apply gradient clipping
                    if self.conf.grad_clip > 0:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(self.model.model.parameters(), self.conf.grad_clip)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                    if scheduler is not None:
                        scheduler.step()
            else:
                loss, gathered_losses = self.train_step(batch, batch_idx)
                loss = loss / self.conf.accum_batches  # Normalize for gradient accumulation
                
                loss.backward()
                
                if self.is_last_accum(batch_idx):
                    # Apply gradient clipping
                    if self.conf.grad_clip > 0:
                        nn.utils.clip_grad_norm_(self.model.model.parameters(), self.conf.grad_clip)
                    
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    if scheduler is not None:
                        scheduler.step()
            
            # Update EMA model
            if self.is_last_accum(batch_idx):
                if self.conf.train_mode == 'latent_diffusion':
                    # Only update latent part for latent diffusion
                    self.model._ema(self.model.model.latent_net, self.model.ema_model.latent_net, self.conf.ema_decay)
                else:
                    self.model.update_ema(self.conf.ema_decay)
                
                # Log samples
                if batch_idx % log_interval == 0:
                    if self.conf.train_mode.require_dataset_infer():
                        imgs = None
                    else:
                        imgs = batch['img'] if isinstance(batch, dict) else None
                    self.log_sample(x_start=imgs)
                
                # Update global step and samples
                self.global_step += 1
                self.num_samples = self.global_step * self.conf.batch_size_effective
                
                # Evaluate metrics periodically
                self.evaluate_scores()
            
            total_loss += loss.item() * self.conf.accum_batches
            num_batches += 1
            
            # Log losses
            if batch_idx % log_interval == 0 and self.global_rank == 0:
                print(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item() * self.conf.accum_batches:.4f}")
                for key, value in gathered_losses.items():
                    print(f"  {key}: {value.item():.4f}")
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self, num_epochs, batch_size=None, fp16=False):
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            batch_size: Batch size (uses conf.batch_size if None)
            fp16: Whether to use mixed precision training
            
        Returns:
            Final model
        """
        # Setup
        if batch_size is None:
            batch_size = self.conf.batch_size // get_world_size()  # Local batch size
        
        # Create dataloaders
        train_loader = self.preprocessor.create_train_dataloader(batch_size)
        
        # Configure optimizers
        optimizer, scheduler = self.configure_optimizers()
        
        # Load checkpoint if exists
        start_epoch = 0
        if os.path.exists(f'{self.conf.logdir}/last.ckpt'):
            self.global_step = self.loader.load_checkpoint(
                self.model.model, optimizer, scheduler
            )
            start_epoch = self.global_step // len(train_loader)
            self.num_samples = self.global_step * self.conf.batch_size_effective
        
        # Setup mixed precision if needed
        scaler = amp.GradScaler() if fp16 else None
        
        # Training loop
        for epoch in range(start_epoch, num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            
            # Train for one epoch
            avg_loss = self.train_epoch(train_loader, optimizer, scheduler, scaler)
            
            # Save checkpoint
            if self.global_rank == 0:
                self.loader.save_last_checkpoint(
                    self.model.model, optimizer, scheduler, self.global_step
                )
                
                if epoch % self.conf.save_epoch_interval == 0:
                    self.loader.save_checkpoint(
                        self.model.model, optimizer, scheduler, self.global_step,
                        f'{self.conf.logdir}/checkpoint_epoch{epoch+1}.ckpt'
                    )
            
            print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
        
        return self.model
    
    def sample(self, N, T=None, T_latent=None):
        """
        Generate samples from the model.
        
        Args:
            N: Number of samples to generate
            T: Number of diffusion steps
            T_latent: Number of latent diffusion steps
            
        Returns:
            Generated images
        """
        if T is None:
            sampler = self.model.eval_sampler
            latent_sampler = self.model.latent_sampler
        else:
            sampler = self.conf._make_diffusion_conf(T).make_sampler()
            latent_sampler = self.conf._make_latent_diffusion_conf(T_latent).make_sampler()

        noise = torch.randn(N,
                            3,
                            self.conf.img_size,
                            self.conf.img_size,
                            device=self.device)
        
        pred_img = render_uncondition(
            self.conf,
            self.model.ema_model,
            noise,
            sampler=sampler,
            latent_sampler=latent_sampler,
            conds_mean=self.conds_mean,
            conds_std=self.conds_std,
        )
        pred_img = (pred_img + 1) / 2
        return pred_img
    
    def render(self, noise, cond=None, T=None):
        """
        Render images from noise with optional conditioning.
        
        Args:
            noise: Input noise
            cond: Conditioning information
            T: Number of diffusion steps
            
        Returns:
            Rendered images
        """
        if T is None:
            sampler = self.model.eval_sampler
        else:
            sampler = self.conf._make_diffusion_conf(T).make_sampler()

        if cond is not None:
            pred_img = render_condition(self.conf,
                                        self.model.ema_model,
                                        noise,
                                        sampler=sampler,
                                        cond=cond)
        else:
            pred_img = render_uncondition(self.conf,
                                          self.model.ema_model,
                                          noise,
                                          sampler=sampler,
                                          latent_sampler=None)
        pred_img = (pred_img + 1) / 2
        return pred_img
    
    def infer_whole_dataset(self, with_render=False, T_render=None, render_save_path=None):
        """
        Infer latents for the entire dataset.
        
        Args:
            with_render: Whether to also render images
            T_render: Number of diffusion steps for rendering
            render_save_path: Path to save rendered images
            
        Returns:
            Inferred conditions
        """
        from tqdm import tqdm
        from contextlib import nullcontext
        from source.lmdb_writer import LMDBImageWriter
        
        data = self.conf.make_dataset()
        if isinstance(data, CelebAlmdb) and data.crop_d2c:
            # Special case where we need the d2c crop
            data.transform = make_transform(self.conf.img_size,
                                            flip_prob=0,
                                            crop_d2c=True)
        else:
            data.transform = make_transform(self.conf.img_size, flip_prob=0)

        loader = self.conf.make_loader(
            data,
            shuffle=False,
            drop_last=False,
            batch_size=self.conf.batch_size_eval,
            parallel=True,
        )
        model = self.model.ema_model
        model.eval()
        conds = []

        if with_render:
            sampler = self.conf._make_diffusion_conf(
                T=T_render or self.conf.T_eval).make_sampler()

            if self.global_rank == 0:
                writer = LMDBImageWriter(render_save_path,
                                         format='webp',
                                         quality=100)
            else:
                writer = nullcontext()
        else:
            writer = nullcontext()

        with writer:
            for batch in tqdm(loader, total=len(loader), desc='infer'):
                with torch.no_grad():
                    # (n, c)
                    cond = model.encoder(batch['img'].to(self.device))

                    # Used for reordering to match the original dataset
                    idx = batch['index']
                    idx = all_gather(idx)
                    if idx.dim() == 2:
                        idx = idx.flatten(0, 1)
                    argsort = idx.argsort()

                    if with_render:
                        noise = torch.randn(len(cond),
                                            3,
                                            self.conf.img_size,
                                            self.conf.img_size,
                                            device=self.device)
                        render = sampler.sample(model, noise=noise, cond=cond)
                        render = (render + 1) / 2
                        # (k, n, c, h, w)
                        render = all_gather(render)
                        if render.dim() == 5:
                            # (k*n, c)
                            render = render.flatten(0, 1)

                        if self.global_rank == 0:
                            writer.put_images(render[argsort])

                    # (k, n, c)
                    cond = all_gather(cond)

                    if cond.dim() == 3:
                        # (k*n, c)
                        cond = cond.flatten(0, 1)

                    conds.append(cond[argsort].cpu())
        
        model.train()
        # (N, c) cpu
        conds = torch.cat(conds).float()
        
        # Calculate and save statistics
        if self.global_rank == 0:
            self.conds = conds
            self.conds_mean = conds.mean(dim=0, keepdim=True).to(self.device)
            self.conds_std = conds.std(dim=0, keepdim=True).to(self.device)
            
            self.loader.save_latent_stats(conds, self.conds_mean.cpu(), self.conds_std.cpu())
        
        return conds
    
    def log_sample(self, x_start):
        """
        Log generated samples to tensorboard.
        
        Args:
            x_start: Real images for comparison (optional)
        """
        def do(model, postfix, use_xstart, save_real=False, no_latent_diff=False, interpolate=False):
            model.eval()
            with torch.no_grad():
                all_x_T = self._split_tensor(self.x_T)
                batch_size = min(len(all_x_T), self.conf.batch_size_eval)
                # Allow for superlarge models
                loader = DataLoader(all_x_T, batch_size=batch_size)

                Gen = []
                for x_T in loader:
                    if use_xstart:
                        _xstart = x_start[:len(x_T)]
                    else:
                        _xstart = None
                    if self.conf.train_mode.is_latent_diffusion() and not use_xstart:
                        # Diffusion of the latent first
                        gen = render_uncondition(
                            conf=self.conf,
                            model=model,
                            x_T=x_T,
                            sampler=self.model.eval_sampler,
                            latent_sampler=self.model.eval_latent_sampler,
                            conds_mean=self.conds_mean,
                            conds_std=self.conds_std)
                    else:
                        if not use_xstart and self.conf.model_type.has_noise_to_cond():
                            # Special case, it may not be stochastic, yet can sample
                            cond = torch.randn(len(x_T),
                                               self.conf.style_ch,
                                               device=self.device)
                            cond = model.noise_to_cond(cond)
                        else:
                            if interpolate:
                                with amp.autocast(self.conf.fp16):
                                    cond = model.encoder(_xstart)
                                    i = torch.randperm(len(cond))
                                    cond = (cond + cond[i]) / 2
                            else:
                                cond = None
                        gen = self.model.eval_sampler.sample(model=model,
                                                            noise=x_T,
                                                            cond=cond,
                                                            x_start=_xstart)
                    Gen.append(gen)

                gen = torch.cat(Gen)
                gen = all_gather(gen)
                if gen.dim() == 5:
                    # (n, c, h, w)
                    gen = gen.flatten(0, 1)

                if save_real and use_xstart:
                    # Save the original images to the tensorboard
                    real = all_gather(_xstart)
                    if real.dim() == 5:
                        real = real.flatten(0, 1)

                    if self.global_rank == 0:
                        grid_real = (make_grid(real) + 1) / 2
                        # Save real images
                        sample_dir = os.path.join(self.conf.logdir, f'sample{postfix}')
                        if not os.path.exists(sample_dir):
                            os.makedirs(sample_dir)
                        save_image(grid_real, os.path.join(sample_dir, f'real_{self.num_samples}.png'))

                if self.global_rank == 0:
                    # Save samples to disk
                    grid = (make_grid(gen) + 1) / 2
                    sample_dir = os.path.join(self.conf.logdir, f'sample{postfix}')
                    if not os.path.exists(sample_dir):
                        os.makedirs(sample_dir)
                    path = os.path.join(sample_dir, f'{self.num_samples}.png')
                    save_image(grid, path)
            model.train()

        if self.conf.sample_every_samples > 0 and self._is_time(
                self.num_samples, self.conf.sample_every_samples,
                self.conf.batch_size_effective):

            if self.conf.train_mode.require_dataset_infer():
                do(self.model.model, '', use_xstart=False)
                do(self.model.ema_model, '_ema', use_xstart=False)
            else:
                if self.conf.model_type.has_autoenc() and self.conf.model_type.can_sample():
                    do(self.model.model, '', use_xstart=False)
                    do(self.model.ema_model, '_ema', use_xstart=False)
                    # Autoencoding mode
                    do(self.model.model, '_enc', use_xstart=True, save_real=True)
                    do(self.model.ema_model, '_enc_ema', use_xstart=True, save_real=True)
                elif self.conf.train_mode.use_latent_net():
                    do(self.model.model, '', use_xstart=False)
                    do(self.model.ema_model, '_ema', use_xstart=False)
                    # Autoencoding mode
                    do(self.model.model, '_enc', use_xstart=True, save_real=True)
                    do(self.model.model, '_enc_nodiff', use_xstart=True, save_real=True, no_latent_diff=True)
                    do(self.model.ema_model, '_enc_ema', use_xstart=True, save_real=True)
                else:
                    do(self.model.model, '', use_xstart=True, save_real=True)
                    do(self.model.ema_model, '_ema', use_xstart=True, save_real=True)
    
    def evaluate_scores(self):
        """
        Evaluate FID and other scores during training.
        """
        def fid(model, postfix):
            score = evaluate_fid(self.model.eval_sampler,
                                 model,
                                 self.conf,
                                 device=self.device,
                                 train_data=self.preprocessor.train_data,
                                 val_data=self.preprocessor.val_data,
                                 latent_sampler=self.model.eval_latent_sampler,
                                 conds_mean=self.conds_mean,
                                 conds_std=self.conds_std)
            if self.global_rank == 0:
                print(f"FID{postfix}: {score}")
                if not os.path.exists(self.conf.logdir):
                    os.makedirs(self.conf.logdir)
                with open(os.path.join(self.conf.logdir, 'eval.txt'), 'a') as f:
                    metrics = {
                        f'FID{postfix}': score,
                        'num_samples': self.num_samples,
                    }
                    f.write(json.dumps(metrics) + "\n")

        def lpips(model, postfix):
            if self.conf.model_type.has_autoenc() and self.conf.train_mode.is_autoenc():
                # {'lpips', 'ssim', 'mse'}
                score = evaluate_lpips(self.model.eval_sampler,
                                       model,
                                       self.conf,
                                       device=self.device,
                                       val_data=self.preprocessor.val_data,
                                       latent_sampler=self.model.eval_latent_sampler)

                if self.global_rank == 0:
                    for key, val in score.items():
                        print(f"{key}{postfix}: {val}")

        if self.conf.eval_every_samples > 0 and self.num_samples > 0 and self._is_time(
                self.num_samples, self.conf.eval_every_samples,
                self.conf.batch_size_effective):
            print(f'Evaluating FID @ {self.num_samples}')
            lpips(self.model.model, '')
            fid(self.model.model, '')

        if self.conf.eval_ema_every_samples > 0 and self.num_samples > 0 and self._is_time(
                self.num_samples, self.conf.eval_ema_every_samples,
                self.conf.batch_size_effective):
            print(f'Evaluating FID EMA @ {self.num_samples}')
            fid(self.model.ema_model, '_ema')
    
    def _split_tensor(self, x):
        """
        Split tensor across workers.
        
        Args:
            x: Tensor to split
            
        Returns:
            Local portion of the tensor
        """
        n = len(x)
        rank = self.global_rank
        world_size = get_world_size()
        per_rank = n // world_size
        return x[rank * per_rank:(rank + 1) * per_rank]
    
    def _is_time(self, num_samples, every, step_size):
        """
        Check if it's time to perform an action based on number of samples.
        
        Args:
            num_samples: Current number of samples
            every: Frequency in samples
            step_size: Step size in samples
            
        Returns:
            Boolean indicating if it's time
        """
        closest = (num_samples // every) * every
        return num_samples - closest < step_size


