import os
import argparse
import torch
from torch.cuda import amp

from model.unet_model import UNetModel
from preprocessing.unet_preprocessing import UNetPreprocessor
from utils.unet_loader import UNetLoader
from training.unet_trainer import UNetTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train or evaluate diffusion models")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"], help="Mode to run")
    parser.add_argument("--gpus", type=str, default="0", help="Comma-separated list of GPU IDs")
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--eval_path", type=str, default=None, help="Path to checkpoint for evaluation")
    parser.add_argument("--eval_programs", type=str, default=None, help="Comma-separated list of evaluation programs")
    return parser.parse_args()

def load_config(config_path):
    from config import TrainConfig
    import importlib.util
    
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    # Assuming the config file defines a 'conf' variable
    return config_module.conf

def main():
    args = parse_args()
    
    # Set visible GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    gpus = list(range(len(args.gpus.split(","))))
    
    # Load configuration
    conf = load_config(args.config)
    
    # Override config with command line arguments
    if args.eval_path:
        conf.eval_path = args.eval_path
    if args.eval_programs:
        conf.eval_programs = args.eval_programs.split(",")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize components
    model = UNetModel(conf)
    preprocessor = UNetPreprocessor(conf)
    loader = UNetLoader(conf)
    trainer = UNetTrainer(model, preprocessor, loader, conf, device=device)
    
    # Run in specified mode
    if args.mode == "train":
        # Setup for training
        preprocessor.setup(seed=conf.seed, global_rank=0)
        
        # Train the model
        trainer.train(
            num_epochs=conf.total_samples // conf.batch_size_effective // len(preprocessor.train_data),
            batch_size=conf.batch_size // len(gpus),
            fp16=conf.fp16
        )
    elif args.mode == "eval":
        # Setup for evaluation
        preprocessor.setup(seed=conf.seed, global_rank=0)
        
        # Load checkpoint
        if conf.eval_path:
            loader.load_checkpoint(model.model, filename=conf.eval_path)
        else:
            loader.load_checkpoint(model.model)
        
        # Run evaluation programs
        for program in conf.eval_programs:
            if program.startswith("fid"):
                # Extract parameters from program string
                if "(" in program and ")" in program:
                    # Format: fid(T,T_latent)
                    params = program[program.find("(")+1:program.find(")")].split(",")
                    T = int(params[0])
                    T_latent = int(params[1]) if len(params) > 1 else None
                else:
                    # Format: fidT
                    T = int(program[3:])
                    T_latent = None
                
                # Evaluate FID
                print(f"Evaluating FID with T={T}, T_latent={T_latent}")
                score = trainer.evaluate_fid(T=T, T_latent=T_latent)
                print(f"FID score: {score}")
            
            elif program.startswith("recon"):
                # Format: reconT
                T = int(program[5:])
                
                # Evaluate reconstruction
                print(f"Evaluating reconstruction with T={T}")
                scores = trainer.evaluate_lpips(T=T)
                for k, v in scores.items():
                    print(f"{k}: {v}")
            
            elif program.startswith("infer"):
                # Infer latents
                if "+" in program:
                    # Format: infer+renderT
                    T = int(program[12:])
                    print(f"Inferring latents and rendering with T={T}")
                    trainer.infer_whole_dataset(
                        with_render=True,
                        T_render=T,
                        render_save_path=f'latent_infer_render{T}/{conf.name}.lmdb'
                    )
                else:
                    # Format: infer
                    print("Inferring latents")
                    trainer.infer_whole_dataset()

if __name__ == "__main__":
    main()
