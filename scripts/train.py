import os
import sys
# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import yaml
import torch
import wandb
from utils.data import load_data
from utils.tools import dotdict
from scripts.trainer import train
import random
import numpy as np

# Set seeds for reproducibility
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
else:
    torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.benchmark = True

def load_config(config_path):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        # Convert nested dicts to dotdict for easy access
        config = dotdict(config)
        config.model = dotdict(config.model)
        config.paths = dotdict(config.paths)
        config.training = dotdict(config.training)
        config.wandb = dotdict(config.wandb)
        return config
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        raise
    except yaml.YAMLError:
        print(f"Error: Unable to parse YAML file at {config_path}")
        raise
    except Exception as e:
        print(f"Unexpected error loading config: {e}")
        raise

if __name__ == "__main__":
    # Load base configuration
    base_config = load_config("config/config.yaml")
    
    # Device handling
    device = torch.device(f"cuda:{base_config.training.gpu}" if base_config.training.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Combine model, paths, and training args for convenience
    args = dotdict({
        **base_config.model,
        **base_config.paths,
        **base_config.training
    })
    
    # Initialize W&B
    print("Initializing Weights & Biases...")
    wandb.init(project="distilled-lstm-pareto", config=base_config.wandb)
    
    # Use sweep config if running a sweep, otherwise fall back to base config
    sweep_config = dotdict(wandb.config) if wandb.run.sweep_id else dotdict(base_config.wandb)
    
    # Run training with sweep config and base args
    print("Starting training process...")
    train(sweep_config, args, device)