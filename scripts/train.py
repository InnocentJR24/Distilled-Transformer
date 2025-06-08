import yaml
import torch
import wandb
from utils.data import load_data
from scripts.trainer import train

# -------- Dotdict for Arguments --------
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

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
    except Exception as e:
        print(f"Error loading config file: {e}")
        raise

if __name__ == "__main__":
    # Load base configuration
    base_config = load_config("config/config.yaml")
    
    # Device handling
    device = torch.device(f"cuda:{base_config.training.gpu}" if base_config.training.use_gpu and torch.cuda.is_available() else "cpu")
    
    # Combine model and training args for convenience
    args = dotdict({
        **base_config.model,
        **base_config.paths,
        **base_config.training
    })
    
    # Initialize W&B sweep agent
    # W&B will override wandb config parameters during the sweep
    wandb.init(project="distilled-lstm-pareto")
    sweep_config = wandb.config  # Hyperparameters from sweep.yaml
    
    # Run training with sweep config and base args
    train(sweep_config, args, device)