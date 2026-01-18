# LSTM Model Improvements and Results Visualization

This repository contains enhancements to an LSTM model, focusing on improving its performance and results visualization. It includes a pre-trained Informer model for knowledge distillation and supports training, evaluation, and logging with Weights & Biases (W&B).

## Overview

- **Improvements**: The LSTM model has been refined with better configuration, data handling, and training scripts. An Informer model serves as a teacher for distillation.
- **Visualization**: Results are logged and visualized using W&B.
- **Structure**: The project is organized into directories for configuration, data, models, scripts, and utilities.

## Repository Structure

- `config/`: Configuration files for model and training settings.
- `data/`: Datasets and related files.
- `informer_checkpoints/`: Pre-trained Informer model checkpoints.
- `models/`: Model definitions (e.g., LSTMModel, Informer).
- `scripts/`: Training and evaluation scripts (e.g., `train.py`, `trainer.py`).
- `utils/`: Utility functions for data loading and tools.
- `.gitignore`: Git ignore file.
- `README.md`: Project documentation.
- `requirements.txt`: Python dependencies.
- `results.ipynb`: Notebook for result analysis.
- `requirements.txt`: Project dependencies.

## Setup

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies**:
   Install the required Python packages listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Weights & Biases**:
   - Install W&B: `pip install wandb`
   - Log in to W&B: `wandb login`
   - Configure your W&B project in `config/config.yaml` under the `wandb` section.

4. **Prepare Data**:
   - Place your dataset in the `data/` directory or update `args.root_path` and `args.data_path` in `config/config.yaml` to point to your data source.

## Usage

1. **Configure the Model**:
   - Edit `config/config.yaml` to adjust model parameters, training settings, and paths.

2. **Run Training**:
   Execute the training script:
   ```bash
   python scripts/train.py
   ```
   - This will initialize W&B, load the configuration, and start the training process with optional sweep configurations.

3. **Evaluate Results**:
   - Check the W&B dashboard for logged metrics (e.g., MSE, inference time, number of parameters).
   - Explore `results.ipynb` for visualizations and further analysis.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Knowledge distillation methodology based on Hinton et al. (2015)
- Informer architecture from Zhou et al. (2021)
- Experiment tracking powered by [Weights & Biases](https://wandb.ai)
