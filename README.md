# Distilling Informer Transformers into LSTMs for Efficient Time-Series Forecasting

This repository contains the implementation of a **Knowledge Distillation (KD) framework** that compresses a state-of-the-art **Informer Transformer** into a lightweight **LSTM** student. By transferring "dark knowledge" from the teacher's soft targets, the distilled LSTM achieves Transformer-level accuracy with up to **72x faster inference**.

## Key Achievements & Contributions

* **Massive Efficiency Gains**: Achieved a **72x speedup** for long-horizon forecasting (reducing latency from 81ms to 1.1ms) and a **22x speedup** for short-horizon tasks.
* **Accuracy Parity & Improvement**: Demonstrated that the distilled LSTM student not only matches but often **outperforms** the Informer teacher (e.g., reducing Test MSE from 1.096 to **0.894**).
* **Automated MLOps Pipeline**: Integrated **Weights & Biases (W&B)** to perform 100+ hyperparameter sweeps, identifying the **Pareto-optimal** configurations for deployment on resource-constrained devices.
* **Robust Statistical Analysis**: Conducted a dual-metric importance analysis (Pearson Correlation & Random Forest) to evaluate the non-linear interactions between distillation weights and model performance.
* **High-Performance Computing**: Optimized and benchmarked the framework on the **DAS-5 Supercomputer cluster** using NVIDIA RTX 2080 Ti GPUs.


## Methodology

### Knowledge Distillation Framework

The student model is trained on a composite loss function that balances ground-truth fidelity with the mimicry of the teacher's inductive biases:

Where:

*  is the Mean Squared Error against the ground truth labels.
*  is the distillation loss against the Informer's soft predictions.
*  is the tunable weight (0.3 to 0.7) that regulates the distillation balance.

### Architectures

* **Teacher**: Informer Transformer with ProbSparse self-attention and a generative decoder.
* **Student**: Multi-layer stacked LSTM (optimized down to ~19k parameters).


## Performance Comparison (ETTh1 Dataset)

| Model | Horizon | Test MSE | Inference Time (s) | Speedup |
| --- | --- | --- | --- | --- |
| **Informer (Teacher)** | Long | 1.096 | 0.0813 | Baseline |
| **Distilled LSTM** | Long | **0.894** | **0.0011** | **72x** |
| **Informer (Teacher)** | Short | 0.5485 | 0.0136 | Baseline |
| **Distilled LSTM** | Short | **0.5094** | **0.0006** | **22x** |


## Repository Structure

```text
├── config/                # YAML files for model, W&B sweeps, and training
├── data/                  # ETTh1 dataset and preprocessing scripts
├── informer_checkpoints/  # Pre-trained weights for the Informer teacher
├── models/                # PyTorch definitions for LSTM and Informer
├── scripts/               
│   ├── train.py           # Main entry point for training/distillation
│   └── trainer.py         # Encapsulated logic for the distillation loop
├── utils/                 # Data loaders, scalers, and timing utilities
└── results.ipynb          # Notebook for post-hoc analysis and Pareto plotting

```


## Usage

### 1. Installation

```bash
git clone <repository-url>
pip install -r requirements.txt

```

### 2. Set Up W&B

```bash
wandb login
# Update project name in config/config.yaml

```

### 3. Run Distillation

To train the student model using the pre-trained Informer:

```bash
python scripts/train.py --config config/config.yaml

```


## Acknowledgments

* **Advisor**: Peter Bloem (Vrije Universiteit Amsterdam).
* **Infrastructure**: DAS-5 Supercomputer Cluster.
* **References**: Based on the seminal work by Hinton et al. (2015) and Zhou et al. (2021).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
