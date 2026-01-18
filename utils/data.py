import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm

def load_data(seq_len, pred_len, root_path, data_path, freq='h'):
    """Load and preprocess ETTh1 dataset with time features and decoder input.
    Reason: StandardScaler normalizes features; time features and x_dec support Informer.
    """
    try:
        # Load and parse CSV
        print(f"Loading data from {root_path}{data_path}...")
        df = pd.read_csv(f"{root_path}{data_path}")
        df['date'] = pd.to_datetime(df['date'])
        
        # Extract time features based on freq='h'
        time_features = {
            'month': df['date'].dt.month,
            'day': df['date'].dt.day,
            'weekday': df['date'].dt.weekday,
            'hour': df['date'].dt.hour
        }
        time_df = pd.DataFrame(time_features)
        df.drop(['date'], axis=1, inplace=True)
        
        # Normalize data
        scaler = StandardScaler()
        data = scaler.fit_transform(df.values)
        time_data = scaler.fit_transform(time_df.values)

        # Pre-allocate arrays for efficiency
        n = len(data) - seq_len - pred_len
        X = np.zeros((n, seq_len, data.shape[1]))
        y = np.zeros((n, pred_len, data.shape[1]))
        x_mark = np.zeros((n, seq_len, time_data.shape[1]))
        y_mark = np.zeros((n, seq_len + pred_len, time_data.shape[1]))
        x_dec = np.zeros((n, seq_len + pred_len, data.shape[1]))

        # Fill arrays with progress bar
        for i in tqdm(range(n), desc="Generating sequences", leave=True):
            X[i] = data[i:i + seq_len]
            y[i] = data[i + seq_len:i + seq_len + pred_len]
            x_mark[i] = time_data[i:i + seq_len]
            y_mark[i] = time_data[i:i + seq_len + pred_len]
            x_dec[i, :seq_len] = X[i]  # Copy encoder input to decoder

        # Convert to tensors
        print("Converting data to tensors...")
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        x_mark = torch.tensor(x_mark, dtype=torch.float32)
        y_mark = torch.tensor(y_mark, dtype=torch.float32)
        x_dec = torch.tensor(x_dec, dtype=torch.float32)
        return X, y, x_mark, y_mark, x_dec
    except FileNotFoundError:
        print(f"Error: Data file not found at {root_path}{data_path}")
        raise
    except pd.errors.ParserError:
        print(f"Error: Unable to parse CSV file at {root_path}{data_path}")
        raise
    except Exception as e:
        print(f"Unexpected error loading data: {e}")
        raise