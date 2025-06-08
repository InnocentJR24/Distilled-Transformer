import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_data(seq_len, pred_len, root_path, data_path, freq='h'):
    """Load and preprocess ETTh1 dataset with time features and decoder input.
    Reason: StandardScaler normalizes features; time features and x_dec support Informer.
    """
    try:
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
        scaler = StandardScaler()
        data = scaler.fit_transform(df.values)
        time_data = scaler.fit_transform(time_df.values)

        xs, ys, x_marks, y_marks, x_decs = [], [], [], [], []
        for i in range(len(data) - seq_len - pred_len):
            x = data[i:i + seq_len]  # Encoder input: [seq_len, features]
            y = data[i + seq_len:i + seq_len + pred_len]  # Target: [pred_len, features]
            x_mark = time_data[i:i + seq_len]  # Time features for encoder
            y_mark = time_data[i + seq_len:i + seq_len + pred_len]  # Time features for decoder
            # Decoder input: last label_len of x_enc + zeros for pred_len
            x_dec = np.zeros((seq_len + pred_len, data.shape[1]))
            x_dec[:seq_len] = x  # Copy last seq_len timesteps
            xs.append(x)
            ys.append(y)
            x_marks.append(x_mark)
            y_marks.append(y_mark)
            x_decs.append(x_dec)
        X = torch.tensor(xs, dtype=torch.float32)
        y = torch.tensor(ys, dtype=torch.float32)
        x_mark = torch.tensor(x_marks, dtype=torch.float32)
        y_mark = torch.tensor(y_marks, dtype=torch.float32)
        x_dec = torch.tensor(x_decs, dtype=torch.float32)
        return X, y, x_mark, y_mark, x_dec
    except Exception as e:
        print(f"Error loading data: {e}")
        raise