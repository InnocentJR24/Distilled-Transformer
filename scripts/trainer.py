import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
import time
import wandb
from models.informer import Informer
from models.lstm import LSTMModel
import os
from utils.data import load_data

def load_informer_model(checkpoints, args, device):
    """Load the pre-trained Informer model.
    Reason: Loads teacher model to provide soft targets for distillation, evaluated as baseline.
    """
    try:
        model = Informer(
            enc_in=args.enc_in,
            dec_in=args.dec_in,
            c_out=args.c_out,
            seq_len=args.seq_len,
            label_len=args.label_len,
            out_len=args.pred_len,
            factor=args.factor,
            d_model=args.d_model,
            n_heads=args.n_heads,
            e_layers=args.e_layers,
            d_layers=args.d_layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
            attn=args.attn,
            embed=args.embed,
            freq=args.freq,
            activation=args.activation,
            distil=args.distil,
            mix=args.mix,
            device=device
        ).to(device)
        checkpoint_path = os.path.join(checkpoints, "informer_ETTh1_ftM_sl336_ll336_pl720_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_Exp_0", "checkpoint.pth")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading Informer model: {e}")
        raise

def evaluate_informer_baseline(model, test_loader, device):
    """Evaluate pre-trained Informer model for MSE and inference time.
    Reason: Provides static baseline to anchor student model’s performance and efficiency.
    """
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for _ in range(3):  # Warm-up iterations
            for xb, _, x_mark, y_mark, x_dec in test_loader:
                xb, x_mark, x_dec, y_mark = xb.to(device), x_mark.to(device), x_dec.to(device), y_mark.to(device)
                _ = model(xb, x_mark, x_dec, y_mark)
    start = time.time()
    with torch.no_grad():
        for xb, yb, x_mark, y_mark, x_dec in test_loader:
            xb, yb, x_mark, y_mark, x_dec = xb.to(device), yb.to(device), x_mark.to(device), y_mark.to(device), x_dec.to(device)
            pred = model(xb, x_mark, x_dec, y_mark)
            preds.append(pred.cpu())
            truths.append(yb.cpu())
    end = time.time()
    inf_time = (end - start) / len(test_loader)  # Average inference time per batch (seconds)
    mse = mean_squared_error(torch.cat(truths).flatten(), torch.cat(preds).flatten())
    return mse, inf_time

def train(sweep_config, args, device):
    # Load pre-trained Informer model (teacher)
    informer_model = load_informer_model(args.checkpoints, args, device)

    # Load ETTh1 data
    X, y, x_mark, y_mark, x_dec = load_data(args.seq_len, args.pred_len, args.root_path, args.data_path, args.freq)
    train_split = int(0.7 * len(X))
    val_split = int(0.85 * len(X))
    train_X, val_X, test_X = X[:train_split], X[train_split:val_split], X[val_split:]
    train_y, val_y, test_y = y[:train_split], y[train_split:val_split], y[val_split:]
    train_x_mark, val_x_mark, test_x_mark = x_mark[:train_split], x_mark[train_split:val_split], x_mark[val_split:]
    train_y_mark, val_y_mark, test_y_mark = y_mark[:train_split], y_mark[train_split:val_split], y_mark[val_split:]
    train_x_dec, val_x_dec, test_x_dec = x_dec[:train_split], x_dec[train_split:val_split], x_dec[val_split:]
    train_loader = DataLoader(TensorDataset(train_X, train_y, train_x_mark, train_y_mark, train_x_dec), batch_size=sweep_config.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_X, val_y, val_x_mark, val_y_mark, val_x_dec), batch_size=sweep_config.batch_size)
    test_loader = DataLoader(TensorDataset(test_X, test_y, test_x_mark, test_y_mark, test_x_dec), batch_size=sweep_config.batch_size)

    # Evaluate Informer baseline
    informer_mse, informer_inf_time = evaluate_informer_baseline(informer_model, test_loader, device)
    wandb.log({"informer_mse": informer_mse, "informer_inference_time": informer_inf_time})

    # Initialize student model
    model = LSTMModel(args.enc_in, sweep_config.hidden_size, sweep_config.num_layers, args.c_out, args.pred_len, sweep_config.dropout_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=sweep_config.lr)
    loss_fn = nn.MSELoss()

    # Training loop with distillation
    for epoch in range(int(sweep_config.epochs)):
        model.train()
        total_loss = 0
        for i, (xb, yb, x_mark, y_mark, x_dec) in enumerate(train_loader):
            xb, yb, x_mark, y_mark, x_dec = xb.to(device), yb.to(device), x_mark.to(device), y_mark.to(device), x_dec.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                soft_target = informer_model(xb, x_mark, x_dec, y_mark)  # Teacher's soft predictions
            pred = model(xb)
            loss = sweep_config.alpha * loss_fn(pred, yb) + (1 - sweep_config.alpha) * loss_fn(pred, soft_target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        wandb.log({"epoch": epoch + 1, "train_loss": total_loss / len(train_loader)})

        # Validation
        model.eval()
        val_preds, val_truths = [], []
        with torch.no_grad():
            for xb, yb, x_mark, y_mark, x_dec in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_preds.append(pred.cpu())
                val_truths.append(yb.cpu())
        val_mse = mean_squared_error(torch.cat(val_truths).flatten(), torch.cat(val_preds).flatten())
        wandb.log({"val_mse": val_mse})

    # Evaluation on test set
    model.eval()
    test_preds, test_truths = [], []
    with torch.no_grad():
        for _ in range(3):  # Warm-up iterations
            for xb, _, x_mark, y_mark, x_dec in test_loader:
                xb = xb.to(device)
                _ = model(xb)
    start = time.time()
    with torch.no_grad():
        for xb, yb, x_mark, y_mark, x_dec in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            test_preds.append(pred.cpu())
            test_truths.append(yb.cpu())
    end = time.time()
    inf_time = (end - start) / len(test_loader)  # Average inference time per batch (seconds)
    test_mse = mean_squared_error(torch.cat(test_truths).flatten(), torch.cat(test_preds).flatten())
    num_params = sum(p.numel() for p in model.parameters())
    wandb.log({"test_mse": test_mse, "inference_time": inf_time, "num_params": num_params})

    # Save model checkpoint
    torch.save(model.state_dict(), "lstm_model.pt")
    wandb.save("lstm_model.pt")