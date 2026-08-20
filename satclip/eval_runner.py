"""
Runs a single evaluation on a given dataset and model. This is the main entry point for running evaluations.
"""
import os
import torch
import torch.nn as nn

from load_eval_models import *
import torch.nn.functional as F

# List of seeds to run
SEEDS = [42, 123, 456, 789]

# List of models to run
# MODELS = ["tsatclip-linear", "gtloc", "climplicit"]
MODELS = ["tsatclip-linear", "gtloc", "naive-sincos"]

# List of datasets to run
#DATASETS = ["ghcnd", "chelsa", "era5"]
DATASETS = ["ghcnd"]

# List of metrics to run
METRICS = ["linear-probe"]

class MLP(nn.Module):
    def __init__(self, input_dim, dim_hidden, num_layers, out_dims):
        super(MLP, self).__init__()

        layers = []
        layers += [nn.Linear(input_dim, dim_hidden, bias=True), nn.ReLU()] # Input layer
        layers += [nn.Linear(dim_hidden, dim_hidden, bias=True), nn.ReLU()] * num_layers # Hidden layers
        layers += [nn.Linear(dim_hidden, out_dims, bias=True)] # Output layer

        self.features = nn.Sequential(*layers)

    def forward(self, x):
        return self.features(x)

class DumbModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_dim = 3  # lat, lon, time

    def forward(self, x):
        # Just passes x through
        return x

class SinCosWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_dim = 4  # lat, lon, sin_time, cos_time

    def forward(self, x):
        # Wrapper expects (N, [lat, lon, time]) and returns embeddings
        # The SinCos model expects (N, [lon, lat, posix_time]) and returns embeddings

        x = x.clone()
        posix_time = x[..., 2]

        # Convert posix time to sin/cos representation
        sin_time = torch.sin(2 * torch.pi * posix_time / 31556926)
        cos_time = torch.cos(2 * torch.pi * posix_time / 31556926)

        # Concatenate lat, lon, sin_time, cos_time
        embeddings = torch.cat([x[..., :2], sin_time.unsqueeze(-1), cos_time.unsqueeze(-1)], dim=-1)
        return embeddings

def create_dataloaders():
    """
    Creates dataloaders for the train and test datasets.
    """
    # Load the train and test datasets
    train_dataset = torch.utils.data.TensorDataset(train_embeddings, train_sample[:, 3].detach().clone().unsqueeze(1).double())
    test_dataset = torch.utils.data.TensorDataset(test_embeddings, test_sample[:, 3].detach().clone().unsqueeze(1).double())

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8096, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8096, shuffle=False)

    return train_loader, test_loader

def run_linear_probe_evaluation(model, seed, train_data, test_data, device="cuda", results_dir="results"):
    from sklearn.linear_model import LinearRegression, Ridge

    # Set the random seed
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)

    # Get embeddings
    model = model.to(device)
    model.eval()
    train_embeddings = model(train_data[:, :3].to(device).detach().clone().double())
    test_embeddings = model(test_data[:, :3].to(device).detach().clone().double())

    X_train = train_embeddings.detach().cpu().numpy()
    y_train = train_data[:, 3].detach().cpu().numpy()

    X_test = test_embeddings.detach().cpu().numpy()
    y_test = test_data[:, 3].detach().cpu().numpy()

    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train, y_train)

    y_pred = ridge_model.predict(X_test)

    #Get RMSE
    rmse = torch.sqrt(F.mse_loss(torch.tensor(y_pred), torch.tensor(y_test))).item()

    results = {}
    results["test_rmse"] = rmse
    return results

def run_mlp_finetune_evaluation(model, train_set, test_set, device="cuda", results_dir="results"):
    """
    Dataset of the form (lat, lon, posix_time, value) where value is the target variable to predict.
    """

    model = model.to(device)
    model.eval()  # Set the model to evaluation mode

    train_dataset = torch.utils.data.TensorDataset(train_set[:, :3].detach().clone().double(), train_set[:, 3].detach().clone().unsqueeze(1).double())
    test_dataset = torch.utils.data.TensorDataset(test_set[:, :3].detach().clone().double(), test_set[:, 3].detach().clone().unsqueeze(1).double())

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8096, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8096, shuffle=False)

    pred_model = MLP(input_dim=model.embedding_dim, dim_hidden=64, num_layers=2, out_dims=1).double().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(pred_model.parameters(), lr=1e-3)

    epoch_losses = []
    epochs = 100
    running_loss = torch.zeros(1, device=device)

    results = {}
    results["epochs"] = []

    for epoch in range(epochs):
        pred_model.train()
        running_loss.zero_()  # Reset running loss for the epoch

        for batch in train_loader:
            optimizer.zero_grad()
            train_coords, values = batch
            # Forward pass
            y_pred = pred_model(model(train_coords.to(device)))
            # Compute the loss
            loss = criterion(y_pred, values.to(device))
            # Backward pass
            loss.backward()
            # Update the parameters
            optimizer.step()
            # Append the loss to the list
            # losses.append(loss.item())
            running_loss += loss.detach() * train_coords.size(0)  # Multiply by batch size to get total loss for the batch

        epoch_loss = (running_loss / len(train_loader.dataset)).item()
        epoch_losses.append(epoch_loss)

        if (epoch) % 10 == 0:
            print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")

        results["epoch_losses"] = epoch_losses
        results["epochs"].append(epoch)

    preds = []
    actuals = []
    with torch.no_grad():
        pred_model.eval()
        model.eval()

        for batch in test_loader:
            test_coords, values = batch
            y_pred_test = pred_model(model(test_coords.to(device)))
            preds.extend(y_pred_test.cpu().detach().numpy())
            actuals.extend(values.float().cpu().detach().numpy())
    test_rmse = torch.sqrt(F.mse_loss(torch.tensor(preds), torch.tensor(actuals))).item()

    results["test_rmse"] = test_rmse
    results["preds"] = preds
    results["actuals"] = actuals

    return results

def run_evaluation(seed, device="cuda", results_dir="results"):
    """
    Runs a single evaluation on a given dataset and model.
    """
    # Set the random seed for reproducibility
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    ts_lin_model = load_temporal_satclip_linear_model()
    ts_doy_model = load_temporal_satclip_doy_model()
    gtloc_model = load_gtloc_model()


    # Load GHCNd dataloaders and appropriate splits

    # Run the evaluations for each combination of model, dataset, metric, and seed

    # Save intermediate plots in results_dir

    return results

# def main():
#     """
#     Main function to run evaluations across all combinations of models, datasets, metrics, and seeds.
#     """
#     seeds = [42]

#     for seed in seeds:
#         run_evaluation(seed, device="cuda", results_dir="results")

# if __name__ == "__main__":
#     main()