"""
Runs a single evaluation on a given dataset and model. This is the main entry point for running evaluations.
"""
import os
from turtle import pd
import torch
import torch.nn as nn
import pandas as pd
from load_eval_models import *
import torch.nn.functional as F
from load_eval_models import *
from tqdm import tqdm

from satclip.splits import *

# List of seeds to run
# SEEDS = [42, 123, 456, 789]
SEEDS = [42]

# List of models to run
# MODELS = ["tsatclip-linear", "gtloc", "climplicit"]
MODELS = [ "tsatclip/doy", "tsatclip/linear", "tsatclip/toroidal", "gtloc", "climplicit", "sin-cos", "simplest"]
# MODELS = ["tsatclip/toroidal"]

# In Lat/Lon Degrees
SPATIAL_DELTAS = [2, 4, 8, 16, 32]

# In seconds
TEMPORAL_DELTAS_DAYS = [1, 7, 30, 90]
TEMPORAL_DELTAS = [days * 24 * 60 * 60 for days in TEMPORAL_DELTAS_DAYS]

# List of datasets to run
#DATASETS = ["ghcnd", "chelsa", "era5"]
DATASETS = ["ghcnd"]

CV_TYPES = ["uar", "spatial", "temporal"]

# List of metrics to run
METRICS = ["ridge-cv"]

RIDGE_ALPHAS = tuple(float(10.0**e) for e in np.arange(-4.0, 6.5, 0.5))

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

def load_ghcnd_dataset(dataset_path="../data/ghcn_2021_2026_high_temp_benchmark.csv"):
    """
    Loads the GHCNd dataset from a CSV file and returns a PyTorch tensor.
    The CSV file is expected to have the following columns: Latitude, Longitude, POSIX_TIME, Value
    """
    import pandas as pd

    df = pd.read_csv(dataset_path, header=None)
    data = torch.tensor(df.values, dtype=torch.float32, device="cpu")
    return data

def run_ridge_cv_probe_eval(model, train_data, test_data, device="cuda", results_dir="results"):
    from sklearn.linear_model import RidgeCV

    # Get embeddings
    model = model.to(device)
    model.eval()
    train_embeddings = model(train_data[:, :3].to(device).detach().clone().double())
    test_embeddings = model(test_data[:, :3].to(device).detach().clone().double())

    X_train = train_embeddings.detach().cpu().numpy()
    y_train = train_data[:, 3].detach().cpu().numpy()

    X_test = test_embeddings.detach().cpu().numpy()
    y_test = test_data[:, 3].detach().cpu().numpy()

    ridge_model = RidgeCV(alphas=RIDGE_ALPHAS, cv=None)
    ridge_model.fit(X_train, y_train)

    y_pred = ridge_model.predict(X_test)

    #Get RMSE
    rmse = torch.sqrt(F.mse_loss(torch.tensor(y_pred), torch.tensor(y_test))).item()

    results = {}
    results["test_rmse"] = rmse
    return results

def run_linear_probe_eval(model, train_data, test_data, device="cuda", results_dir="results"):
    from sklearn.linear_model import LinearRegression, Ridge

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

def run_mlp_finetune_eval(model, train_set, test_set, device="cuda", results_dir="results"):
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

_REGISTERED_DATASET = {
    "ghcnd": load_ghcnd_dataset,
}

_REGISTERED_MODELS = { 
    "tsatclip/linear": load_temporal_satclip_linear_model,
    "tsatclip/doy": load_temporal_satclip_doy_model,
    "tsatclip/toroidal": load_temporal_satclip_toroidal_model,
    "gtloc": load_gtloc_model,
    "climplicit": lambda device="cuda": ClimplicitWrapper().to(device),
    "sin-cos": lambda device="cuda": SinCosWrapper().to(device),
    "simplest": lambda device="cuda": SimplestModelWrapper().to(device),
}

def run_evaluation(seed, model, dataset, cv_type, metric, delta=None, device="cuda", results_dir="results"):
    """
    Runs a single evaluation on a given dataset and model.
    """
    print(f"Running eval: model {model} | dataset {dataset} | cv_type {cv_type} | delta {delta} | metric {metric}.")

    # Set the random seed for reproducibility
    import random
    import numpy as np
    import torch

    results = {}
    results["seed"] = seed
    results["model"] = model
    results["dataset"] = dataset
    results["cv_type"] = cv_type
    results["metric"] = metric

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    model_loader = _REGISTERED_MODELS[model]
    dataset_loader = _REGISTERED_DATASET[dataset]

    model = model_loader()
    dataset = dataset_loader()

    orig_dataset_shape = dataset.shape

    # Subsample the dataset for faster evaluation (optional)
    dataset = dataset[torch.randperm(dataset.shape[0])[:500_000]]  # Subsample to 10,000 points
    # print(f"Subsampled dataset shape from {orig_dataset_shape} to {dataset.shape}")

    # Split the dataset into train and test sets based on cv_type
    if cv_type == "uar":
        # Uniformly split 50/50 into train and test sets
        num_samples = dataset.shape[0]
        indices = torch.randperm(num_samples)
        split = num_samples // 2
        train_indices = indices[:split]
        test_indices = indices[split:]

    elif cv_type == "spatial":
        # Spatially split the dataset into train and test sets
        spatial_grid_size = delta if delta is not None else 1.0  # degrees
        splits = checkerboard_splits(dataset[:, :2], spatial_grid_size, dim=-1)
        train_indices = (splits == 0).squeeze()
        test_indices = (splits == 1).squeeze()

    elif cv_type == "temporal":
        # Temporally split the dataset into train and test sets
        # temporal_grid_size = 365.0 * 24 * 60 * 60  # seconds in a year
        temporal_grid_size = delta if delta is not None else 30.0 * 24 * 60 * 60  # seconds in a month
        splits = temporal_splits(dataset[:, 2], temporal_grid_size)
        train_indices = (splits == 0).squeeze()
        test_indices = (splits == 1).squeeze()

    train_data = dataset[train_indices]
    test_data = dataset[test_indices]

    # Run the eval
    if metric == "linear-probe":
        evals = run_linear_probe_eval(model, train_data, test_data, device=device, results_dir=results_dir)
    elif metric == "mlp-finetune":
        evals = run_mlp_finetune_eval(model, train_data, test_data, device=device, results_dir=results_dir)
    elif metric == "ridge-cv":
        evals = run_ridge_cv_probe_eval(model, train_data, test_data, device=device, results_dir=results_dir)

    # Save intermediate plots in results_dir
    results["eval"] = evals
    return results

def main():
    """
    Main function to run evaluations across all combinations of models, datasets, metrics, and seeds.
    """

    cv_types = CV_TYPES
    seeds = SEEDS
    datasets = DATASETS
    models = MODELS
    metrics = METRICS

    results = {}

    results_df = pd.DataFrame(columns=["seed", "model", "dataset", "cv_type", "metric", "test_rmse", "delta"])

    for seed in tqdm(seeds, desc="Seeds"):
        for model in tqdm(models, desc="Models", leave=False):
            for dataset in tqdm(datasets, desc="Datasets", leave=False):
                for cv_type in cv_types:
                    for metric in metrics:
                        if cv_type == "spatial":
                            for delta in SPATIAL_DELTAS:
                                evals = run_evaluation(seed, model, dataset, cv_type, metric, delta=delta, device="cuda", results_dir="results")
                                results[(seed, model, dataset, cv_type, metric, delta)] = evals["eval"]["test_rmse"]

                                new_row = pd.DataFrame([{
                                    "seed": seed,
                                    "model": model,
                                    "dataset": dataset,
                                    "cv_type": cv_type,
                                    "metric": metric,
                                    "delta": delta if cv_type in ["spatial", "temporal"] else None,
                                    "test_rmse": evals["eval"]["test_rmse"]/10
                                }])
                                results_df = pd.concat([results_df, new_row], ignore_index=True)
                        elif cv_type == "temporal":
                            for delta in TEMPORAL_DELTAS:
                                evals = run_evaluation(seed, model, dataset, cv_type, metric, delta=delta, device="cuda", results_dir="results")
                                results[(seed, model, dataset, cv_type, metric, delta)] = evals["eval"]["test_rmse"]

                                new_row = pd.DataFrame([{
                                    "seed": seed,
                                    "model": model,
                                    "dataset": dataset,
                                    "cv_type": cv_type,
                                    "metric": metric,
                                    "delta": delta if cv_type in ["spatial", "temporal"] else None,
                                    "test_rmse": evals["eval"]["test_rmse"]/10
                                }])
                                results_df = pd.concat([results_df, new_row], ignore_index=True)
                        else:
                            evals = run_evaluation(seed, model, dataset, cv_type, metric, device="cuda", results_dir="results")
                            results[(seed, model, dataset, cv_type, metric)] = evals["eval"]["test_rmse"]

                            new_row = pd.DataFrame([{
                                "seed": seed,
                                "model": model,
                                "dataset": dataset,
                                "cv_type": cv_type,
                                "metric": metric,
                                "delta": delta if cv_type in ["spatial", "temporal"] else None,
                                "test_rmse": evals["eval"]["test_rmse"]/10
                            }])

                            results_df = pd.concat([results_df, new_row], ignore_index=True)


    # Save results to a file
    import pickle
    with open("results/evaluation_results.pkl", "wb") as f:
        pickle.dump(results, f)

    results_df.to_csv("results/evaluation_results.csv", index=False)

    print("Evaluation completed. Results saved to results/evaluation_results.pkl")
    print("Results DataFrame:")
    print(results_df)

if __name__ == "__main__":
    main()