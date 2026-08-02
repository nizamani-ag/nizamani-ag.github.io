"""
LST Prediction using a CNN with multi‑source remote sensing data and urban features.
This script trains a convolutional neural network (CNN) to predict land surface
temperature (LST) from NDVI, emissivity, coordinates, and urban land cover masks
It uses PyTorch for training and rasterio/geopandas for data handling
"""
import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from shapely.geometry import Point
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# Configuration
@dataclass
class Config:
    """Configuration parameters for the LST prediction pipeline"""
    # Data paths
    lst_path: Path
    ndvi_path: Path
    emissivity_path: Path
    urban_features_paths: Dict[str, Path]
    output_dir: Path
    # Training hyperparameters
    patch_size: int = 7
    batch_size: int = 64
    epochs: int = 200
    learning_rate: float = 1e-3
    # Data split
    train_ratio: float = 0.8
    # Reproducibility
    random_seed: int = 42
    # Device
    device: torch.device = None
    def __post_init__(self):
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

# Data Loading Utilities
def load_raster(path: Path) -> Tuple[np.ndarray, Any, Any, Any]:
    """
    Load a single-band raster file
    Args:
        path: Path to the raster file
    Returns:
        Tuple containing:
            data: 2D numpy array of the raster values
            transform: Affine transformation
            bounds: Raster bounds
            crs: Coordinate reference system
    """
    try:
        with rasterio.open(path) as src:
            data = src.read(1)
            transform = src.transform
            bounds = src.bounds
            crs = src.crs
        logger.info(f"Loaded raster: {path} (shape: {data.shape})")
        return data, transform, bounds, crs
    except Exception as e:
        logger.error(f"Failed to load raster {path}: {e}")
        raise

def load_geojson(path: Path) -> gpd.GeoDataFrame:
    """Load a GeoJSON file as a GeoDataFrame"""
    try:
        gdf = gpd.read_file(path)
        logger.info(f"Loaded GeoJSON: {path} (features: {len(gdf)})")
        return gdf
    except Exception as e:
        logger.error(f"Failed to load GeoJSON {path}: {e}")
        raise

def normalize(data: np.ndarray) -> np.ndarray:
    """Min‑max normalization to [0,1]"""
    min_val, max_val = np.nanmin(data), np.nanmax(data)
    if max_val == min_val:
        return np.zeros_like(data)
    return (data - min_val) / (max_val - min_val)

def geojson_to_array(geojson_path: Path,
                     x_coords: np.ndarray,
                     y_coords: np.ndarray) -> np.ndarray:
    """
    Convert a GeoJSON polygon layer to a binary mask at the given coordinates
    Args:
        geojson_path: Path to the GeoJSON file
        x_coords: 2D array of x coordinates
        y_coords: 2D array of y coordinates
    Returns:
        Boolean mask of the same shape as x_coords indicating whether each point
        lies inside any polygon of the GeoJSON
    """
    gdf = load_geojson(geojson_path)
    points = [Point(x, y) for x, y in zip(x_coords.ravel(), y_coords.ravel())]
    contains = np.array([gdf.contains(p).any() for p in points])
    return contains.reshape(x_coords.shape).astype(np.float32)

# Dataset Preparation
def create_coordinate_grid(shape: Tuple[int, int], transform: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a meshgrid of coordinates from the raster transform
    Args:
        shape: (height, width) of the raster
        transform: Affine transform from rasterio
    Returns:
        (x_coords, y_coords) 2D arrays
    """
    height, width = shape
    x = np.arange(width) * transform[0] + transform[2]
    y = np.arange(height) * transform[4] + transform[5]
    return np.meshgrid(x, y)

class LSTPatchDataset(Dataset):
    """PyTorch Dataset that extracts patches around given centers"""
    def __init__(self, centers: np.ndarray, X_img: np.ndarray, Y_img: np.ndarray, patch_size: int):
        """
        Args:
            centers: Array of (row, col) coordinates for patch centers
            X_img: Feature image of shape (H, W, C)
            Y_img: Target image of shape (H, W)
            patch_size: Side length of the square patch (odd number)
        """
        self.centers = centers
        self.X_img = X_img
        self.Y_img = Y_img
        self.patch_size = patch_size
        self.offset = patch_size // 2
    def __len__(self) -> int:
        return len(self.centers)
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        r, c = self.centers[idx]
        # Extract patch
        patch = self.X_img[r - self.offset:r + self.offset + 1,
                           c - self.offset:c + self.offset + 1, :]  # (patch_size, patch_size, C)
        target = self.Y_img[r - self.offset:r + self.offset + 1,
                            c - self.offset:c + self.offset + 1]    # (patch_size, patch_size)
        # Convert to tensors and rearrange to (C, H, W)
        patch_t = torch.from_numpy(patch).permute(2, 0, 1).float()
        target_t = torch.from_numpy(target).unsqueeze(0).float()    # (1, H, W)
        return patch_t, target_t

# CNN Model Definition
class LSTCNN(nn.Module):
    """A simple 3‑layer CNN for LST prediction from multi‑channel patches"""
    def __init__(self, in_channels: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(128, 1, kernel_size=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.conv_out(x)
        return x

# Training & Evaluation Functions
def train_one_epoch(model: nn.Module, loader: DataLoader,
                    criterion: nn.Module, optimizer: optim.Optimizer,
                    device: torch.device) -> Tuple[float, float]:
    """Run one epoch of training and return (loss, MAE)"""
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    for X, y in tqdm(loader, desc="Training", leave=False):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        total_mae += torch.abs(outputs - y).sum().item()
    n_samples = len(loader.dataset)
    return total_loss / n_samples, total_mae / n_samples

def validate_one_epoch(model: nn.Module, loader: DataLoader,
                       criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    """Run one epoch of validation and return (loss, MAE)"""
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    with torch.no_grad():
        for X, y in tqdm(loader, desc="Validation", leave=False):
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            total_loss += loss.item() * X.size(0)
            total_mae += torch.abs(outputs - y).sum().item()
    n_samples = len(loader.dataset)
    return total_loss / n_samples, total_mae / n_samples

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute R, R^2, RMSE, MAE between two flattened arrays"""
    y_true_flat = y_true.ravel()
    y_pred_flat = y_pred.ravel()
    ss_res = np.sum((y_true_flat - y_pred_flat) ** 2)
    ss_tot = np.sum((y_true_flat - np.mean(y_true_flat)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
    corr = np.corrcoef(y_true_flat, y_pred_flat)[0, 1]
    rmse = np.sqrt(np.mean((y_true_flat - y_pred_flat) ** 2))
    mae = np.mean(np.abs(y_true_flat - y_pred_flat))
    return {"R": corr, "R^2": r2, "RMSE": rmse, "MAE": mae}

# Visualization
def plot_comparison(true_lst: np.ndarray, pred_lst: np.ndarray,
                    bounds: Any, output_path: Path) -> None:
    """
    Create a side‑by‑side plot of true LST, predicted LST, and absolute difference
    Args:
        true_lst: True LST array (Celsius)
        pred_lst: Predicted LST array (Celsius)
        bounds: Bounds object from rasterio (with left, right, bottom, top)
        output_path: Where to save the figure
    """
    abs_diff = np.abs(true_lst - pred_lst)
    fig, axes = plt.subplots(1, 3, figsize=(20, 18))
    titles = ("True LST (Landsat 8)", "Predicted LST (CNN)", "Absolute Difference")
    cmaps = ("YlOrRd", "YlOrRd", "coolwarm")
    for ax, data, cmap, title in zip(axes, [true_lst, pred_lst, abs_diff], cmaps, titles):
        im = ax.imshow(data, cmap=cmap, extent=(bounds.left, bounds.right, bounds.bottom, bounds.top))
        ax.set_title(title, fontsize=22)
        if title == titles[0]:  # first subplot only
            ax.set_xticks([bounds.left, bounds.right])
            ax.set_yticks([bounds.bottom, bounds.top])
            ax.xaxis.set_major_formatter(FuncFormatter(
                lambda x, _: f"{abs(x):.2f}°{'E' if x >= 0 else 'W'}"))
            ax.yaxis.set_major_formatter(FuncFormatter(
                lambda y, _: f"{abs(y):.2f}°{'N' if y >= 0 else 'S'}"))
            ax.tick_params(axis='both', labelsize=22)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.04, aspect=15)
        cbar.set_label('')
        vmin, vmax = im.get_clim()
        ticks = np.linspace(vmin, vmax, 5)
        cbar.set_ticks(ticks)
        tick_labels = [f"{tick:.2f}" if title == titles[2] else f"{tick:.1f}" for tick in ticks]
        cbar.ax.set_xticklabels(tick_labels, fontsize=22)
        if title in titles[:2]:  # LST plots
            cbar.ax.text(1.02, 0.5, '°C', transform=cbar.ax.transAxes,
                         va='center', ha='left', fontsize=22)
        else:  # difference plot
            cbar.ax.text(1.02, 0.5, 'ΔT (°C)', transform=cbar.ax.transAxes,
                         va='center', ha='left', fontsize=22)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Comparison plot saved to {output_path}")

# Main Pipeline
def prepare_data(config: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
    """
    Load, normalize, and stack all features. Also create train/test masks
    Returns:
        X_img: Feature image (H, W, C)
        Y_img: Target image (H, W)
        train_mask: Boolean mask for training pixels
        test_mask: Boolean mask for test pixels
        bounds: Raster bounds for plotting
    """
    # Load rasters
    lst, lst_transform, lst_bounds, lst_crs = load_raster(config.lst_path)
    ndvi, _, _, _ = load_raster(config.ndvi_path)
    emissivity, _, _, _ = load_raster(config.emissivity_path)
    # Ensure all rasters have the same shape
    if not (lst.shape == ndvi.shape == emissivity.shape):
        raise ValueError("LST, NDVI, and emissivity rasters must have the same dimensions")
    # Generate coordinate grid
    x_coords, y_coords = create_coordinate_grid(lst.shape, lst_transform)
    # Load urban features
    urban_masks = {}
    for key, path in config.urban_features_paths.items():
        urban_masks[key] = geojson_to_array(path, x_coords, y_coords)
    # Normalize continuous features
    x_coords_norm = normalize(x_coords)
    y_coords_norm = normalize(y_coords)
    ndvi_norm = normalize(ndvi)
    emissivity_norm = normalize(emissivity)
    # Stack all features into a multi‑channel image
    feature_list = [
        x_coords_norm,
        y_coords_norm,
        ndvi_norm,
        emissivity_norm,
        urban_masks['buildings'],
        urban_masks['roads'],
        urban_masks['rails'],
        urban_masks['parks'],
        urban_masks['water'],
        urban_masks['trees']]
    X_img = np.stack(feature_list, axis=-1).astype(np.float32)
    # Normalize target LST
    Y_img = normalize(lst).astype(np.float32)
    logger.info(f"Input image shape: {X_img.shape}")
    logger.info(f"Target image shape: {Y_img.shape}")
    # Create train/test split
    np.random.seed(config.random_seed)
    height, width = lst.shape
    pixel_indices = np.arange(height * width)
    np.random.shuffle(pixel_indices)
    split_idx = int(config.train_ratio * len(pixel_indices))
    train_pixels = pixel_indices[:split_idx]
    test_pixels = pixel_indices[split_idx:]
    train_mask = np.zeros((height, width), dtype=bool)
    test_mask = np.zeros((height, width), dtype=bool)
    train_mask.flat[train_pixels] = True
    test_mask.flat[test_pixels] = True
    logger.info(f"Training pixels: {train_mask.sum()}")
    logger.info(f"Test pixels: {test_mask.sum()}")
    return X_img, Y_img, train_mask, test_mask, lst_bounds

def extract_patch_centers(mask: np.ndarray, patch_size: int) -> np.ndarray:
    """Return (row, col) indices of centers that allow a full patch"""
    h, w = mask.shape
    offset = patch_size // 2
    valid = np.zeros_like(mask, dtype=bool)
    valid[offset:h - offset, offset:w - offset] = True
    valid_centers = valid & mask
    return np.argwhere(valid_centers)

def train_model(config: Config,
                train_loader: DataLoader,
                test_loader: DataLoader) -> Tuple[nn.Module, pd.DataFrame]:
    """Train the CNN and return the trained model and history dataframe"""
    model = LSTCNN(in_channels=10).to(config.device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    logger.info(f"Starting training on {config.device}")
    history = {"loss": [], "val_loss": [], "mae": [], "val_mae": []}
    for epoch in range(1, config.epochs + 1):
        train_loss, train_mae = train_one_epoch(model, train_loader, criterion, optimizer, config.device)
        val_loss, val_mae = validate_one_epoch(model, test_loader, criterion, config.device)
        history["loss"].append(train_loss)
        history["mae"].append(train_mae)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)
        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch}/{config.epochs} | "
                f"Train Loss: {train_loss:.6f}, Train MAE: {train_mae:.6f} | "
                f"Val Loss: {val_loss:.6f}, Val MAE: {val_mae:.6f}")
    # Save history as CSV
    history_df = pd.DataFrame(history)
    history_df.to_csv(config.output_dir / "training_history.csv", index=False)
    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["loss"], label="Train Loss")
    ax1.plot(history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (MSE)")
    ax1.legend()
    ax1.set_title("Loss over Epochs")
    #---
    ax2.plot(history["mae"], label="Train MAE")
    ax2.plot(history["val_mae"], label="Val MAE")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MAE")
    ax2.legend()
    ax2.set_title("MAE over Epochs")
    plt.tight_layout()
    plt.savefig(config.output_dir / "training_history.png", dpi=150)
    plt.close()
    logger.info("Training history saved")
    return model, history_df

def predict_full_image(model: nn.Module, X_img: np.ndarray,
                       lst_min: float, lst_max: float,
                       device: torch.device) -> np.ndarray:
    """Predict LST for the whole image and denormalize"""
    model.eval()
    # Add batch and channel dimensions: (1, C, H, W)
    X_tensor = torch.from_numpy(X_img).permute(2, 0, 1).unsqueeze(0).float().to(device)
    with torch.no_grad():
        pred_norm = model(X_tensor).cpu().numpy()[0, 0, :, :]  # (H, W)
    # Denormalize
    return pred_norm * (lst_max - lst_min) + lst_min

def main(config: Config):
    """Run the entire LST prediction pipeline"""
    logger.info("Starting LST prediction pipeline")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Using device: {config.device}")
    # Prepare data
    X_img, Y_img, train_mask, test_mask, bounds = prepare_data(config)
    # Extract patch centers
    train_centers = extract_patch_centers(train_mask, config.patch_size)
    test_centers = extract_patch_centers(test_mask, config.patch_size)
    logger.info(f"Train patches: {len(train_centers)}")
    logger.info(f"Test patches: {len(test_centers)}")
    # Create datasets and dataloaders
    train_dataset = LSTPatchDataset(train_centers, X_img, Y_img, config.patch_size)
    test_dataset = LSTPatchDataset(test_centers, X_img, Y_img, config.patch_size)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)
    # Train model
    model, history = train_model(config, train_loader, test_loader)
    # Full image prediction
    # Get original LST min/max for denormalization
    lst, _, _, _ = load_raster(config.lst_path)
    lst_min, lst_max = np.min(lst), np.max(lst)
    pred_lst_norm = predict_full_image(model, X_img, lst_min, lst_max, config.device)
    # Convert to Celsius
    pred_lst_c = pred_lst_norm - 273.15
    true_lst_c = lst - 273.15
    # Evaluate metrics
    metrics = compute_metrics(true_lst_c, pred_lst_c)
    logger.info(f"Overall metrics: {metrics}")
    with open(config.output_dir / "metrics_cnn.txt", "w") as f:
        for name, value in metrics.items():
            f.write(f"{name}: {value:.4f}\n")
    # Save predictions
    np.save(config.output_dir / "lst_pred_denorm_C.npy", pred_lst_c)
    # Plot comparison
    plot_comparison(true_lst_c, pred_lst_c, bounds, config.output_dir / "prediction_cnn.png")
    # Save model
    torch.save(model.state_dict(), config.output_dir / "cnn_model.pth")
    logger.info("Model saved")
    logger.info("Pipeline completed successfully")

# Command‑Line Interface
def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train a CNN for LST prediction.")
    parser.add_argument("--lst", required=True, type=Path, help="Path to LST raster")
    parser.add_argument("--ndvi", required=True, type=Path, help="Path to NDVI raster")
    parser.add_argument("--emissivity", required=True, type=Path, help="Path to emissivity raster")
    parser.add_argument("--buildings", required=True, type=Path, help="GeoJSON of buildings")
    parser.add_argument("--roads", required=True, type=Path, help="GeoJSON of roads")
    parser.add_argument("--rails", required=True, type=Path, help="GeoJSON of rails")
    parser.add_argument("--parks", required=True, type=Path, help="GeoJSON of parks")
    parser.add_argument("--water", required=True, type=Path, help="GeoJSON of water bodies")
    parser.add_argument("--trees", required=True, type=Path, help="GeoJSON of trees")
    parser.add_argument("--output_dir", required=True, type=Path, help="Directory to save outputs")
    parser.add_argument("--patch_size", type=int, default=7,help="Patch size (odd number)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Fraction of pixels used for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    # Build urban features dictionary
    urban_paths = {
        "buildings": args.buildings,
        "roads": args.roads,
        "rails": args.rails,
        "parks": args.parks,
        "water": args.water,
        "trees": args.trees}
    return Config(
        lst_path=args.lst,
        ndvi_path=args.ndvi,
        emissivity_path=args.emissivity,
        urban_features_paths=urban_paths,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        train_ratio=args.train_ratio,
        random_seed=args.seed)

if __name__ == "__main__":
    cfg = parse_args()
    # Set random seeds for reproducibility
    torch.manual_seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    main(cfg)