"""
GCN-based land surface temperature (LST) prediction from satellite imagery and urban features.
This script trains a graph convolutional network (GCN) on image patches, then performs full-image prediction
"""
import os
import logging
import warnings
from dataclasses import dataclass, field
from typing import Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
import geopandas as gpd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
from shapely.geometry import Point
from matplotlib.ticker import FuncFormatter
# Suppress user warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Configuration
@dataclass
class Config:
    """Configuration parameters for the experiment"""
    # Paths
    lst_path: str = "/path_directory/LST_PR.tif"
    ndvi_path: str = "/path_directory/NDVI_PR.tif"
    emissivity_path: str = "/path_directory/EM_PR.tif"
    urban_geojson_dir: str = "/path_directory/paris_GEE/"
    output_dir: str = "/path_directory/GCN_model/"
    # Urban feature files (relative to urban_geojson_dir)
    urban_features: Tuple[str, ...] = (
        "buildings.geojson", "roads.geojson", "rails.geojson",
        "parks.geojson", "water.geojson", "trees.geojson")
    # Model hyperparameters
    patch_size: int = 7
    batch_size: int = 64
    epochs: int = 200
    hidden_channels: int = 128
    learning_rate: float = 0.001
    random_seed: int = 42
    # Device
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Helper functions
def load_raster(path: str) -> Tuple[np.ndarray, rasterio.Affine, rasterio.coords.BoundingBox, rasterio.crs.CRS]:
    """
    Load a single-band raster file
    Args:
        path: Path to the raster file
    Returns:
        Tuple of (data array, transform, bounds, crs)
    """
    with rasterio.open(path) as src:
        data = src.read(1)
        transform = src.transform
        bounds = src.bounds
        crs = src.crs
    return data, transform, bounds, crs

def load_geojson(path: str) -> gpd.GeoDataFrame:
    """Load a GeoJSON file"""
    return gpd.read_file(path)

def normalize(data: np.ndarray) -> np.ndarray:
    """Min‑max normalization ignoring NaN values"""
    min_val = np.nanmin(data)
    max_val = np.nanmax(data)
    if max_val - min_val == 0:
        return np.zeros_like(data)
    return (data - min_val) / (max_val - min_val)

def geojson_to_array(geojson_path: str, x_coords: np.ndarray, y_coords: np.ndarray) -> np.ndarray:
    """
    Convert a GeoJSON file containing polygons into a binary mask array
    Args:
        geojson_path: Path to the GeoJSON file
        x_coords: 2D array of x coordinates
        y_coords: 2D array of y coordinates
    Returns:
        Binary mask (1 where a point lies inside any polygon, 0 otherwise)
    """
    gdf = load_geojson(geojson_path)
    points = [Point(x, y) for x, y in zip(x_coords.ravel(), y_coords.ravel())]
    return np.array([gdf.contains(p).any() for p in points]).reshape(x_coords.shape)

def extract_patch_center(mask: np.ndarray, patch_size: int) -> np.ndarray:
    """
    Create a boolean mask of valid patch center positions
    A center is valid if a full patch of size patch_size fits inside the image
    """
    h, w = mask.shape
    offset = patch_size // 2
    valid_center = np.zeros_like(mask, dtype=bool)
    valid_center[offset:h - offset, offset:w - offset] = True
    return valid_center & mask

def get_pixel_edges(patch_size: int) -> torch.Tensor:
    """
    Return edge indices for an 8‑connected grid of size patch_size x patch_size
    Args:
        patch_size: Width/height of the square patch
    Returns:
        Edge index tensor of shape (2, E) (undirected)
    """
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    edges = []
    for r in range(patch_size):
        for c in range(patch_size):
            idx = r * patch_size + c
            for dr, dc in offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < patch_size and 0 <= nc < patch_size:
                    nidx = nr * patch_size + nc
                    edges.append([idx, nidx])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    # Add reverse edges to make it undirected
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return edge_index

def denormalize(pred_norm: np.ndarray, original_min: float, original_max: float) -> np.ndarray:
    """Reverse min‑max normalization"""
    return pred_norm * (original_max - original_min) + original_min

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics: R^2, Pearson correlation, RMSE, MAE"""
    # Ensure 1D
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    return {"R^2": r2, "R": corr, "RMSE": rmse, "MAE": mae}

# Dataset class
class GraphPatchDataset(Dataset):
    """
    Dataset that returns a torch_geometric.data.Data object for each patch
    Each patch is a small graph where nodes are pixels (flattened) and edges are
    defined by an 8‑connected grid
    """
    def __init__(self, centers: np.ndarray, X_img: np.ndarray, Y_img: np.ndarray,
                 patch_size: int, edge_index: torch.Tensor):
        """
        Args:
            centers: (N, 2) array of (row, col) center coordinates
            X_img: Input image (H, W, C)
            Y_img: Target image (H, W)
            patch_size: Width/height of the square patch
            edge_index: Precomputed edge indices for the patch graph
        """
        self.centers = centers
        self.X_img = X_img
        self.Y_img = Y_img
        self.patch_size = patch_size
        self.offset = patch_size // 2
        self.edge_index = edge_index
    def __len__(self) -> int:
        return len(self.centers)
    def __getitem__(self, idx: int) -> Data:
        r, c = self.centers[idx]
        # Extract patch
        patch = self.X_img[r - self.offset:r + self.offset + 1,
                           c - self.offset:c + self.offset + 1, :]  # (P, P, C)
        target = self.Y_img[r - self.offset:r + self.offset + 1,
                            c - self.offset:c + self.offset + 1]    # (P, P)
        # Flatten to nodes
        node_features = patch.reshape(-1, patch.shape[-1])  # (N, C)
        node_targets = target.reshape(-1)                   # (N,)
        x = torch.from_numpy(node_features).float()
        y = torch.from_numpy(node_targets).float()
        return Data(x=x, edge_index=self.edge_index, y=y)

def collate_graphs(batch: List[Data]) -> Batch:
    """Collate a list of Data objects into a Batch"""
    return Batch.from_data_list(batch)

# Model definition
class GCNRegressor(nn.Module):
    """GCN for node‑wise regression"""
    def __init__(self, in_channels: int, hidden_channels: int = 128, out_channels: int = 1):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)
    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index) # (N, out_channels)
        return x.squeeze(-1)  # (N,)

# Training and evaluation functions
def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer,
                    criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    """Train the model for one epoch and return average loss and MAE"""
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    num_nodes = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)  # (total_nodes,)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_nodes
        total_mae += torch.abs(out - batch.y).sum().item()
        num_nodes += batch.num_nodes
    avg_loss = total_loss / num_nodes
    avg_mae = total_mae / num_nodes
    return avg_loss, avg_mae
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             device: torch.device) -> Tuple[float, float]:
    """Evaluate the model and return average loss and MAE"""
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    num_nodes = 0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        loss = criterion(out, batch.y)
        total_loss += loss.item() * batch.num_nodes
        total_mae += torch.abs(out - batch.y).sum().item()
        num_nodes += batch.num_nodes
    avg_loss = total_loss / num_nodes
    avg_mae = total_mae / num_nodes
    return avg_loss, avg_mae

def full_image_prediction(model: nn.Module, X_img: np.ndarray, patch_size: int,
                          edge_index: torch.Tensor, device: torch.device) -> np.ndarray:
    """
    Perform sliding‑window prediction over the full image and average overlapping predictions
    Returns:
        Normalized predicted image (same shape as Y_img)
    """
    model.eval()
    h, w, _ = X_img.shape
    offset = patch_size // 2
    pred_img = np.zeros((h, w), dtype=np.float64)
    weight_img = np.zeros((h, w), dtype=np.float64)
    for r in range(offset, h - offset):
        for c in range(offset, w - offset):
            # Extract patch
            patch = X_img[r - offset:r + offset + 1,
                          c - offset:c + offset + 1, :]  # (P, P, C)
            node_features = patch.reshape(-1, patch.shape[-1]) # (N, C)
            x = torch.from_numpy(node_features).float().to(device)
            data = Data(x=x, edge_index=edge_index.to(device))
            with torch.no_grad():
                out = model(data)  # (N,)
            # Map node predictions back to global coordinates
            for node_idx, pred_val in enumerate(out.cpu().numpy()):
                local_r = node_idx // patch_size
                local_c = node_idx % patch_size
                global_r = r - offset + local_r
                global_c = c - offset + local_c
                pred_img[global_r, global_c] += pred_val
                weight_img[global_r, global_c] += 1
    # Average overlapping predictions
    valid = weight_img > 0
    pred_img[valid] /= weight_img[valid]
    pred_img[~valid] = np.nan
    return pred_img

# Main execution
def main(config: Config) -> None:
    """Run the entire pipeline: data loading, training, evaluation, and saving"""
    # Set seeds for reproducibility
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    logger.info(f"Output directory: {config.output_dir}")
    # Load data
    logger.info("Loading raster data")
    lst, lst_transform, lst_bounds, lst_crs = load_raster(config.lst_path)
    ndvi, _, ndvi_bounds, _ = load_raster(config.ndvi_path)
    emissivity, _, emissivity_bounds, _ = load_raster(config.emissivity_path)
    # Generate coordinate grid from LST transform
    height, width = lst.shape
    x_coords, y_coords = np.meshgrid(
        np.arange(width) * lst_transform[0] + lst_transform[2],
        np.arange(height) * lst_transform[4] + lst_transform[5])
    # Load urban features as binary masks
    logger.info("Loading urban features")
    urban_masks = {}
    for fname in config.urban_features:
        fpath = os.path.join(config.urban_geojson_dir, fname)
        urban_masks[fname.split('.')[0]] = geojson_to_array(fpath, x_coords, y_coords)
    # Normalize continuous features
    x_coords_norm = normalize(x_coords)
    y_coords_norm = normalize(y_coords)
    ndvi_norm = normalize(ndvi)
    emissivity_norm = normalize(emissivity)
    # Stack all input features (10 channels)
    X_img = np.stack([
        x_coords_norm, y_coords_norm,
        ndvi_norm, emissivity_norm,
        urban_masks['buildings'].astype(np.float32),
        urban_masks['roads'].astype(np.float32),
        urban_masks['rails'].astype(np.float32),
        urban_masks['parks'].astype(np.float32),
        urban_masks['water'].astype(np.float32),
        urban_masks['trees'].astype(np.float32)], axis=-1)
    Y_img = normalize(lst).astype(np.float32)
    logger.info(f"Input image shape: {X_img.shape}")
    logger.info(f"Target image shape: {Y_img.shape}")
    # Train/test split (pixel‑wise)
    pixel_indices = np.arange(height * width)
    np.random.shuffle(pixel_indices)
    split_idx = int(0.8 * len(pixel_indices))
    train_pixels = pixel_indices[:split_idx]
    test_pixels = pixel_indices[split_idx:]
    train_mask = np.zeros((height, width), dtype=bool)
    test_mask = np.zeros((height, width), dtype=bool)
    train_mask.flat[train_pixels] = True
    test_mask.flat[test_pixels] = True
    logger.info(f"Training pixels: {train_mask.sum()}")
    logger.info(f"Test pixels: {test_mask.sum()}")
    # Build patch datasets
    logger.info("Building patch datasets")
    train_center_mask = extract_patch_center(train_mask, config.patch_size)
    test_center_mask = extract_patch_center(test_mask, config.patch_size)
    train_centers = np.argwhere(train_center_mask)
    test_centers = np.argwhere(test_center_mask)
    logger.info(f"Number of train patches: {len(train_centers)}")
    logger.info(f"Number of test patches: {len(test_centers)}")
    edge_index = get_pixel_edges(config.patch_size)  # Precomputed for all patches
    train_dataset = GraphPatchDataset(train_centers, X_img, Y_img, config.patch_size, edge_index)
    test_dataset = GraphPatchDataset(test_centers, X_img, Y_img, config.patch_size, edge_index)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
        collate_fn=collate_graphs, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
        collate_fn=collate_graphs, num_workers=4, pin_memory=True)
    # Model initialization
    logger.info("Initializing model")
    model = GCNRegressor(in_channels=X_img.shape[-1],
        hidden_channels=config.hidden_channels).to(config.device)
    logger.info(model)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    # Training loop
    logger.info("Starting training")
    train_losses, train_maes = [], []
    val_losses, val_maes = [], []
    for epoch in range(config.epochs):
        train_loss, train_mae = train_one_epoch(model, train_loader, optimizer, criterion, config.device)
        val_loss, val_mae = evaluate(model, test_loader, criterion, config.device)
        train_losses.append(train_loss)
        train_maes.append(train_mae)
        val_losses.append(val_loss)
        val_maes.append(val_mae)
        logger.info(
            f"Epoch {epoch+1:3d}/{config.epochs} | "
            f"Train Loss: {train_loss:.6f} | Train MAE: {train_mae:.6f} | "
            f"Val Loss: {val_loss:.6f} | Val MAE: {val_mae:.6f}")
    # Save training history
    history_df = pd.DataFrame({
        'loss': train_losses,
        'val_loss': val_losses,
        'mae': train_maes,
        'val_mae': val_maes})
    history_path = os.path.join(config.output_dir, 'training_history.csv')
    history_df.to_csv(history_path, index=False)
    logger.info(f"Training history saved to {history_path}")
    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.legend()
    ax1.set_title('Loss over Epochs')
    #----
    ax2.plot(train_maes, label='Train MAE')
    ax2.plot(val_maes, label='Val MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.set_title('MAE over Epochs')
    plt.tight_layout()
    plt.savefig(os.path.join(config.output_dir, 'training_history.png'), dpi=150)
    plt.close()
    # Full‑image prediction
    logger.info("Generating full‑image prediction")
    pred_norm = full_image_prediction(model, X_img, config.patch_size, edge_index, config.device)
    # Denormalize to Kelvin then to Celsius
    lst_min, lst_max = lst.min(), lst.max()
    lst_pred_kelvin = denormalize(pred_norm, lst_min, lst_max)
    lst_pred_celsius = lst_pred_kelvin - 273.15
    lst_true_celsius = lst - 273.15
    abs_diff = np.abs(lst_pred_celsius - lst_true_celsius)
    # Save predicted array
    np.save(os.path.join(config.output_dir, "lst_pred_denorm_C.npy"), lst_pred_celsius)
    # Metrics
    metrics = compute_metrics(lst_true_celsius, lst_pred_celsius)
    logger.info(f"Overall metrics: R^2={metrics['R2']:.4f}, R={metrics['R']:.4f}, "
                f"RMSE={metrics['RMSE']:.4f}°C, MAE={metrics['MAE']:.4f}°C")
    with open(os.path.join(config.output_dir, "metrics_gcn.txt"), "w") as f:
        f.write(f"R: {metrics['R']:.4f}\n")
        f.write(f"R^2: {metrics['R2']:.4f}\n")
        f.write(f"RMSE: {metrics['RMSE']:.4f} °C\n")
        f.write(f"MAE: {metrics['MAE']:.4f} °C\n")
    # Visualization
    logger.info("Creating comparison plot")
    fig, axes = plt.subplots(1, 3, figsize=(20, 18))
    cmaps = ['YlOrRd', 'YlOrRd', 'coolwarm']
    titles = ("True LST (Landsat 8)", "Predicted LST (GCN)", "Absolute Difference")
    data_list = [lst_true_celsius, lst_pred_celsius, abs_diff]
    for ax, data, cmap, title in zip(axes, data_list, cmaps, titles):
        im = ax.imshow(data, cmap=cmap, extent=(lst_bounds.left, lst_bounds.right,
                                                lst_bounds.bottom, lst_bounds.top))
        ax.set_title(title, fontsize=22)
        # Only add coordinate ticks on the first subplot
        if ax == axes[0]:
            ax.set_xticks([lst_bounds.left, lst_bounds.right])
            ax.set_yticks([lst_bounds.bottom, lst_bounds.top])
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{abs(x):.2f}°{'E' if x>=0 else 'W'}"))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{abs(y):.2f}°{'N' if y>=0 else 'S'}"))
            ax.tick_params(axis='both', labelsize=22)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.04, aspect=15)
        cbar.set_label('')
        vmin, vmax = im.get_clim()
        ticks = np.linspace(vmin, vmax, 5)
        cbar.set_ticks(ticks)
        tick_labels = [f"{tick:.2f}" if title == "Absolute Difference" else f"{tick:.1f}" for tick in ticks]
        cbar.ax.set_xticklabels(tick_labels, fontsize=22)
        if title != "Absolute Difference":
            cbar.ax.text(1.02, 0.5, '°C', transform=cbar.ax.transAxes, va='center', ha='left', fontsize=22)
        else:
            cbar.ax.text(1.02, 0.5, 'ΔT (°C)', transform=cbar.ax.transAxes, va='center', ha='left', fontsize=22)
    plt.tight_layout()
    plt.savefig(os.path.join(config.output_dir, "prediction_gcn.png"), dpi=300, bbox_inches='tight')
    plt.close()
    # Save model
    model_path = os.path.join(config.output_dir, "gcn_model.pth")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
    logger.info("GCN training and evaluation completed successfully")

if __name__ == "__main__":
    cfg = Config()
    main(cfg)