"""
LST Prediction with U-Net using multi‑source remote sensing data
"""
import argparse
import logging
import warnings
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from shapely.geometry import Point
from tqdm import tqdm
# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="rasterio")

# Configuration
class Config:
    """Central configuration for paths and hyperparameters"""
    # Data paths
    lst_path = Path("/path_directory/LST_PR.tif")
    ndvi_path = Path("/path_directory/NDVI_PR.tif")
    emissivity_path = Path("/path_directory/EM_PR.tif")
    geojson_dir = Path("/path_directory/paris_GEE")
    output_dir = Path("/path_directory/UNET_model")
    # Urban feature file names
    urban_features = ['buildings', 'roads', 'rails', 'parks', 'water', 'trees']
    # Model hyperparameters
    patch_size = 7
    batch_size = 64
    epochs = 200
    learning_rate = 1e-3
    weight_decay = 0
    early_stopping_patience = 20
    scheduler_factor = 0.5
    scheduler_patience = 10
    # Training/validation split
    train_ratio = 0.8
    random_seed = 42
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # U-Net architecture
    unet_features = [64, 128, 256]
    # Logging
    log_level = logging.INFO

# Helper functions
def setup_logging(log_level: int) -> None:
    """Configure logging"""
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")

def load_raster(path: Path) -> Tuple[np.ndarray, rasterio.Affine, rasterio.coords.BoundingBox, str]:
    """
    Load a single‑band raster
    Returns:
        data (np.ndarray): 2D array of raster values
        transform: Geotransform
        bounds: Bounding box
        crs (str): Coordinate reference system
    """
    if not path.exists():
        raise FileNotFoundError(f"Raster not found: {path}")
    with rasterio.open(path) as src:
        data = src.read(1)
        transform = src.transform
        bounds = src.bounds
        crs = src.crs
    return data, transform, bounds, crs

def load_geojson(path: Path) -> gpd.GeoDataFrame:
    """Load a GeoJSON file"""
    if not path.exists():
        raise FileNotFoundError(f"GeoJSON not found: {path}")
    return gpd.read_file(path)

def normalize(data: np.ndarray) -> np.ndarray:
    """Min‑max normalization to [0,1]"""
    min_val = np.nanmin(data)
    max_val = np.nanmax(data)
    if max_val - min_val == 0:
        return np.zeros_like(data)
    return (data - min_val) / (max_val - min_val)

def geojson_to_array(geojson_path: Path, x_coords: np.ndarray, y_coords: np.ndarray) -> np.ndarray:
    """
    Convert GeoJSON polygon features to a binary array aligned with the coordinate grid
    Args:
        geojson_path: Path to GeoJSON file
        x_coords: 2D array of x coordinates
        y_coords: 2D array of y coordinates
    Returns:
        Boolean array of same shape as x_coords, True where point lies inside any feature
    """
    gdf = load_geojson(geojson_path)
    points = [Point(x, y) for x, y in zip(x_coords.ravel(), y_coords.ravel())]
    inside = np.array([gdf.contains(p).any() for p in points])
    return inside.reshape(x_coords.shape)

def extract_patch_center(mask: np.ndarray, patch_size: int) -> np.ndarray:
    """
    Return a boolean mask where a patch of given size can be extracted
    centered at each pixel
    """
    h, w = mask.shape
    offset = patch_size // 2
    valid_center = np.zeros_like(mask, dtype=bool)
    valid_center[offset:h - offset, offset:w - offset] = True
    return valid_center & mask

# Dataset class
class LSTPatchDataset(Dataset):
    """PyTorch dataset for extracting patches from multi‑channel input and target images"""
    def __init__(self, centers: np.ndarray, X_img: np.ndarray, Y_img: np.ndarray, patch_size: int):
        """
        Args:
            centers: N x 2 array of (row, col) center coordinates
            X_img: Input image of shape (H, W, C)
            Y_img: Target image of shape (H, W)
            patch_size: Size of square patch
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
        # Convert to tensor and rearrange to (C, H, W)
        patch = torch.from_numpy(patch).permute(2, 0, 1).float()
        target = torch.from_numpy(target).unsqueeze(0).float()
        return patch, target

# U-Net model
class UNet(nn.Module):
    """Simple U‑Net architecture for patch‑based regression"""
    def __init__(self, in_channels: int, out_channels: int = 1, features: list = [64, 128, 256]):
        super().__init__()
        self.enc1 = self._block(in_channels, features[0])
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self._block(features[0], features[1])
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self._block(features[1], features[2])
        self.bottleneck = self._block(features[2], features[2] * 2)
        # Upsampling using bilinear interpolation
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = self._block(features[2] * 2 + features[1], features[2])
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = self._block(features[2] + features[0], features[1])
        self.final = nn.Conv2d(features[1], out_channels, 1)
    @staticmethod
    def _block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        # Bottleneck
        b = self.bottleneck(e3)
        # Decoder
        d2 = self.up2(b)
        if d2.shape != e2.shape:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        if d1.shape != e1.shape:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.final(d1)

# Training and evaluation functions
def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                    optimizer: torch.optim.Optimizer, device: torch.device) -> Tuple[float, float]:
    """Train the model for one epoch and return average loss and MAE"""
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    n_samples = 0
    for batch_X, batch_y in tqdm(loader, desc="Training", leave=False):
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        batch_size = batch_X.size(0)
        total_loss += loss.item() * batch_size
        total_mae += torch.abs(outputs - batch_y).sum().item()
        n_samples += batch_size
    return total_loss / n_samples, total_mae / n_samples

def validate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             device: torch.device) -> Tuple[float, float]:
    """Validate the model and return average loss and MAE"""
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    n_samples = 0
    with torch.no_grad():
        for batch_X, batch_y in tqdm(loader, desc="Validation", leave=False):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            batch_size = batch_X.size(0)
            total_loss += loss.item() * batch_size
            total_mae += torch.abs(outputs - batch_y).sum().item()
            n_samples += batch_size
    return total_loss / n_samples, total_mae / n_samples

def evaluate_full_image(model: nn.Module, X_img: np.ndarray, Y_img: np.ndarray,
                        lst_min: float, lst_max: float, device: torch.device,
                        output_dir: Path) -> Tuple[np.ndarray, dict]:
    """
    Predict on the entire image and compute global metrics
    Returns:
        pred_celsius: Predicted LST in Celsius (H, W)
        metrics: Dictionary with R^2, R, RMSE, MAE
    """
    model.eval()
    # Convert full image to tensor (1, C, H, W)
    X_tensor = torch.from_numpy(X_img).permute(2, 0, 1).unsqueeze(0).float().to(device)
    with torch.no_grad():
        pred_norm = model(X_tensor).cpu().numpy()[0, 0, :, :]  # (H, W)
    # Denormalize
    pred_celsius = pred_norm * (lst_max - lst_min) + lst_min - 273.15
    true_celsius = Y_img - 273.15
    # Compute metrics
    true_flat = true_celsius.ravel()
    pred_flat = pred_celsius.ravel()
    ss_res = np.sum((true_flat - pred_flat) ** 2)
    ss_tot = np.sum((true_flat - np.mean(true_flat)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
    correlation = np.corrcoef(true_flat, pred_flat)[0, 1]
    rmse = np.sqrt(np.mean((true_flat - pred_flat) ** 2))
    mae = np.mean(np.abs(true_flat - pred_flat))
    metrics = {
        'R^2': r2,
        'R': correlation,
        'RMSE': rmse,
        'MAE': mae}
    # Save metrics
    with open(output_dir / "metrics_unet.txt", "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")
    # Save predicted array
    np.save(output_dir / "lst_pred_denorm_C.npy", pred_celsius)
    return pred_celsius, metrics

def plot_results(true: np.ndarray, pred: np.ndarray, diff: np.ndarray,
                 bounds: rasterio.coords.BoundingBox, output_path: Path) -> None:
    """Create a three‑panel comparison plot"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 18))
    titles = ("True LST (Landsat 8)", "Predicted LST (U-Net)", "Absolute Difference")
    cmaps = ('YlOrRd', 'YlOrRd', 'coolwarm')
    for ax, data, cmap, title in zip(axes, [true, pred, diff], cmaps, titles):
        im = ax.imshow(data, cmap=cmap, extent=(bounds.left, bounds.right, bounds.bottom, bounds.top))
        ax.set_title(title, fontsize=22)
        if title == titles[0]:
            # Show coordinate ticks only for the true map
            ax.set_xticks([bounds.left, bounds.right])
            ax.set_yticks([bounds.bottom, bounds.top])
            ax.xaxis.set_major_formatter(
                FuncFormatter(lambda x, _: f"{abs(x):.2f}°{'E' if x >= 0 else 'W'}"))
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda y, _: f"{abs(y):.2f}°{'N' if y >= 0 else 'S'}"))
            ax.tick_params(axis='x', pad=15, labelsize=22)
            ax.tick_params(axis='y', labelsize=22)
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
        # Add units
        if title != titles[2]:
            cbar.ax.text(1.02, 0.5, '°C', transform=cbar.ax.transAxes,
                         va='center', ha='left', fontsize=22)
        else:
            cbar.ax.text(1.02, 0.5, 'ΔT (°C)', transform=cbar.ax.transAxes,
                         va='center', ha='left', fontsize=22)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# Main pipeline
def main(config: Config) -> None:
    """Run the complete workflow"""
    setup_logging(config.log_level)
    logger = logging.getLogger(__name__)
    logger.info("Starting LST prediction pipeline")
    logger.info(f"Using device: {config.device}")
    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)
    # Load rasters and extract metadata
    logger.info("Loading rasters")
    lst, lst_transform, lst_bounds, lst_crs = load_raster(config.lst_path)
    ndvi, _, ndvi_bounds, _ = load_raster(config.ndvi_path)
    emissivity, _, emissivity_bounds, _ = load_raster(config.emissivity_path)
    # Generate coordinate grid (from LST)
    height, width = lst.shape
    x_coords, y_coords = np.meshgrid(
        np.arange(width) * lst_transform[0] + lst_transform[2],
        np.arange(height) * lst_transform[4] + lst_transform[5])
    # Load urban features from GeoJSON
    logger.info("Loading urban features")
    urban_arrays = {}
    for feature in config.urban_features:
        geojson_path = config.geojson_dir / f"{feature}.geojson"
        urban_arrays[feature] = geojson_to_array(geojson_path, x_coords, y_coords)
    # Normalize continuous features
    ndvi_min, ndvi_max = np.min(ndvi), np.max(ndvi)
    emissivity_min, emissivity_max = np.min(emissivity), np.max(emissivity)
    logger.info(f"NDVI range: [{ndvi_min}, {ndvi_max}]")
    logger.info(f"Emissivity range: [{emissivity_min}, {emissivity_max}]")
    x_coords_norm = normalize(x_coords)
    y_coords_norm = normalize(y_coords)
    ndvi_norm = normalize(ndvi)
    emissivity_norm = normalize(emissivity)
    # Stack features: (H, W, C)
    X_img = np.stack([
        x_coords_norm, y_coords_norm,
        ndvi_norm, emissivity_norm,
        urban_arrays['buildings'].astype(np.float32),
        urban_arrays['roads'].astype(np.float32),
        urban_arrays['rails'].astype(np.float32),
        urban_arrays['parks'].astype(np.float32),
        urban_arrays['water'].astype(np.float32),
        urban_arrays['trees'].astype(np.float32)], axis=-1)
    # Normalize target
    Y_img = normalize(lst).astype(np.float32)
    lst_min, lst_max = np.min(lst), np.max(lst)
    logger.info(f"Input shape: {X_img.shape}")
    logger.info(f"Target shape: {Y_img.shape}")
    # Train/test split (pixel indices)
    np.random.seed(config.random_seed)
    pixel_indices = np.arange(height * width)
    np.random.shuffle(pixel_indices)
    split = int(config.train_ratio * len(pixel_indices))
    train_pixels = pixel_indices[:split]
    test_pixels = pixel_indices[split:]
    train_mask = np.zeros((height, width), dtype=bool)
    test_mask = np.zeros((height, width), dtype=bool)
    train_mask.flat[train_pixels] = True
    test_mask.flat[test_pixels] = True
    logger.info(f"Training pixels: {train_mask.sum()}")
    logger.info(f"Test pixels: {test_mask.sum()}")
    # Extract valid patch centers
    train_center_mask = extract_patch_center(train_mask, config.patch_size)
    test_center_mask = extract_patch_center(test_mask, config.patch_size)
    train_centers = np.argwhere(train_center_mask)
    test_centers = np.argwhere(test_center_mask)
    logger.info(f"Training patches: {len(train_centers)}")
    logger.info(f"Test patches: {len(test_centers)}")
    # Create datasets and dataloaders
    train_dataset = LSTPatchDataset(train_centers, X_img, Y_img, config.patch_size)
    test_dataset = LSTPatchDataset(test_centers, X_img, Y_img, config.patch_size)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)
    # Model, optimizer, scheduler, criterion
    model = UNet(in_channels=X_img.shape[-1], out_channels=1,
                 features=config.unet_features).to(config.device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate,
                           weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=config.scheduler_factor,
        patience=config.scheduler_patience, verbose=True)
    # Training loop with early stopping
    logger.info("Starting training")
    train_losses, val_losses = [], []
    train_maes, val_maes = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = config.output_dir / "best_model.pth"
    for epoch in range(1, config.epochs + 1):
        logger.info(f"Epoch {epoch}/{config.epochs}")
        train_loss, train_mae = train_one_epoch(model, train_loader, criterion,
                                                optimizer, config.device)
        val_loss, val_mae = validate(model, test_loader, criterion, config.device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_maes.append(train_mae)
        val_maes.append(val_mae)
        logger.info(
            f"Train Loss: {train_loss:.6f}, Train MAE: {train_mae:.6f} | "
            f"Val Loss: {val_loss:.6f}, Val MAE: {val_mae:.6f}")
        # Learning rate scheduling
        scheduler.step(val_loss)
        # Early stopping and checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"New best model saved (val_loss = {val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
    # Load best model for final evaluation
    model.load_state_dict(torch.load(best_model_path))
    logger.info(f"Best model loaded from {best_model_path}")
    # Save training history
    history_df = pd.DataFrame({
        'loss': train_losses,
        'val_loss': val_losses,
        'mae': train_maes,
        'val_mae': val_maes})
    history_df.to_csv(config.output_dir / "training_history.csv", index=False)
    # Plot history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.title('Loss over Epochs')
    #----
    plt.subplot(1, 2, 2)
    plt.plot(train_maes, label='Train MAE')
    plt.plot(val_maes, label='Val MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.title('MAE over Epochs')
    plt.tight_layout()
    plt.savefig(config.output_dir / "training_history.png", dpi=150)
    plt.close()
    # Full image prediction and evaluation
    pred_celsius, metrics = evaluate_full_image(
        model, X_img, Y_img, lst_min, lst_max, config.device, config.output_dir)
    logger.info("Full image evaluation metrics:")
    for k, v in metrics.items():
        logger.info(f"{k}: {v:.4f}")
    # Visualization
    true_celsius = lst - 273.15
    abs_diff = np.abs(pred_celsius - true_celsius)
    plot_results(true_celsius, pred_celsius, abs_diff, lst_bounds,
                 config.output_dir / "prediction_unet.png")
    # Save final model (full state dict)
    torch.save(model.state_dict(), config.output_dir / "unet_model.pth")
    logger.info("Pipeline finished successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LST prediction with U-Net")
    args = parser.parse_args()
    cfg = Config()
    main(cfg)