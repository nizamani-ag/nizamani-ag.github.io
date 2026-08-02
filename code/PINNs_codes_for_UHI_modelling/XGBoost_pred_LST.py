"""
XGBoost-based land surface temperature (LST) prediction using satellite data
and urban features (e.g., buildings, roads). The script loads raster data,
extracts coordinates, normalizes features, trains an XGBoost model, evaluates
it, and produces visualizations and feature importance plots
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import geopandas as gpd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import xgboost as xgb
from matplotlib.ticker import FuncFormatter
from rasterio.transform import Affine
from shapely.geometry import Point
from sklearn.metrics import mean_squared_error, r2_score

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

# Helper functions
def load_raster(path: Path) -> Tuple[np.ndarray, Affine, rasterio.coords.BoundingBox, str]:
    """
    Load a single-band raster file
    Args:
        path: Path to the raster file (e.g., GeoTIFF)
    Returns:
        Tuple containing:
            - data: 2D array of raster values.
            - transform: Affine transformation of the raster
            - bounds: BoundingBox object (left, bottom, right, top)
            - crs: Coordinate reference system as a string
    """
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        transform = src.transform
        bounds = src.bounds
        crs = src.crs.to_string()
    return data, transform, bounds, crs

def load_geojson(path: Path) -> gpd.GeoDataFrame:
    """Load a GeoJSON file into a GeoDataFrame"""
    return gpd.read_file(path)

def geojson_to_array(
    geojson_path: Path, x_coords: np.ndarray, y_coords: np.ndarray) -> np.ndarray:
    """
    Convert a GeoJSON of polygons to a binary array indicating whether each point
    lies inside any polygon
    Args:
        geojson_path: Path to the GeoJSON file
        x_coords: 2D array of x‑coordinates (e.g., from meshgrid)
        y_coords: 2D array of y‑coordinates
    Returns:
        2D binary array with 1 where point is inside any polygon, 0 otherwise
    """
    gdf = gpd.read_file(geojson_path)
    points = [Point(x, y) for x, y in zip(x_coords.ravel(), y_coords.ravel())]
    inside = np.array([gdf.contains(p).any() for p in points])
    return inside.reshape(x_coords.shape)

def normalize(data: np.ndarray) -> np.ndarray:
    """Normalize a 2D array to [0, 1] using min‑max scaling, ignoring NaNs"""
    min_val, max_val = np.nanmin(data), np.nanmax(data)
    if max_val == min_val:
        return np.zeros_like(data)
    return (data - min_val) / (max_val - min_val)

def denormalize(data_norm: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """Convert normalized data back to original scale"""
    return data_norm * (max_val - min_val) + min_val

# Data preparation
def prepare_data(
    lst_path: Path,
    ndvi_path: Path,
    emissivity_path: Path,
    urban_feature_paths: Dict[str, Path]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load all inputs, create coordinate grid, build feature matrix and target vector
    Args:
        lst_path: Path to LST raster (Kelvin)
        ndvi_path: Path to NDVI raster
        emissivity_path: Path to emissivity raster
        urban_feature_paths: Dictionary mapping feature names to GeoJSON paths
    Returns:
        Tuple of:
            - inputs: Feature matrix (N_samples x N_features)
            - targets: Normalized LST target vector (1D)
            - metadata: Dictionary containing:
                - lst_transform: Affine transform of LST
                - lst_bounds: BoundingBox of LST
                - lst_min, lst_max: Min/max LST values (original scale)
                - ndvi_min, ndvi_max: Min/max NDVI values (original)
                - emissivity_min, emissivity_max: Min/max emissivity values
                - x_coords, y_coords: 2D coordinate arrays
                - shape: (rows, cols) of the raster
    """
    # Load rasters
    lst, lst_transform, lst_bounds, lst_crs = load_raster(lst_path)
    ndvi, _, ndvi_bounds, _ = load_raster(ndvi_path)
    emissivity, _, emissivity_bounds, _ = load_raster(emissivity_path)
    # Store normalization parameters
    ndvi_min, ndvi_max = np.min(ndvi), np.max(ndvi)
    emissivity_min, emissivity_max = np.min(emissivity), np.max(emissivity)
    logger.info(f"NDVI range: [{ndvi_min:.4f}, {ndvi_max:.4f}]")
    logger.info(f"Emissivity range: [{emissivity_min:.4f}, {emissivity_max:.4f}]")
    # Generate coordinate grid (same extent as LST)
    rows, cols = lst.shape
    x_coords = np.arange(cols) * lst_transform[0] + lst_transform[2]
    y_coords = np.arange(rows) * lst_transform[4] + lst_transform[5]
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    # Load urban features
    urban_arrays = {}
    for name, path in urban_feature_paths.items():
        logger.info(f"Loading urban feature: {name} from {path}")
        urban_arrays[name] = geojson_to_array(path, x_grid, y_grid)
    # Build feature matrix
    features = [
        normalize(x_grid).ravel(),
        normalize(y_grid).ravel(),
        normalize(ndvi).ravel(),
        normalize(emissivity).ravel()]
    for name in urban_feature_paths.keys():
        features.append(urban_arrays[name].ravel())
    inputs = np.column_stack(features)
    targets = normalize(lst).ravel()
    metadata = {
        "lst_transform": lst_transform,
        "lst_bounds": lst_bounds,
        "lst_min": np.min(lst),
        "lst_max": np.max(lst),
        "ndvi_min": ndvi_min,
        "ndvi_max": ndvi_max,
        "emissivity_min": emissivity_min,
        "emissivity_max": emissivity_max,
        "x_coords": x_grid,
        "y_coords": y_grid,
        "shape": (rows, cols),
        "feature_names": ["X_coord", "Y_coord", "NDVI", "Emissivity"] + list(urban_feature_paths.keys())}
    return inputs, targets, metadata

# Model training and evaluation
def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    params: Dict[str, Any],
    random_state: int = 42) -> Tuple[xgb.XGBRegressor, Dict[str, float]]:
    """
    Train an XGBoost regressor and evaluate on test set
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        params: XGBoost hyperparameters (n_estimators, max_depth, learning_rate)
        random_state: Seed for reproducibility
    Returns:
        Tuple of:
            - trained model (XGBRegressor)
            - dictionary with training and test metrics (rmse, r2, pearson_r)
    """
    model = xgb.XGBRegressor(**params, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    # Training metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_r2 = r2_score(y_train, y_pred_train)
    train_r = np.corrcoef(y_train, y_pred_train)[0, 1]
    # Test metrics
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_r2 = r2_score(y_test, y_pred_test)
    test_r = np.corrcoef(y_test, y_pred_test)[0, 1]
    metrics = {
        "train_rmse": train_rmse,
        "train_r2": train_r2,
        "train_r": train_r,
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "test_r": test_r}
    logger.info(f"Training RMSE: {train_rmse:.4f}, R^2: {train_r2:.4f}, R: {train_r:.4f}")
    logger.info(f"Testing RMSE:  {test_rmse:.4f}, R^2: {test_r2:.4f}, R: {test_r:.4f}")
    return model, metrics

# Visualization
def plot_comparison(
    true_lst: np.ndarray,
    pred_lst: np.ndarray,
    abs_diff: np.ndarray,
    bounds: rasterio.coords.BoundingBox,
    output_path: Path,
    titles: Tuple[str, str, str] = ("True LST (Landsat 8)", "Predicted LST (XGBoost)", "Absolute Difference"),
    cmap_abs_diff: str = "coolwarm") -> None:
    """
    Create a side‑by‑side plot comparing true LST, predicted LST, and absolute difference
    Args:
        true_lst: True LST in Celsius
        pred_lst: Predicted LST in Celsius
        abs_diff: Absolute difference (|true - pred|)
        bounds: BoundingBox of the raster (left, bottom, right, top)
        output_path: Where to save the figure (png)
        titles: Titles for the three subplots
        cmap_abs_diff: Colormap for the absolute difference plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 18))
    cmaps = ["YlOrRd", "YlOrRd", cmap_abs_diff]
    for i, (ax, data, cmap, title) in enumerate(zip(axes, [true_lst, pred_lst, abs_diff], cmaps, titles)):
        im = ax.imshow(data, cmap=cmap, extent=(bounds.left, bounds.right, bounds.bottom, bounds.top))
        ax.set_title(title, fontsize=22)
        if i == 0:  # first subplot gets ticks and labels
            ax.set_xticks([bounds.left, bounds.right])
            ax.set_yticks([bounds.bottom, bounds.top])
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{abs(x):.2f}°{'E' if x >= 0 else 'W'}"))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{abs(y):.2f}°{'N' if y >= 0 else 'S'}"))
            ax.tick_params(axis="x", pad=15)
            ax.tick_params(axis="both", labelsize=22)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.04, aspect=15)
        cbar.set_label("")
        vmin, vmax = im.get_clim()
        ticks = np.linspace(vmin, vmax, 5)
        cbar.set_ticks(ticks)
        tick_labels = [f"{tick:.2f}" if i == 2 else f"{tick:.1f}" for tick in ticks]
        cbar.ax.set_xticklabels(tick_labels)
        cbar.ax.tick_params(labelsize=22)
        # Unit label
        label = "ΔT (°C)" if i == 2 else "°C"
        cbar.ax.text(1.02, 0.5, label, transform=cbar.ax.transAxes, va="center", ha="left", fontsize=22)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Comparison plot saved to {output_path}")

def plot_feature_importance(
    importance_scores: np.ndarray,
    feature_names: List[str],
    output_path: Path) -> None:
    """
    Plot and save feature importance bar chart
    Args:
        importance_scores: Array of importance values
        feature_names: Corresponding feature names
        output_path: Where to save the figure
    """
    plt.figure(figsize=(12, 8))
    indices = np.argsort(importance_scores)[::-1]
    plt.bar(range(len(importance_scores)), importance_scores[indices])
    plt.xticks(range(len(importance_scores)), [feature_names[i] for i in indices], rotation=45)
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Feature importance plot saved to {output_path}")

def save_metrics(metrics: Dict[str, float], output_path: Path) -> None:
    """
    Save evaluation metrics to a text file
    Args:
        metrics: Dictionary with keys 'train_rmse', 'train_r2', 'train_r',
                 'test_rmse', 'test_r2', 'test_r', and optionally others
        output_path: Path to the output text file
    """
    with open(output_path, "w") as f:
        f.write("Model Performance Metrics\n")
        f.write(f"Training RMSE: {metrics['train_rmse']:.4f}\n")
        f.write(f"Training R^2:   {metrics['train_r2']:.4f}\n")
        f.write(f"Training R:    {metrics['train_r']:.4f}\n")
        f.write(f"Testing RMSE:  {metrics['test_rmse']:.4f}\n")
        f.write(f"Testing R^2:    {metrics['test_r2']:.4f}\n")
        f.write(f"Testing R:     {metrics['test_r']:.4f}\n")
    logger.info(f"Metrics saved to {output_path}")

# Main pipeline
def main(args: argparse.Namespace) -> None:
    """Run the entire LST prediction pipeline"""
    # Set random seeds for reproducibility
    np.random.seed(args.random_seed)
    # Resolve input paths
    lst_path = Path(args.lst_path)
    ndvi_path = Path(args.ndvi_path)
    emissivity_path = Path(args.emissivity_path)
    urban_feature_paths = {
        "Buildings": Path(args.buildings_path),
        "Roads": Path(args.roads_path),
        "Rails": Path(args.rails_path),
        "Parks": Path(args.parks_path),
        "Water": Path(args.water_path),
        "Trees": Path(args.trees_path)}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Data preparation
    logger.info("Loading and preparing data")
    inputs, targets, metadata = prepare_data(lst_path, ndvi_path, emissivity_path, urban_feature_paths)
    # Train/test split (80/20)
    n_samples = len(inputs)
    indices = np.random.permutation(n_samples)
    split_idx = int(0.8 * n_samples)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    X_train, X_test = inputs[train_indices], inputs[test_indices]
    y_train, y_test = targets[train_indices], targets[test_indices]
    logger.info(f"Training samples: {X_train.shape[0]}")
    logger.info(f"Testing samples:  {X_test.shape[0]}")
    # Model training
    logger.info("Training XGBoost model")
    xgb_params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree}
    model, metrics = train_xgboost(X_train, y_train, X_test, y_test, xgb_params, args.random_seed)
    # Full prediction and evaluation
    logger.info("Predicting on entire dataset")
    lst_pred_norm = model.predict(inputs)
    lst_pred = lst_pred_norm.reshape(metadata["shape"])
    lst_original = denormalize(lst_pred, metadata["lst_min"], metadata["lst_max"])
    # Convert to Celsius
    lst_true_c = metadata["lst_min"] - 273.15
    lst_pred_c = lst_original - 273.15
    abs_diff_c = np.abs(lst_true_c - lst_pred_c)
    # Save predicted LST as numpy array
    np.save(output_dir / "lst_pred_celsius.npy", lst_pred_c)
    logger.info(f"Predicted LST array saved to {output_dir / 'lst_pred_celsius.npy'}")
    # Compute full‑dataset metrics (in Celsius)
    true_flat = lst_true_c.ravel()
    pred_flat = lst_pred_c.ravel()
    rmse = np.sqrt(np.mean((true_flat - pred_flat) ** 2))
    mae = np.mean(np.abs(true_flat - pred_flat))
    r2 = r2_score(true_flat, pred_flat)
    r = np.corrcoef(true_flat, pred_flat)[0, 1]
    logger.info(f"Full dataset RMSE: {rmse:.4f}°C, MAE: {mae:.4f}°C, R²: {r2:.4f}, R: {r:.4f}")
    # Save extended metrics
    metrics.update({
        "full_rmse_c": rmse,
        "full_mae_c": mae,
        "full_r2": r2,
        "full_r": r})
    save_metrics(metrics, output_dir / "metrics.txt")
    # Feature importance
    importance = model.feature_importances_
    feature_names = metadata["feature_names"]
    plot_feature_importance(importance, feature_names, output_dir / "feature_importance.png")
    # Log top features
    sorted_idx = np.argsort(importance)[::-1]
    logger.info("Feature importance (top 5):")
    for i in sorted_idx[:5]:
        logger.info(f"  {feature_names[i]}: {importance[i]:.4f}")
    # Visualization
    plot_comparison(
        lst_true_c,
        lst_pred_c,
        abs_diff_c,
        metadata["lst_bounds"],
        output_dir / "prediction_comparison.png")
    # Save model and metadata
    joblib.dump(model, output_dir / "xgboost_model.pkl")
    joblib.dump((feature_names, importance), output_dir / "feature_importance.pkl")
    logger.info(f"Model saved to {output_dir / 'xgboost_model.pkl'}")
    logger.info("XGBoost training and evaluation completed successfully")

# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict LST using XGBoost and urban features")
    parser.add_argument("--lst_path", required=True, help="Path to LST raster (GeoTIFF)")
    parser.add_argument("--ndvi_path", required=True, help="Path to NDVI raster")
    parser.add_argument("--emissivity_path", required=True, help="Path to emissivity raster")
    parser.add_argument("--buildings_path", required=True, help="GeoJSON of buildings")
    parser.add_argument("--roads_path", required=True, help="GeoJSON of roads")
    parser.add_argument("--rails_path", required=True, help="GeoJSON of rails")
    parser.add_argument("--parks_path", required=True, help="GeoJSON of parks")
    parser.add_argument("--water_path", required=True, help="GeoJSON of water bodies")
    parser.add_argument("--trees_path", required=True, help="GeoJSON of trees")
    parser.add_argument("--output_dir", required=True, help="Directory to save outputs")
    # XGBoost hyperparameters
    parser.add_argument("--n_estimators", type=int, default=1000, help="Number of trees")
    parser.add_argument("--max_depth", type=int, default=8, help="Maximum tree depth")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--subsample", type=float, default=0.8, help="Subsample ratio")
    parser.add_argument("--colsample_bytree", type=float, default=0.8, help="Column subsample ratio")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    main(args)