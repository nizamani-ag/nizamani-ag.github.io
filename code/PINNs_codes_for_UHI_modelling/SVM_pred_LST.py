import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import joblib
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.ticker import FuncFormatter
from shapely.geometry import Point
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
import geopandas as gpd

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# Configuration
@dataclass
class Config:
    """Configuration parameters for the SVM training pipeline"""
    # Paths
    lst_raster: Path = Path("/path_directory/LST_PR.tif")
    ndvi_raster: Path = Path("/path_directory/NDVI_PR.tif")
    emissivity_raster: Path = Path("/path_directory/EM_PR.tif")
    geojson_dir: Path = Path("/path_directory/paris_GEE/")
    output_dir: Path = Path("/path_directory/SVM_model/")
    # Urban feature GeoJSON files (relative to geojson_dir)
    urban_features: Dict[str, str] = None
    # SVM hyperparameters
    svm_kernel: str = "rbf" # change kernel type: "linear" or "poly"
    svm_C: float = 100.0
    svm_epsilon: float = 0.1
    svm_gamma: str = "scale"
    # Data split
    train_ratio: float = 0.8
    random_seed: int = 42
    def __post_init__(self):
        if self.urban_features is None:
            self.urban_features = {
                "buildings": "buildings.geojson",
                "roads": "roads.geojson",
                "rails": "rails.geojson",
                "parks": "parks.geojson",
                "water": "water.geojson",
                "trees": "trees.geojson"}
        self.output_dir.mkdir(parents=True, exist_ok=True)

# Data Loaders
class DataLoader:
    """Load raster and vector data with consistency checks"""
    @staticmethod
    def load_raster(path: Path) -> Tuple[np.ndarray, rasterio.transform.Affine, rasterio.coords.BoundingBox, rasterio.crs.CRS]:
        """Load a single-band raster and its metadata"""
        if not path.exists():
            raise FileNotFoundError(f"Raster not found: {path}")
        with rasterio.open(path) as src:
            data = src.read(1)
            transform = src.transform
            bounds = src.bounds
            crs = src.crs
        logger.info(f"Loaded {path}: shape {data.shape}, bounds {bounds}, CRS {crs}")
        return data, transform, bounds, crs
    @staticmethod
    def load_geojson(path: Path) -> gpd.GeoDataFrame:
        """Load a GeoJSON file"""
        if not path.exists():
            raise FileNotFoundError(f"GeoJSON not found: {path}")
        gdf = gpd.read_file(path)
        logger.info(f"Loaded {path} with {len(gdf)} features")
        return gdf
    @staticmethod
    def check_raster_consistency(raster_list: list) -> None:
        """Ensure all rasters have the same shape, bounds and CRS"""
        if len(raster_list) < 2:
            return
        first = raster_list[0]
        for r in raster_list[1:]:
            if r[0].shape != first[0].shape:
                raise ValueError("Raster shapes differ")
            if r[1] != first[1]:
                raise ValueError("Raster transforms differ")
            if r[2] != first[2]:
                raise ValueError("Raster bounds differ")
            if r[3] != first[3]:
                raise ValueError("Raster CRS differ")
        logger.info("Raster consistency check passed")

class UrbanFeaturesExtractor:
    """Extract binary masks from GeoJSONs using a coordinate grid"""
    def __init__(self, geojson_dir: Path, x_coords: np.ndarray, y_coords: np.ndarray):
        self.geojson_dir = Path(geojson_dir)
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.shape = x_coords.shape
    def geojson_to_array(self, filename: str) -> np.ndarray:
        """Convert a GeoJSON file to a binary array based on point containment"""
        path = self.geojson_dir / filename
        gdf = DataLoader.load_geojson(path)
        # Project points if CRS differs
        points = [Point(x, y) for x, y in zip(self.x_coords.ravel(), self.y_coords.ravel())]
        contained = np.array([gdf.contains(p).any() for p in points])
        return contained.reshape(self.shape).astype(np.float32)
    def extract_all(self, feature_files: Dict[str, str]) -> Dict[str, np.ndarray]:
        """Extract all requested features"""
        features = {}
        for name, filename in feature_files.items():
            features[name] = self.geojson_to_array(filename)
            logger.info(f"Extracted {name} feature, shape {features[name].shape}")
        return features

# Preprocessing Utilities
def normalize(data: np.ndarray, min_val: Optional[float] = None, max_val: Optional[float] = None) -> Tuple[np.ndarray, float, float]:
    """Min-max normalize array to [0,1]. Returns normalized array, min, max"""
    if min_val is None:
        min_val = np.nanmin(data)
    if max_val is None:
        max_val = np.nanmax(data)
    if max_val == min_val:
        logger.warning("Constant array detected, returning zeros")
        return np.zeros_like(data), min_val, max_val
    norm = (data - min_val) / (max_val - min_val)
    return norm, min_val, max_val

def build_coordinate_grid(transform: rasterio.transform.Affine, shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Generate coordinate arrays for each pixel"""
    rows, cols = shape
    x_coords = transform[2] + np.arange(cols) * transform[0]
    y_coords = transform[5] + np.arange(rows) * transform[4]
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    return x_grid, y_grid

# SVM Model Trainer
class SVMTrainer:
    """Train and evaluate an SVR model"""
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred_train = None
        self.y_pred_test = None
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Split data and train the SVM"""
        logger.info("Splitting data into train/test sets")
        np.random.seed(self.config.random_seed)
        indices = np.random.permutation(len(X))
        split_idx = int(self.config.train_ratio * len(X))
        train_idx, test_idx = indices[:split_idx], indices[split_idx:]
        self.X_train = X[train_idx]
        self.X_test = X[test_idx]
        self.y_train = y[train_idx]
        self.y_test = y[test_idx]
        logger.info(f"Training set size: {self.X_train.shape[0]}")
        logger.info(f"Testing set size: {self.X_test.shape[0]}")
        # Create SVM regressor
        self.model = SVR(
            kernel=self.config.svm_kernel,
            C=self.config.svm_C,
            epsilon=self.config.svm_epsilon,
            gamma=self.config.svm_gamma,
            verbose=False)
        logger.info(f"Training SVM with kernel={self.config.svm_kernel}, C={self.config.svm_C}, epsilon={self.config.svm_epsilon}")
        self.model.fit(self.X_train, self.y_train)
        # Predictions
        self.y_pred_train = self.model.predict(self.X_train)
        self.y_pred_test = self.model.predict(self.X_test)

    def evaluate(self) -> Dict[str, float]:
        """Compute metrics and return them as dict"""
        metrics = {}
        metrics["train_rmse"] = np.sqrt(mean_squared_error(self.y_train, self.y_pred_train))
        metrics["train_r2"] = r2_score(self.y_train, self.y_pred_train)
        metrics["train_r"] = np.corrcoef(self.y_train, self.y_pred_train)[0, 1]
        metrics["test_rmse"] = np.sqrt(mean_squared_error(self.y_test, self.y_pred_test))
        metrics["test_r2"] = r2_score(self.y_test, self.y_pred_test)
        metrics["test_r"] = np.corrcoef(self.y_test, self.y_pred_test)[0, 1]
        logger.info("Training metrics: RMSE=%.4f, R^2=%.4f, R=%.4f",
                    metrics["train_rmse"], metrics["train_r2"], metrics["train_r"])
        logger.info("Testing metrics:  RMSE=%.4f, R^2=%.4f, R=%.4f",
                    metrics["test_rmse"], metrics["test_r2"], metrics["test_r"])
        return metrics

    def predict_full(self, X_full: np.ndarray) -> np.ndarray:
        """Predict on the entire dataset"""
        return self.model.predict(X_full)

# Visualization and Saving
class Visualizer:
    """Create comparison plots and save metrics"""
    @staticmethod
    def plot_comparison(
        true: np.ndarray,
        pred: np.ndarray,
        abs_diff: np.ndarray,
        bounds: rasterio.coords.BoundingBox,
        titles: Tuple[str, str, str],
        output_path: Path,
        cmap_abs_diff: str = "coolwarm") -> None:
        """Generate three-panel comparison plot"""
        fig, axes = plt.subplots(1, 3, figsize=(20, 18))
        cmaps = ["YlOrRd", "YlOrRd", cmap_abs_diff]
        data_list = [true, pred, abs_diff]
        for ax, data, cmap, title in zip(axes, data_list, cmaps, titles):
            im = ax.imshow(data, cmap=cmap, extent=(bounds.left, bounds.right, bounds.bottom, bounds.top))
            ax.set_title(title, fontsize=22)
            if ax == axes[0]:  # first subplot
                ax.set_xticks([bounds.left, bounds.right])
                ax.set_yticks([bounds.bottom, bounds.top])
                ax.xaxis.set_major_formatter(FuncFormatter(
                    lambda x, _: f"{abs(x):.2f}°{'E' if x >= 0 else 'W'}"))
                ax.yaxis.set_major_formatter(FuncFormatter(
                    lambda y, _: f"{abs(y):.2f}°{'N' if y >= 0 else 'S'}"))
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
            tick_labels = [f"{tick:.2f}" if title.startswith("Absolute") else f"{tick:.1f}" for tick in ticks]
            cbar.ax.set_xticklabels(tick_labels, fontsize=22)
            if title in ("True LST (Landsat 8)", "Predicted LST (SVM)"):
                cbar.ax.text(1.02, 0.5, "°C", transform=cbar.ax.transAxes,
                             va="center", ha="left", fontsize=22)
            elif title == "Absolute Difference":
                cbar.ax.text(1.02, 0.5, "ΔT (°C)", transform=cbar.ax.transAxes,
                             va="center", ha="left", fontsize=22)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved plot to {output_path}")
    @staticmethod
    def save_metrics(metrics: Dict[str, float], output_path: Path) -> None:
        """Save metrics to a text file"""
        with open(output_path, "w") as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value:.4f}\n")
        logger.info(f"Saved metrics to {output_path}")
    @staticmethod
    def compute_overall_metrics(true: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
        """Compute overall metrics on flattened arrays"""
        true_flat = true.ravel()
        pred_flat = pred.ravel()
        ss_res = np.sum((true_flat - pred_flat) ** 2)
        ss_tot = np.sum((true_flat - np.mean(true_flat)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        r = np.corrcoef(true_flat, pred_flat)[0, 1]
        rmse = np.sqrt(np.mean((true_flat - pred_flat) ** 2))
        mae = np.mean(np.abs(true_flat - pred_flat))
        return {"R": r, "R^2": r2, "RMSE (°C)": rmse, "MAE (°C)": mae}

# Main Pipeline
def main(config: Config):
    """Run the full SVM training and evaluation pipeline"""
    logger.info("Starting SVM pipeline")
    logger.info(f"Configuration: {config}")
    # Load rasters
    loader = DataLoader()
    lst, lst_transform, lst_bounds, lst_crs = loader.load_raster(config.lst_raster)
    ndvi, ndvi_transform, ndvi_bounds, ndvi_crs = loader.load_raster(config.ndvi_raster)
    emissivity, em_transform, em_bounds, em_crs = loader.load_raster(config.emissivity_raster)
    # Check consistency
    loader.check_raster_consistency([
        (lst, lst_transform, lst_bounds, lst_crs),
        (ndvi, ndvi_transform, ndvi_bounds, ndvi_crs),
        (emissivity, em_transform, em_bounds, em_crs)])
    # Normalize NDVI and emissivity
    ndvi_norm, ndvi_min, ndvi_max = normalize(ndvi)
    em_norm, em_min, em_max = normalize(emissivity)
    logger.info(f"NDVI range: min={ndvi_min:.4f}, max={ndvi_max:.4f}")
    logger.info(f"Emissivity range: min={em_min:.4f}, max={em_max:.4f}")
    # Build coordinate grid
    x_coords, y_coords = build_coordinate_grid(lst_transform, lst.shape)
    # Extract urban features
    extractor = UrbanFeaturesExtractor(config.geojson_dir, x_coords, y_coords)
    urban_features = extractor.extract_all(config.urban_features)
    # Build input feature matrix
    inputs = np.column_stack((
        normalize(x_coords)[0].ravel(), # x-coordinate normalized
        normalize(y_coords)[0].ravel(), # y-coordinate normalized
        ndvi_norm.ravel(),
        em_norm.ravel(),
        urban_features["buildings"].ravel(),
        urban_features["roads"].ravel(),
        urban_features["rails"].ravel(),
        urban_features["parks"].ravel(),
        urban_features["water"].ravel(),
        urban_features["trees"].ravel()))
    targets = normalize(lst)[0].ravel()
    # Train SVM
    trainer = SVMTrainer(config)
    trainer.train(inputs, targets)
    # Evaluate on train/test
    metrics = trainer.evaluate()
    # Predict full map
    lst_pred_norm = trainer.predict_full(inputs)
    lst_pred = lst_pred_norm.reshape(lst.shape)
    # Denormalize (back to Kelvin, then Celsius)
    lst_min, lst_max = lst.min(), lst.max()
    lst_pred_denorm_K = lst_pred * (lst_max - lst_min) + lst_min
    lst_pred_denorm_C = lst_pred_denorm_K - 273.15
    true_lst_C = lst - 273.15
    abs_diff = np.abs(lst_pred_denorm_C - true_lst_C)
    # Save predicted LST
    output_pred_path = config.output_dir / f"lst_pred_{config.svm_kernel}_kernel.npy"
    np.save(output_pred_path, lst_pred_denorm_C)
    logger.info(f"Saved predicted LST to {output_pred_path}")
    # Compute overall metrics and save
    overall_metrics = Visualizer.compute_overall_metrics(true_lst_C, lst_pred_denorm_C)
    overall_metrics.update(metrics)  # include train/test metrics
    metrics_path = config.output_dir / f"metrics_{config.svm_kernel}_kernel.txt"
    Visualizer.save_metrics(overall_metrics, metrics_path)
    # Generate comparison plot
    plot_path = config.output_dir / f"comparison_{config.svm_kernel}_kernel.png"
    Visualizer.plot_comparison(
        true_lst_C,
        lst_pred_denorm_C,
        abs_diff,
        lst_bounds,
        titles=("True LST (Landsat 8)", "Predicted LST (SVM)", "Absolute Difference"),
        output_path=plot_path)
    # Save model
    model_path = config.output_dir / f"svm_model_{config.svm_kernel}_kernel.pkl"
    joblib.dump(trainer.model, model_path)
    logger.info(f"Model saved to {model_path}")
    logger.info("Pipeline completed successfully")

if __name__ == "__main__":
    cfg = Config()
    # to change kernel to linear, please uncomment:
    # cfg.svm_kernel = "linear"
    main(cfg)