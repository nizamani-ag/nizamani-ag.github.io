"""
UHI Prediction using PINNs
This script loads trained PINN models to predict Land Surface Temperature (LST)
and generates comparative visualizations between true and predicted values
"""
import os
import warnings
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from shapely.geometry import Point
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class Config:
    """Configuration class for model parameters and file paths"""
    # Input data paths
    DATA_PATHS = {
        'lst': "/your_directory/paris_GEE/LST_PR.tif",
        'ndvi': "/your_directory/paris_GEE/NDVI_PR.tif",
        'emissivity': "/your_directory/paris_GEE/EM_PR.tif",
        'buildings': "/your_directory/paris_GEE/buildings.geojson",
        'roads': "/your_directory/paris_GEE/roads.geojson",
        'rails': "/your_directory/paris_GEE/rails.geojson",
        'parks': "/your_directory/paris_GEE/parks.geojson",
        'water': "/your_directory/paris_GEE/water.geojson",
        'trees': "/your_directory/paris_GEE/trees.geojson",
    }
    # Model and output paths
    MODEL_DIR = "/your_directory/Paris/"
    OUTPUT_DIR = "/your_directory/Paris_true_pred_histo_plot/"
    # Plotting parameters
    PLOT_CONFIG = {
        'figsize': (20, 18),
        'dpi': 300,
        'fontsize': 22,
        'cmap_true': 'autumn_r',
        'cmap_pred': 'autumn_r',
        'cmap_diff': 'coolwarm'
    }

class DataLoader:
    """Handles loading and preprocessing of raster and vector data"""
    @staticmethod
    def load_raster(file_path: str) -> Tuple[np.ndarray, rasterio.Affine, rasterio.coords.BoundingBox, rasterio.CRS]:
        """
        Load raster data and metadata
        Args:
            file_path: Path to raster file
        Returns:
            Tuple of (data, transform, bounds, crs)
        """
        try:
            with rasterio.open(file_path) as src:
                data = src.read(1)
                transform = src.transform
                bounds = src.bounds
                crs = src.crs
            logger.info(f"Successfully loaded raster: {os.path.basename(file_path)}")
            return data, transform, bounds, crs
        except Exception as e:
            logger.error(f"Error loading raster {file_path}: {e}")
            raise
    @staticmethod
    def load_geojson(file_path: str) -> gpd.GeoDataFrame:
        """
        Load GeoJSON file as GeoDataFrame
        Args:
            file_path: Path to GeoJSON file
        Returns:
            GeoDataFrame
        """
        try:
            gdf = gpd.read_file(file_path)
            logger.info(f"Successfully loaded GeoJSON: {os.path.basename(file_path)}")
            return gdf
        except Exception as e:
            logger.error(f"Error loading GeoJSON {file_path}: {e}")
            raise
    @staticmethod
    def geojson_to_mask(geojson_path: str, x_coords: np.ndarray, y_coords: np.ndarray,
                        target_crs: rasterio.CRS) -> np.ndarray:
        """
        Convert GeoJSON features to binary mask array
        Args:
            geojson_path: Path to GeoJSON file
            x_coords: X coordinate grid
            y_coords: Y coordinate grid
            target_crs: Target CRS for reprojection
        Returns:
            Binary mask array
        """
        gdf = DataLoader.load_geojson(geojson_path)
        # Reproject if necessary
        if gdf.crs != target_crs:
            logger.info(f"Reprojecting {os.path.basename(geojson_path)} to match target CRS")
            gdf = gdf.to_crs(target_crs)
        # Create mask
        points = [Point(x, y) for x, y in zip(x_coords.ravel(), y_coords.ravel())]
        mask = np.array([gdf.contains(p).any() for p in points]).reshape(x_coords.shape)
        return mask

class UrbanHeatPINN(nn.Module):
    """
    PINNs for UHI prediction
    Architecture features:
    - Input processing with Tanh for smooth feature extraction
    - Middle layers with LeakyReLU for better gradient flow
    - Physics-informed layers with Tanh for smooth derivatives
    - Output layers with Tanh for final smooth output
    """
    def __init__(self, input_dim: int):
        """
        Initialize the PINN model
        Args:
            input_dim: Dimension of input features
        """
        super().__init__()
        # Input processing layers
        self.input_layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Tanh(),  # Smooth activation for initial processing
            nn.Linear(256, 256),
            nn.Tanh()
        )
        # Middle layers
        self.middle_layers = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.01),  # Small negative slope for gradient flow
            nn.Linear(256, 128),
            nn.LeakyReLU(0.01)
        )
        # Physics-informed layers
        self.physics_layers = nn.Sequential(
            nn.Linear(128, 128),
            nn.Tanh(),  # Critical for physics loss calculations
            nn.Linear(128, 64),
            nn.Tanh()
        )
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),  # Keep smooth for final output
            nn.Linear(32, 1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        x = self.input_layers(x)
        x = self.middle_layers(x)
        x = self.physics_layers(x)
        return self.output_layers(x)

class PINNPredictor:
    """Main class for loading models and generating predictions"""
    def __init__(self, config: Config):
        """
        Initialize predictor with configuration
        Args:
            config: Configuration object
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        # Create output directory
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    def load_data(self) -> Dict[str, np.ndarray]:
        """
        Load and preprocess all input data
        Returns:
            Dictionary containing all loaded data arrays
        """
        logger.info("Loading input data")
        # Load base rasters
        lst_data, lst_transform, lst_bounds, lst_crs = DataLoader.load_raster(self.config.DATA_PATHS['lst'])
        ndvi_data, _, _, _ = DataLoader.load_raster(self.config.DATA_PATHS['ndvi'])
        emissivity_data, _, _, _ = DataLoader.load_raster(self.config.DATA_PATHS['emissivity'])
        logger.info("CRS Consistency Check")
        logger.info("=" * 40)
        logger.info(f"LST CRS: {lst_crs}")
        # Generate coordinate grid
        x_coords, y_coords = np.meshgrid(
            np.arange(lst_data.shape[1]) * lst_transform[0] + lst_transform[2],
            np.arange(lst_data.shape[0]) * lst_transform[4] + lst_transform[5]
        )
        # Load urban features
        logger.info("Loading urban features")
        urban_features = {}
        for name, path in self.config.DATA_PATHS.items():
            if name in ['lst', 'ndvi', 'emissivity']:
                continue
            logger.info(f"Processing {name}")
            urban_features[name] = DataLoader.geojson_to_mask(path, x_coords, y_coords, lst_crs)
        return {
            'lst': lst_data,
            'ndvi': ndvi_data,
            'emissivity': emissivity_data,
            'x_coords': x_coords,
            'y_coords': y_coords,
            'transform': lst_transform,
            'bounds': lst_bounds,
            'crs': lst_crs,
            'urban_features': urban_features
        }

    def prepare_features(self, data_dict: Dict) -> np.ndarray:
        """
        Prepare input features for model prediction
        Args:
            data_dict: Dictionary containing all data arrays
        Returns:
            Normalized input features array
        """
        logger.info("Preparing input features")
        lst_data = data_dict['lst']
        ndvi_data = data_dict['ndvi']
        emissivity_data = data_dict['emissivity']
        x_coords = data_dict['x_coords']
        y_coords = data_dict['y_coords']
        urban_features = data_dict['urban_features']
        # Precompute normalization factors
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        ndvi_min, ndvi_max = ndvi_data.min(), ndvi_data.max()
        em_min, em_max = emissivity_data.min(), emissivity_data.max()
        def normalize(data: np.ndarray, dmin: float, dmax: float) -> np.ndarray:
            """Normalize data to [0, 1] range"""
            return (data - dmin) / (dmax - dmin)
        # Prepare input features
        inputs = np.column_stack((
            normalize(x_coords, x_min, x_max).ravel(),
            normalize(y_coords, y_min, y_max).ravel(),
            normalize(ndvi_data, ndvi_min, ndvi_max).ravel(),
            normalize(emissivity_data, em_min, em_max).ravel(),
            urban_features['buildings'].ravel(),
            urban_features['roads'].ravel(),
            urban_features['rails'].ravel(),
            urban_features['parks'].ravel(),
            urban_features['water'].ravel(),
            urban_features['trees'].ravel(),
        ))
        return inputs

    def load_model(self, input_dim: int) -> UrbanHeatPINN:
        """
        Load trained model from checkpoint
        Args:
            input_dim: Dimension of input features
        Returns:
            Loaded and configured model
        """
        logger.info("Loading trained model")
        model_path = os.path.join(self.config.MODEL_DIR, "pinn_model.pth")
        history_path = os.path.join(self.config.MODEL_DIR, "training_history.pth")
        try:
            checkpoint = torch.load(model_path, weights_only=False, map_location=self.device)
            history = torch.load(history_path, weights_only=False, map_location=self.device)
            model = UrbanHeatPINN(input_dim).to(self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            logger.info("Model successfully loaded")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    def predict(self, model: UrbanHeatPINN, inputs: np.ndarray,
                original_shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
        """
        Generate predictions using the loaded model
        Args:
            model: Trained PINN model
            inputs: Input features array
            original_shape: Original shape for reshaping predictions
        Returns:
            Dictionary containing prediction results
        """
        logger.info("Generating predictions...")
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            lst_pred = model(inputs_tensor).cpu().numpy().reshape(original_shape)
        # Denormalize predictions
        lst_data, _, _, _ = DataLoader.load_raster(self.config.DATA_PATHS['lst'])
        lst_min, lst_max = lst_data.min(), lst_data.max()
        lst_pred_denorm_K = lst_pred * (lst_max - lst_min) + lst_min
        lst_pred_denorm_C = lst_pred_denorm_K - 273.15
        # Convert true LST to Celsius
        lst_C = lst_data - 273.15
        abs_diff = np.abs(lst_pred_denorm_C - lst_C)
        logger.info(f"Absolute difference - Min: {np.min(abs_diff):.4f}, Max: {np.max(abs_diff):.4f}")
        return {
            'true_lst_c': lst_C,
            'pred_lst_c': lst_pred_denorm_C,
            'absolute_diff': abs_diff,
            'bounds': DataLoader.load_raster(self.config.DATA_PATHS['lst'])[2]
        }

    def plot_comparison(self, results: Dict, output_name: str = "prediction_comparison"):
        """
        Create comparison plot between true and predicted LST
        Args:
            results: Dictionary containing prediction results
            output_name: Name for output file
        """
        logger.info("Generating comparison plot")
        true_lst = results['true_lst_c']
        pred_lst = results['pred_lst_c']
        abs_diff = results['absolute_diff']
        bounds = results['bounds']
        fig, axes = plt.subplots(1, 3, figsize=self.config.PLOT_CONFIG['figsize'])
        cmaps = [self.config.PLOT_CONFIG['cmap_true'],
                 self.config.PLOT_CONFIG['cmap_pred'],
                 self.config.PLOT_CONFIG['cmap_diff']]

        titles = ["Ground Truth LST", "PINN-Modelled LST", "Absolute Error"]
        left, right, bottom, top = bounds.left, bounds.right, bounds.bottom, bounds.top
        for i, (ax, data, cmap, title) in enumerate(zip(axes, [true_lst, pred_lst, abs_diff], cmaps, titles)):
            im = ax.imshow(data, cmap=cmap, extent=(left, right, bottom, top))
            ax.set_title(title, fontsize=self.config.PLOT_CONFIG['fontsize'], pad=12)
            # Configure ticks and labels for first subplot only
            if i == 0:
                ax.set_xticks([left, right])
                ax.set_yticks([bottom, top])
                ax.xaxis.set_major_formatter(FuncFormatter(
                    lambda x, _: f"{abs(x):.2f}°{'E' if x >= 0 else 'W'}"
                ))
                ax.yaxis.set_major_formatter(FuncFormatter(
                    lambda y, _: f"{abs(y):.2f}°{'N' if y >= 0 else 'S'}"
                ))
                ax.tick_params(axis='x', pad=15)
                ax.tick_params(axis='both', labelsize=self.config.PLOT_CONFIG['fontsize'])
            else:
                ax.set_xticks([])
                ax.set_yticks([])
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.04, aspect=15)
            cbar.set_label('')
            # Format colorbar ticks
            vmin, vmax = im.get_clim()
            ticks = np.linspace(vmin, vmax, 5)
            cbar.set_ticks(ticks)
            if i == 2:  # Absolute error plot
                tick_labels = [f"{tick:.2f}" for tick in ticks]
            else:
                tick_labels = [f"{tick:.1f}" for tick in ticks]
            cbar.ax.set_xticklabels(tick_labels)
            cbar.ax.tick_params(labelsize=self.config.PLOT_CONFIG['fontsize'])
            # Add unit labels
            if i in [0, 1]:
                cbar.ax.text(1.02, 0.5, '°C', transform=cbar.ax.transAxes,
                             va='center', ha='left', fontsize=self.config.PLOT_CONFIG['fontsize'])
            elif i == 2:
                cbar.ax.text(1.02, 0.5, 'ΔT (°C)', transform=cbar.ax.transAxes,
                             va='center', ha='left', fontsize=self.config.PLOT_CONFIG['fontsize'])
        plt.tight_layout()
        output_path = os.path.join(self.config.OUTPUT_DIR, f"{output_name}.png")
        plt.savefig(output_path, dpi=self.config.PLOT_CONFIG['dpi'], bbox_inches='tight')
        plt.close()

        logger.info(f"Comparison plot saved to: {output_path}")

    def plot_histograms(self, results: Dict, output_name: str = "histogram_comparison"):
        """
        Create histogram comparison between true and predicted LST
        Args:
            results: Dictionary containing prediction results
            output_name: Name for output file
        """
        logger.info("Generating histogram plot")
        true_lst = results['true_lst_c']
        pred_lst = results['pred_lst_c']
        abs_diff = results['absolute_diff']
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        # True vs Predicted histogram
        axes[0].hist(true_lst.flatten(), bins=50, density=True, histtype='step',
                     color='magenta', label='Ground Truth LST', linewidth=3, alpha=0.85)
        axes[0].hist(pred_lst.flatten(), density=True, bins=50, histtype='step',
                     color='green', linestyle="--", label='PINN-Modelled LST', linewidth=3, alpha=0.85)
        axes[0].set_xlabel('Surface Temperature (°C)', fontsize=24)
        axes[0].set_ylabel('Density', fontsize=24)
        axes[0].tick_params(axis='both', which='major', labelsize=24)
        axes[0].grid(True, linestyle='-', alpha=0.7)
        axes[0].legend(loc="best", fontsize=24)
        # Absolute difference histogram
        axes[1].hist(abs_diff.flatten(), bins=50, density=True, color='blue',
                     label='Absolute Error', linewidth=3, alpha=0.75)
        axes[1].set_xlabel('ΔT (°C)', fontsize=24)
        axes[1].set_ylabel('Density', fontsize=24)
        axes[1].tick_params(axis='both', which='major', labelsize=24)
        axes[1].grid(True, linestyle='-', alpha=0.5)
        axes[1].legend(loc="best", fontsize=24)
        plt.tight_layout()
        output_path = os.path.join(self.config.OUTPUT_DIR, f"{output_name}.png")
        plt.savefig(output_path, dpi=self.config.PLOT_CONFIG['dpi'], bbox_inches='tight')
        plt.close()
   
    def run(self):
        """Main execution method"""
        logger.info("Starting PINN prediction pipeline")
        try:
            # Load and preprocess data
            data_dict = self.load_data()
            # Prepare features
            inputs = self.prepare_features(data_dict)
            # Load model
            model = self.load_model(input_dim=inputs.shape[1])
            # Generate predictions
            results = self.predict(model, inputs, data_dict['urban_features']['buildings'].shape)
            # Create visualizations
            self.plot_comparison(results, "pinn_prediction_comparison")
            self.plot_histograms(results, "temperature_histograms")
            logger.info("PINN prediction pipeline completed successfully!")
        except Exception as e:
            logger.error(f"Error in prediction pipeline: {e}")
            raise

def main():
    """Main function to run the PINN predictor"""
    config = Config()
    predictor = PINNPredictor(config)
    predictor.run()

if __name__ == "__main__":
    main()