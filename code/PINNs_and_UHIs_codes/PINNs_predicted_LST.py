"""
Urban Heat Island Model Prediction and Evaluation

This module loads a trained PINN model and generates predictions for urban heat island analysis,
including visualization and performance metrics.
"""

import warnings
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import rasterio
import geopandas as gpd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from shapely.geometry import Point
warnings.filterwarnings("ignore", category=UserWarning)


class UrbanHeatPredictor:
    """
    Loads trained Urban Heat PINN model and generates predictions.

    This class handles data loading, preprocessing, model inference,
    and result visualization for urban heat island analysis.
    """

    def __init__(self, model_config: Dict):
        """
        Initialize predictor with configuration.

        Args:
            model_config: Dictionary containing model paths and data parameters
        """
        self.config = model_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.norm_params = None
        print(f"Using device: {self.device}")
        self._validate_paths()

    def _validate_paths(self):
        """Validate that all required paths exist."""
        required_paths = [
            'model_path', 'data_paths.lst_path', 'data_paths.ndvi_path',
            'data_paths.emissivity_path', 'data_paths.urban_features_dir'
        ]
        for path_key in required_paths:
            path = self._get_nested_config(path_key)
            if path and not Path(path).exists():
                raise FileNotFoundError(f"Required path not found: {path}")

    def _get_nested_config(self, key: str):
        """Get value from nested configuration dictionary."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, {})
        return value if value != {} else None

    def load_model(self):
        """Load trained model and normalization parameters."""
        print("Loading trained model...")
        checkpoint = torch.load(self.config['model_path'], map_location=self.device)
        self.norm_params = checkpoint.get('norm_params', {})
        # Initialize model architecture
        self.model = UrbanHeatPINN(checkpoint['input_dim']).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"Model loaded successfully with input dimension: {checkpoint['input_dim']}")

    def load_and_preprocess_data(self) -> Dict:
        """
        Load and preprocess input data for prediction.

        Returns:
            Dictionary containing processed data and metadata
        """
        print("Loading and preprocessing data...")
        # Load raster data
        lst_data, lst_transform, lst_bounds, lst_crs = self._load_raster(
            self.config['data_paths']['lst_path']
        )
        ndvi_data, _, _, _ = self._load_raster(self.config['data_paths']['ndvi_path'])
        emissivity_data, _, _, _ = self._load_raster(self.config['data_paths']['emissivity_path'])
        # Generate coordinate grid
        x_coords, y_coords = self._generate_coordinate_grid(lst_data, lst_transform)
        # Load and validate urban features
        urban_features = self._load_urban_features(
            x_coords, y_coords, lst_crs, self.config['data_paths']['urban_features_dir']
        )
        # Prepare input features
        input_features = self._prepare_input_features(
            x_coords, y_coords, ndvi_data, emissivity_data, urban_features
        )
        return {
            'input_features': input_features,
            'lst_data': lst_data,
            'coordinates': (x_coords, y_coords),
            'transform': lst_transform,
            'bounds': lst_bounds,
            'crs': lst_crs,
            'urban_features': urban_features,
            'shape': lst_data.shape
        }
    @staticmethod
    def _load_raster(path: str) -> Tuple[np.ndarray, rasterio.Affine,
                                         rasterio.coords.BoundingBox, rasterio.CRS]:
        """Load raster file and return data with metadata."""
        with rasterio.open(path) as src:
            data = src.read(1)
            transform = src.transform
            bounds = src.bounds
            crs = src.crs
        return data, transform, bounds, crs

    def _generate_coordinate_grid(self, data: np.ndarray,
                                  transform: rasterio.Affine) -> Tuple[np.ndarray, np.ndarray]:
        """Generate coordinate grids from raster transform."""
        height, width = data.shape
        x_coords, y_coords = np.meshgrid(
            np.arange(width) * transform[0] + transform[2],
            np.arange(height) * transform[4] + transform[5]
        )
        return x_coords, y_coords

    def _load_urban_features(self, x_coords: np.ndarray, y_coords: np.ndarray,
                             target_crs: rasterio.CRS, urban_features_dir: str) -> Dict[str, np.ndarray]:
        """Load urban features with CRS validation and conversion."""
        feature_files = {
            'buildings': 'buildings.geojson',
            'roads': 'roads.geojson',
            'rails': 'rails.geojson',
            'parks': 'parks.geojson',
            'water': 'water.geojson',
            'trees': 'trees.geojson'
        }
        print("Loading urban features with CRS validation...")
        urban_features = {}
        for feature_name, filename in feature_files.items():
            file_path = Path(urban_features_dir) / filename
            gdf = gpd.read_file(file_path)

            # Check and reproject CRS if necessary
            if gdf.crs != target_crs:
                print(f"  Reprojecting {feature_name} from {gdf.crs} to {target_crs}")
                gdf = gdf.to_crs(target_crs)
            urban_features[feature_name] = self._geojson_to_array(gdf, x_coords, y_coords)
        return urban_features

    @staticmethod
    def _geojson_to_array(gdf: gpd.GeoDataFrame, x_coords: np.ndarray,
                          y_coords: np.ndarray) -> np.ndarray:
        """Convert GeoJSON features to binary array mask."""
        points = [Point(x, y) for x, y in zip(x_coords.ravel(), y_coords.ravel())]
        return np.array([gdf.contains(p).any() for p in points]).reshape(x_coords.shape)

    def _prepare_input_features(self, x_coords: np.ndarray, y_coords: np.ndarray,
                                ndvi: np.ndarray, emissivity: np.ndarray,
                                urban_features: Dict[str, np.ndarray]) -> np.ndarray:
        """Prepare normalized input features for model prediction."""
        # Use stored normalization parameters or compute from data
        if self.norm_params:
            x_min, x_max = self.norm_params.get('x', (x_coords.min(), x_coords.max()))
            y_min, y_max = self.norm_params.get('y', (y_coords.min(), y_coords.max()))
            ndvi_min, ndvi_max = self.norm_params.get('ndvi', (ndvi.min(), ndvi.max()))
            em_min, em_max = self.norm_params.get('emissivity', (emissivity.min(), emissivity.max()))
        else:
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()
            ndvi_min, ndvi_max = ndvi.min(), ndvi.max()
            em_min, em_max = emissivity.min(), emissivity.max()

        def normalize(data, min_val, max_val):
            return (data - min_val) / (max_val - min_val)
        # Prepare feature list
        features_list = [
            normalize(x_coords, x_min, x_max).ravel(),
            normalize(y_coords, y_min, y_max).ravel(),
            normalize(ndvi, ndvi_min, ndvi_max).ravel(),
            normalize(emissivity, em_min, em_max).ravel(),
        ]
        # Add urban features
        for feature_name in ['buildings', 'roads', 'rails', 'parks', 'water', 'trees']:
            features_list.append(urban_features[feature_name].ravel())
        return np.column_stack(features_list)

    def predict(self, input_features: np.ndarray) -> np.ndarray:
        """
        Generate predictions using the trained model.

        Args:
            input_features: Preprocessed input features

        Returns:
            Model predictions
        """
        print("Generating predictions...")
        input_tensor = torch.tensor(input_features, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            predictions = self.model(input_tensor).cpu().numpy()
        return predictions

    def denormalize_predictions(self, predictions: np.ndarray,
                                original_lst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Denormalize predictions to original temperature scale.

        Args:
            predictions: Normalized model predictions
            original_lst: Original LST data for scaling

        Returns:
            Tuple of (predictions in Kelvin, predictions in Celsius)
        """
        if self.norm_params and 'lst' in self.norm_params:
            lst_min = self.norm_params['lst']['min']
            lst_max = self.norm_params['lst']['max']
        else:
            lst_min = np.nanmin(original_lst)
            lst_max = np.nanmax(original_lst)
        # Denormalize to Kelvin
        pred_kelvin = predictions * (lst_max - lst_min) + lst_min
        # Convert to Celsius
        pred_celsius = pred_kelvin - 273.15
        return pred_kelvin, pred_celsius


class ResultVisualizer:
    """Handles visualization of prediction results and metrics."""

    def __init__(self, output_dir: str):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save output plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Set professional styling
        plt.style.use('default')
        self.colors = {
            'total': '#1f77b4',
            'data': '#ff7f0e',
            'physics': '#2ca02c',
            'bcs': '#9467bd'
        }

    def plot_training_history(self, history: Dict[str, List[float]],
                              location_name: str = "Study Area"):
        """
        Plot training history with professional styling.

        Args:
            history: Training history dictionary
            location_name: Name of study area for title
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        # Plot each loss component
        for loss_type, color in self.colors.items():
            if loss_type in history and history[loss_type]:
                ax.plot(history[loss_type], color=color, linewidth=3,
                        label=f'{loss_type.title()} Loss')
        # Styling
        ax.set_xlabel('Epoch', fontsize=20)
        ax.set_ylabel('Loss', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.legend(fontsize=16, framealpha=0.9)
        ax.set_title(f"Training History - {location_name}", fontsize=22, pad=20)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = self.output_dir / "training_history.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Training history plot saved to: {save_path}")

    def plot_comparison(self, true_lst: np.ndarray, pred_lst: np.ndarray,
                        absolute_diff: np.ndarray, bounds: rasterio.coords.BoundingBox,
                        titles: Tuple[str, str, str] = ("True LST", "Predicted LST", "Absolute Difference"),
                        location_name: str = "Study Area",
                        cmap_abs_diff: str = 'coolwarm'):
        """
        Create comparison plot between true and predicted LST.

        Args:
            true_lst: True land surface temperature in Celsius
            pred_lst: Predicted land surface temperature in Celsius
            absolute_diff: Absolute difference between true and predicted
            bounds: Geographic bounds for plotting
            titles: Titles for each subplot
            location_name: Name of study area
            cmap_abs_diff: Colormap for difference plot
        """
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        # Define colormaps
        cmaps = ['YlOrRd', 'YlOrRd', cmap_abs_diff]
        # Extract bounds
        left, bottom, right, top = bounds.left, bounds.bottom, bounds.right, bounds.top
        for i, (ax, data, cmap, title) in enumerate(zip(axes, [true_lst, pred_lst, absolute_diff],
                                                        cmaps, titles)):
            # Plot data
            im = ax.imshow(data, cmap=cmap, extent=(left, right, bottom, top))
            ax.set_title(title, fontsize=18, pad=15)
            # Configure axes
            self._configure_axis(ax, bounds, i)
            # Add colorbar
            self._add_colorbar(fig, im, ax, i, data)
        plt.suptitle(f"Land Surface Temperature Comparison - {location_name}",
                     fontsize=22, y=0.95)
        plt.tight_layout()
        save_path = self.output_dir / "lst_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"LST comparison plot saved to: {save_path}")

    def _configure_axis(self, ax: plt.Axes, bounds: rasterio.coords.BoundingBox,
                        subplot_index: int):
        """Configure axis ticks and labels."""
        left, bottom, right, top = bounds.left, bounds.bottom, bounds.right, bounds.top

        if subplot_index == 0:  # Only show coordinates on first subplot
            ax.set_xticks([left, right])
            ax.set_yticks([bottom, top])
            ax.xaxis.set_major_formatter(FuncFormatter(
                lambda x, _: f"{abs(x):.2f}°{'E' if x >= 0 else 'W'}"
            ))
            ax.yaxis.set_major_formatter(FuncFormatter(
                lambda y, _: f"{abs(y):.2f}°{'N' if y >= 0 else 'S'}"
            ))
            ax.tick_params(axis='x', pad=10, labelsize=12)
            ax.tick_params(axis='y', labelsize=12)
        else:
            ax.set_xticks([])
            ax.set_yticks([])

    def _add_colorbar(self, fig: plt.Figure, im: plt.AxesImage, ax: plt.Axes,
                      subplot_index: int, data: np.ndarray):
        """Add formatted colorbar to subplot."""
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, aspect=30)
        cbar.ax.tick_params(labelsize=12)
        # Set colorbar label based on subplot
        if subplot_index in [0, 1]:  # Temperature plots
            cbar.set_label('Temperature (°C)', fontsize=14, labelpad=10)
        else:  # Difference plot
            cbar.set_label('ΔT (°C)', fontsize=14, labelpad=10)

class ModelEvaluator:
    """Evaluates model performance using various metrics."""

    def __init__(self, output_dir: str):
        """
        Initialize evaluator.

        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def calculate_metrics(self, true_values: np.ndarray, pred_values: np.ndarray,
                          location_name: str = "Study Area") -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.

        Args:
            true_values: Ground truth values
            pred_values: Model predictions
            location_name: Name of study area

        Returns:
            Dictionary of evaluation metrics
        """
        # Flatten arrays for calculation
        true_flat = true_values.ravel()
        pred_flat = pred_values.ravel()
        # Remove NaN values for metric calculation
        mask = ~(np.isnan(true_flat) | np.isnan(pred_flat))
        true_clean = true_flat[mask]
        pred_clean = pred_flat[mask]
        if len(true_clean) == 0:
            raise ValueError("No valid data points for metric calculation")
        # Calculate metrics
        metrics = {}
        # R² Score
        ss_res = np.sum((true_clean - pred_clean) ** 2)
        ss_tot = np.sum((true_clean - np.mean(true_clean)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        # RMSE
        metrics['rmse'] = np.sqrt(np.mean((true_clean - pred_clean) ** 2))
        # MAE
        metrics['mae'] = np.mean(np.abs(true_clean - pred_clean))
        # Mean Absolute Percentage Error
        metrics['mape'] = np.mean(np.abs((true_clean - pred_clean) / true_clean)) * 100
        # Maximum Absolute Error
        metrics['max_ae'] = np.max(np.abs(true_clean - pred_clean))
        # Print results
        self._print_metrics(metrics, location_name)
        return metrics

    def _print_metrics(self, metrics: Dict[str, float], location_name: str):
        """Print formatted metrics to console."""
        print("=" * 50)
        print(f"MODEL PERFORMANCE - {location_name.upper()}")
        print("=" * 50)
        print(f"R² Score:        {metrics['r2']:.4f}")
        print(f"RMSE:            {metrics['rmse']:.4f} °C")
        print(f"MAE:             {metrics['mae']:.4f} °C")
        print(f"MAPE:            {metrics['mape']:.4f} %")
        print(f"Max Absolute Error: {metrics['max_ae']:.4f} °C")
        print("=" * 50)

    def save_metrics(self, metrics: Dict[str, float], filename: str = "model_metrics"):
        """Save metrics to text file."""
        file_path = self.output_dir / f"{filename}.txt"
        with open(file_path, 'w') as f:
            f.write("Urban Heat Island Model Performance Metrics\n")
            f.write("=" * 50 + "\n")
            for metric, value in metrics.items():
                if metric == 'r2':
                    f.write(f"R² Score: {value:.4f}\n")
                elif metric == 'rmse':
                    f.write(f"RMSE: {value:.4f} °C\n")
                elif metric == 'mae':
                    f.write(f"MAE: {value:.4f} °C\n")
                elif metric == 'mape':
                    f.write(f"MAPE: {value:.4f} %\n")
                elif metric == 'max_ae':
                    f.write(f"Max Absolute Error: {value:.4f} °C\n")
        print(f"Metrics saved to: {file_path}")

class UrbanHeatPINN(nn.Module):
    """
    Physics-Informed Neural Network for Urban Heat Island modeling.

    This matches the architecture used during training.
    """
    def __init__(self, input_dim: int):
        super().__init__()
        # Input processing layers
        self.input_layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
        )
        # Middle layers
        self.middle_layers = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.01),
        )
        # Physics-informed layers
        self.physics_layers = nn.Sequential(
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
        )
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = self.input_layers(x)
        x = self.middle_layers(x)
        x = self.physics_layers(x)
        return self.output_layers(x)

def main():
    """Main function to run model prediction and evaluation."""
    # Configuration
    config = {
        'model_path': "/your_PINNs_model_saving_directory/pinn_model.pth",
        'training_history_path': "/your_PINNs_model_saving_directory/training_history.pth",
        'data_paths': {
            'lst_path': "/your_GEE_data_directory/LS.tif",
            'ndvi_path': "/your_GEE_data_directory/NDVI.tif",
            'emissivity_path': "/your_GEE_data_directory/EM.tif",
            'urban_features_dir': "/your_urban_features_data_directory/",
        },
        'output': {
            'plots_dir': "/your_directory_saving_plots/",
            'location_name': "city_study_area"
        }
    }
    try:
        # Initialize components
        predictor = UrbanHeatPredictor(config)
        visualizer = ResultVisualizer(config['output']['plots_dir'])
        evaluator = ModelEvaluator(config['output']['plots_dir'])
        # Load model
        predictor.load_model()
        # Load training history for plotting
        history = torch.load(config['training_history_path'])
        # Load and preprocess data
        data = predictor.load_and_preprocess_data()
        # Generate predictions
        predictions = predictor.predict(data['input_features'])
        predictions_reshaped = predictions.reshape(data['shape'])
        # Denormalize predictions
        pred_kelvin, pred_celsius = predictor.denormalize_predictions(
            predictions_reshaped, data['lst_data']
        )
        # Convert true LST to Celsius for comparison
        true_celsius = data['lst_data'] - 273.15
        # Calculate absolute difference
        absolute_diff = np.abs(pred_celsius - true_celsius)
        print(f"Prediction range: {pred_celsius.min():.2f}°C to {pred_celsius.max():.2f}°C")
        print(f"Absolute difference range: {absolute_diff.min():.4f}°C to {absolute_diff.max():.4f}°C")
        # Generate visualizations
        visualizer.plot_training_history(history, config['output']['location_name'])
        visualizer.plot_comparison(
            true_celsius, pred_celsius, absolute_diff, data['bounds'],
            location_name=config['output']['location_name']
        )
        # Calculate and save metrics
        metrics = evaluator.calculate_metrics(true_celsius, pred_celsius,
                                              config['output']['location_name'])
        evaluator.save_metrics(metrics, "model_performance_metrics")
        print(f"\nAll results saved to: {config['output']['plots_dir']}")
    except Exception as e:
        print(f"Error during model prediction: {e}")
        raise

if __name__ == "__main__":
    main()