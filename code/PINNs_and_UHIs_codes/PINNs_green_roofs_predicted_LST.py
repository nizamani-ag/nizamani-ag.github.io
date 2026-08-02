"""
Green Roof Temperature Reduction Analysis using Physics-Informed Neural Networks

This module analyzes the cooling effects of green roof implementation
using a trained PINN model to predict temperature reductions.
"""

import warnings
from pathlib import Path
from typing import Dict, Tuple, Any
import numpy as np
import rasterio
import geopandas as gpd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.ndimage import binary_dilation
from shapely.geometry import Point
warnings.filterwarnings("ignore", category=UserWarning)


class GreenRoofAnalyzer:
    """
    Analyzes temperature reduction effects of green roof implementation.

    This class uses a trained PINN model to simulate the cooling effects
    of converting building rooftops to green roofs in urban heat islands.
    """

    def __init__(self, config: Dict):
        """
        Initialize analyzer with configuration.

        Args:
            config: Dictionary containing paths and analysis parameters
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.data = {}
        self.norm_params = {}
        print(f"Using device: {self.device}")
        self._validate_config()

    def _validate_config(self):
        """Validate configuration parameters."""
        required_paths = [
            'model_path', 'data_paths.lst_path', 'data_paths.ndvi_path',
            'data_paths.emissivity_path', 'data_paths.urban_features_dir',
            'output.plots_dir'
        ]
        for path_key in required_paths:
            path = self._get_nested_config(path_key)
            if path and not Path(path).exists():
                raise FileNotFoundError(f"Required path not found: {path}")

    def _get_nested_config(self, key: str) -> Any:
        """Get value from nested configuration dictionary."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, {})
        return value if value != {} else None

    def load_model(self):
        """Load trained PINN model for prediction."""
        print("Loading trained model...")
        checkpoint = torch.load(self.config['model_path'], map_location=self.device)
        # Initialize model with same architecture
        self.model = UrbanHeatPINN(checkpoint['input_dim']).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"Model loaded with input dimension: {checkpoint['input_dim']}")

    def load_and_preprocess_data(self):
        """Load and preprocess all required data with CRS validation."""
        print("Loading and preprocessing data...")
        # Load raster data
        self.data['lst'], lst_transform, self.data['bounds'], lst_crs = self._load_raster(
            self.config['data_paths']['lst_path']
        )
        self.data['ndvi'], _, _, _ = self._load_raster(self.config['data_paths']['ndvi_path'])
        self.data['emissivity'], _, _, _ = self._load_raster(self.config['data_paths']['emissivity_path'])
        # Generate coordinate grid
        self.data['x_coords'], self.data['y_coords'] = self._generate_coordinate_grid(
            self.data['lst'], lst_transform
        )
        # Load urban features with CRS validation
        self.data['urban_features'] = self._load_urban_features(
            self.data['x_coords'], self.data['y_coords'], lst_crs,
            self.config['data_paths']['urban_features_dir']
        )
        # Store normalization parameters
        self._compute_normalization_parameters()
        print("Data loading and preprocessing completed successfully")

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
            # Validate and reproject CRS if necessary
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

    def _compute_normalization_parameters(self):
        """Compute and store normalization parameters for all features."""
        self.norm_params = {
            'x': {'min': self.data['x_coords'].min(), 'max': self.data['x_coords'].max()},
            'y': {'min': self.data['y_coords'].min(), 'max': self.data['y_coords'].max()},
            'ndvi': {'min': self.data['ndvi'].min(), 'max': self.data['ndvi'].max()},
            'emissivity': {'min': self.data['emissivity'].min(), 'max': self.data['emissivity'].max()},
            'lst': {'min': np.nanmin(self.data['lst']), 'max': np.nanmax(self.data['lst'])}
        }

    def _normalize_with(self, data: np.ndarray, feature: str) -> np.ndarray:
        """Normalize data using stored parameters."""
        params = self.norm_params[feature]
        return (data - params['min']) / (params['max'] - params['min'])

    def prepare_input_features(self, ndvi_data: np.ndarray = None,
                               emissivity_data: np.ndarray = None) -> np.ndarray:
        """
        Prepare input features for model prediction.

        Args:
            ndvi_data: Custom NDVI data (uses original if None)
            emissivity_data: Custom emissivity data (uses original if None)

        Returns:
            Normalized input features
        """
        if ndvi_data is None:
            ndvi_data = self.data['ndvi']
        if emissivity_data is None:
            emissivity_data = self.data['emissivity']
        # Prepare feature list
        features_list = [
            self._normalize_with(self.data['x_coords'], 'x').ravel(),
            self._normalize_with(self.data['y_coords'], 'y').ravel(),
            self._normalize_with(ndvi_data, 'ndvi').ravel(),
            self._normalize_with(emissivity_data, 'emissivity').ravel(),
        ]
        # Add urban features
        for feature_name in ['buildings', 'roads', 'rails', 'parks', 'water', 'trees']:
            features_list.append(self.data['urban_features'][feature_name].ravel())
        return np.column_stack(features_list)

    def predict_temperature(self, input_features: np.ndarray) -> np.ndarray:
        """
        Generate temperature predictions using the trained model.

        Args:
            input_features: Preprocessed input features

        Returns:
            Temperature predictions in Celsius
        """
        input_tensor = torch.tensor(input_features, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            predictions_normalized = self.model(input_tensor).cpu().numpy()
        # Denormalize predictions
        predictions_kelvin = (predictions_normalized *
                              (self.norm_params['lst']['max'] - self.norm_params['lst']['min']) +
                              self.norm_params['lst']['min'])
        predictions_celsius = predictions_kelvin - 273.15
        return predictions_celsius.reshape(self.data['lst'].shape)

class HotspotAnalyzer:
    """Analyzes urban heat islands and identifies hotspots."""

    def __init__(self):
        """Initialize hotspot analyzer."""
        self.classification_thresholds = {}

    def classify_thermal_spots(self, temperature_data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Classify thermal spots into cool and hot categories.

        Args:
            temperature_data: Temperature data in Celsius

        Returns:
            Tuple of (classified_array, threshold_dictionary)
        """
        classified = np.zeros_like(temperature_data, dtype=int)
        # Define percentile thresholds
        cool_percentiles = [5, 10, 15]  # Cool spots: 5th, 10th, 15th percentiles
        hot_percentiles = [95, 90, 85]  # Hot spots: 95th, 90th, 85th percentiles
        # Calculate thresholds
        for i, pct in enumerate(cool_percentiles):
            self.classification_thresholds[f'cool_level_{i + 1}'] = {
                'percentile': pct,
                'temperature': np.percentile(temperature_data, pct)
            }
        for i, pct in enumerate(hot_percentiles):
            self.classification_thresholds[f'hot_level_{i + 1}'] = {
                'percentile': pct,
                'temperature': np.percentile(temperature_data, pct)
            }
        # Apply classification
        # Cool spots (1-3)
        classified[temperature_data <= self.classification_thresholds['cool_level_1']['temperature']] = 1
        classified[(temperature_data > self.classification_thresholds['cool_level_1']['temperature']) &
                   (temperature_data <= self.classification_thresholds['cool_level_2']['temperature'])] = 2
        classified[(temperature_data > self.classification_thresholds['cool_level_2']['temperature']) &
                   (temperature_data <= self.classification_thresholds['cool_level_3']['temperature'])] = 3
        # Hot spots (4-6)
        classified[temperature_data >= self.classification_thresholds['hot_level_1']['temperature']] = 6
        classified[(temperature_data < self.classification_thresholds['hot_level_1']['temperature']) &
                   (temperature_data >= self.classification_thresholds['hot_level_2']['temperature'])] = 5
        classified[(temperature_data < self.classification_thresholds['hot_level_2']['temperature']) &
                   (temperature_data >= self.classification_thresholds['hot_level_3']['temperature'])] = 4
        return classified, self.classification_thresholds

    def calculate_uhti(self, lst_celsius: np.ndarray, ndvi: np.ndarray,
                       emissivity: np.ndarray) -> np.ndarray:
        """
        Calculate Urban Heat Thermal Index (UHTI).

        Args:
            lst_celsius: Land surface temperature in Celsius
            ndvi: Normalized Difference Vegetation Index
            emissivity: Surface emissivity

        Returns:
            UHTI values
        """
        # Ensure emissivity has safe values
        emissivity_safe = np.where(emissivity <= 0, 0.01, emissivity)
        # UHTI formula: LST * (1 - NDVI) * (1 / emissivity)
        return lst_celsius * (1 - ndvi) * (1 / emissivity_safe)

    def create_hotspot_masks(self, classified_data: np.ndarray) -> np.ndarray:
        """
        Create binary mask for hotspot areas.

        Args:
            classified_data: Classified thermal spots

        Returns:
            Binary mask where True indicates hotspots
        """
        return np.isin(classified_data, [4, 5, 6])  # Hot spot categories


class GreenRoofScenario:
    """Manages green roof implementation scenarios."""

    def __init__(self, ndvi_target: float = 0.65):
        """
        Initialize green roof scenario.

        Args:
            ndvi_target: Target NDVI value for green roofs
        """
        self.ndvi_target = ndvi_target
        self.emissivity_target = self._calculate_green_roof_emissivity(ndvi_target)
        print(f"Green roof parameters - NDVI: {ndvi_target:.3f}, Emissivity: {self.emissivity_target:.3f}")

    @staticmethod
    def _calculate_green_roof_emissivity(ndvi: float) -> float:
        """
        Calculate emissivity for green roofs based on NDVI.

        Args:
            ndvi: Target NDVI value

        Returns:
            Calculated emissivity
        """
        # Physical parameters
        NDVI_soil = 0.2
        NDVI_veg = 0.5
        epsilon_veg = 0.985
        epsilon_soil = 0.96
        # Calculate vegetation fraction
        pv = np.clip((ndvi - NDVI_soil) / (NDVI_veg - NDVI_soil), 0, 1)
        # Calculate composite emissivity
        return epsilon_veg * pv + epsilon_soil * (1 - pv)

    def create_building_near_rail_mask(self, buildings_mask: np.ndarray,
                                       rails_mask: np.ndarray, buffer_pixels: int = 15) -> np.ndarray:
        """
        Create mask for buildings near railway lines.

        Args:
            buildings_mask: Binary mask of buildings
            rails_mask: Binary mask of railway lines
            buffer_pixels: Buffer distance in pixels

        Returns:
            Binary mask of buildings near railways
        """
        # Dilate rails mask to create buffer
        structure = np.ones((buffer_pixels * 2 + 1, buffer_pixels * 2 + 1))
        dilated_rails = binary_dilation(rails_mask, structure=structure)
        # Buildings near rails are those overlapping with dilated rails
        return buildings_mask & dilated_rails

    def analyze_building_characteristics(self, building_mask: np.ndarray, hotspot_mask: np.ndarray,
                                         ndvi_data: np.ndarray, emissivity_data: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray]:
        """
        Analyze NDVI and emissivity characteristics for buildings in hotspots.

        Args:
            building_mask: Binary mask of buildings
            hotspot_mask: Binary mask of hotspots
            ndvi_data: NDVI values
            emissivity_data: Emissivity values

        Returns:
            Tuple of (NDVI values, emissivity values) for selected buildings
        """
        combined_mask = building_mask.astype(bool) & hotspot_mask
        ndvi_vals = ndvi_data[combined_mask]
        em_vals = emissivity_data[combined_mask]
        return ndvi_vals, em_vals

    @staticmethod
    def remove_outliers_iqr(data: np.ndarray) -> np.ndarray:
        """
        Remove outliers using Interquartile Range method.

        Args:
            data: Input data array

        Returns:
            Data with outliers removed
        """
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return data[(data >= lower_bound) & (data <= upper_bound)]

    def create_green_roof_mask(self, building_mask: np.ndarray, hotspot_mask: np.ndarray,
                               ndvi_data: np.ndarray, emissivity_data: np.ndarray) -> np.ndarray:
        """
        Create mask for buildings suitable for green roof implementation.

        Args:
            building_mask: Binary mask of buildings
            hotspot_mask: Binary mask of hotspots
            ndvi_data: NDVI values
            emissivity_data: Emissivity values

        Returns:
            Binary mask for green roof implementation
        """
        # Analyze building characteristics
        ndvi_vals, em_vals = self.analyze_building_characteristics(
            building_mask, hotspot_mask, ndvi_data, emissivity_data
        )
        # Remove outliers
        ndvi_clean = self.remove_outliers_iqr(ndvi_vals)
        em_clean = self.remove_outliers_iqr(em_vals)
        # Calculate IQR bounds
        ndvi_q1, ndvi_q3 = np.percentile(ndvi_clean, [25, 75])
        ndvi_iqr = ndvi_q3 - ndvi_q1
        ndvi_lower = ndvi_q1 - 1.5 * ndvi_iqr
        ndvi_upper = ndvi_q3 + 1.5 * ndvi_iqr
        em_q1, em_q3 = np.percentile(em_clean, [25, 75])
        em_iqr = em_q3 - em_q1
        em_lower = em_q1 - 1.5 * em_iqr
        em_upper = em_q3 + 1.5 * em_iqr
        # Create IQR filter mask
        iqr_mask = (
                (ndvi_data >= ndvi_lower) & (ndvi_data <= ndvi_upper) &
                (emissivity_data >= em_lower) & (emissivity_data <= em_upper)
        )
        # Combine with building and hotspot masks
        combined_mask = building_mask.astype(bool) & hotspot_mask
        return combined_mask & iqr_mask


class ResultVisualizer:
    """Handles visualization of green roof analysis results."""

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

    def plot_comparison(self, original_lst: np.ndarray, green_roof_lst: np.ndarray,
                        temperature_reduction: np.ndarray, bounds: rasterio.coords.BoundingBox,
                        scenario_name: str, location_name: str = "Study Area"):
        """
        Create comparison plot between original and green roof scenarios.

        Args:
            original_lst: Original temperature predictions
            green_roof_lst: Green roof scenario predictions
            temperature_reduction: Temperature reduction values
            bounds: Geographic bounds for plotting
            scenario_name: Name of the green roof scenario
            location_name: Name of study area
        """
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        # Extract bounds
        left, bottom, right, top = bounds.left, bounds.bottom, bounds.right, bounds.top
        # Define subplot data and titles
        plot_data = [original_lst, green_roof_lst, temperature_reduction]
        titles = ["Predicted LST (PINN)", "Green Roof Predicted LST", "Temperature Reduction"]
        cmaps = ['YlOrRd', 'YlOrRd', 'GnBu_r']
        for i, (ax, data, cmap, title) in enumerate(zip(axes, plot_data, cmaps, titles)):
            # Plot data
            im = ax.imshow(data, cmap=cmap, extent=(left, right, bottom, top))
            ax.set_title(title, fontsize=16, pad=10)

            # Configure axes
            self._configure_axis(ax, bounds, i)

            # Add colorbar
            self._add_colorbar(fig, im, ax, i, data)
        plt.suptitle(f"Green Roof Analysis - {scenario_name}\n{location_name}",
                     fontsize=18, y=0.95)
        plt.tight_layout()
        # Save figure
        safe_name = scenario_name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
        save_path = self.output_dir / f"{safe_name}_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Comparison plot saved to: {save_path}")

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
        cbar.ax.tick_params(labelsize=10)
        # Set colorbar label and formatting
        if subplot_index in [0, 1]:  # Temperature plots
            cbar.set_label('Temperature (°C)', fontsize=12, labelpad=10)
            # Format ticks with 1 decimal
            vmin, vmax = im.get_clim()
            ticks = np.linspace(vmin, vmax, 5)
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f"{tick:.1f}" for tick in ticks])
        else:  # Difference plot
            cbar.set_label('ΔT (°C)', fontsize=12, labelpad=10)
            # Format ticks with 2 decimals
            vmin, vmax = im.get_clim()
            ticks = np.linspace(vmin, vmax, 5)
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f"{tick:.2f}" for tick in ticks])

    def plot_temperature_distributions(self, original_temps: np.ndarray,
                                       green_roof_temps: np.ndarray,
                                       temperature_reduction: np.ndarray,
                                       scenario_name: str):
        """
        Plot temperature distribution histograms.

        Args:
            original_temps: Original temperature values
            green_roof_temps: Green roof scenario temperatures
            temperature_reduction: Temperature reduction values
            scenario_name: Name of the scenario
        """
        fig, axes = plt.subplots(1, 3, figsize=(22, 8))
        # Set larger font sizes
        plt.rcParams.update({'font.size': 14})
        # Plot configurations
        plot_configs = [
            (original_temps, 'red', 'Predicted LST (PINN)', '°C'),
            (green_roof_temps, 'coral', 'Green Roof Predicted LST', '°C'),
            (temperature_reduction, 'blue', 'Temperature Reduction', 'ΔT (°C)')
        ]
        for i, (data, color, title, xlabel) in enumerate(plot_configs):
            self._plot_histogram(axes[i], data.flatten(), color, title, xlabel)
        plt.suptitle(f"Temperature Distributions - {scenario_name}", fontsize=18)
        plt.tight_layout()
        # Save figure
        safe_name = scenario_name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
        save_path = self.output_dir / f"{safe_name}_distributions.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Distribution plot saved to: {save_path}")

    def _plot_histogram(self, ax: plt.Axes, data: np.ndarray, color: str,
                        title: str, xlabel: str):
        """Plot histogram with percentage values."""
        # Define bins
        vmin, vmax = np.min(data), np.max(data)
        bins = np.linspace(vmin, vmax, 6)  # 5 bins
        # Calculate percentages
        counts, _ = np.histogram(data, bins=bins)
        percentages = (counts / len(data)) * 100
        # Plot histogram
        ax.bar(bins[:-1], percentages, width=np.diff(bins), alpha=0.7,
               color=color, edgecolor='black', align='edge')
        # Styling
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel('Percentage of Pixels (%)', fontsize=14)
        ax.set_xticks(bins)
        ax.set_xticklabels([f"{tick:.1f}" for tick in bins])
        ax.tick_params(axis='both', labelsize=12)
        ax.legend([title], fontsize=12, loc='best')
        ax.grid(True, alpha=0.3)
        # Add percentage labels
        for j, (p, width) in enumerate(zip(percentages, np.diff(bins))):
            if p > 0:
                ax.annotate(f'{p:.1f}%',
                            (bins[j] + width / 2, p),
                            ha='center', va='bottom', fontsize=11, fontweight='bold')

class UrbanHeatPINN(nn.Module):
    """
    Physics-Informed Neural Network for Urban Heat Island modeling.

    Matches the architecture used during training.
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
    """Main function to run green roof temperature reduction analysis."""
    # Configuration
    config = {
        'model_path': "/your_PINNs_model_saving_directory/pinn_model.pth",
        'data_paths': {
            'lst_path': "/your_GEE_data_directory/LS.tif",
            'ndvi_path': "/your_GEE_data_directory/NDVI.tif",
            'emissivity_path': "/your_GEE_data_directory/EM.tif",
            'urban_features_dir': "/your_urban_features_data_directory/",
        },
        'green_roof': {
            'ndvi_target': 0.65,
            'rail_buffer_pixels': 15
        },
        'output': {
            'plots_dir': "/your_directory_to_save_plots/",
            'location_name': "city_study_area"
        }
    }
    try:
        # Initialize components
        analyzer = GreenRoofAnalyzer(config)
        hotspot_analyzer = HotspotAnalyzer()
        green_roof_scenario = GreenRoofScenario(config['green_roof']['ndvi_target'])
        visualizer = ResultVisualizer(config['output']['plots_dir'])
        # Load model and data
        analyzer.load_model()
        analyzer.load_and_preprocess_data()
        # Get original predictions
        original_inputs = analyzer.prepare_input_features()
        original_temps = analyzer.predict_temperature(original_inputs)
        # Convert true LST to Celsius for analysis
        true_lst_celsius = analyzer.data['lst'] - 273.15
        # Analyze hotspots
        lst_classified, lst_thresholds = hotspot_analyzer.classify_thermal_spots(true_lst_celsius)
        uhti = hotspot_analyzer.calculate_uhti(
            true_lst_celsius, analyzer.data['ndvi'], analyzer.data['emissivity']
        )
        uhti_classified, uhti_thresholds = hotspot_analyzer.classify_thermal_spots(uhti)
        lst_hotspots = hotspot_analyzer.create_hotspot_masks(lst_classified)
        uhti_hotspots = hotspot_analyzer.create_hotspot_masks(uhti_classified)
        # Create building masks
        buildings_near_rails = green_roof_scenario.create_building_near_rail_mask(
            analyzer.data['urban_features']['buildings'],
            analyzer.data['urban_features']['rails'],
            config['green_roof']['rail_buffer_pixels']
        )
        # Remove overlaps
        buildings_no_rails = (analyzer.data['urban_features']['buildings'] &
                              ~buildings_near_rails)
        # Define analysis cases
        analysis_cases = [
            ("Buildings with LST Hotspots", buildings_no_rails, lst_hotspots),
            ("Buildings with UHTI Hotspots", buildings_no_rails, uhti_hotspots),
            ("Buildings near Rails with LST Hotspots", buildings_near_rails, lst_hotspots),
            ("Buildings near Rails with UHTI Hotspots", buildings_near_rails, uhti_hotspots)
        ]
        print("\n" + "=" * 60)
        print("GREEN ROOF TEMPERATURE REDUCTION ANALYSIS")
        print("=" * 60)
        # Analyze each case
        for case_name, building_mask, hotspot_mask in analysis_cases:
            print(f"\nAnalyzing: {case_name}")
            # Create green roof mask
            green_roof_mask = green_roof_scenario.create_green_roof_mask(
                building_mask, hotspot_mask,
                analyzer.data['ndvi'], analyzer.data['emissivity']
            )
            # Apply green roof modifications
            ndvi_gr = analyzer.data['ndvi'].copy()
            emissivity_gr = analyzer.data['emissivity'].copy()
            ndvi_gr[green_roof_mask] = green_roof_scenario.ndvi_target
            emissivity_gr[green_roof_mask] = green_roof_scenario.emissivity_target
            # Get predictions with green roofs
            green_roof_inputs = analyzer.prepare_input_features(ndvi_gr, emissivity_gr)
            green_roof_temps = analyzer.predict_temperature(green_roof_inputs)
            # Calculate temperature reduction
            temp_reduction = green_roof_temps - original_temps
            print(f"  Temperature reduction range: {temp_reduction.min():.3f}°C to {temp_reduction.max():.3f}°C")
            print(f"  Mean temperature reduction: {temp_reduction.mean():.3f}°C")
            print(f"  Pixels modified: {np.sum(green_roof_mask)}")
            # Generate visualizations
            visualizer.plot_comparison(
                original_temps, green_roof_temps, temp_reduction,
                analyzer.data['bounds'], case_name, config['output']['location_name']
            )
            visualizer.plot_temperature_distributions(
                original_temps, green_roof_temps, temp_reduction, case_name
            )
        print(f"\nAnalysis completed successfully!")
        print(f"All results saved to: {config['output']['plots_dir']}")
    except Exception as e:
        print(f"Error during green roof analysis: {e}")
        raise

if __name__ == "__main__":
    main()