"""
UHI Mitigation: Green Roof Impact Analysis using PINNs
This module implements a PINN-based approach to simulate the cooling effects of green roofs
on urban heat islands using satellite and geospatial data
"""
import os
import warnings
from typing import Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from shapely.geometry import Point
from scipy.ndimage import binary_dilation
# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

class Config:
    """Configuration class for model parameters and file paths"""
    # File paths
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
    # Model paths
    MODEL_DIR = "/your_directory/Paris/"
    OUTPUT_DIR = "/your_directory/Plots_save/"
    # Physical parameters
    NDVI_SOIL = 0.2
    NDVI_VEG = 0.5
    EPSILON_V = 0.985
    EPSILON_S = 0.96
    NDVI_GREEN_ROOF = 0.75
    BUILDING_REDUCTION_FACTOR = 0.3
    RAIL_BUFFER_PIXELS = 15
    # Analysis parameters
    COOL_PERCENTILES = [5, 10, 15]
    HOT_PERCENTILES = [95, 90, 85]
    # Plotting parameters
    PLOT_FONT_SIZE = 22
    FIGURE_SIZE = (20, 18)
    DPI = 300

class DataLoader:
    """Handles loading and preprocessing of geospatial data"""
    @staticmethod
    def load_raster(path: str) -> Tuple[np.ndarray, Any, Any, Any]:
        """
        Load raster data from file
        Args:
            path: Path to raster file
        Returns:
            Tuple of (data, transform, bounds, crs)
        """
        with rasterio.open(path) as src:
            data = src.read(1)
            transform = src.transform
            bounds = src.bounds
            crs = src.crs
        return data, transform, bounds, crs
    @staticmethod
    def load_geojson(path: str) -> gpd.GeoDataFrame:
        """
        Load GeoJSON data
        Args:
            path: Path to GeoJSON file
        Returns:
            GeoDataFrame containing the data
        """
        return gpd.read_file(path)
    @staticmethod
    def geojson_to_array(geojson_path: str, x_coords: np.ndarray, y_coords: np.ndarray) -> np.ndarray:
        """
        Convert GeoJSON features to binary array mask
        Args:
            geojson_path: Path to GeoJSON file
            x_coords: X coordinate grid
            y_coords: Y coordinate grid
        Returns:
            Binary mask array
        """
        gdf = gpd.read_file(geojson_path)
        points = [Point(x, y) for x, y in zip(x_coords.ravel(), y_coords.ravel())]
        return np.array([gdf.contains(p).any() for p in points]).reshape(x_coords.shape)

class UrbanHeatPINN(nn.Module):
    """
    PINN for UHI modeling
    Architecture:
    - Input layers with Tanh activation
    - Middle layers with LeakyReLU activation
    - Physics-informed layers with Tanh activation
    - Output layers with Tanh activation
    """
    def __init__(self, input_dim: int):
        """
        Initialize the PINN model
        Args:
            input_dim: Number of input features
        """
        super().__init__()
        self.input_layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh()
        )
        self.middle_layers = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.01)
        )
        self.physics_layers = nn.Sequential(
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh()
        )
        self.output_layers = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network
        Args:
            x: Input tensor
        Returns:
            Model predictions
        """
        x = self.input_layers(x)
        x = self.middle_layers(x)
        x = self.physics_layers(x)
        return self.output_layers(x)

class UrbanHeatAnalyzer:
    """Main class for UHI analysis and green roof simulation"""
    def __init__(self, config: Config = Config()):
        """
        Initialize the analyzer with configuration
        Args:
            config: Configuration object
        """
        self.config = config
        self.data_loader = DataLoader()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.normalization_params = None
        self.lst_range = None
        # Create output directory
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
    @staticmethod
    def calculate_emissivity(ndvi: np.ndarray) -> np.ndarray:
        """
        Calculate emissivity based on NDVI values
        Args:
            ndvi: NDVI array
        Returns:
            Emissivity array
        """
        Pv = np.clip(
            (ndvi - Config.NDVI_SOIL) / (Config.NDVI_VEG - Config.NDVI_SOIL), 0, 1
        )
        return Config.EPSILON_V * Pv + Config.EPSILON_S * (1 - Pv)
    @staticmethod
    def remove_outliers_iqr(data: np.ndarray) -> np.ndarray:
        """
        Remove outliers using Interquartile Range method (IQR)
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
    @staticmethod
    def create_building_near_rail_mask(buildings_mask: np.ndarray,
                                       rails_mask: np.ndarray,
                                       buffer_pixels: int = Config.RAIL_BUFFER_PIXELS) -> np.ndarray:
        """
        Create mask for buildings near railways
        Args:
            buildings_mask: Binary building mask
            rails_mask: Binary railway mask
            buffer_pixels: Buffer distance in pixels
        Returns:
            Binary mask of buildings near railways
        """
        structure = np.ones((buffer_pixels * 2 + 1, buffer_pixels * 2 + 1))
        dilated_rails = binary_dilation(rails_mask, structure=structure)
        return buildings_mask & dilated_rails
    @staticmethod
    def classify_hotspots(data: np.ndarray, levels: int = 3) -> Tuple[np.ndarray, Dict]:
        """
        Classify hotspots based on percentiles
        Args:
            data: Input data for classification
            levels: Number of classification levels
        Returns:
            Tuple of (classified_array, threshold_dict)
        """
        classified = np.zeros_like(data, dtype=int)
        thresholds = {}
        cool_percentiles = Config.COOL_PERCENTILES
        hot_percentiles = Config.HOT_PERCENTILES
        # Calculate cool thresholds
        for i, pct in enumerate(cool_percentiles):
            key = f'cool_level_{i + 1}'
            thresholds[key] = {
                'percentile': pct,
                'temperature': np.percentile(data, pct)
            }
        # Calculate hot thresholds
        for i, pct in enumerate(hot_percentiles):
            key = f'hot_level_{i + 1}'
            thresholds[key] = {
                'percentile': pct,
                'temperature': np.percentile(data, pct)
            }
        # Apply classification
        classified[data <= thresholds['cool_level_1']['temperature']] = 1
        classified[(data > thresholds['cool_level_1']['temperature']) &
                   (data <= thresholds['cool_level_2']['temperature'])] = 2
        classified[(data > thresholds['cool_level_2']['temperature']) &
                   (data <= thresholds['cool_level_3']['temperature'])] = 3

        classified[data >= thresholds['hot_level_1']['temperature']] = 6
        classified[(data < thresholds['hot_level_1']['temperature']) &
                   (data >= thresholds['hot_level_2']['temperature'])] = 5
        classified[(data < thresholds['hot_level_2']['temperature']) &
                   (data >= thresholds['hot_level_3']['temperature'])] = 4
        return classified, thresholds

    def load_data(self) -> Dict[str, Any]:
        """Load and prepare all required data"""
        print("Loading data")
        # Load base rasters
        lst, lst_transform, lst_bounds, lst_crs = self.data_loader.load_raster(
            self.config.DATA_PATHS['lst']
        )
        ndvi, _, ndvi_bounds, _ = self.data_loader.load_raster(
            self.config.DATA_PATHS['ndvi']
        )
        emissivity, _, emissivity_bounds, _ = self.data_loader.load_raster(
            self.config.DATA_PATHS['emissivity']
        )
        # Generate coordinate grid
        x_coords, y_coords = np.meshgrid(
            np.arange(lst.shape[1]) * lst_transform[0] + lst_transform[2],
            np.arange(lst.shape[0]) * lst_transform[4] + lst_transform[5]
        )
        # Load urban features
        urban_features = {}
        for name, path in self.config.DATA_PATHS.items():
            if name in ['buildings', 'roads', 'rails', 'parks', 'water', 'trees']:
                urban_features[name] = self.data_loader.geojson_to_array(
                    path, x_coords, y_coords
                )
        # Store normalization parameters
        self.normalization_params = (
            x_coords.min(), x_coords.max(),
            y_coords.min(), y_coords.max(),
            ndvi.min(), ndvi.max(),
            emissivity.min(), emissivity.max()
        )
        self.lst_range = (lst.min(), lst.max())
        return {
            'lst': lst, 'lst_transform': lst_transform, 'lst_bounds': lst_bounds, 'lst_crs': lst_crs,
            'ndvi': ndvi, 'emissivity': emissivity,
            'x_coords': x_coords, 'y_coords': y_coords,
            'urban_features': urban_features
        }
    def load_model(self) -> None:
        """Load the trained PINN model"""
        print("Loading model")
        checkpoint_path = os.path.join(self.config.MODEL_DIR, "pinn_model.pth")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        self.model = UrbanHeatPINN(checkpoint['input_dim']).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
    @staticmethod
    def normalize_data(data: np.ndarray, dmin: float, dmax: float) -> np.ndarray:
        """Normalize data to [0, 1] range"""
        return (data - dmin) / (dmax - dmin)
    def prepare_input_features(self, data_dict: Dict[str, Any]) -> np.ndarray:
        """Prepare normalized input features for model prediction"""
        x_coords = data_dict['x_coords']
        y_coords = data_dict['y_coords']
        ndvi = data_dict['ndvi']
        emissivity = data_dict['emissivity']
        urban_features = data_dict['urban_features']
        x_min, x_max, y_min, y_max, ndvi_min, ndvi_max, em_min, em_max = self.normalization_params
        inputs = np.column_stack((
            self.normalize_data(x_coords, x_min, x_max).ravel(),
            self.normalize_data(y_coords, y_min, y_max).ravel(),
            self.normalize_data(ndvi, ndvi_min, ndvi_max).ravel(),
            self.normalize_data(emissivity, em_min, em_max).ravel(),
            urban_features['buildings'].ravel(),
            urban_features['roads'].ravel(),
            urban_features['rails'].ravel(),
            urban_features['parks'].ravel(),
            urban_features['water'].ravel(),
            urban_features['trees'].ravel(),
        ))
        return inputs
    def apply_green_roof_modifications(self, building_hotspot_mask: np.ndarray,
                                       ndvi: np.ndarray, emissivity: np.ndarray,
                                       buildings: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply green roof modifications to surface properties
        Args:
            building_hotspot_mask: Mask of buildings identified for green roofs
            ndvi: Original NDVI array
            emissivity: Original emissivity array
            buildings: Original building fraction array
        Returns:
            Tuple of modified (ndvi, emissivity, buildings)
        """
        ndvi_gr = ndvi.copy()
        emissivity_gr = emissivity.copy()
        buildings_gr = buildings.copy()
        # Apply green roof surface properties
        ndvi_gr[building_hotspot_mask] = self.config.NDVI_GREEN_ROOF
        emissivity_gr[building_hotspot_mask] = self.calculate_emissivity(
            self.config.NDVI_GREEN_ROOF
        )
        # Reduce building fraction to simulate green roof coverage
        buildings_gr[building_hotspot_mask] = (
                buildings_gr[building_hotspot_mask] * self.config.BUILDING_REDUCTION_FACTOR
        )
        return ndvi_gr, emissivity_gr, buildings_gr

    def predict_green_roof_effect(self, building_hotspot_mask: np.ndarray,
                                  inputs_original: np.ndarray, data_dict: Dict[str, Any]) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict the cooling effect of green roofs
        Args:
            building_hotspot_mask: Mask of buildings for green roof application
            inputs_original: Original input features
            data_dict: Dictionary containing all data arrays
        Returns:
            Tuple of (original_LST, green_roof_LST, temperature_reduction)
        """
        urban_features = data_dict['urban_features']
        x_coords = data_dict['x_coords']
        y_coords = data_dict['y_coords']
        ndvi = data_dict['ndvi']
        emissivity = data_dict['emissivity']
        buildings = urban_features['buildings']
        # Apply green roof modifications
        ndvi_gr, emissivity_gr, buildings_gr = self.apply_green_roof_modifications(
            building_hotspot_mask, ndvi, emissivity, buildings
        )
        # Prepare green roof inputs
        x_min, x_max, y_min, y_max, ndvi_min, ndvi_max, em_min, em_max = self.normalization_params
        inputs_gr = np.column_stack((
            self.normalize_data(x_coords, x_min, x_max).ravel(),
            self.normalize_data(y_coords, y_min, y_max).ravel(),
            self.normalize_data(ndvi_gr, ndvi_min, ndvi_max).ravel(),
            self.normalize_data(emissivity_gr, em_min, em_max).ravel(),
            buildings_gr.ravel(),
            urban_features['roads'].ravel(),
            urban_features['rails'].ravel(),
            urban_features['parks'].ravel(),
            urban_features['water'].ravel(),
            urban_features['trees'].ravel(),
        ))
        # Convert to tensors
        inputs_original_tensor = torch.tensor(inputs_original, dtype=torch.float32).to(self.device)
        inputs_gr_tensor = torch.tensor(inputs_gr, dtype=torch.float32).to(self.device)
        # Generate predictions
        with torch.no_grad():
            lst_pred_original = self.model(inputs_original_tensor).cpu().numpy().reshape(buildings.shape)
            lst_pred_gr = self.model(inputs_gr_tensor).cpu().numpy().reshape(buildings.shape)
        # Denormalize and convert to Celsius
        lst_min, lst_max = self.lst_range
        lst_pred_denorm_K_original = lst_pred_original * (lst_max - lst_min) + lst_min
        lst_pred_denorm_C_original = lst_pred_denorm_K_original - 273.15
        lst_pred_denorm_K_gr = lst_pred_gr * (lst_max - lst_min) + lst_max
        lst_pred_denorm_C_gr = lst_pred_denorm_K_gr - 273.15
        # Calculate temperature reduction
        temp_reduction = lst_pred_denorm_C_gr - lst_pred_denorm_C_original
        # Ensure cooling effect for green roof pixels
        temp_reduction[building_hotspot_mask] = np.minimum(temp_reduction[building_hotspot_mask], 0)
        return lst_pred_denorm_C_original, lst_pred_denorm_C_gr, temp_reduction

    def analyze_case(self, name: str, building_mask: np.ndarray,
                     hotspot_mask: np.ndarray, data_dict: Dict[str, Any],
                     inputs_original: np.ndarray) -> Dict[str, Any]:
        """
        Analyze green roof impact for a specific case
        Args:
            name: Case name
            building_mask: Building mask for the case
            hotspot_mask: Hotspot mask for the case
            data_dict: Data dictionary
            inputs_original: Original input features
        Returns:
            Dictionary with analysis results
        """
        print(f"\n{'=' * 60}")
        print(f"Processing: {name}")
        print(f"{'=' * 60}")
        # Create combined mask
        combined_mask = building_mask.astype(bool) & hotspot_mask
        # Analyze NDVI and emissivity with IQR outlier removal
        ndvi_vals = data_dict['ndvi'][combined_mask]
        em_vals = data_dict['emissivity'][combined_mask]
        ndvi_vals_no_outliers = self.remove_outliers_iqr(ndvi_vals)
        em_vals_no_outliers = self.remove_outliers_iqr(em_vals)
        # Calculate IQR bounds
        ndvi_q1, ndvi_q3 = np.percentile(ndvi_vals_no_outliers, [25, 75])
        ndvi_iqr = ndvi_q3 - ndvi_q1
        ndvi_lower = ndvi_q1 - 1.5 * ndvi_iqr
        ndvi_upper = ndvi_q3 + 1.5 * ndvi_iqr
        em_q1, em_q3 = np.percentile(em_vals_no_outliers, [25, 75])
        em_iqr = em_q3 - em_q1
        em_lower = em_q1 - 1.5 * em_iqr
        em_upper = em_q3 + 1.5 * em_iqr
        # Create IQR filter mask
        iqr_mask = (
                (data_dict['ndvi'] >= ndvi_lower) & (data_dict['ndvi'] <= ndvi_upper) &
                (data_dict['emissivity'] >= em_lower) & (data_dict['emissivity'] <= em_upper)
        )
        # Identify specific building pixels for green roofs
        building_hotspot_iqr_mask = combined_mask & iqr_mask
        print(f"Pixels identified for green roofs: {np.sum(building_hotspot_iqr_mask)}")
        # Predict green roof effect
        lst_original, lst_gr, temp_reduction = self.predict_green_roof_effect(
            building_hotspot_iqr_mask, inputs_original, data_dict
        )
        # Calculate cooling metrics
        cooling_effect = -temp_reduction
        avg_cooling = np.mean(cooling_effect[building_hotspot_iqr_mask])
        max_cooling = np.max(cooling_effect[building_hotspot_iqr_mask])
        print(f"Average cooling: {avg_cooling:.3f}°C")
        print(f"Maximum cooling: {max_cooling:.3f}°C")
        print(f"Pixels with cooling: {np.sum(cooling_effect[building_hotspot_iqr_mask] > 0)}")
        print(f"Pixels with warming: {np.sum(temp_reduction[building_hotspot_iqr_mask] > 0)}")
        return {
            'lst_original': lst_original,
            'lst_green_roof': lst_gr,
            'temp_reduction': temp_reduction,
            'cooling_effect': cooling_effect,
            'avg_cooling': avg_cooling,
            'max_cooling': max_cooling,
            'pixel_count': np.sum(building_hotspot_iqr_mask),
            'building_hotspot_mask': building_hotspot_iqr_mask
        }
    def run_analysis(self) -> None:
        """Run complete green roof impact analysis"""
        print("=" * 50)
        print("GREEN ROOF PREDICTION ANALYSIS")
        print("=" * 50)
        # Load data and model
        data_dict = self.load_data()
        self.load_model()
        # Prepare input features
        inputs_original = self.prepare_input_features(data_dict)
        # Generate baseline predictions
        inputs_tensor = torch.tensor(inputs_original, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            lst_pred = self.model(inputs_tensor).cpu().numpy().reshape(data_dict['urban_features']['buildings'].shape)
            lst_pred_denorm_K = lst_pred * (self.lst_range[1] - self.lst_range[0]) + self.lst_range[0]
            lst_pred_denorm_C = lst_pred_denorm_K - 273.15
        # Convert true LST to Celsius
        lst_C = data_dict['lst'] - 273.15
        abs_diff = np.abs(lst_pred_denorm_C - lst_C)
        print(f"Absolute difference - Min: {np.min(abs_diff):.3f}, Max: {np.max(abs_diff):.3f}")
        # Calculate UHTI and classify hotspots
        emissivity_safe = np.where(data_dict['emissivity'] <= 0, 0.01, data_dict['emissivity'])
        uhti = lst_C * (1 - data_dict['ndvi']) * (1 / emissivity_safe)
        lst_classified, _ = self.classify_hotspots(lst_C)
        uhti_classified, _ = self.classify_hotspots(uhti)
        lst_hotspots_mask = np.isin(lst_classified, [4, 5, 6])
        uhti_hotspots_mask = np.isin(uhti_classified, [4, 5, 6])
        # Create building masks
        buildings = data_dict['urban_features']['buildings']
        rails = data_dict['urban_features']['rails']
        buildings_near_rails = self.create_building_near_rail_mask(buildings, rails)
        buildings_no_rails = buildings & ~buildings_near_rails
        # Define analysis cases
        cases = [
            ("Buildings with LST Hotspots", buildings_no_rails, lst_hotspots_mask),
            ("Buildings with UHTI Hotspots", buildings_no_rails, uhti_hotspots_mask),
        ]
        # Run analysis for each case
        results = {}
        for name, building_mask, hotspot_mask in cases:
            result = self.analyze_case(name, building_mask, hotspot_mask, data_dict, inputs_original)
            results[name] = result
            # Generate visualizations
            self.plot_comparison(
                result['lst_original'], result['lst_green_roof'], result['temp_reduction'],
                data_dict['lst_bounds'], name
            )
            self.plot_frequency_polygon(
                result['lst_original'], result['lst_green_roof'], result['temp_reduction'], name
            )
        # Print summary
        self.print_summary(results)

    def plot_comparison(self, lst_original: np.ndarray, lst_green_roof: np.ndarray,
                        temp_reduction: np.ndarray, bounds: Any, name: str) -> None:
        """Plot comparison between original and green roof scenarios"""
        fig, axes = plt.subplots(1, 3, figsize=self.config.FIGURE_SIZE)
        cmaps = ['OrRd', 'OrRd', 'BuGn_r']
        titles = ("PINN Predicted LST", "Green Roof LST Simulation", "Cooling Effect")
        left, right, bottom, top = bounds.left, bounds.right, bounds.bottom, bounds.top
        for i, (ax, data, cmap, title) in enumerate(
                zip(axes, [lst_original, lst_green_roof, temp_reduction], cmaps, titles)):
            im = ax.imshow(data, cmap=cmap, extent=(left, right, bottom, top))
            ax.set_title(title, fontsize=self.config.PLOT_FONT_SIZE, pad=12)
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
                ax.tick_params(axis='both', labelsize=self.config.PLOT_FONT_SIZE)
            else:
                ax.set_xticks([])
                ax.set_yticks([])
            cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.04, aspect=15)
            cbar.set_label('')
            vmin, vmax = im.get_clim()
            ticks = np.linspace(vmin, vmax, 5)
            cbar.set_ticks(ticks)
            tick_labels = [f"{tick:.2f}" for tick in ticks] if i == 2 else [f"{tick:.1f}" for tick in ticks]
            cbar.ax.set_xticklabels(tick_labels)
            cbar.ax.tick_params(labelsize=self.config.PLOT_FONT_SIZE)
            unit = 'ΔT (°C)' if i == 2 else '°C'
            cbar.ax.text(1.02, 0.5, unit, transform=cbar.ax.transAxes,
                         va='center', ha='left', fontsize=self.config.PLOT_FONT_SIZE)
        plt.tight_layout()
        safe_name = self._sanitize_filename(name)
        plt.savefig(os.path.join(self.config.OUTPUT_DIR, f"{safe_name}_comparison.png"),
                    dpi=self.config.DPI, bbox_inches='tight')
        plt.close()

    def plot_frequency_polygon(self, lst_original: np.ndarray, lst_green_roof: np.ndarray,
                               temp_reduction: np.ndarray, name: str) -> None:
        """Create frequency polygon plots"""
        fig, axes = plt.subplots(1, 3, figsize=(22, 12))
        plt.rcParams.update({'font.size': 26})
        datasets = [lst_original, lst_green_roof, temp_reduction]
        labels = ["PINN Predicted LST", "Green Roof LST Simulation", "Cooling Effect"]
        colors = ['#A23B72', '#F18F01', '#2E86AB']
        units = ['°C', '°C', 'ΔT (°C)']
        markers = ['o', 's', '^']
        for i, (ax, data, label, color, unit, marker) in enumerate(zip(axes, datasets, labels, colors, units, markers)):
            vmin, vmax = np.min(data), np.max(data)
            ticks = np.linspace(vmin, vmax, 5)
            percentages = self._calculate_percentages(data.flatten(), ticks)
            midpoints = (ticks[:-1] + ticks[1:]) / 2
            ax.plot(midpoints, percentages, marker=marker, markersize=13, linewidth=3,
                    color=color, label=label)
            ax.fill_between(midpoints, percentages, alpha=0.4, color=color)
            ax.set_xlabel(unit, fontsize=26)
            ax.set_ylabel('Percentage of Pixels (%)', fontsize=26)
            ax.set_xticks(ticks)
            tick_format = '.2f' if i == 2 else '.1f'
            ax.set_xticklabels([f"{tick:{tick_format}}" for tick in ticks])
            ax.tick_params(axis='x', labelsize=26)
            ax.tick_params(axis='y', labelsize=26)
            ax.legend(loc="upper left", fontsize=22)
            ax.grid(True, alpha=0.3)
            # Add percentage labels
            for p, x in zip(percentages, midpoints):
                if p > 0:
                    ax.annotate(f'{p:.1f}', (x, p), ha='center', va='bottom',
                                fontsize=22, fontweight='bold', xytext=(0, 9),
                                textcoords='offset points')
        plt.tight_layout()
        safe_name = self._sanitize_filename(name)
        plt.savefig(os.path.join(self.config.OUTPUT_DIR, f'frequency_polygon_{safe_name}.png'),
                    dpi=self.config.DPI, bbox_inches='tight')
        plt.close()
    @staticmethod
    def _calculate_percentages(data: np.ndarray, bins: np.ndarray) -> np.ndarray:
        """Calculate percentage of data in each bin"""
        counts, _ = np.histogram(data, bins=bins)
        return (counts / len(data)) * 100
    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """Sanitize filename by removing special characters"""
        return name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace("=", "")
    def print_summary(self, results: Dict[str, Any]) -> None:
        """Print analysis summary"""
        print("\n" + "=" * 60)
        print("GREEN ROOF ANALYSIS SUMMARY")
        print("=" * 60)
        for name, data in results.items():
            print(f"\n{name}:")
            print(f"  Pixels: {data['pixel_count']}")
            print(f"  Avg Cooling: {data['avg_cooling']:.3f}°C")
            print(f"  Max Cooling: {data['max_cooling']:.3f}°C")

def main():
    """Main execution function"""
    analyzer = UrbanHeatAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main()