"""
PINNs for UHI Modelling
Predicts LST using satellite data and urban features
"""

import os
import warnings
import numpy as np
import rasterio
import geopandas as gpd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from shapely.geometry import Point
from torch.utils.data import DataLoader, TensorDataset
from matplotlib.ticker import FuncFormatter
from rasterio.features import rasterize
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import MinMaxScaler
from scipy import ndimage
import json


class Config:
    """Configuration class for model parameters and file paths"""
    # File paths
    INPUT_DIR = "/your_directory/paris_GEE/"
    OUTPUT_DIR = "/your_directory/Paris/"
    # Model parameters
    EPOCHS = 200
    BATCH_SIZE = 512
    LEARNING_RATE = 0.001
    BOUNDARY_WIDTH = 5
    # Physics parameters
    DATA_LOSS_WEIGHT = 1.0
    PHYSICS_LOSS_WEIGHT = 0.1
    BOUNDARY_LOSS_WEIGHT = 0.3
    # Thermal diffusivity values (normalized)
    K_URBAN = 1.0  # Reference urban thermal diffusivity
    K_VEG = 0.33  # Vegetation thermal diffusivity
    K_WATER = 0.19  # Water thermal diffusivity

class DataProcessor:
    """Handles data loading and preprocessing"""
    @staticmethod
    def load_raster(path):
        """Load raster file and return data with metadata"""
        with rasterio.open(path) as src:
            data = src.read(1)
            transform = src.transform
            bounds = src.bounds
            crs = src.crs
        return data, transform, bounds, crs
    @staticmethod
    def normalize(data):
        """Normalize data to [0, 1] range"""
        return (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))
    @staticmethod
    def geojson_to_array(geojson_path, x_coords, y_coords):
        """Convert GeoJSON features to binary array mask"""
        gdf = gpd.read_file(geojson_path)
        points = [Point(x, y) for x, y in zip(x_coords.ravel(), y_coords.ravel())]
        return np.array([gdf.contains(p).any() for p in points]).reshape(x_coords.shape)
    @staticmethod
    def compute_distance_transform(geometries, target_shape, transform):
        """Compute distance transform to vector features"""
        presence_raster = rasterize(
            [(geom, 1) for geom in geometries],
            out_shape=target_shape,
            transform=transform,
            fill=0,
            dtype=np.float32
        )
        distance_transform = ndimage.distance_transform_edt(1 - presence_raster)
        return distance_transform

class UrbanHeatPINN(nn.Module):
    """PINN for UHI modelling"""
    def __init__(self, input_dim):
        super().__init__()
        # Input processing layers with Tanh for smooth feature extraction
        self.input_layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
        )
        # Middle layers with LeakyReLU for better gradient flow
        self.middle_layers = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.01),
        )
        # Physics-informed layers with Tanh for smooth derivatives
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
    def forward(self, x):
        x = self.input_layers(x)
        x = self.middle_layers(x)
        x = self.physics_layers(x)
        return self.output_layers(x)
    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class PhysicsInformedLoss:
    """Handles physics-informed loss calculations"""
    @staticmethod
    def material_properties(materials_tensor):
        """
        Compute thermal diffusivity based on urban material composition
        Args:
            materials_tensor: torch.Tensor of shape (batch_size, num_features)
                            [buildings, roads, rails, parks, water, trees]
        Returns:
            k: torch.Tensor of thermal diffusivity values (batch_size, 1)
        """
        buildings = materials_tensor[:, 0]
        roads = materials_tensor[:, 1]
        rails = materials_tensor[:, 2]
        parks = materials_tensor[:, 3]
        water = materials_tensor[:, 4]
        trees = materials_tensor[:, 5]
        # Create material masks
        urban_mask = (buildings > 0.5) | (roads > 0.5) | (rails > 0.5)
        veg_mask = (parks > 0.5) | (trees > 0.5)
        water_mask = water > 0.5
        # Initialize with urban default
        k = torch.ones_like(buildings) * Config.K_URBAN
        # Override with vegetation and water values
        k = torch.where(veg_mask, Config.K_VEG, k)
        k = torch.where(water_mask, Config.K_WATER, k)
        # Handle mixed pixels using weighted average
        urban_weight = buildings + roads + rails
        veg_weight = parks + trees
        total_weight = urban_weight + veg_weight + water

        valid_mask = total_weight > 0
        mixed_k = torch.zeros_like(k)
        mixed_k[valid_mask] = (
                                      (urban_weight[valid_mask] * Config.K_URBAN) +
                                      (veg_weight[valid_mask] * Config.K_VEG) +
                                      (water[valid_mask] * Config.K_WATER)
                              ) / total_weight[valid_mask]

        # Use mixed_k where multiple materials coexist
        mixed_pixel_mask = ((urban_mask & veg_mask) |
                            (urban_mask & water_mask) |
                            (veg_mask & water_mask))
        k = torch.where(mixed_pixel_mask, mixed_k, k)
        return k.unsqueeze(1)

    @staticmethod
    def create_heat_source_field(lst_shape, lst_transform, output_dir):
        """
        Create balanced heat source field with both sources and sinks using OSM data
        """
        print("Creating balanced OSM-based heat source field")
        # Load OSM data
        processor = DataProcessor()
        buildings_gdf = gpd.read_file(os.path.join(Config.INPUT_DIR, "buildings.geojson"))
        roads_gdf = gpd.read_file(os.path.join(Config.INPUT_DIR, "roads.geojson"))
        parks_gdf = gpd.read_file(os.path.join(Config.INPUT_DIR, "parks.geojson"))
        water_gdf = gpd.read_file(os.path.join(Config.INPUT_DIR, "water.geojson"))
        trees_gdf = gpd.read_file(os.path.join(Config.INPUT_DIR, "trees.geojson"))
        # Heat Sources (Positive Q)
        print("Computing heat sources")
        # Building density
        building_density = rasterize(
            [(geom, 1) for geom in buildings_gdf.geometry],
            out_shape=lst_shape, transform=lst_transform, fill=0
        )
        building_density_smooth = gaussian_filter(building_density.astype(float), sigma=2)
        # Road network density
        road_density = rasterize(
            [(geom, 1) for geom in roads_gdf.geometry],
            out_shape=lst_shape, transform=lst_transform, fill=0
        )
        road_density_smooth = gaussian_filter(road_density.astype(float), sigma=1.5)
        # Distance to major roads
        major_roads = roads_gdf[roads_gdf['highway'].isin(['motorway', 'trunk', 'primary'])]
        if len(major_roads) == 0:
            major_roads = roads_gdf
        distance_to_major_roads = processor.compute_distance_transform(
            major_roads.geometry, lst_shape, lst_transform
        )
        max_distance = np.max(distance_to_major_roads)
        inverse_distance = max_distance - distance_to_major_roads
        # Urban canyon effect
        building_complexity = np.zeros(lst_shape)
        for geom in buildings_gdf.geometry:
            if geom.area > 0:
                perimeter_area_ratio = geom.length / geom.area
                mask = rasterize([geom], out_shape=lst_shape, transform=lst_transform)
                building_complexity += mask * min(perimeter_area_ratio, 0.1)
        urban_canyon = gaussian_filter(building_complexity, sigma=1)
        # Heat Sinks (Negative Q)
        print("Computing heat sinks")
        # Park cooling effect
        park_density = rasterize(
            [(geom, 1) for geom in parks_gdf.geometry],
            out_shape=lst_shape, transform=lst_transform, fill=0
        )
        park_density_smooth = gaussian_filter(park_density.astype(float), sigma=3)
        # Water body cooling
        water_density = rasterize(
            [(geom, 1) for geom in water_gdf.geometry],
            out_shape=lst_shape, transform=lst_transform, fill=0
        )
        water_density_smooth = gaussian_filter(water_density.astype(float), sigma=2)
        # Tree canopy cooling
        tree_density = rasterize(
            [(geom, 1) for geom in trees_gdf.geometry],
            out_shape=lst_shape, transform=lst_transform, fill=0
        )
        tree_density_smooth = gaussian_filter(tree_density.astype(float), sigma=2)
        # Distance to cooling features
        cooling_features = (list(parks_gdf.geometry) +
                            list(water_gdf.geometry) +
                            list(trees_gdf.geometry))
        distance_to_cooling = processor.compute_distance_transform(
            cooling_features, lst_shape, lst_transform
        )
        cooling_proximity = 1.0 / (1.0 + distance_to_cooling)
        # Normalize and combine components
        print("Combining sources and sinks")
        scaler = MinMaxScaler()
        # Normalize heat sources
        bd_norm = scaler.fit_transform(building_density_smooth.reshape(-1, 1)).reshape(lst_shape)
        rd_norm = scaler.fit_transform(road_density_smooth.reshape(-1, 1)).reshape(lst_shape)
        uc_norm = scaler.fit_transform(urban_canyon.reshape(-1, 1)).reshape(lst_shape)
        id_norm = scaler.fit_transform(inverse_distance.reshape(-1, 1)).reshape(lst_shape)
        # Normalize heat sinks
        pd_norm = scaler.fit_transform(park_density_smooth.reshape(-1, 1)).reshape(lst_shape)
        wd_norm = scaler.fit_transform(water_density_smooth.reshape(-1, 1)).reshape(lst_shape)
        td_norm = scaler.fit_transform(tree_density_smooth.reshape(-1, 1)).reshape(lst_shape)
        cp_norm = scaler.fit_transform(cooling_proximity.reshape(-1, 1)).reshape(lst_shape)

        # Urban energy balance weights
        building_w, road_w, id_w, uc_w = 0.3, 0.25, 0.20, 0.15
        park_w, water_w, tree_w, cooling_w = 0.15, 0.10, 0.10, 0.10
        Q_sources = (building_w * bd_norm +  # Building density
                     road_w * rd_norm +  # Road density
                     id_w * id_norm +  # Proximity to major roads
                     uc_w * uc_norm)  # Urban canyon

        Q_sinks = (park_w * pd_norm +  # Park cooling
                   water_w * wd_norm +  # Water cooling
                   tree_w * td_norm +  # Tree cooling
                   cooling_w * cp_norm)  # Cooling proximity

        # Final balanced Q: Sources - Sinks
        Q_balanced = Q_sources - Q_sinks
        # Visualize components
        PhysicsInformedLoss._visualize_heat_components(
            [building_density_smooth, road_density_smooth, inverse_distance, urban_canyon,
             park_density_smooth, water_density_smooth, tree_density_smooth, cooling_proximity,
             Q_sources, Q_balanced],
            ['Building Density (Source)', 'Road Density (Source)',
             'Inverse Distance to Roads (Source)', 'Urban Canyon (Source)',
             'Park Density (Sink)', 'Water Density (Sink)',
             'Tree Density (Sink)', 'Cooling Proximity (Sink)',
             'Heat Sources', 'Final Balanced Q'],
            output_dir
        )
        # Print balance statistics
        positive_fraction = np.sum(Q_balanced > 0) / Q_balanced.size
        negative_fraction = np.sum(Q_balanced < 0) / Q_balanced.size
        print(f"Balanced Q: {positive_fraction:.1%} sources, {negative_fraction:.1%} sinks")
        print(f"Q range: [{Q_balanced.min():.3f}, {Q_balanced.max():.3f}]")
        return Q_balanced.ravel()

    @staticmethod
    def _visualize_heat_components(components, titles, output_dir):
        """Visualize heat source/sink components"""
        fig, axes = plt.subplots(4, 3, figsize=(15, 12))
        for idx, (data, title) in enumerate(zip(components, titles)):
            ax = axes[idx // 3, idx % 3]
            cmap = 'hot' if 'Source' in title or 'Final' in title else 'cool'
            im = ax.imshow(data, cmap=cmap)
            ax.set_title(title, fontsize=10)
            plt.colorbar(im, ax=ax, orientation='horizontal')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "Q_balanced_components.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()

    @staticmethod
    def create_fallback_heat_source(batch_inputs):
        """Create fallback heat source field using limited features"""
        # Sources (positive)
        albedo = 1.0 - batch_inputs[:, 3:4]  # 1 - emissivity
        urban_density = batch_inputs[:, 4:5]  # buildings
        # Sinks (negative)
        impervious_fraction = batch_inputs[:, 4:5] + batch_inputs[:, 5:6]  # buildings + roads
        green_fraction = batch_inputs[:, 7:8]  # parks only
        # Balanced Q: sources - sinks
        albedo_w, impervious_fraction_w, urban_density_w, green_fraction_w = 0.5, 0.25, 0.3, 0.4
        Q_fallback = (albedo_w * albedo +  # Solar absorption
                      impervious_fraction_w * impervious_fraction +
                      urban_density_w * urban_density -  # Anthropogenic heat
                      green_fraction_w * green_fraction)  # Limited cooling
        return Q_fallback

    @staticmethod
    def physics_loss(batch_inputs, batch_pred, interior_mask, batch_indices, Q_independent=None):
        """Compute physics loss with balanced heat sources and sinks"""
        batch_inputs.requires_grad_(True)
        # Compute gradients
        grads = torch.autograd.grad(
            batch_pred, batch_inputs,
            grad_outputs=torch.ones_like(batch_pred),
            create_graph=True,
            allow_unused=True,
            retain_graph=True,
        )[0]
        if grads is None:
            return torch.tensor(0.0, device=batch_pred.device)
        dTdx = grads[:, 0:1]
        dTdy = grads[:, 1:2]
        # Second derivatives
        d2Tdx2 = torch.autograd.grad(
            dTdx.sum(), batch_inputs,
            create_graph=True,
            allow_unused=True,
            retain_graph=True,
        )[0]
        d2Tdx2 = d2Tdx2[:, 0:1] if d2Tdx2 is not None else torch.zeros_like(dTdx)
        d2Tdy2 = torch.autograd.grad(
            dTdy.sum(), batch_inputs,
            create_graph=True,
            allow_unused=True,
            retain_graph=True,
        )[0]
        d2Tdy2 = d2Tdy2[:, 1:2] if d2Tdy2 is not None else torch.zeros_like(dTdy)
        # Material properties
        materials = batch_inputs[:, 4:10]
        k = PhysicsInformedLoss.material_properties(materials)
        # Choose Q source
        if Q_independent is not None:
            batch_Q = Q_independent[batch_indices].unsqueeze(1)
        else:
            batch_Q = PhysicsInformedLoss.create_fallback_heat_source(batch_inputs)
        # Heat equation: k∇²T = Q
        laplacian_T = d2Tdx2 + d2Tdy2
        residual = k * laplacian_T - batch_Q
        # Apply interior mask
        residual = residual[interior_mask]
        if len(residual) == 0:
            return torch.tensor(0.0, device=batch_pred.device)
        physics_loss_val = torch.mean(residual ** 2)
        return physics_loss_val

    @staticmethod
    def boundary_loss(batch_inputs, batch_pred, batch_targets, boundary_mask):
        """Compute boundary loss with mixed boundary conditions"""
        # Extract features
        parks = batch_inputs[:, 7] > 0.5
        water = batch_inputs[:, 8] > 0.5
        # Create masks
        neumann_mask = boundary_mask & (water | parks)  # Water OR parks
        dirichlet_mask = boundary_mask & ~(water | parks)  # Urban areas
        mse_loss = nn.MSELoss()
        # Dirichlet Loss (temperature match for urban boundaries)
        dirichlet_loss = torch.tensor(0.0, device=batch_pred.device)
        if torch.any(dirichlet_mask):
            dirichlet_loss = mse_loss(
                batch_pred[dirichlet_mask],
                batch_targets[dirichlet_mask]
            )
        # Neumann Loss (zero heat flux for water/parks)
        neumann_loss = torch.tensor(0.0, device=batch_pred.device)
        if torch.any(neumann_mask):
            neumann_points = batch_inputs[neumann_mask]
            grads = torch.autograd.grad(
                batch_pred[neumann_mask],
                neumann_points,
                grad_outputs=torch.ones_like(batch_pred[neumann_mask]),
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0]
            if grads is not None:
                flux_magnitude = torch.norm(grads[:, :2], dim=1)
                neumann_loss = torch.mean(flux_magnitude ** 2)
        return dirichlet_loss + 0.1 * neumann_loss

class Visualization:
    """Handles visualization and plotting"""
    @staticmethod
    def plot_training_history(history, output_dir):
        """Plot training loss history"""
        plt.figure(figsize=(10, 8))
        plt.plot(history['total'], color='#1f77b4', lw=3, label='Total Loss')
        plt.plot(history['data'], color='#ff7f0e', lw=3, label='Data Loss')
        plt.plot(history['physics'], color='#2ca02c', lw=3, label='Physics Loss')
        plt.plot(history['bcs'], color='#9467bd', lw=3, label='Boundary Loss')
        plt.xlabel('Epoch', fontsize=22)
        plt.ylabel('Loss', fontsize=22)
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.legend(fontsize=22)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "training_loss.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_comparison(lst_C, lst_pred_denorm_C, abs_diff, lst_bounds, output_dir):
        """Plot comparison between true and predicted LST"""
        fig, axes = plt.subplots(1, 3, figsize=(20, 18))
        cmaps = ['YlOrRd', 'YlOrRd', 'coolwarm']
        titles = ("True LST (Landsat 8)", "Predicted LST (PINN)", "Absolute Difference")
        for i, (ax, data, cmap, title) in enumerate(zip(axes, [lst_C, lst_pred_denorm_C, abs_diff], cmaps, titles)):
            im = ax.imshow(data, cmap=cmap,
                           extent=(lst_bounds.left, lst_bounds.right,
                                   lst_bounds.bottom, lst_bounds.top))
            ax.set_title(title, fontsize=22)
            # Configure ticks and labels only for first subplot
            if i == 0:
                ax.set_xticks([lst_bounds.left, lst_bounds.right])
                ax.set_yticks([lst_bounds.bottom, lst_bounds.top])
                ax.xaxis.set_major_formatter(FuncFormatter(
                    lambda x, _: f"{abs(x):.2f}°{'E' if x >= 0 else 'W'}"
                ))
                ax.yaxis.set_major_formatter(FuncFormatter(
                    lambda y, _: f"{abs(y):.2f}°{'N' if y >= 0 else 'S'}"
                ))
                ax.tick_params(axis='x', pad=15)
                ax.tick_params(axis='both', labelsize=22)
            else:
                ax.set_xticks([])
                ax.set_yticks([])
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.04, aspect=15)
            cbar.set_label('')
            vmin, vmax = im.get_clim()
            ticks = np.linspace(vmin, vmax, 5)
            cbar.set_ticks(ticks)
            if i == 2:  # Absolute difference plot
                tick_labels = [f"{tick:.2f}" for tick in ticks]
            else:
                tick_labels = [f"{tick:.1f}" for tick in ticks]
            cbar.ax.set_xticklabels(tick_labels)
            cbar.ax.tick_params(labelsize=22)
            # Add unit labels
            if i in [0, 1]:
                cbar.ax.text(1.02, 0.5, '°C', transform=cbar.ax.transAxes,
                             va='center', ha='left', fontsize=22)
            elif i == 2:
                cbar.ax.text(1.02, 0.5, 'ΔT (°C)', transform=cbar.ax.transAxes,
                             va='center', ha='left', fontsize=22)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "prediction_comparison.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()

class Metrics:
    """Handles model evaluation metrics"""
    @staticmethod
    def compute_metrics(lst_true, lst_pred, output_dir):
        """Compute and save evaluation metrics"""
        lst_flat = lst_true.ravel()
        pred_flat = lst_pred.ravel()
        # R² Score
        ss_res = np.sum((lst_flat - pred_flat) ** 2)
        ss_tot = np.sum((lst_flat - np.mean(lst_flat)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        # Additional metrics
        rmse = np.sqrt(np.mean((lst_flat - pred_flat) ** 2))
        mae = np.mean(np.abs(lst_flat - pred_flat))
        print(f"R² Score: {r2:.4f}")
        print(f"RMSE: {rmse:.4f} °C")
        print(f"MAE: {mae:.4f} °C")
        # Save metrics
        metrics = {
            'r2': float(r2),
            'rmse': float(rmse),
            'mae': float(mae)
        }
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)
        return metrics

class UrbanHeatPINNTrainer:
    """Main trainer class for Urban Heat PINN model"""
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = DataProcessor()
        # Create output directory
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
        print(f"Output directory: {config.OUTPUT_DIR}")

    def prepare_data(self):
        """Prepare training data and features"""
        print("Loading and preparing data")
        # Load raster data
        lst, lst_transform, lst_bounds, lst_crs = self.processor.load_raster(
            os.path.join(self.config.INPUT_DIR, "LST_PR.tif"))
        ndvi, _, ndvi_bounds, _ = self.processor.load_raster(
            os.path.join(self.config.INPUT_DIR, "NDVI_PR.tif"))
        emissivity, _, emissivity_bounds, _ = self.processor.load_raster(
            os.path.join(self.config.INPUT_DIR, "EM_PR.tif"))
        # Store normalization parameters
        self.ndvi_min, self.ndvi_max = np.min(ndvi), np.max(ndvi)
        self.emissivity_min, self.emissivity_max = np.min(emissivity), np.max(emissivity)
        self.lst_min, self.lst_max = np.nanmin(lst), np.nanmax(lst)
        print(f"NDVI range: [{self.ndvi_min:.4f}, {self.ndvi_max:.4f}]")
        print(f"Emissivity range: [{self.emissivity_min:.4f}, {self.emissivity_max:.4f}]")
        print(f"LST range: [{self.lst_min:.2f}, {self.lst_max:.2f}] K")
        # Generate coordinate grid
        x_coords, y_coords = np.meshgrid(
            np.arange(lst.shape[1]) * lst_transform[0] + lst_transform[2],
            np.arange(lst.shape[0]) * lst_transform[4] + lst_transform[5])
        # Load urban features
        urban_features = {}
        feature_files = ['buildings', 'roads', 'rails', 'parks', 'water', 'trees']
        for feature in feature_files:
            path = os.path.join(self.config.INPUT_DIR, f"{feature}.geojson")
            urban_features[feature] = self.processor.geojson_to_array(path, x_coords, y_coords)
        # Create boundary mask
        boundary_mask = np.zeros_like(lst, dtype=bool)
        boundary_mask[:self.config.BOUNDARY_WIDTH, :] = True
        boundary_mask[-self.config.BOUNDARY_WIDTH:, :] = True
        boundary_mask[:, :self.config.BOUNDARY_WIDTH] = True
        boundary_mask[:, -self.config.BOUNDARY_WIDTH:] = True
        # Prepare input features
        inputs = np.column_stack((
            self.processor.normalize(x_coords).ravel(),
            self.processor.normalize(y_coords).ravel(),
            self.processor.normalize(ndvi).ravel(),
            self.processor.normalize(emissivity).ravel(),
            urban_features['buildings'].ravel(),
            urban_features['roads'].ravel(),
            urban_features['rails'].ravel(),
            urban_features['parks'].ravel(),
            urban_features['water'].ravel(),
            urban_features['trees'].ravel(),
        ))
        targets = self.processor.normalize(lst).ravel()
        # Convert to PyTorch tensors
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
        # Store masks as tensors
        self.boundary_mask_tensor = torch.tensor(boundary_mask.ravel(), dtype=torch.bool)
        self.interior_mask_tensor = torch.tensor(~boundary_mask.ravel(), dtype=torch.bool)
        # Store original data for visualization
        self.lst_original = lst
        self.lst_bounds = lst_bounds
        self.lst_transform = lst_transform
        print(f"Input features shape: {inputs.shape}")
        print(f"Targets shape: {targets.shape}")
        return inputs.shape[1]  # Return input dimension

    def create_heat_source_field(self):
        """Create independent heat source field"""
        print("Preparing balanced independent heat source field")
        try:
            Q_independent_array = PhysicsInformedLoss.create_heat_source_field(
                self.lst_original.shape, self.lst_transform, self.config.OUTPUT_DIR)
            self.Q_independent_tensor = torch.tensor(Q_independent_array, dtype=torch.float32)
            print("Using balanced OSM-based heat source field")
            return True
        except Exception as e:
            print(f"OSM-based heat source field failed: {e}")
            print("Using balanced fallback formulation")
            self.Q_independent_tensor = None
            return False
    def train(self):
        """Train the PINN model"""
        print("Starting training")
        # Prepare data
        input_dim = self.prepare_data()
        # Create heat source field
        self.create_heat_source_field()
        # Initialize model
        self.model = UrbanHeatPINN(input_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5)
        self.mse_loss = nn.MSELoss()
        print(f"Model parameters: {self.model.count_parameters():,}")
        # Move data to device
        inputs = self.inputs.to(self.device)
        targets = self.targets.to(self.device)
        boundary_mask_tensor = self.boundary_mask_tensor.to(self.device)
        interior_mask_tensor = self.interior_mask_tensor.to(self.device)
        if self.Q_independent_tensor is not None:
            Q_independent_tensor = self.Q_independent_tensor.to(self.device)
        else:
            Q_independent_tensor = None
        # Create dataset with indices
        indices = torch.arange(len(inputs))
        dataset = TensorDataset(inputs, targets, indices)
        dataloader = DataLoader(dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
        # Training history
        history = {'total': [], 'data': [], 'physics': [], 'bcs': []}
        # Training loop
        for epoch in range(self.config.EPOCHS):
            epoch_loss = 0.0
            epoch_data_loss = 0.0
            epoch_physics_loss = 0.0
            epoch_bcs_loss = 0.0
            for batch_inputs, batch_targets, batch_indices in dataloader:
                batch_indices = batch_indices.to(self.device)
                batch_inputs.requires_grad_(True)
                self.optimizer.zero_grad()
                # Forward pass
                pred = self.model(batch_inputs)
                # Get masks for current batch
                batch_boundary_mask = boundary_mask_tensor[batch_indices]
                batch_interior_mask = interior_mask_tensor[batch_indices]
                # Boundary loss
                if torch.any(batch_boundary_mask):
                    bc_loss = PhysicsInformedLoss.boundary_loss(
                        batch_inputs[batch_boundary_mask],
                        pred[batch_boundary_mask],
                        batch_targets[batch_boundary_mask],
                        batch_boundary_mask
                    )
                else:
                    bc_loss = torch.tensor(0.0, device=self.device)
                # Interior losses
                if torch.any(batch_interior_mask):
                    data_loss = self.mse_loss(
                        pred[batch_interior_mask],
                        batch_targets[batch_interior_mask]
                    )
                    phys_loss = PhysicsInformedLoss.physics_loss(
                        batch_inputs,
                        pred,
                        batch_interior_mask,
                        batch_indices,
                        Q_independent_tensor
                    )
                else:
                    data_loss = torch.tensor(0.0, device=self.device)
                    phys_loss = torch.tensor(0.0, device=self.device)
                # Total loss
                total_loss = (self.config.DATA_LOSS_WEIGHT * data_loss +
                              self.config.PHYSICS_LOSS_WEIGHT * phys_loss +
                              self.config.BOUNDARY_LOSS_WEIGHT * bc_loss)

                total_loss.backward()
                self.optimizer.step()
                # Accumulate losses
                epoch_loss += total_loss.item()
                epoch_data_loss += data_loss.item()
                epoch_physics_loss += phys_loss.item()
                epoch_bcs_loss += bc_loss.item()
            # Epoch statistics
            avg_loss = epoch_loss / len(dataloader)
            self.scheduler.step(avg_loss)
            history['total'].append(avg_loss)
            history['data'].append(epoch_data_loss / len(dataloader))
            history['physics'].append(epoch_physics_loss / len(dataloader))
            history['bcs'].append(epoch_bcs_loss / len(dataloader))
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.config.EPOCHS}: "
                      f"Loss={avg_loss:.4f} "
                      f"(Data={history['data'][-1]:.4f}, "
                      f"Physics={history['physics'][-1]:.4f}, "
                      f"BCs={history['bcs'][-1]:.4f})")
        self.history = history
        return history

    def evaluate(self):
        """Evaluate the trained model"""
        print("Evaluating model")
        # Generate predictions
        with torch.no_grad():
            inputs = self.inputs.to(self.device)
            lst_pred = self.model(inputs).cpu().numpy().reshape(self.lst_original.shape)
            # Denormalize predictions
            lst_pred_denorm_K = lst_pred * (self.lst_max - self.lst_min) + self.lst_min
            lst_pred_denorm_C = lst_pred_denorm_K - 273.15
            # Convert original LST to Celsius
            lst_C = self.lst_original - 273.15
            abs_diff = np.abs(lst_pred_denorm_C - lst_C)
        print(f"Absolute difference - Min: {np.min(abs_diff):.2f}°C, "
              f"Max: {np.max(abs_diff):.2f}°C")
        return lst_C, lst_pred_denorm_C, abs_diff

    def save_results(self):
        """Save model and results"""
        print("Saving results")
        # Save model
        model_path = os.path.join(self.config.OUTPUT_DIR, "pinn_model.pth")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.model.input_layers[0].in_features,
            'lst_min': self.lst_min,
            'lst_max': self.lst_max,
            'ndvi_min': self.ndvi_min,
            'ndvi_max': self.ndvi_max,
            'emissivity_min': self.emissivity_min,
            'emissivity_max': self.emissivity_max,
            'config': self.config.__dict__
        }, model_path)
        # Save training history
        history_path = os.path.join(self.config.OUTPUT_DIR, "training_history.pth")
        torch.save(self.history, history_path)
        print(f"Model saved to: {model_path}")
        print(f"Training history saved to: {history_path}")

    def run(self):
        """Run complete training and evaluation pipeline"""
        # Train model
        history = self.train()
        # Evaluate model
        lst_C, lst_pred_denorm_C, abs_diff = self.evaluate()
        # Save results
        self.save_results()
        # Generate visualizations
        Visualization.plot_training_history(history, self.config.OUTPUT_DIR)
        Visualization.plot_comparison(lst_C, lst_pred_denorm_C, abs_diff,
                                      self.lst_bounds, self.config.OUTPUT_DIR)
        # Compute metrics
        metrics = Metrics.compute_metrics(lst_C, lst_pred_denorm_C, self.config.OUTPUT_DIR)
        print("Training and evaluation completed successfully!")
        return metrics

def main():
    """Main function to run the Urban Heat PINN training"""
    # Suppress warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    try:
        # Initialize configuration
        config = Config()
        # Create and run trainer
        trainer = UrbanHeatPINNTrainer(config)
        metrics = trainer.run()
        print("\n=== Final Results ===")
        print(f"R² Score: {metrics['r2']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f} °C")
        print(f"MAE: {metrics['mae']:.4f} °C")
    except Exception as e:
        print(f"Error during execution: {e}")
        raise

if __name__ == "__main__":
    main()