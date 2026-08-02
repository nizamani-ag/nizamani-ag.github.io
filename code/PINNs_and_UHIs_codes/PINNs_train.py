"""
Physics-Informed Neural Network for Urban Heat Island Modeling

This module implements a PINN to model urban heat island effects using
remote sensing data and urban feature data.
"""

import warnings
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import rasterio
import geopandas as gpd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from shapely.geometry import Point, Polygon
warnings.filterwarnings("ignore", category=UserWarning)


class UrbanHeatDataProcessor:
    """Processes urban heat island data from multiple sources."""
    def __init__(self, data_config: Dict):
        """
        Initialize data processor.

        Args:
            data_config: Dictionary containing data paths and parameters
        """
        self.config = data_config
        self._validate_paths()

    def _validate_paths(self):
        """Validate that all required data paths exist."""
        required_paths = ['lst_path', 'ndvi_path', 'emissivity_path', 'urban_features_dir']
        for path_key in required_paths:
            path = Path(self.config[path_key])
            if not path.exists():
                raise FileNotFoundError(f"Required data path not found: {path}")

    def load_geospatial_data(self) -> Tuple[Dict, Dict]:
        """
        Load and preprocess all geospatial data.

        Returns:
            Tuple of (raster_data, normalization_params) dictionaries
        """
        print("Loading geospatial data...")
        # Load raster data
        lst_data, lst_transform, lst_bounds, lst_crs = self._load_raster(self.config['lst_path'])
        ndvi_data, _, _, _ = self._load_raster(self.config['ndvi_path'])
        emissivity_data, _, _, _ = self._load_raster(self.config['emissivity_path'])
        # Store normalization parameters before normalization
        norm_params = {
            'ndvi': {'min': np.min(ndvi_data), 'max': np.max(ndvi_data)},
            'emissivity': {'min': np.min(emissivity_data), 'max': np.max(emissivity_data)},
            'lst': {'min': np.nanmin(lst_data), 'max': np.nanmax(lst_data)}
        }
        self._validate_data_shapes(lst_data, ndvi_data, emissivity_data)
        self._check_data_quality(lst_data, ndvi_data, emissivity_data)
        # Generate coordinate grid
        x_coords, y_coords = self._generate_coordinate_grid(lst_data, lst_transform)
        # Load urban features
        urban_features = self._load_urban_features(x_coords, y_coords)
        raster_data = {
            'lst': lst_data,
            'ndvi': ndvi_data,
            'emissivity': emissivity_data,
            'transform': lst_transform,
            'bounds': lst_bounds,
            'crs': lst_crs,
            'coordinates': (x_coords, y_coords),
            'urban_features': urban_features
        }
        return raster_data, norm_params

    @staticmethod
    def _load_raster(path: str) -> Tuple[np.ndarray, rasterio.Affine, rasterio.coords.BoundingBox, rasterio.CRS]:
        """Load raster file and return data with metadata."""
        with rasterio.open(path) as src:
            data = src.read(1)
            transform = src.transform
            bounds = src.bounds
            crs = src.crs
        return data, transform, bounds, crs

    def _validate_data_shapes(self, *arrays):
        """Validate that all input arrays have the same shape."""
        shapes = [arr.shape for arr in arrays]
        if len(set(shapes)) > 1:
            raise ValueError(f"Inconsistent data shapes: {shapes}")

    def _check_data_quality(self, *arrays):
        """Check data quality and report statistics."""
        for i, arr in enumerate(arrays):
            nan_count = np.isnan(arr).sum()
            print(f"Array {i}: NaN values = {nan_count} ({nan_count / arr.size:.2%})")

    def _generate_coordinate_grid(self, data: np.ndarray, transform: rasterio.Affine) -> Tuple[np.ndarray, np.ndarray]:
        """Generate coordinate grids from raster transform."""
        height, width = data.shape
        x_coords, y_coords = np.meshgrid(
            np.arange(width) * transform[0] + transform[2],
            np.arange(height) * transform[4] + transform[5]
        )
        return x_coords, y_coords

    def _load_urban_features(self, x_coords: np.ndarray, y_coords: np.ndarray) -> Dict[str, np.ndarray]:
        """Convert urban feature GeoJSON files to raster masks."""
        feature_files = {
            'buildings': 'buildings.geojson',
            'roads': 'roads.geojson',
            'rails': 'rails.geojson',
            'parks': 'parks.geojson',
            'water': 'water.geojson',
            'trees': 'trees.geojson'
        }
        urban_features = {}
        for feature_name, filename in feature_files.items():
            file_path = Path(self.config['urban_features_dir']) / filename
            urban_features[feature_name] = self._geojson_to_array(file_path, x_coords, y_coords)
        return urban_features

    @staticmethod
    def _geojson_to_array(geojson_path: Path, x_coords: np.ndarray, y_coords: np.ndarray) -> np.ndarray:
        """Convert GeoJSON features to binary array mask."""
        gdf = gpd.read_file(geojson_path)
        points = [Point(x, y) for x, y in zip(x_coords.ravel(), y_coords.ravel())]
        return np.array([gdf.contains(p).any() for p in points]).reshape(x_coords.shape)

class UrbanHeatPINN(nn.Module):
    """
    Physics-Informed Neural Network for Urban Heat Island modeling.

    This network incorporates physical constraints through custom loss functions
    and material-dependent thermal properties.
    """
    def __init__(self, input_dim: int):
        """
        Initialize the PINN model.

        Args:
            input_dim: Number of input features
        """
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
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = self.input_layers(x)
        x = self.middle_layers(x)
        x = self.physics_layers(x)
        return self.output_layers(x)

class PhysicsInformedLoss:
    """Implements physics-informed loss functions for urban heat modeling."""

    # Thermal diffusivity constants (normalized)
    THERMAL_DIFFUSIVITY = {
        'urban': 1.0,  # Reference value
        'vegetation': 0.33,  # ~0.25e-6 / 0.75e-6
        'water': 0.19  # ~0.14e-6 / 0.75e-6
    }
    def __init__(self):
        self.mse_loss = nn.MSELoss()
    def material_properties(self, materials_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute thermal diffusivity based on urban material composition.

        Args:
            materials_tensor: Shape (batch_size, 6) containing
                            [buildings, roads, rails, parks, water, trees]

        Returns:
            Thermal diffusivity values (batch_size, 1)
        """
        buildings, roads, rails, parks, water, trees = [
            materials_tensor[:, i] for i in range(6)
        ]
        # Create material masks
        urban_mask = (buildings > 0.5) | (roads > 0.5) | (rails > 0.5)
        veg_mask = (parks > 0.5) | (trees > 0.5)
        water_mask = water > 0.5
        # Initialize with urban default
        k = torch.ones_like(buildings) * self.THERMAL_DIFFUSIVITY['urban']
        # Apply material-specific values
        k = torch.where(veg_mask, self.THERMAL_DIFFUSIVITY['vegetation'], k)
        k = torch.where(water_mask, self.THERMAL_DIFFUSIVITY['water'], k)
        # Handle mixed pixels with weighted average
        mixed_k = self._compute_mixed_thermal_diffusivity(
            buildings, roads, rails, parks, water, trees
        )
        mixed_pixel_mask = self._get_mixed_pixel_mask(
            urban_mask, veg_mask, water_mask
        )
        k = torch.where(mixed_pixel_mask, mixed_k, k)
        return k.unsqueeze(1)

    def _compute_mixed_thermal_diffusivity(self, *material_weights) -> torch.Tensor:
        """Compute thermal diffusivity for mixed pixels using weighted average."""
        urban_weight = material_weights[0] + material_weights[1] + material_weights[2]
        veg_weight = material_weights[3] + material_weights[4]
        water_weight = material_weights[5]
        total_weight = urban_weight + veg_weight + water_weight
        valid_mask = total_weight > 0
        mixed_k = torch.zeros_like(urban_weight)
        mixed_k[valid_mask] = (
                                      urban_weight[valid_mask] * self.THERMAL_DIFFUSIVITY['urban'] +
                                      veg_weight[valid_mask] * self.THERMAL_DIFFUSIVITY['vegetation'] +
                                      water_weight[valid_mask] * self.THERMAL_DIFFUSIVITY['water']
                              ) / total_weight[valid_mask]
        return mixed_k

    @staticmethod
    def _get_mixed_pixel_mask(urban_mask: torch.Tensor, veg_mask: torch.Tensor,
                              water_mask: torch.Tensor) -> torch.Tensor:
        """Identify pixels with multiple material types."""
        return (urban_mask & veg_mask) | (urban_mask & water_mask) | (veg_mask & water_mask)

    def physics_loss(self, batch_inputs: torch.Tensor, batch_pred: torch.Tensor,
                     interior_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute physics-informed loss based on heat equation.

        Args:
            batch_inputs: Input features
            batch_pred: Model predictions
            interior_mask: Mask for interior points

        Returns:
            Physics loss value
        """
        batch_inputs.requires_grad_(True)
        # Compute first derivatives
        grads = torch.autograd.grad(
            batch_pred, batch_inputs,
            grad_outputs=torch.ones_like(batch_pred),
            create_graph=True,
            retain_graph=True,
        )[0]
        if grads is None:
            return torch.tensor(0.0, device=batch_pred.device)
        dTdx = grads[:, 0:1]
        dTdy = grads[:, 1:2]
        # Compute second derivatives
        d2Tdx2 = torch.autograd.grad(
            dTdx.sum(), batch_inputs, create_graph=True, retain_graph=True
        )[0][:, 0:1]
        d2Tdy2 = torch.autograd.grad(
            dTdy.sum(), batch_inputs, create_graph=True, retain_graph=True
        )[0][:, 1:2]
        # Handle None gradients
        d2Tdx2 = d2Tdx2 if d2Tdx2 is not None else torch.zeros_like(dTdx)
        d2Tdy2 = d2Tdy2 if d2Tdy2 is not None else torch.zeros_like(dTdy)
        # Extract features and compute thermal properties
        materials = batch_inputs[:, 4:10]
        k = self.material_properties(materials)
        # Compute heat source terms
        Q = self._compute_heat_sources(batch_inputs)
        # Heat equation residual: k∇²T = Q
        laplacian_T = d2Tdx2 + d2Tdy2
        residual = k * laplacian_T - Q
        # Apply interior mask
        residual = residual[interior_mask]
        if len(residual) == 0:
            return torch.tensor(0.0, device=batch_pred.device)
        return torch.mean(residual ** 2)

    def _compute_heat_sources(self, batch_inputs: torch.Tensor) -> torch.Tensor:
        """Compute urban heat source terms."""
        ndvi = batch_inputs[:, 2:3]
        emissivity = batch_inputs[:, 3:4]
        buildings = batch_inputs[:, 4:5]
        roads = batch_inputs[:, 5:6]
        rails = batch_inputs[:, 6:7]
        parks = batch_inputs[:, 7:8]
        water = batch_inputs[:, 8:9]
        trees = batch_inputs[:, 9:10]
        ndvi_coeff, emissivity_coeff, buildings_coeff, roads_coeff, rails_coeff = 0.2, 0.05, 0.15, 0.08, 0.05
        parks_coeff, water_coeff, trees_coeff = 0.2, 0.2, 0.25
        return (ndvi_coeff * (1 - ndvi) +
                emissivity_coeff * (1 - emissivity) +
                buildings_coeff * buildings +
                roads_coeff * roads +
                rails_coeff * rails -
                parks_coeff * parks -
                water_coeff * water -
                trees_coeff * trees)

    def boundary_loss(self, batch_inputs: torch.Tensor, batch_pred: torch.Tensor,
                      batch_targets: torch.Tensor, boundary_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute boundary condition losses.

        Args:
            batch_inputs: Input features
            batch_pred: Model predictions
            batch_targets: Target values
            boundary_mask: Boundary point mask

        Returns:
            Combined boundary loss
        """
        parks = batch_inputs[:, 7] > 0.5
        water = batch_inputs[:, 8] > 0.5
        # Create boundary masks
        neumann_mask = boundary_mask & (water | parks)  # Zero flux for water/parks
        dirichlet_mask = boundary_mask & ~(water | parks)  # Fixed temp for urban
        # Dirichlet loss (temperature match)
        dirichlet_loss = torch.tensor(0.0, device=batch_pred.device)
        if torch.any(dirichlet_mask):
            dirichlet_loss = self.mse_loss(
                batch_pred[dirichlet_mask],
                batch_targets[dirichlet_mask]
            )
        # Neumann loss (zero heat flux)
        neumann_loss = torch.tensor(0.0, device=batch_pred.device)
        if torch.any(neumann_mask):
            neumann_points = batch_inputs[neumann_mask]
            grads = torch.autograd.grad(
                batch_pred[neumann_mask], neumann_points,
                grad_outputs=torch.ones_like(batch_pred[neumann_mask]),
                create_graph=True, retain_graph=True
            )[0]
            if grads is not None:
                flux_magnitude = torch.norm(grads[:, :2], dim=1)
                neumann_loss = torch.mean(flux_magnitude ** 2)
        neumann_weight = 0.1
        return dirichlet_loss + neumann_weight * neumann_loss


class UrbanHeatTrainer:
    """Handles training of the Urban Heat PINN model."""

    def __init__(self, model: nn.Module, physics_loss: PhysicsInformedLoss,
                 device: torch.device, config: Dict):
        """
        Initialize trainer.

        Args:
            model: PINN model
            physics_loss: Physics loss calculator
            device: Training device
            config: Training configuration
        """
        self.model = model
        self.physics_loss = physics_loss
        self.device = device
        self.config = config
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.get('lr', 0.001))
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', patience=config.get('scheduler_patience', 5)
        )
        self.history = {
            'total': [], 'data': [], 'physics': [], 'bcs': [], 'learning_rate': []
        }

    def prepare_data(self, raster_data: Dict, norm_params: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare training data from raster data."""
        x_coords, y_coords = raster_data['coordinates']
        urban_features = raster_data['urban_features']
        # Normalize features
        def normalize(data, min_val, max_val):
            return (data - min_val) / (max_val - min_val)
        # Prepare input features
        features_list = [
            normalize(x_coords, x_coords.min(), x_coords.max()).ravel(),
            normalize(y_coords, y_coords.min(), y_coords.max()).ravel(),
            normalize(raster_data['ndvi'], norm_params['ndvi']['min'], norm_params['ndvi']['max']).ravel(),
            normalize(raster_data['emissivity'], norm_params['emissivity']['min'],
                      norm_params['emissivity']['max']).ravel(),
        ]
        # Add urban features
        for feature_name in ['buildings', 'roads', 'rails', 'parks', 'water', 'trees']:
            features_list.append(urban_features[feature_name].ravel())

        inputs = np.column_stack(features_list)
        targets = normalize(raster_data['lst'], norm_params['lst']['min'], norm_params['lst']['max']).ravel()
        return (
            torch.tensor(inputs, dtype=torch.float32).to(self.device),
            torch.tensor(targets, dtype=torch.float32).unsqueeze(1).to(self.device)
        )

    def create_boundary_mask(self, shape: Tuple[int, int], boundary_width: int = 5) -> torch.Tensor:
        """Create boundary mask tensor."""
        boundary_mask = np.zeros(shape, dtype=bool)
        boundary_mask[:boundary_width, :] = True
        boundary_mask[-boundary_width:, :] = True
        boundary_mask[:, :boundary_width] = True
        boundary_mask[:, -boundary_width:] = True
        boundary_mask_flat = boundary_mask.ravel()
        return torch.tensor(boundary_mask_flat, dtype=torch.bool).to(self.device)

    def train(self, inputs: torch.Tensor, targets: torch.Tensor,
              boundary_mask: torch.Tensor, epochs: int) -> Dict[str, List[float]]:
        """
        Train the PINN model.

        Args:
            inputs: Input features
            targets: Target values
            boundary_mask: Boundary mask
            epochs: Number of training epochs

        Returns:
            Training history
        """
        print(f"Starting training for {epochs} epochs on {self.device}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters())}")
        # Create dataset with indices
        indices = torch.arange(len(inputs))
        dataset = TensorDataset(inputs, targets, indices)
        num_batch_size = 512
        dataloader = DataLoader(dataset, batch_size=self.config.get('batch_size', num_batch_size), shuffle=True)
        interior_mask = ~boundary_mask
        for epoch in range(epochs):
            epoch_metrics = self._train_epoch(dataloader, interior_mask, boundary_mask)
            self._update_history(epoch_metrics)
            self._log_epoch(epoch, epoch_metrics)
            # Update learning rate
            self.scheduler.step(epoch_metrics['total'])
        return self.history

    def _train_epoch(self, dataloader: DataLoader, interior_mask: torch.Tensor,
                     boundary_mask: torch.Tensor) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_loss = 0.0
        epoch_data_loss = 0.0
        epoch_physics_loss = 0.0
        epoch_bcs_loss = 0.0
        for batch_inputs, batch_targets, batch_indices in dataloader:
            batch_inputs.requires_grad_(True)
            self.optimizer.zero_grad()
            pred = self.model(batch_inputs)
            # Get masks for current batch
            batch_boundary_mask = boundary_mask[batch_indices]
            batch_interior_mask = interior_mask[batch_indices]
            # Compute losses
            bc_loss = self._compute_boundary_loss(
                batch_inputs, pred, batch_targets, batch_boundary_mask
            )
            data_loss, phys_loss = self._compute_interior_losses(
                batch_inputs, pred, batch_targets, batch_interior_mask
            )
            # Combine losses
            physics_weight, boundary_weight = 0.1, 0.3
            total_loss = data_loss + physics_weight * phys_loss + boundary_weight * bc_loss
            # Backward pass
            total_loss.backward()
            self.optimizer.step()
            # Accumulate metrics
            epoch_loss += total_loss.item()
            epoch_data_loss += data_loss.item()
            epoch_physics_loss += phys_loss.item()
            epoch_bcs_loss += bc_loss.item()
        num_batches = len(dataloader)
        return {
            'total': epoch_loss / num_batches,
            'data': epoch_data_loss / num_batches,
            'physics': epoch_physics_loss / num_batches,
            'bcs': epoch_bcs_loss / num_batches
        }

    def _compute_boundary_loss(self, batch_inputs: torch.Tensor, pred: torch.Tensor,
                               batch_targets: torch.Tensor, batch_boundary_mask: torch.Tensor) -> torch.Tensor:
        """Compute boundary condition loss."""
        if torch.any(batch_boundary_mask):
            return self.physics_loss.boundary_loss(
                batch_inputs[batch_boundary_mask],
                pred[batch_boundary_mask],
                batch_targets[batch_boundary_mask],
                torch.ones_like(batch_boundary_mask[batch_boundary_mask])
            )
        return torch.tensor(0.0, device=self.device)

    def _compute_interior_losses(self, batch_inputs: torch.Tensor, pred: torch.Tensor,
                                 batch_targets: torch.Tensor, batch_interior_mask: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """Compute interior data and physics losses."""
        if torch.any(batch_interior_mask):
            data_loss = self.physics_loss.mse_loss(
                pred[batch_interior_mask], batch_targets[batch_interior_mask]
            )
            phys_loss = self.physics_loss.physics_loss(
                batch_inputs, pred, batch_interior_mask
            )
            return data_loss, phys_loss
        return torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)

    def _update_history(self, epoch_metrics: Dict[str, float]):
        """Update training history."""
        for key in self.history:
            if key in epoch_metrics:
                self.history[key].append(epoch_metrics[key])
        self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])

    def _log_epoch(self, epoch: int, metrics: Dict[str, float]):
        """Log epoch progress."""
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}: "
                  f"Loss={metrics['total']:.4f} "
                  f"(Data={metrics['data']:.4f}, "
                  f"Physics={metrics['physics']:.4f}, "
                  f"BCs={metrics['bcs']:.4f})")

    def save_model(self, save_dir: Path, model_name: str, history: Dict, norm_params: Dict):
        """Save trained model and training history."""
        save_dir.mkdir(parents=True, exist_ok=True)
        # Save model
        model_path = save_dir / f"{model_name}.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.model.input_layers[0].in_features,
            'norm_params': norm_params,
            'training_config': self.config
        }, model_path)
        # Save history
        history_path = save_dir / "training_history.pth"
        torch.save(history, history_path)
        print(f"Model saved to {model_path}")
        print(f"Training history saved to {history_path}")


def main():
    """Example usage of the Urban Heat Island PINN pipeline."""
    # Configuration
    config = {
        'data_paths': {
            'lst_path': "/your_GEE_data_directory/LS.tif",
            'ndvi_path': "/your_GEE_data_directory/NDVI.tif",
            'emissivity_path': "/your_GEE_data_directory/EM.tif",
            'urban_features_dir': "/your_urban_features_data_directory/",
        },
        'training': {
            'epochs': 300,
            'batch_size': 512,
            'lr': 0.001,
            'scheduler_patience': 5,
            'boundary_width': 5
        },
        'output': {
            'save_dir': "/your_PINNs_model_saving_directory/",
            'model_name': "urban_heat_pinn_model"
        }
    }
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    try:
        # Initialize components
        data_processor = UrbanHeatDataProcessor(config['data_paths'])
        physics_loss = PhysicsInformedLoss()
        # Load and process data
        raster_data, norm_params = data_processor.load_geospatial_data()
        # Initialize model
        input_dim = 10  # x, y, ndvi, emissivity, 6 urban features
        model = UrbanHeatPINN(input_dim).to(device)
        # Initialize trainer
        trainer = UrbanHeatTrainer(model, physics_loss, device, config['training'])
        # Prepare training data
        inputs, targets = trainer.prepare_data(raster_data, norm_params)
        boundary_mask = trainer.create_boundary_mask(
            raster_data['lst'].shape,
            config['training']['boundary_width']
        )
        # Train model
        history = trainer.train(inputs, targets, boundary_mask, config['training']['epochs'])
        # Save results
        trainer.save_model(
            Path(config['output']['save_dir']),
            config['output']['model_name'],
            history,
            norm_params
        )
        print("Urban Heat Island PINN training completed successfully!")
    except Exception as e:
        print(f"Error in Urban Heat Island pipeline: {e}")
        raise

if __name__ == "__main__":
    main()