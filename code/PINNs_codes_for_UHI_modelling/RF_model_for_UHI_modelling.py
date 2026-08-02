"""
Random Forest for Land Surface Temperature (LST) Prediction
This script implements a Random Forest model to predict LST
using satellite imagery and urban feature data
"""
import numpy as np
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import os
import warnings
from matplotlib.ticker import FuncFormatter
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)


class LSTPredictor:
    """
    This class handles data loading, preprocessing, model training, prediction,
    and visualization for LST prediction tasks
    """
    def __init__(self, output_dir):
        """
        Initialize the LST Predictor
        Args:
            output_dir (str): Directory to save results and models
        """
        self.output_dir = output_dir
        self.rf_model = None
        self.feature_names = [
            'x_coords', 'y_coords', 'ndvi', 'emissivity',
            'buildings', 'roads', 'rails', 'parks', 'water', 'trees'
        ]
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

    def load_raster(self, path):
        """
        Load raster data and metadata
        Args:
            path (str): Path to raster file
        Returns:
            tuple: (data, transform, bounds, crs)
        """
        try:
            with rasterio.open(path) as src:
                data = src.read(1)
                transform = src.transform
                bounds = src.bounds
                crs = src.crs
            return data, transform, bounds, crs
        except Exception as e:
            raise ValueError(f"Error loading raster {path}: {str(e)}")

    def normalize(self, data):
        """
        Normalize data to [0, 1] range
        Args:
            data (np.array): Input data array
        Returns:
            np.array: Normalized data
        """
        data_min = np.nanmin(data)
        data_max = np.nanmax(data)
        return (data - data_min) / (data_max - data_min)

    def geojson_to_array(self, geojson_path, x_coords, y_coords):
        """
        Convert GeoJSON features to binary array mask
        Args:
            geojson_path (str): Path to GeoJSON file
            x_coords (np.array): X coordinate grid
            y_coords (np.array): Y coordinate grid
        Returns:
            np.array: Binary mask array
        """
        try:
            gdf = gpd.read_file(geojson_path)
            points = [Point(x, y) for x, y in zip(x_coords.ravel(), y_coords.ravel())]
            return np.array([gdf.contains(p).any() for p in points]).reshape(x_coords.shape)
        except Exception as e:
            raise ValueError(f"Error processing GeoJSON {geojson_path}: {str(e)}")

    def prepare_data(self, data_config):
        """
        Prepare training data from input files
        Args:
            data_config (dict): Dictionary containing paths to input files
        Returns:
            tuple: (inputs, targets, interior_mask_flat, normalization_params)
        """
        print("Loading and preprocessing data")
        # Load raster data
        lst, lst_transform, lst_bounds, lst_crs = self.load_raster(data_config['lst_path'])
        ndvi, _, ndvi_bounds, _ = self.load_raster(data_config['ndvi_path'])
        emissivity, _, emissivity_bounds, _ = self.load_raster(data_config['emissivity_path'])
        # Store normalization parameters
        normalization_params = {
            'ndvi': {'min': np.min(ndvi), 'max': np.max(ndvi)},
            'emissivity': {'min': np.min(emissivity), 'max': np.max(emissivity)},
            'lst': {'min': np.min(lst), 'max': np.max(lst)}
        }
        print(f"NDVI range: {normalization_params['ndvi']['min']:.4f} to {normalization_params['ndvi']['max']:.4f}")
        print(
            f"Emissivity range: {normalization_params['emissivity']['min']:.4f} to {normalization_params['emissivity']['max']:.4f}")
        # Generate coordinate grid
        x_coords, y_coords = np.meshgrid(
            np.arange(lst.shape[1]) * lst_transform[0] + lst_transform[2],
            np.arange(lst.shape[0]) * lst_transform[4] + lst_transform[5]
        )
        # Load urban features
        print("Loading urban features")
        urban_features = {}
        for feature_name in ['buildings', 'roads', 'rails', 'parks', 'water', 'trees']:
            feature_path = data_config[f'{feature_name}_path']
            urban_features[feature_name] = self.geojson_to_array(feature_path, x_coords, y_coords)
        # Create boundary mask
        boundary_width = 5
        boundary_mask = np.zeros_like(lst, dtype=bool)
        boundary_mask[:boundary_width, :] = True
        boundary_mask[-boundary_width:, :] = True
        boundary_mask[:, :boundary_width] = True
        boundary_mask[:, -boundary_width:] = True
        interior_mask_flat = ~boundary_mask.ravel()
        # Prepare input features
        inputs = np.column_stack((
            self.normalize(x_coords).ravel(),
            self.normalize(y_coords).ravel(),
            self.normalize(ndvi).ravel(),
            self.normalize(emissivity).ravel(),
            urban_features['buildings'].ravel(),
            urban_features['roads'].ravel(),
            urban_features['rails'].ravel(),
            urban_features['parks'].ravel(),
            urban_features['water'].ravel(),
            urban_features['trees'].ravel(),
        ))
        targets = self.normalize(lst).ravel()
        print(f"Input shape: {inputs.shape}")
        print(f"Target shape: {targets.shape}")
        return inputs, targets, interior_mask_flat, normalization_params, lst, lst_bounds

    def train_model(self, inputs, targets, interior_mask):
        """
        Train Random Forest model
        Args:
            inputs (np.array): Input features
            targets (np.array): Target values
            interior_mask (np.array): Mask for interior pixels
        Returns:
            RandomForestRegressor: Trained model
        """
        print("Training Random Forest model")
        # Initialize Random Forest model
        rf_model = RandomForestRegressor(
            n_estimators=1000,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
        interior_indices = np.where(interior_mask)[0]
        rf_model.fit(inputs[interior_indices], targets[interior_indices])
        print("Random Forest training completed!")
        return rf_model

    def save_feature_importance(self, model):
        """
        Save feature importance plot
        Args:
            model: Trained Random Forest model
        """
        feature_importances = model.feature_importances_
        plt.figure(figsize=(10, 6))
        plt.barh(self.feature_names, feature_importances)
        plt.xlabel('Feature Importance')
        plt.title('Random Forest Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "feature_importance.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("Feature importances:")
        for name, importance in zip(self.feature_names, feature_importances):
            print(f"  {name}: {importance:.4f}")

    def plot_comparison(self, lst_true, lst_pred, abs_diff, bounds,
                        titles=("True LST", "Predicted LST", "Absolute Difference"),
                        filename="prediction_comparison", cmap_abs_diff='coolwarm'):
        """
        Create comparison plot between true and predicted LST
        Args:
            lst_true (np.array): True LST values
            lst_pred (np.array): Predicted LST values
            abs_diff (np.array): Absolute difference
            bounds: Raster bounds for georeferencing
            titles (tuple): Plot titles
            filename (str): Output filename
            cmap_abs_diff (str): Colormap for difference plot
        """
        fig, axes = plt.subplots(1, 3, figsize=(20, 18))
        cmaps = ['YlOrRd', 'YlOrRd', cmap_abs_diff]
        for i, (ax, data, cmap, title) in enumerate(zip(axes, [lst_true, lst_pred, abs_diff], cmaps, titles)):
            im = ax.imshow(data, cmap=cmap,
                           extent=(bounds.left, bounds.right, bounds.bottom, bounds.top))
            ax.set_title(title, fontsize=22)
            # Configure ticks and labels only for first subplot
            if i == 0:
                ax.set_xticks([bounds.left, bounds.right])
                ax.set_yticks([bounds.bottom, bounds.top])

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
            if i == 2:
                tick_labels = [f"{tick:.2f}" for tick in ticks]
            else:
                tick_labels = [f"{tick:.1f}" for tick in ticks]
            cbar.ax.set_xticklabels(tick_labels)
            cbar.ax.tick_params(labelsize=22)
            # Add units to colorbar
            if i in [0, 1]:
                cbar.ax.text(1.02, 0.5, '°C', transform=cbar.ax.transAxes,
                             va='center', ha='left', fontsize=22)
            elif i == 2:
                cbar.ax.text(1.02, 0.5, 'ΔT (°C)', transform=cbar.ax.transAxes,
                             va='center', ha='left', fontsize=22)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{filename}.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def compute_metrics(self, y_true, y_pred, filename="metrics"):
        """
        Compute and save evaluation metrics
        Args:
            y_true (np.array): True values
            y_pred (np.array): Predicted values
            filename (str): Output filename
        Returns:
            tuple: (r2, rmse, mae)
        """
        y_true_flat = y_true.ravel()
        y_pred_flat = y_pred.ravel()
        r2 = r2_score(y_true_flat, y_pred_flat)
        rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
        mae = np.mean(np.abs(y_true_flat - y_pred_flat))
        print(f"R² Score: {r2:.4f}")
        print(f"RMSE: {rmse:.4f} °C")
        print(f"MAE: {mae:.4f} °C")
        # Save metrics to file
        with open(os.path.join(self.output_dir, f"{filename}.txt"), "w") as f:
            f.write(f"R²: {r2:.4f}\n")
            f.write(f"RMSE: {rmse:.4f} °C\n")
            f.write(f"MAE: {mae:.4f} °C\n")
        return r2, rmse, mae

    def run_pipeline(self, data_config):
        """
        Run the complete LST prediction pipeline
        Args:
            data_config (dict): Dictionary containing paths to input files
        """
        print("Starting LST Prediction Pipeline")
        print("=" * 50)
        # Data Preparation
        inputs, targets, interior_mask, norm_params, lst, lst_bounds = self.prepare_data(data_config)
        # Model Training
        self.rf_model = self.train_model(inputs, targets, interior_mask)
        # Predictions
        print("Generating predictions...")
        lst_pred_normalized = self.rf_model.predict(inputs)
        lst_pred_normalized = lst_pred_normalized.reshape(lst.shape)
        # Denormalize predictions
        lst_pred_denorm_K = lst_pred_normalized * (norm_params['lst']['max'] - norm_params['lst']['min']) + \
                            norm_params['lst']['min']
        lst_pred_denorm_C = lst_pred_denorm_K - 273.15 # convert from Kelvin to Celcius
        # Save Model and Results
        joblib.dump(self.rf_model, os.path.join(self.output_dir, "random_forest_model.pkl"))
        self.save_feature_importance(self.rf_model)
        # Visualization
        lst_C = lst - 273.15
        abs_diff = np.abs(lst_pred_denorm_C - lst_C)
        print(f"Absolute difference - Min: {np.min(abs_diff):.4f}, Max: {np.max(abs_diff):.4f}")
        self.plot_comparison(
            lst_C, lst_pred_denorm_C, abs_diff, lst_bounds,
            titles=("True LST (Landsat 8)", "Predicted LST (Random Forest)", "Absolute Difference"),
            filename="prediction_RF"
        )
        #Compute Metrics
        r2, rmse, mae = self.compute_metrics(lst_C, lst_pred_denorm_C, "metrics_RF")
        print("\n" + "=" * 50)
        print("RANDOM FOREST TRAINING COMPLETED!")
        print("=" * 50)
        print(f"Model saved to: {self.output_dir}")
        print(f"R² Score: {r2:.4f}")
        print(f"RMSE: {rmse:.4f} °C")
        print(f"MAE: {mae:.4f} °C")
        print("Feature importance plot saved")
        print("=" * 50)

def main():
    """Main function to run the LST prediction pipeline"""
    # Configuration
    data_config = {
        'lst_path': "/your_directory/paris_GEE/LST_PR.tif",
        'ndvi_path': "/your_directory/paris_GEE/NDVI_PR.tif",
        'emissivity_path': "/your_directory/paris_GEE/EM_PR.tif",
        'buildings_path': "/your_directory/paris_GEE/buildings.geojson",
        'roads_path': "/your_directory/paris_GEE/roads.geojson",
        'rails_path': "/your_directory/paris_GEE/rails.geojson",
        'parks_path': "/your_directory/paris_GEE/parks.geojson",
        'water_path': "/your_directory/paris_GEE/water.geojson",
        'trees_path': "/your_directory/paris_GEE/trees.geojson",
    }
    output_dir = "/your_directory/Paris_RF_plots/"
    # Initialize and run pipeline
    predictor = LSTPredictor(output_dir)
    predictor.run_pipeline(data_config)

if __name__ == "__main__":
    main()