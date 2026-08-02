"""
Urban Heat Mitigation Analysis: Green Roofs and Tree Implementation

This module analyzes the temperature reduction effects of implementing green roofs
and new trees using a trained Physics-Informed Neural Networks (PINNs) model.
"""

import warnings
from pathlib import Path
from typing import Dict, Tuple, List, Any
import numpy as np
import rasterio
import geopandas as gpd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches
import osmnx as ox
from shapely.geometry import Point, Polygon, shape
from shapely.ops import unary_union
from rasterio.features import rasterize
from rasterio.mask import mask
warnings.filterwarnings("ignore", category=UserWarning)


class UrbanHeatMitigationAnalyzer:
    """
    Analyzes urban heat mitigation strategies using green roofs and tree implementation.

    This class uses a trained PINN model to simulate temperature reduction effects
    of various urban greening strategies.
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

    def load_geospatial_data(self):
        """Load and preprocess all required geospatial data."""
        print("Loading geospatial data...")
        # Load raster data
        self.data['lst'], lst_transform, self.data['bounds'], self.data['crs'], _ = self._load_raster(
            self.config['data_paths']['lst_path']
        )
        self.data['ndvi_ds'] = rasterio.open(self.config['data_paths']['ndvi_path'])
        self.data['emissivity_ds'] = rasterio.open(self.config['data_paths']['emissivity_path'])
        # Read raster arrays
        self.data['ndvi'] = self.data['ndvi_ds'].read(1)
        self.data['emissivity'] = self.data['emissivity_ds'].read(1)
        # Generate coordinate grid
        self.data['x_coords'], self.data['y_coords'] = self._generate_coordinate_grid(
            self.data['lst'], lst_transform
        )
        # Load urban features
        self.data['urban_features'] = self._load_urban_features()
        # Create study area polygon
        self.data['study_area'] = self._create_study_area_polygon()
        # Store normalization parameters
        self._compute_normalization_parameters()
        print("Geospatial data loading completed successfully")

    @staticmethod
    def _load_raster(path: str) -> Tuple[np.ndarray, rasterio.Affine,
                                         rasterio.coords.BoundingBox, rasterio.CRS, Any]:
        """Load raster file and return data with metadata."""
        with rasterio.open(path) as src:
            data = src.read(1)
            transform = src.transform
            bounds = src.bounds
            crs = src.crs
            nodata = src.nodata
        return data, transform, bounds, crs, nodata

    def _generate_coordinate_grid(self, data: np.ndarray,
                                  transform: rasterio.Affine) -> Tuple[np.ndarray, np.ndarray]:
        """Generate coordinate grids from raster transform."""
        height, width = data.shape
        x_coords, y_coords = np.meshgrid(
            np.arange(width) * transform[0] + transform[2],
            np.arange(height) * transform[4] + transform[5]
        )
        return x_coords, y_coords

    def _load_urban_features(self) -> Dict[str, np.ndarray]:
        """Load urban features and convert to raster arrays."""
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
            file_path = Path(self.config['data_paths']['urban_features_dir']) / filename
            urban_features[feature_name] = self._geojson_to_array(file_path)
        return urban_features

    def _geojson_to_array(self, geojson_path: Path) -> np.ndarray:
        """Convert GeoJSON features to binary array mask."""
        gdf = gpd.read_file(geojson_path)
        if gdf.crs != self.data['crs']:
            gdf = gdf.to_crs(self.data['crs'])
        points = [Point(x, y) for x, y in zip(
            self.data['x_coords'].ravel(),
            self.data['y_coords'].ravel()
        )]
        return np.array([gdf.contains(p).any() for p in points]).reshape(self.data['x_coords'].shape)

    def _create_study_area_polygon(self) -> gpd.GeoDataFrame:
        """Create study area polygon from raster bounds."""
        bounds = self.data['bounds']
        study_area_poly = Polygon([
            (bounds.left, bounds.bottom),
            (bounds.right, bounds.bottom),
            (bounds.right, bounds.top),
            (bounds.left, bounds.top)
        ])
        return gpd.GeoDataFrame(geometry=[study_area_poly], crs=self.data['crs'])

    def _compute_normalization_parameters(self):
        """Compute and store normalization parameters."""
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
                               emissivity_data: np.ndarray = None,
                               trees_data: np.ndarray = None) -> np.ndarray:
        """
        Prepare input features for model prediction.

        Args:
            ndvi_data: Custom NDVI data
            emissivity_data: Custom emissivity data
            trees_data: Custom trees data

        Returns:
            Normalized input features
        """
        if ndvi_data is None:
            ndvi_data = self.data['ndvi']
        if emissivity_data is None:
            emissivity_data = self.data['emissivity']
        if trees_data is None:
            trees_data = self.data['urban_features']['trees']
        features_list = [
            self._normalize_with(self.data['x_coords'], 'x').ravel(),
            self._normalize_with(self.data['y_coords'], 'y').ravel(),
            self._normalize_with(ndvi_data, 'ndvi').ravel(),
            self._normalize_with(emissivity_data, 'emissivity').ravel(),
            self.data['urban_features']['buildings'].ravel(),
            self.data['urban_features']['roads'].ravel(),
            self.data['urban_features']['rails'].ravel(),
            self.data['urban_features']['parks'].ravel(),
            self.data['urban_features']['water'].ravel(),
            trees_data.ravel(),
        ]
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

    def close_datasets(self):
        """Close raster datasets."""
        self.data['ndvi_ds'].close()
        self.data['emissivity_ds'].close()


class GreenRoofAnalyzer:
    """Analyzes and implements green roof strategies."""

    def __init__(self, ndvi_target: float = 0.70):
        """
        Initialize green roof analyzer.

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
        NDVI_soil = 0.2
        NDVI_veg = 0.5
        epsilon_veg = 0.985
        epsilon_soil = 0.96
        # Calculate vegetation fraction
        pv = np.clip((ndvi - NDVI_soil) / (NDVI_veg - NDVI_soil), 0, 1)
        return epsilon_veg * pv + epsilon_soil * (1 - pv)

    @staticmethod
    def remove_outliers_iqr(data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Remove outliers using Interquartile Range method.

        Args:
            data: Input data array

        Returns:
            Tuple of (filtered_data, statistics)
        """
        data_array = np.array(data)
        data_array = data_array[~np.isnan(data_array)]
        if data_array.size == 0:
            return np.array([]), {}
        Q1 = np.percentile(data_array, 25)
        Q3 = np.percentile(data_array, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        filtered_data = data_array[(data_array >= lower_bound) & (data_array <= upper_bound)]
        stats = {
            'original_count': len(data_array),
            'filtered_count': len(filtered_data),
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
        return filtered_data, stats

    def apply_iqr_aligned(self, ndvi_data: np.ndarray, em_data: np.ndarray,
                          indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict, Dict]:
        """Apply IQR filtering while keeping NDVI-emissivity pairs aligned."""
        # Apply IQR to NDVI
        ndvi_filtered, ndvi_stats = self.remove_outliers_iqr(ndvi_data)
        if len(ndvi_filtered) == 0:
            return np.array([]), np.array([]), np.array([]), ndvi_stats, {}
        # Find surviving indices
        surviving_mask = np.isin(ndvi_data, ndvi_filtered)
        surviving_indices = indices[surviving_mask]
        surviving_ndvi = ndvi_data[surviving_mask]
        surviving_em = em_data[surviving_mask]
        # Apply IQR to emissivity
        em_filtered, em_stats = self.remove_outliers_iqr(surviving_em)
        if len(em_filtered) == 0:
            return np.array([]), np.array([]), np.array([]), ndvi_stats, em_stats
        # Final alignment
        final_mask = np.isin(surviving_em, em_filtered)
        final_indices = surviving_indices[final_mask]
        final_ndvi = surviving_ndvi[final_mask]
        final_em = surviving_em[final_mask]
        return final_ndvi, final_em, final_indices, ndvi_stats, em_stats

    def extract_raster_values(self, buildings_gdf: gpd.GeoDataFrame,
                              raster_ds: rasterio.DatasetReader, value_type: str) -> Tuple[List, List, List]:
        """
        Extract raster values for building geometries.

        Args:
            buildings_gdf: Buildings GeoDataFrame
            raster_ds: Raster dataset
            value_type: Type of values ('ndvi' or 'emissivity')

        Returns:
            Tuple of (values, building_ids, issues)
        """
        values = []
        building_ids = []
        issues = []
        for idx, building in buildings_gdf.iterrows():
            geom = building.geometry
            if not geom.is_valid:
                geom = geom.buffer(0)
            try:
                geom_dict = [{'type': 'Feature', 'geometry': geom.__geo_interface__}]
                out_image, out_transform = mask(
                    raster_ds, geom_dict, crop=True, all_touched=True, filled=True
                )
                masked_data = out_image[0]
                valid_mask = (~np.isnan(masked_data)) & (masked_data != raster_ds.nodata)
                if valid_mask.any():
                    mean_value = np.mean(masked_data[valid_mask])
                    # Validate based on value type
                    if value_type == 'emissivity' and (mean_value < 0.8 or mean_value > 1.0):
                        issues.append(f"Building {idx}: emissivity {mean_value:.3f} outside normal range")
                    elif value_type == 'ndvi' and (mean_value < -1.0 or mean_value > 1.0):
                        issues.append(f"Building {idx}: NDVI {mean_value:.3f} outside normal range")
                    values.append(mean_value)
                else:
                    values.append(np.nan)
                    issues.append(f"Building {idx}: No valid raster values found")
                building_ids.append(idx)

            except Exception as e:
                print(f"Error processing building {idx}: {e}")
                values.append(np.nan)
                building_ids.append(idx)
        return values, building_ids, issues

    def identify_green_roof_candidates(self, buildings_gdf: gpd.GeoDataFrame,
                                       ndvi_ds: rasterio.DatasetReader,
                                       emissivity_ds: rasterio.DatasetReader,
                                       ndvi_threshold: float = 0.2) -> gpd.GeoDataFrame:
        """
        Identify buildings suitable for green roof implementation.

        Args:
            buildings_gdf: Buildings GeoDataFrame
            ndvi_ds: NDVI raster dataset
            emissivity_ds: Emissivity raster dataset
            ndvi_threshold: NDVI threshold for candidate selection

        Returns:
            GeoDataFrame of green roof candidate buildings
        """
        if buildings_gdf.empty:
            return gpd.GeoDataFrame()

        # Extract NDVI and emissivity values
        ndvi_values, ndvi_ids, ndvi_issues = self.extract_raster_values(
            buildings_gdf, ndvi_ds, 'ndvi'
        )
        emissivity_values, em_ids, em_issues = self.extract_raster_values(
            buildings_gdf, emissivity_ds, 'emissivity'
        )
        print(f"Extracted {len(ndvi_values)} NDVI values, {len(emissivity_values)} emissivity values")
        # Convert to arrays and remove NaNs
        ndvi_arr = np.array(ndvi_values)
        em_arr = np.array(emissivity_values)
        valid_mask = (~np.isnan(ndvi_arr)) & (~np.isnan(em_arr)) & \
                     (ndvi_arr >= -1.0) & (ndvi_arr <= 1.0) & \
                     (em_arr >= 0.5) & (em_arr <= 1.0)
        ndvi_valid = ndvi_arr[valid_mask]
        em_valid = em_arr[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        print(f"After physical range filtering: {len(ndvi_valid)} valid pairs")
        if len(ndvi_valid) == 0:
            return gpd.GeoDataFrame()
        # Apply IQR filtering
        ndvi_iqr, em_iqr, iqr_indices, ndvi_stats, em_stats = self.apply_iqr_aligned(
            ndvi_valid, em_valid, valid_indices
        )
        print(f"After IQR filtering: {len(ndvi_iqr)} buildings remain")
        if len(ndvi_iqr) == 0:
            return gpd.GeoDataFrame()
        # Apply NDVI threshold
        low_ndvi_mask = ndvi_iqr <= ndvi_threshold
        low_ndvi_indices = iqr_indices[low_ndvi_mask]
        print(f"Found {len(low_ndvi_indices)} buildings with NDVI ≤ {ndvi_threshold}")
        if len(low_ndvi_indices) > 0:
            candidates = buildings_gdf.iloc[low_ndvi_indices].copy()
            candidates['original_ndvi'] = ndvi_iqr[low_ndvi_mask]
            candidates['original_emissivity'] = em_iqr[low_ndvi_mask]
            candidates['ndvi_target'] = self.ndvi_target
            candidates['emissivity_target'] = self.emissivity_target
            return candidates
        else:
            return gpd.GeoDataFrame()

class TreeImplementationAnalyzer:
    """Analyzes and implements tree planting strategies."""

    def __init__(self, tree_increase_percentage: float = 2.0):
        """
        Initialize tree implementation analyzer.

        Args:
            tree_increase_percentage: Percentage increase in tree coverage
        """
        self.tree_increase_percentage = tree_increase_percentage

    def identify_main_roads(self, roads_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Identify main roads from roads dataset.

        Args:
            roads_gdf: Roads GeoDataFrame

        Returns:
            Main roads GeoDataFrame
        """
        main_road_types = [
            'motorway', 'motorway_link', 'trunk', 'trunk_link',
            'primary', 'primary_link', 'secondary', 'secondary_link'
        ]
        return roads_gdf[roads_gdf['highway'].isin(main_road_types)]

    def find_traffic_lights(self, study_area_gdf: gpd.GeoDataFrame,
                            main_road_types: List[str],
                            railways_gdf: gpd.GeoDataFrame = None) -> gpd.GeoDataFrame:
        """
        Find traffic lights using OSMnx.

        Args:
            study_area_gdf: Study area GeoDataFrame
            main_road_types: List of main road types
            railways_gdf: Railways GeoDataFrame for filtering

        Returns:
            Traffic lights GeoDataFrame
        """
        if study_area_gdf.crs != 'EPSG:4326':
            study_area_gdf = study_area_gdf.to_crs('EPSG:4326')
        # Query traffic signals directly from OSM
        tags = {'highway': 'traffic_signals'}
        try:
            traffic_lights = ox.features_from_polygon(
                study_area_gdf.geometry.iloc[0], tags=tags
            )
            print(f"Found {len(traffic_lights)} direct traffic signals in OSM")
        except Exception as e:
            print(f"Error querying traffic signals: {e}")
            traffic_lights = gpd.GeoDataFrame()
        # If no direct traffic lights found, use intersections
        if traffic_lights.empty:
            print("No direct traffic lights found, looking for intersections...")
            traffic_lights = self._find_intersections(study_area_gdf, main_road_types)
        # Filter by railways if provided
        if not traffic_lights.empty and railways_gdf is not None:
            traffic_lights = self._filter_by_railways(traffic_lights, railways_gdf)
        return traffic_lights

    def _find_intersections(self, study_area_gdf: gpd.GeoDataFrame,
                            main_road_types: List[str]) -> gpd.GeoDataFrame:
        """Find intersections as potential traffic light locations."""
        try:
            G = ox.graph_from_polygon(
                study_area_gdf.geometry.iloc[0], network_type='drive', simplify=True
            )
            intersections = ox.consolidate_intersections(
                G, tolerance=5, rebuild_graph=False, dead_ends=False
            )
            intersections_gdf = gpd.GeoDataFrame(
                geometry=[Point(data['x'], data['y']) for _, data in intersections.items()],
                crs='EPSG:4326'
            )
            print(f"Found {len(intersections_gdf)} potential traffic light locations")
            return intersections_gdf
        except Exception as e:
            print(f"Error finding intersections: {e}")
            return gpd.GeoDataFrame()

    def _filter_by_railways(self, traffic_lights: gpd.GeoDataFrame,
                            railways_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Filter traffic lights near railways."""
        # Convert to projected CRS for accurate distance measurement
        avg_lat = traffic_lights.geometry.y.mean()
        avg_lon = traffic_lights.geometry.x.mean()
        utm_zone = int((avg_lon + 180) / 6) + 1
        hemisphere = 'south' if avg_lat < 0 else 'north'
        proj_crs = f"+proj=utm +zone={utm_zone} +{hemisphere} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
        traffic_lights_proj = traffic_lights.to_crs(proj_crs)
        railways_proj = railways_gdf.to_crs(proj_crs)
        # Create buffer around railways
        rail_buffer = unary_union(railways_proj.geometry).buffer(5)  # 5 meter buffer
        # Filter out traffic lights in railway buffer
        filtered_lights = traffic_lights_proj[
            ~traffic_lights_proj.geometry.intersects(rail_buffer)
        ]
        print(f"After railway filtering: {len(filtered_lights)} traffic lights remain")
        return filtered_lights.to_crs(traffic_lights.crs)

    def implement_new_trees(self, main_roads_gdf: gpd.GeoDataFrame,
                            traffic_lights_gdf: gpd.GeoDataFrame,
                            railways_gdf: gpd.GeoDataFrame,
                            existing_trees_gdf: gpd.GeoDataFrame,
                            study_area_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Implement new trees along main roads.

        Args:
            main_roads_gdf: Main roads GeoDataFrame
            traffic_lights_gdf: Traffic lights GeoDataFrame
            railways_gdf: Railways GeoDataFrame
            existing_trees_gdf: Existing trees GeoDataFrame
            study_area_gdf: Study area GeoDataFrame

        Returns:
            New trees GeoDataFrame
        """
        # Calculate number of new trees
        num_existing_trees = len(existing_trees_gdf)
        num_new_trees = int((self.tree_increase_percentage / 100) * num_existing_trees)
        if num_new_trees <= 0:
            print("No existing trees found, cannot add new trees")
            return gpd.GeoDataFrame()
        print(
            f"Adding {num_new_trees} new trees ({self.tree_increase_percentage}% of {num_existing_trees} existing trees)")
        # Convert to projected CRS for accurate measurements
        proj_crs = "EPSG:3857"
        main_roads_proj = main_roads_gdf.to_crs(proj_crs)
        study_area_proj = study_area_gdf.to_crs(proj_crs)
        # Create avoidance zones
        avoidance_union = self._create_avoidance_zones(
            traffic_lights_gdf, railways_gdf, existing_trees_gdf, main_roads_proj, proj_crs
        )
        # Generate tree points along main roads
        new_trees_points = self._generate_tree_points(
            main_roads_proj, study_area_proj, avoidance_union, num_new_trees
        )
        if new_trees_points:
            new_trees_gdf = gpd.GeoDataFrame(geometry=new_trees_points, crs=proj_crs)
            return new_trees_gdf.to_crs(study_area_gdf.crs)
        else:
            return gpd.GeoDataFrame()

    def _create_avoidance_zones(self, traffic_lights_gdf: gpd.GeoDataFrame,
                                railways_gdf: gpd.GeoDataFrame,
                                existing_trees_gdf: gpd.GeoDataFrame,
                                main_roads_proj: gpd.GeoDataFrame,
                                proj_crs: str) -> Any:
        """Create avoidance zones for tree placement."""
        avoidance_zones = []
        buffer_distance = 10  # meters
        # Buffer around traffic lights
        if not traffic_lights_gdf.empty:
            traffic_lights_proj = traffic_lights_gdf.to_crs(proj_crs)
            traffic_lights_buffer = traffic_lights_proj.buffer(buffer_distance)
            avoidance_zones.append(unary_union(traffic_lights_buffer.geometry))
        # Buffer around existing trees
        if not existing_trees_gdf.empty:
            existing_trees_proj = existing_trees_gdf.to_crs(proj_crs)
            existing_trees_buffer = existing_trees_proj.buffer(buffer_distance)
            avoidance_zones.append(unary_union(existing_trees_buffer.geometry))
        # Buffer around railways and road-railway overlaps
        if not railways_gdf.empty:
            railways_proj = railways_gdf.to_crs(proj_crs)
            railways_buffer = railways_proj.buffer(buffer_distance)
            avoidance_zones.append(unary_union(railways_buffer.geometry))
            # Avoid areas where roads and railways overlap
            railways_buffer_small = railways_proj.buffer(2)
            railways_union_small = unary_union(railways_buffer_small.geometry)
            main_roads_union = unary_union(main_roads_proj.geometry)
            overlap_areas = main_roads_union.intersection(railways_union_small)
            if not overlap_areas.is_empty:
                if hasattr(overlap_areas, 'geoms'):
                    overlap_buffer = unary_union([geom.buffer(buffer_distance) for geom in overlap_areas.geoms])
                else:
                    overlap_buffer = overlap_areas.buffer(buffer_distance)
                avoidance_zones.append(overlap_buffer)
        return unary_union(avoidance_zones) if avoidance_zones else None

    def _generate_tree_points(self, main_roads_proj: gpd.GeoDataFrame,
                              study_area_proj: gpd.GeoDataFrame,
                              avoidance_union: Any,
                              num_trees: int) -> List[Point]:
        """Generate tree points along main roads."""
        new_trees_points = []
        #main_roads_union = unary_union(main_roads_proj.geometry)
        # Calculate total road length and spacing
        total_road_length = sum(road.length for road in main_roads_proj.geometry)
        spacing = max(total_road_length / num_trees, 5)  # Minimum 5m spacing
        print(f"Total road length: {total_road_length:.2f}m, tree spacing: {spacing:.2f}m")

        # Generate points along each road segment
        for road in main_roads_proj.geometry:
            if len(new_trees_points) >= num_trees:
                break
            road_length = road.length
            num_points = max(1, int(road_length / spacing))
            for i in range(num_points):
                if len(new_trees_points) >= num_trees:
                    break
                distance = (i / num_points) * road_length
                try:
                    point = road.interpolate(distance)
                    if ((avoidance_union is None or not avoidance_union.contains(point)) and
                            study_area_proj.contains(point).any()):
                        new_trees_points.append(point)
                except Exception as e:
                    print(f"Error interpolating point: {e}")
        print(f"Generated {len(new_trees_points)} new tree locations")
        return new_trees_points


class MitigationVisualizer:
    """Handles visualization of mitigation strategy results."""

    def __init__(self, output_dir: str):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save output plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_urban_features(self, study_area_polygon: Polygon,
                            urban_features: Dict[str, gpd.GeoDataFrame],
                            main_roads_gdf: gpd.GeoDataFrame = None,
                            traffic_lights_gdf: gpd.GeoDataFrame = None,
                            new_trees_gdf: gpd.GeoDataFrame = None,
                            green_roofs_gdf: gpd.GeoDataFrame = None,
                            scenario_name: str = "study_area"):
        """
        Plot urban features with mitigation strategies.

        Args:
            study_area_polygon: Study area polygon
            urban_features: Dictionary of urban feature GeoDataFrames
            main_roads_gdf: Main roads GeoDataFrame
            traffic_lights_gdf: Traffic lights GeoDataFrame
            new_trees_gdf: New trees GeoDataFrame
            green_roofs_gdf: Green roofs GeoDataFrame
            scenario_name: Name of the scenario
        """
        fig, ax = plt.subplots(figsize=(15, 15), constrained_layout=True)
        # Configure plot style
        plt.rcParams.update({
            'xtick.labelsize': 20,
            'ytick.labelsize': 20,
            'legend.fontsize': 20
        })
        legend_handles = []
        legend_labels = []
        # Plot features in order of visibility
        self._plot_water(urban_features['water'], ax, legend_handles, legend_labels)
        self._plot_parks(urban_features['parks'], ax, legend_handles, legend_labels)
        self._plot_trees(urban_features['trees'], ax, legend_handles, legend_labels)
        self._plot_roads(urban_features['roads'], ax, legend_handles, legend_labels)
        if main_roads_gdf is not None:
            self._plot_main_roads(main_roads_gdf, ax, legend_handles, legend_labels)
        self._plot_railways(urban_features['rails'], ax, legend_handles, legend_labels)
        if traffic_lights_gdf is not None:
            self._plot_traffic_lights(traffic_lights_gdf, ax, legend_handles, legend_labels)
        self._plot_buildings(urban_features['buildings'], ax, legend_handles, legend_labels)
        if new_trees_gdf is not None:
            self._plot_new_trees(new_trees_gdf, ax, legend_handles, legend_labels)
        if green_roofs_gdf is not None:
            self._plot_green_roofs(green_roofs_gdf, ax, legend_handles, legend_labels)
        # Configure plot appearance
        self._configure_plot_appearance(ax, study_area_polygon)
        # Add legend
        if legend_handles:
            legend = ax.legend(
                legend_handles, legend_labels,
                loc='center left', bbox_to_anchor=(1.0, 0.5),
                frameon=True, framealpha=1.0, edgecolor='silver', facecolor='white'
            )
            legend.get_frame().set_linewidth(1.5)
        # Save plot
        safe_name = scenario_name.replace(" ", "_").lower()
        filename = f"urban_features_{safe_name}.png"
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight',
                    pad_inches=0.5, facecolor='white')
        plt.close()
        print(f"Urban features plot saved: {filename}")

    def _plot_water(self, water_gdf: gpd.GeoDataFrame, ax: plt.Axes,
                    legend_handles: List, legend_labels: List):
        """Plot water bodies."""
        water_color = "dodgerblue"
        water_edge = "darkblue"
        water_gdf.plot(ax=ax, color=water_color, edgecolor=water_edge,
                       linewidth=1.0, alpha=0.95, zorder=10)
        legend_handles.append(mpatches.Patch(facecolor=water_color, edgecolor=water_edge,
                                             linewidth=1.0, alpha=0.95))
        legend_labels.append('Water bodies')

    def _plot_parks(self, parks_gdf: gpd.GeoDataFrame, ax: plt.Axes,
                    legend_handles: List, legend_labels: List):
        """Plot parks."""
        parks_gdf.plot(ax=ax, color='#e6f0e6', alpha=0.7, edgecolor='none')
        legend_handles.append(mpatches.Patch(facecolor='#e6f0e6', alpha=0.7, edgecolor='black'))
        legend_labels.append('Parks')

    def _plot_trees(self, trees_gdf: gpd.GeoDataFrame, ax: plt.Axes,
                    legend_handles: List, legend_labels: List):
        """Plot existing trees."""
        if not trees_gdf.empty:
            trees_gdf.plot(ax=ax, color='#002200', markersize=4, marker='o', alpha=0.7)
            legend_handles.append(
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#002200',
                           markersize=8, linestyle='None')
            )
            legend_labels.append('Existing Trees')

    def _plot_roads(self, roads_gdf: gpd.GeoDataFrame, ax: plt.Axes,
                    legend_handles: List, legend_labels: List):
        """Plot roads."""
        roads_gdf.plot(ax=ax, color='white', linewidth=1.0)
        legend_handles.append(mpatches.Patch(facecolor='white', edgecolor='black', alpha=0.7))
        legend_labels.append('Streets')

    def _plot_main_roads(self, main_roads_gdf: gpd.GeoDataFrame, ax: plt.Axes,
                         legend_handles: List, legend_labels: List):
        """Plot main roads."""
        main_roads_gdf.plot(ax=ax, color='red', linewidth=1.5, alpha=0.5)
        legend_handles.append(plt.Line2D([0], [0], color='red', lw=2))
        legend_labels.append('Main Roads')

    def _plot_railways(self, railways_gdf: gpd.GeoDataFrame, ax: plt.Axes,
                       legend_handles: List, legend_labels: List):
        """Plot railways."""
        railways_gdf.plot(ax=ax, color='#800080', linewidth=1.4, linestyle='--', alpha=0.4)
        legend_handles.append(plt.Line2D([0], [0], color='#800080', lw=2, linestyle='--'))
        legend_labels.append('Railways')

    def _plot_traffic_lights(self, traffic_lights_gdf: gpd.GeoDataFrame, ax: plt.Axes,
                             legend_handles: List, legend_labels: List):
        """Plot traffic lights."""
        traffic_lights_gdf.plot(ax=ax, color='yellow', markersize=150, marker='*',
                                edgecolor='black', linewidth=0.5, zorder=150)
        legend_handles.append(
            plt.Line2D([0], [0], marker='*', color='black', markerfacecolor='yellow',
                       markersize=20, linestyle='None')
        )
        legend_labels.append('Traffic Lights')

    def _plot_buildings(self, buildings_gdf: gpd.GeoDataFrame, ax: plt.Axes,
                        legend_handles: List, legend_labels: List):
        """Plot buildings."""
        if not buildings_gdf.empty:
            buildings_gdf.plot(ax=ax, color='#e0e0e0', alpha=0.8, edgecolor='none')
            legend_handles.append(mpatches.Patch(facecolor='#e0e0e0', alpha=0.8, edgecolor='black'))
            legend_labels.append('Buildings')

    def _plot_new_trees(self, new_trees_gdf: gpd.GeoDataFrame, ax: plt.Axes,
                        legend_handles: List, legend_labels: List):
        """Plot new trees."""
        new_trees_gdf.plot(ax=ax, color='#00ff00', markersize=100, marker='^',
                           alpha=0.85, edgecolor='black', zorder=100)
        legend_handles.append(
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#00ff00',
                       markersize=20, markeredgecolor='black', markeredgewidth=0.5, linestyle='None')
        )
        legend_labels.append('New Trees')

    def _plot_green_roofs(self, green_roofs_gdf: gpd.GeoDataFrame, ax: plt.Axes,
                          legend_handles: List, legend_labels: List):
        """Plot green roofs."""
        green_roofs_gdf.plot(ax=ax, color='#ff9900', alpha=0.99,
                             edgecolor='black', linewidth=0.7)
        legend_handles.append(mpatches.Patch(facecolor='#ff9900', alpha=0.99, edgecolor='black'))
        legend_labels.append('Green Roofs')

    def _configure_plot_appearance(self, ax: plt.Axes, study_area_polygon: Polygon):
        """Configure plot appearance and coordinates."""
        minx, miny, maxx, maxy = study_area_polygon.bounds
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_aspect('equal')
        # Set ticks
        ax.set_xticks([minx, maxx])
        ax.set_yticks([miny, maxy])
        # Format coordinates
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda x, _: f"{abs(x):.2f}°{'E' if x >= 0 else 'W'}"
        ))
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda y, _: f"{abs(y):.2f}°{'N' if y >= 0 else 'S'}"
        ))

        ax.tick_params(axis='x', pad=15)
        ax.tick_params(axis='both', which='major', labelsize=20,
                       bottom=True, top=True, left=True, right=True,
                       labelbottom=True, labelleft=True)
        # Add border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)

    def plot_temperature_comparison(self, original_temps: np.ndarray,
                                    mitigation_temps: np.ndarray,
                                    temperature_reduction: np.ndarray,
                                    bounds: rasterio.coords.BoundingBox,
                                    scenario_name: str,
                                    location_name: str = "study_area"):
        """
        Plot temperature comparison between original and mitigation scenarios.

        Args:
            original_temps: Original temperature predictions
            mitigation_temps: Mitigation scenario temperature predictions
            temperature_reduction: Temperature reduction values
            bounds: Geographic bounds for plotting
            scenario_name: Name of the mitigation scenario
            location_name: Name of study area
        """
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        # Extract bounds
        left, bottom, right, top = bounds.left, bounds.bottom, bounds.right, bounds.top
        plot_data = [original_temps, mitigation_temps, temperature_reduction]
        titles = ["Predicted LST", f"{scenario_name} Predicted LST", "Temperature Reduction"]
        cmaps = ['OrRd', 'OrRd', 'BuGn_r']
        for i, (ax, data, cmap, title) in enumerate(zip(axes, plot_data, cmaps, titles)):
            im = ax.imshow(data, cmap=cmap, extent=(left, right, bottom, top))
            ax.set_title(title, fontsize=16, pad=10)
            self._configure_comparison_axis(ax, bounds, i)
            self._add_comparison_colorbar(fig, im, ax, i, data)
        plt.suptitle(f"Temperature Analysis - {scenario_name}\n{location_name}",
                     fontsize=18, y=0.95)
        plt.tight_layout()
        # Save plot
        safe_name = scenario_name.replace(" ", "_").lower()
        filename = f"temperature_comparison_{safe_name}.png"
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Temperature comparison plot saved: {filename}")

    def _configure_comparison_axis(self, ax: plt.Axes, bounds: rasterio.coords.BoundingBox,
                                   subplot_index: int):
        """Configure comparison plot axis."""
        left, bottom, right, top = bounds.left, bounds.bottom, bounds.right, bounds.top

        if subplot_index == 0:
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

    def _add_comparison_colorbar(self, fig: plt.Figure, im: plt.AxesImage, ax: plt.Axes,
                                 subplot_index: int, data: np.ndarray):
        """Add colorbar to comparison plot."""
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, aspect=30)
        cbar.ax.tick_params(labelsize=10)
        if subplot_index in [0, 1]:
            cbar.set_label('Temperature (°C)', fontsize=12, labelpad=10)
            vmin, vmax = im.get_clim()
            ticks = np.linspace(vmin, vmax, 5)
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f"{tick:.1f}" for tick in ticks])
        else:
            cbar.set_label('ΔT (°C)', fontsize=12, labelpad=10)
            vmin, vmax = im.get_clim()
            ticks = np.linspace(vmin, vmax, 5)
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f"{tick:.2f}" for tick in ticks])


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
    """Main function to run urban heat mitigation analysis."""
    # Configuration
    config = {
        'model_path': "/your_PINNs_model_saving_directory/pinn_model.pth",
        'data_paths': {
            'lst_path': "/your_GEE_data_directory/LST.tif",
            'ndvi_path': "/your_GEE_data_directory/NDVI.tif",
            'emissivity_path': "/your_GEE_data_directory/EM.tif",
            'urban_features_dir': "/your_urban_features_data_directory/",
        },
        'mitigation_strategies': {
            'green_roofs': {
                'ndvi_target': 0.70,
                'ndvi_threshold': 0.2
            },
            'tree_implementation': {
                'tree_increase_percentage': 2.0,
                'buffer_distance': 10
            }
        },
        'analysis_cases': [
            {
                'name': 'Railways',
                'building_buffer_distance': 50
            }
        ],
        'output': {
            'plots_dir': "/your_directory_for_saving_plots/",
            'location_name': "study_area_name"
        }
    }
    try:
        # Initialize components
        heat_analyzer = UrbanHeatMitigationAnalyzer(config)
        green_roof_analyzer = GreenRoofAnalyzer(
            config['mitigation_strategies']['green_roofs']['ndvi_target']
        )
        tree_analyzer = TreeImplementationAnalyzer(
            config['mitigation_strategies']['tree_implementation']['tree_increase_percentage']
        )
        visualizer = MitigationVisualizer(config['output']['plots_dir'])
        # Load model and data
        heat_analyzer.load_model()
        heat_analyzer.load_geospatial_data()
        # Load additional urban feature data
        buildings_gdf = gpd.read_file(
            Path(config['data_paths']['urban_features_dir']) / "buildings.geojson"
        )
        roads_gdf = gpd.read_file(
            Path(config['data_paths']['urban_features_dir']) / "roads.geojson"
        )
        trees_gdf = gpd.read_file(
            Path(config['data_paths']['urban_features_dir']) / "trees.geojson"
        )
        # Clip to study area
        buildings_gdf = gpd.clip(buildings_gdf, heat_analyzer.data['study_area'])
        roads_gdf = gpd.clip(roads_gdf, heat_analyzer.data['study_area'])
        trees_gdf = gpd.clip(trees_gdf, heat_analyzer.data['study_area'])
        # Identify main roads
        main_roads_gdf = tree_analyzer.identify_main_roads(roads_gdf)
        # Find traffic lights
        main_road_types = [
            'motorway', 'motorway_link', 'trunk', 'trunk_link',
            'primary', 'primary_link', 'secondary', 'secondary_link'
        ]
        # Get railway data
        study_area_4326 = heat_analyzer.data['study_area'].to_crs('EPSG:4326')
        tags = {'railway': True}
        try:
            railways_gdf = ox.features_from_polygon(study_area_4326.geometry.iloc[0], tags=tags)
            if not railways_gdf.empty:
                railways_gdf = railways_gdf[railways_gdf.geometry.type.isin(['LineString', 'MultiLineString'])]
                railways_gdf = railways_gdf.to_crs(heat_analyzer.data['crs'])
                print(f"Found {len(railways_gdf)} railway features")
            else:
                railways_gdf = gpd.GeoDataFrame()
        except Exception as e:
            print(f"Error fetching railway data: {e}")
            railways_gdf = gpd.GeoDataFrame()

        traffic_lights_gdf = tree_analyzer.find_traffic_lights(
            study_area_4326, main_road_types, railways_gdf
        )
        if not traffic_lights_gdf.empty:
            traffic_lights_gdf = traffic_lights_gdf.to_crs(heat_analyzer.data['crs'])

        print(f"Found {len(traffic_lights_gdf)} traffic lights")
        # Implement new trees
        new_trees_gdf = tree_analyzer.implement_new_trees(
            main_roads_gdf, traffic_lights_gdf, railways_gdf,
            trees_gdf, heat_analyzer.data['study_area']
        )
        # Process analysis cases
        for case_config in config['analysis_cases']:
            case_name = case_config['name']
            print(f"\nProcessing case: {case_name}")
            # Find buildings near features
            if case_name == 'Railways' and not railways_gdf.empty:
                buildings_near_features = _find_buildings_near_features(
                    buildings_gdf, railways_gdf,
                    case_config['building_buffer_distance'],
                    heat_analyzer.data['crs']
                )
                print(f"Found {len(buildings_near_features)} buildings near {case_name}")
            else:
                print(f"No features found for case: {case_name}")
                continue
            # Identify green roof candidates
            green_roofs_gdf = green_roof_analyzer.identify_green_roof_candidates(
                buildings_near_features,
                heat_analyzer.data['ndvi_ds'],
                heat_analyzer.data['emissivity_ds'],
                config['mitigation_strategies']['green_roofs']['ndvi_threshold']
            )
            print(f"Green roof candidates for {case_name}: {len(green_roofs_gdf)} buildings")
            # Prepare scenarios
            scenarios = {}
            # Original scenario
            scenarios['original'] = heat_analyzer.prepare_input_features()
            # Combined scenario (green roofs + new trees)
            if not green_roofs_gdf.empty:
                # Rasterize green roofs
                shapes = [(geom, 1) for geom in green_roofs_gdf.geometry]
                green_roof_mask = rasterize(
                    shapes, out_shape=heat_analyzer.data['ndvi'].shape,
                    transform=heat_analyzer.data['ndvi_ds'].transform,
                    fill=0, default_value=1, dtype=np.uint8
                ).astype(bool)
                # Apply green roof values
                ndvi_modified = np.where(
                    green_roof_mask,
                    green_roof_analyzer.ndvi_target,
                    heat_analyzer.data['ndvi']
                )
                emissivity_modified = np.where(
                    green_roof_mask,
                    green_roof_analyzer.emissivity_target,
                    heat_analyzer.data['emissivity']
                )
            else:
                ndvi_modified = heat_analyzer.data['ndvi']
                emissivity_modified = heat_analyzer.data['emissivity']
            # Combine trees
            if not new_trees_gdf.empty:
                new_trees_array = _geojson_to_array_from_gdf(
                    new_trees_gdf,
                    heat_analyzer.data['x_coords'],
                    heat_analyzer.data['y_coords']
                )
                combined_trees = np.logical_or(
                    heat_analyzer.data['urban_features']['trees'],
                    new_trees_array
                ).astype(int)
            else:
                combined_trees = heat_analyzer.data['urban_features']['trees']

            scenarios['combined'] = heat_analyzer.prepare_input_features(
                ndvi_modified, emissivity_modified, combined_trees
            )
            # Generate predictions
            predictions = {}
            for scenario_name, inputs in scenarios.items():
                predictions[scenario_name] = heat_analyzer.predict_temperature(inputs)
            # Calculate temperature reduction
            temp_reduction = predictions['combined'] - predictions['original']
            print(f"Temperature reduction range: {temp_reduction.min():.3f}°C to {temp_reduction.max():.3f}°C")
            print(f"Mean temperature reduction: {temp_reduction.mean():.3f}°C")
            # Generate visualizations
            visualizer.plot_temperature_comparison(
                predictions['original'], predictions['combined'], temp_reduction,
                heat_analyzer.data['bounds'],
                f"Combined: Green Roofs + New Trees - {case_name}",
                config['output']['location_name']
            )
            # Load all urban features for visualization
            urban_features_gdf = {}
            for feature_name in ['buildings', 'roads', 'rails', 'parks', 'water', 'trees']:
                feature_path = Path(config['data_paths']['urban_features_dir']) / f"{feature_name}.geojson"
                urban_features_gdf[feature_name] = gpd.read_file(feature_path)
                if urban_features_gdf[feature_name].crs != heat_analyzer.data['crs']:
                    urban_features_gdf[feature_name] = urban_features_gdf[feature_name].to_crs(
                        heat_analyzer.data['crs'])
                urban_features_gdf[feature_name] = gpd.clip(
                    urban_features_gdf[feature_name], heat_analyzer.data['study_area']
                )
            visualizer.plot_urban_features(
                heat_analyzer.data['study_area'].geometry.iloc[0],
                urban_features_gdf,
                main_roads_gdf,
                traffic_lights_gdf,
                new_trees_gdf,
                green_roofs_gdf,
                f"{case_name} Mitigation Strategy"
            )
        # Close datasets
        heat_analyzer.close_datasets()
        print(f"\nAnalysis completed successfully!")
        print(f"All results saved to: {config['output']['plots_dir']}")
    except Exception as e:
        print(f"Error during urban heat mitigation analysis: {e}")
        raise

def _find_buildings_near_features(buildings_gdf: gpd.GeoDataFrame,
                                  features_gdf: gpd.GeoDataFrame,
                                  buffer_distance: float,
                                  crs: rasterio.CRS) -> gpd.GeoDataFrame:
    """Find buildings within buffer distance of features."""
    proj_crs = "EPSG:3857"
    buildings_proj = buildings_gdf.to_crs(proj_crs)
    features_proj = features_gdf.to_crs(proj_crs)
    features_buffer = features_proj.buffer(buffer_distance)
    features_buffer_union = unary_union(features_buffer.geometry)
    nearby_buildings = buildings_proj[buildings_proj.geometry.intersects(features_buffer_union)]
    return nearby_buildings.to_crs(crs)

def _geojson_to_array_from_gdf(gdf: gpd.GeoDataFrame, x_coords: np.ndarray,
                               y_coords: np.ndarray) -> np.ndarray:
    """Convert GeoDataFrame to binary array mask."""
    points = [Point(x, y) for x, y in zip(x_coords.ravel(), y_coords.ravel())]
    return np.array([gdf.contains(p).any() for p in points]).reshape(x_coords.shape)

if __name__ == "__main__":
    main()