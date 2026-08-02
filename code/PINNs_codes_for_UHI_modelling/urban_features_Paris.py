"""
A comprehensive tool for downloading, managing, and visualizing urban features
from OpenStreetMap within specified geographic boundaries
"""

from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging
import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches
from shapely.geometry import Polygon


class UrbanFeatureManager:
    """
    This class handles the complete workflow from downloading OSM data,
    saving it to GeoJSON files, and creating professional visualizations
    """
    # Default color scheme for urban features
    COLOR_SCHEME = {
        'parks': "#4CAF50",  # Vibrant green
        'water': "#2196F3",  # Bright blue
        'water_edge': "#0D47A1",
        'trees': "#2E7D32",  # Dark green
        'roads': "white",
        'railways': "#E91E63",  # Pink
        'buildings': "#C0C0C0",  # Silver
    }
    # Default OSM tags for feature extraction
    DEFAULT_TAGS = {
        'building': True,
        'highway': True,
        'railway': True,
        'leisure': 'park',
        'natural': ['water', 'tree'],
        'amenity': 'swimming_pool'
    }
    # File names for saved GeoJSON files
    FEATURE_FILES = {
        'buildings': 'buildings.geojson',
        'roads': 'roads.geojson',
        'rails': 'rails.geojson',
        'parks': 'parks.geojson',
        'water': 'water.geojson',
        'trees': 'trees.geojson'
    }
    def __init__(self, timeout: int = 300, log_level: str = 'INFO'):
        """
        Initialize the urban feature manager
        Args:
            timeout: Timeout in seconds for OSM API requests
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        ox.settings.timeout = timeout
        self._setup_logging(log_level)
        self._setup_plot_style()
        self.features = {}
        self.study_area = None
    def _setup_logging(self, log_level: str) -> None:
        """Configure logging settings"""
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    def _setup_plot_style(self) -> None:
        """Configure matplotlib style settings"""
        plt.rcParams.update({
            'xtick.labelsize': 22,
            'ytick.labelsize': 22,
            'legend.fontsize': 22
        })
    def _clip_to_boundary(self, features: gpd.GeoDataFrame,
                          boundary: Polygon) -> gpd.GeoDataFrame:
        """
        Clip features to study area boundary
        Args:
            features: GeoDataFrame to clip
            boundary: Polygon defining the study area
        Returns:
            Clipped GeoDataFrame
        """
        return gpd.clip(features, boundary)
    def _download_osm_data(self, study_area: Polygon) -> gpd.GeoDataFrame:
        """
        Download OSM features within the study area
        Args:
            study_area: Polygon defining the study area
        Returns:
            GeoDataFrame containing all downloaded features
        """
        self.logger.info("Downloading OSM data")
        gdf = ox.features_from_polygon(study_area, tags=self.DEFAULT_TAGS)
        self.logger.info(f"Downloaded {len(gdf)} features from OSM")
        return gdf

    def _extract_and_clip_features(self, gdf: gpd.GeoDataFrame,
                                   study_area: Polygon) -> Dict[str, gpd.GeoDataFrame]:
        """
        Extract and clip different feature types from OSM data
        Args:
            gdf: Raw OSM GeoDataFrame
            study_area: Study area boundary polygon
        Returns:
            Dictionary of clipped feature GeoDataFrames
        """
        features = {}
        self.logger.info("Extracting and clipping features")
        # Extract and clip each feature type
        feature_definitions = {
            'buildings': ('building', 'notna', ['Polygon', 'MultiPolygon']),
            'roads': ('highway', 'notna', None),
            'rails': ('railway', 'notna', None),
            'parks': ('leisure', '==', 'park'),
            'water': (['natural', 'amenity'], 'complex', None),
            'trees': ('natural', '==', 'tree')
        }
        for feature_name, (tag, condition, geometry_types) in feature_definitions.items():
            if feature_name == 'water':
                # Special handling for water features
                water_mask = (gdf['natural'] == 'water') | (gdf['amenity'] == 'swimming_pool')
                features['water'] = self._clip_to_boundary(gdf[water_mask], study_area)
                self.logger.info(f"Extracted {len(features['water'])} water features")
            else:
                if condition == 'notna':
                    mask = gdf[tag].notna()
                elif condition == '==':
                    mask = gdf[tag] == condition
                else:
                    mask = gdf[tag].notna()
                clipped = self._clip_to_boundary(gdf[mask], study_area)
                # Filter by geometry type if specified
                if geometry_types and not clipped.empty:
                    clipped = clipped[clipped.geometry.type.isin(geometry_types)]
                features[feature_name] = clipped
                self.logger.info(f"Extracted {len(clipped)} {feature_name}")
        return features

    def download_and_save_features(
            self,
            polygon_vertices: List[List[float]],
            output_path: str = "",
            force_download: bool = False
    ) -> Dict[str, gpd.GeoDataFrame]:
        """
        Download OSM features and save them as GeoJSON files
        Args:
            polygon_vertices: List of [lon, lat] pairs defining the study area
            output_path: Directory to save the GeoJSON files
            force_download: Whether to re-download if files already exist
        Returns:
            Dictionary of feature GeoDataFrames
        Raises:
            ValueError: If polygon_vertices doesn't form a valid polygon
            Exception: For OSM download or file saving errors
        """
        # Validate input
        if len(polygon_vertices) < 3:
            raise ValueError("Polygon must have at least 3 vertices")
        # Create output directory if it doesn't exist
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Define study area polygon
        self.study_area = Polygon(polygon_vertices)
        self.logger.info(f"Study area bounds: {self.study_area.bounds}")
        # Check if files already exist
        all_files_exist = all(
            (output_dir / filename).exists()
            for filename in self.FEATURE_FILES.values()
        )
        if all_files_exist and not force_download:
            self.logger.info("Loading existing GeoJSON files")
            return self.load_features_from_disk(output_path)
        try:
            # Download OSM data
            gdf = self._download_osm_data(self.study_area)
            # Extract and clip features
            self.features = self._extract_and_clip_features(gdf, self.study_area)
            # Save to GeoJSON files
            self._save_features_to_disk(output_path)
            self.logger.info(f"All features saved to {output_path}")
            return self.features
        except Exception as e:
            self.logger.error(f"Failed to download and save features: {str(e)}")
            raise Exception(f"Failed to download and save features: {str(e)}")

    def _save_features_to_disk(self, output_path: str) -> None:
        """
        Save all features to GeoJSON files
        Args:
            output_path: Directory to save the GeoJSON files
        """
        output_dir = Path(output_path)
        for feature_type, gdf in self.features.items():
            if not gdf.empty:
                filename = self.FEATURE_FILES[feature_type]
                filepath = output_dir / filename
                gdf.to_file(filepath, driver="GeoJSON")
                self.logger.info(f"Saved {len(gdf)} {feature_type} to {filename}")
            else:
                self.logger.warning(f"No {feature_type} found in the study area")
    def load_features_from_disk(self, input_path: str) -> Dict[str, gpd.GeoDataFrame]:
        """
        Load features from existing GeoJSON files
        Args:
            input_path: Directory containing the GeoJSON files
        Returns:
            Dictionary of feature GeoDataFrames
        """
        input_dir = Path(input_path)
        self.features = {}
        for feature_type, filename in self.FEATURE_FILES.items():
            filepath = input_dir / filename
            if filepath.exists():
                self.features[feature_type] = gpd.read_file(filepath)
                self.logger.info(f"Loaded {len(self.features[feature_type])} {feature_type} from {filename}")
            else:
                self.logger.warning(f"File not found: {filename}")
                self.features[feature_type] = gpd.GeoDataFrame()
        return self.features

    def _create_legend_handles(self) -> Tuple[List[mpatches.Patch], List[str]]:
        """
        Create legend handles and labels
        Returns:
            Tuple of (handles, labels) for matplotlib legend
        """
        handles = []
        labels = []
        legend_config = [
            ('Parks', mpatches.Patch(facecolor=self.COLOR_SCHEME['parks'], alpha=0.8)),
            ('Water Surfaces', mpatches.Patch(
                facecolor=self.COLOR_SCHEME['water'],
                edgecolor=self.COLOR_SCHEME['water_edge'],
                linewidth=1.0,
                alpha=0.9
            )),
            ('Tree Canopies', plt.Line2D(
                [0], [0], marker='o', color='w',
                markerfacecolor=self.COLOR_SCHEME['trees'],
                markersize=12, linestyle='None'
            )),
            ('Road Networks', mpatches.Patch(
                facecolor=self.COLOR_SCHEME['roads'],
                edgecolor='black',
                alpha=0.7
            )),
            ('Rail Corridors', plt.Line2D(
                [0], [0], color=self.COLOR_SCHEME['railways'],
                lw=2.5, linestyle='--'
            )),
            ('Built Structures', mpatches.Patch(
                facecolor=self.COLOR_SCHEME['buildings'],
                alpha=0.7
            )),
        ]
        for label, handle in legend_config:
            handles.append(handle)
            labels.append(label)
        return handles, labels

    def _configure_axes(self, ax: plt.Axes, study_area: Polygon,
                        labelsize: int = 22) -> None:
        """
        Configure plot axes with proper bounds and formatting
        Args:
            ax: Matplotlib axes object
            study_area: Study area polygon for bounds
            labelsize: Font size for axis labels
        """
        minx, miny, maxx, maxy = study_area.bounds
        # Set bounds and aspect ratio
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_aspect('equal')
        # Configure ticks
        ax.set_xticks([minx, maxx])
        ax.set_yticks([miny, maxy])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # Coordinate formatters
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda x, _: f"{abs(x):.2f}°{'E' if x >= 0 else 'W'}"
        ))
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda y, _: f"{abs(y):.2f}°{'N' if y >= 0 else 'S'}"
        ))
        ax.tick_params(axis='x', pad=10)
        ax.tick_params(
            axis='both', which='major', labelsize=labelsize,
            bottom=True, top=True, left=True, right=True,
            labelbottom=False, labeltop=True, labelleft=False, labelright=True
        )
        for spine in ax.spines.values():
            spine.set_visible(True)
    def _plot_features(self, ax: plt.Axes, features: Dict[str, gpd.GeoDataFrame]) -> None:
        """
        Plot all urban features on the axes
        Args:
            ax: Matplotlib axes object
            features: Dictionary of feature GeoDataFrames
        """
        if not features['parks'].empty:
            features['parks'].plot(
                ax=ax, color=self.COLOR_SCHEME['parks'], alpha=0.7
            )
        if not features['water'].empty:
            features['water'].plot(
                ax=ax, color=self.COLOR_SCHEME['water'],
                edgecolor=self.COLOR_SCHEME['water_edge'],
                linewidth=1.0, alpha=0.9, zorder=10
            )
        if not features['trees'].empty:
            features['trees'].plot(
                ax=ax, color=self.COLOR_SCHEME['trees'],
                markersize=8, marker='o', alpha=0.7
            )
        if not features['roads'].empty:
            features['roads'].plot(
                ax=ax, color=self.COLOR_SCHEME['roads'], linewidth=0.8
            )
        if not features['rails'].empty:
            features['rails'].plot(
                ax=ax, color=self.COLOR_SCHEME['railways'],
                linewidth=2.5, linestyle='--'
            )
        if not features['buildings'].empty:
            features['buildings'].plot(
                ax=ax, color=self.COLOR_SCHEME['buildings'], alpha=0.7
            )
    def visualize_urban_features(
            self,
            output_path: str = "",
            filename: str = "urban_features",
            labelsize: int = 22,
            include_legend: bool = True,
            figsize: Tuple[int, int] = (12, 12),
            dpi: int = 300,
            polygon_vertices: Optional[List[List[float]]] = None,
            input_path: Optional[str] = None
    ) -> str:
        # Load features
        if not self.features:
            if input_path:
                self.load_features_from_disk(input_path)
            elif polygon_vertices:
                if not output_path:
                    raise ValueError("output_path must be provided when downloading new data")
                self.download_and_save_features(polygon_vertices, output_path)
            else:
                raise ValueError("Either provide input_path or polygon_vertices")
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Use stored study area
        if self.study_area is None and self.features:
            # Create study area from combined bounds of all features
            all_bounds = [feature.total_bounds for feature in self.features.values() if not feature.empty]
            if all_bounds:
                minx = min(bounds[0] for bounds in all_bounds)
                miny = min(bounds[1] for bounds in all_bounds)
                maxx = max(bounds[2] for bounds in all_bounds)
                maxy = max(bounds[3] for bounds in all_bounds)
                self.study_area = Polygon([
                    [minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]
                ])
        if self.study_area is None:
            raise ValueError("Could not determine study area bounds")
        try:
            fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
            self._plot_features(ax, self.features)
            self._configure_axes(ax, self.study_area, labelsize)
            if include_legend:
                handles, labels = self._create_legend_handles()
                ax.legend(
                    handles, labels,
                    loc='center left',
                    bbox_to_anchor=(1.0, 0.3),
                    framealpha=1
                )
            output_file = output_dir / f"{filename}.png"
            plt.savefig(
                output_file,
                dpi=dpi,
                bbox_inches='tight',
                facecolor='white'
            )
            plt.close()
            self.logger.info(f"Visualization saved to: {output_file}")
            return str(output_file)
        except Exception as e:
            plt.close()
            self.logger.error(f"Failed to generate visualization: {str(e)}")
            raise

def main():
    OUTPUT_PATH = "/your_directory/paris_GEE/"
    # Paris study area coordinates [lon, lat]
    PARIS_POLYGON = [
        [2.2556070936757155, 48.84508567477442],
        [2.309508766038997, 48.84508567477442],
        [2.309508766038997, 48.88212608424269],
        [2.2556070936757155, 48.88212608424269],
        [2.2556070936757155, 48.84508567477442],
    ]
    try:
        manager = UrbanFeatureManager(timeout=300, log_level='INFO')
        print("Downloading and visualizing urban features")
        output_file = manager.visualize_urban_features(
            polygon_vertices=PARIS_POLYGON,
            output_path=OUTPUT_PATH,
            filename="urban_features_paris",
            labelsize=22,
            include_legend=True
        )
        print(f"Urban features visualization saved to: {output_file}")
    except Exception as e:
        print(f"Error processing urban features: {e}")
        raise

if __name__ == "__main__":
    main()