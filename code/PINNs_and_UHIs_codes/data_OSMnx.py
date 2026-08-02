"""
Urban Feature Extraction and Visualization

This module provides functionality to extract urban features from OpenStreetMap
and create professional visualizations with consistent styling and data export.
"""

from pathlib import Path
from typing import List, Tuple, Dict, Any
import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches
from shapely.geometry import Polygon


class UrbanFeatureExtractor:
    """
    A class to extract and visualize urban features from OpenStreetMap data.

    This class handles the extraction of buildings, roads, railways, parks,
    water bodies, and trees within a specified study area, and generates
    both data exports and professional visualizations.
    """

    # Class constants for consistent styling
    DEFAULT_STYLING = {
        'water': {'color': 'dodgerblue', 'edgecolor': 'darkblue', 'alpha': 0.95},
        'parks': {'color': '#b2df8a', 'alpha': 0.7},
        'trees': {'color': '#33a02c', 'markersize': 8, 'alpha': 0.7},
        'roads': {'color': 'white', 'linewidth': 0.8},
        'railways': {'color': '#e31a1c', 'linewidth': 1.5, 'linestyle': '--'},
        'buildings': {'color': '#636363', 'alpha': 0.7}
    }
    OSM_TAGS = {
        'building': True,
        'highway': True,
        'railway': True,
        'leisure': 'park',
        'natural': ['water', 'tree'],
        'amenity': 'swimming_pool'
    }
    def __init__(self, timeout: int = 300):
        """
        Initialize the UrbanFeatureExtractor.

        Args:
            timeout: Timeout in seconds for OSM data download (default: 300)
        """
        ox.settings.timeout = timeout
        self._validate_dependencies()
    @staticmethod
    def _validate_dependencies():
        """Validate that required dependencies are available."""
        required_packages = ['osmnx', 'geopandas', 'matplotlib', 'shapely']
        for package in required_packages:
            try:
                __import__(package)
            except ImportError as e:
                raise ImportError(
                    f"Required package {package} is not installed. "
                    f"Please install it using: pip install {package}"
                ) from e
    def extract_features(
            self,
            polygon_vertices: List[Tuple[float, float]],
            output_dir: str = "output"
    ) -> Dict[str, gpd.GeoDataFrame]:
        """
        Extract urban features from OSM within the specified polygon.

        Args:
            polygon_vertices: List of (lon, lat) tuples defining the study area
            output_dir: Directory to save extracted GeoJSON files

        Returns:
            Dictionary containing GeoDataFrames for each feature type
        """
        study_area = Polygon(polygon_vertices)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print("Downloading OSM data...")
        gdf = ox.features_from_polygon(study_area, tags=self.OSM_TAGS)
        print("Processing features...")
        features = self._process_features(gdf, study_area)
        print("Saving GeoJSON files...")
        self._save_geojson_files(features, output_path)
        return features

    def _process_features(
            self,
            gdf: gpd.GeoDataFrame,
            study_area: Polygon
    ) -> Dict[str, gpd.GeoDataFrame]:
        """Process and clip OSM features to study area boundary."""
        features = {}
        feature_filters = {
            'buildings': gdf['building'].notna(),
            'roads': gdf['highway'].notna(),
            'railways': gdf['railway'].notna(),
            'parks': gdf['leisure'] == 'park',
            'water': (gdf['natural'] == 'water') | (gdf['amenity'] == 'swimming_pool'),
            'trees': gdf['natural'] == 'tree'
        }
        for feature_name, filter_condition in feature_filters.items():
            filtered_gdf = gdf[filter_condition]
            if not filtered_gdf.empty:
                features[feature_name] = gpd.clip(filtered_gdf, study_area)
            else:
                features[feature_name] = gpd.GeoDataFrame()
        return features

    def _save_geojson_files(
            self,
            features: Dict[str, gpd.GeoDataFrame],
            output_path: Path
    ):
        """Save features as GeoJSON files."""
        for feature_name, gdf in features.items():
            if not gdf.empty:
                file_path = output_path / f"{feature_name}.geojson"
                gdf.to_file(file_path, driver="GeoJSON")
                print(f"Saved {file_path}")

    def create_visualization(
            self,
            features: Dict[str, gpd.GeoDataFrame],
            study_area: Polygon,
            output_path: Path,
            filename: str = "urban_features",
            labelsize: int = 22,
            include_legend: bool = True,
            figsize: Tuple[int, int] = (15, 15)
    ):
        """
        Create a professional visualization of urban features.

        Args:
            features: Dictionary of GeoDataFrames from extract_features()
            study_area: Shapely Polygon defining the study area
            output_path: Directory to save the visualization
            filename: Output filename (without extension)
            labelsize: Font size for axis labels
            include_legend: Whether to include legend
            figsize: Figure size as (width, height)
        """
        # Set up plot
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        self._configure_plot_style(ax, labelsize)
        # Create legend elements
        legend_handles, legend_labels = self._create_legend_elements()
        # Plot features in specific order for proper layering
        self._plot_features(ax, features, legend_handles, legend_labels)
        # Configure plot appearance
        self._configure_plot_appearance(ax, study_area)
        # Add legend if requested
        if include_legend:
            self._add_legend(ax, legend_handles, legend_labels)
        # Save figure
        self._save_figure(fig, output_path, filename)
        plt.close()

    def _configure_plot_style(self, ax: plt.Axes, labelsize: int):
        """Configure the overall plot style and typography."""
        plt.rcParams.update({
            'xtick.labelsize': labelsize,
            'ytick.labelsize': labelsize,
            'legend.fontsize': labelsize
        })
        # Configure tick appearance
        ax.tick_params(
            axis='both', which='major', labelsize=labelsize,
            bottom=True, top=True, left=True, right=True,
            labelbottom=True, labelleft=True
        )
        ax.tick_params(axis='x', pad=15)  # Add padding to prevent label overlap

    def _create_legend_elements(self) -> Tuple[List[Any], List[str]]:
        """Create legend handles and labels."""
        legend_handles = []
        legend_labels = []
        # Water
        legend_handles.append(mpatches.Patch(
            facecolor=self.DEFAULT_STYLING['water']['color'],
            edgecolor=self.DEFAULT_STYLING['water']['edgecolor'],
            linewidth=1.0,
            alpha=self.DEFAULT_STYLING['water']['alpha']
        ))
        legend_labels.append('Water bodies')
        # Parks
        legend_handles.append(mpatches.Patch(
            facecolor=self.DEFAULT_STYLING['parks']['color'],
            alpha=self.DEFAULT_STYLING['parks']['alpha']
        ))
        legend_labels.append('Parks')
        # Trees
        legend_handles.append(plt.Line2D(
            [0], [0], marker='o', color='w',
            markerfacecolor=self.DEFAULT_STYLING['trees']['color'],
            markersize=12, linestyle='None'
        ))
        legend_labels.append('Trees')
        # Roads
        legend_handles.append(mpatches.Patch(
            facecolor=self.DEFAULT_STYLING['roads']['color'],
            edgecolor='black', alpha=0.7
        ))
        legend_labels.append('Roads')
        # Railways
        legend_handles.append(plt.Line2D(
            [0], [0], color=self.DEFAULT_STYLING['railways']['color'],
            lw=2, linestyle=self.DEFAULT_STYLING['railways']['linestyle']
        ))
        legend_labels.append('Railways')
        # Buildings
        legend_handles.append(mpatches.Patch(
            facecolor=self.DEFAULT_STYLING['buildings']['color'],
            alpha=self.DEFAULT_STYLING['buildings']['alpha']
        ))
        legend_labels.append('Buildings')
        return legend_handles, legend_labels

    def _plot_features(
            self,
            ax: plt.Axes,
            features: Dict[str, gpd.GeoDataFrame],
            legend_handles: List[Any],
            legend_labels: List[str]
    ):
        """Plot all urban features with consistent styling."""
        # Plot water
        if 'water' in features and not features['water'].empty:
            features['water'].plot(
                ax=ax,
                color=self.DEFAULT_STYLING['water']['color'],
                edgecolor=self.DEFAULT_STYLING['water']['edgecolor'],
                linewidth=1.0,
                alpha=self.DEFAULT_STYLING['water']['alpha'],
                zorder=10
            )
        # Plot parks
        if 'parks' in features and not features['parks'].empty:
            features['parks'].plot(
                ax=ax,
                color=self.DEFAULT_STYLING['parks']['color'],
                alpha=self.DEFAULT_STYLING['parks']['alpha']
            )
        # Plot trees
        if 'trees' in features and not features['trees'].empty:
            features['trees'].plot(
                ax=ax,
                color=self.DEFAULT_STYLING['trees']['color'],
                markersize=self.DEFAULT_STYLING['trees']['markersize'],
                marker='o',
                alpha=self.DEFAULT_STYLING['trees']['alpha']
            )
        # Plot roads
        if 'roads' in features and not features['roads'].empty:
            features['roads'].plot(
                ax=ax,
                color=self.DEFAULT_STYLING['roads']['color'],
                linewidth=self.DEFAULT_STYLING['roads']['linewidth']
            )
        # Plot railways
        if 'railways' in features and not features['railways'].empty:
            features['railways'].plot(
                ax=ax,
                color=self.DEFAULT_STYLING['railways']['color'],
                linewidth=self.DEFAULT_STYLING['railways']['linewidth'],
                linestyle=self.DEFAULT_STYLING['railways']['linestyle']
            )
        # Plot buildings
        if 'buildings' in features and not features['buildings'].empty:
            features['buildings'].plot(
                ax=ax,
                color=self.DEFAULT_STYLING['buildings']['color'],
                alpha=self.DEFAULT_STYLING['buildings']['alpha']
            )
    def _configure_plot_appearance(self, ax: plt.Axes, study_area: Polygon):
        """Configure plot bounds, aspect ratio, and formatting."""
        # Set plot bounds
        minx, miny, maxx, maxy = study_area.bounds
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_aspect('equal')
        # Set ticks
        ax.set_xticks([minx, maxx])
        ax.set_yticks([miny, maxy])
        # Format coordinate labels
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda x, _: f"{abs(x):.2f}°{'E' if x >= 0 else 'W'}"
        ))
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda y, _: f"{abs(y):.2f}°{'N' if y >= 0 else 'S'}"
        ))
        # Add border
        for spine in ax.spines.values():
            spine.set_visible(True)

    def _add_legend(self, ax: plt.Axes, handles: List[Any], labels: List[str]):
        """Add legend to the plot."""
        ax.legend(
            handles,
            labels,
            loc='center left',
            bbox_to_anchor=(1.0, 0.5),
            framealpha=1
        )
    def _save_figure(self, fig: plt.Figure, output_path: Path, filename: str):
        """Save figure to file."""
        filepath = output_path / f"{filename}.png"
        fig.savefig(
            filepath,
            dpi=300,
            bbox_inches='tight',
            facecolor='white'
        )
        print(f"Saved visualization to {filepath}")

def main():
    """Example usage of the UrbanFeatureExtractor class."""
    # Define study area polygon vertices (replace with your coordinates)
    polygon_vertices = [
        (-122.4194, 37.7749),  # San Francisco coordinates - replace with your area
        (-122.4194, 37.7849),
        (-122.4094, 37.7849),
        (-122.4094, 37.7749)
    ]
    # Set output directory
    output_dir = "urban_analysis_output"
    try:
        # Initialize extractor
        extractor = UrbanFeatureExtractor(timeout=300)
        # Extract features
        features = extractor.extract_features(polygon_vertices, output_dir)
        # Create study area polygon for visualization
        study_area = Polygon(polygon_vertices)
        # Create visualization
        extractor.create_visualization(
            features=features,
            study_area=study_area,
            output_path=Path(output_dir),
            filename="urban_features_map",
            labelsize=22,
            include_legend=True
        )
        print("Urban feature extraction and visualization completed successfully!")
    except Exception as e:
        print(f"Error during processing: {e}")
        raise

if __name__ == "__main__":
    main()