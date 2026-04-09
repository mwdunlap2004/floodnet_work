"""
Floodnet NYC Maximum Depth Map Generator
Author: Michael Dunlap

This script processes a dataset of NYC Floodnet sensor readings, calculates the 
maximum recorded water depth per sensor, and generates a publication-ready 
spatial map using GeoPandas.

The resulting figure is saved as a high-resolution PNG, suitable for inclusion 
in academic journals or professional MSDS reports.
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import os

def process_flood_data(csv_path):
    """
    Loads flood data, calculates the maximum depth per sensor, and converts 
    it to a GeoDataFrame.

    Args:
        csv_path (str): File path to the merged Floodnet CSV.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing sensor names, max depth, 
                          and geometry (Points).
    """
    # Load the full combined dataset
    df = pd.read_csv(csv_path)
    
    # Filter out any rows missing coordinate data
    df = df.dropna(subset=['latitude', 'longitude'])
    
    # Group by the sensor location to find the maximum flood depth recorded
    # We keep the latitude and longitude by taking the first instance (they are static)
    summary_df = df.groupby('name').agg({
        'depth_inches': 'max',
        'longitude': 'first',
        'latitude': 'first'
    }).reset_index()
    
    # Filter out sensors that never recorded any flooding (depth <= 0)
    # to focus the map on actual flood events
    summary_df = summary_df[summary_df['depth_inches'] > 0]

    # Convert the pandas DataFrame to a GeoPandas GeoDataFrame
    geometry = [Point(xy) for xy in zip(summary_df.longitude, summary_df.latitude)]
    gdf_sensors = gpd.GeoDataFrame(summary_df, geometry=geometry, crs="EPSG:4326")
    
    return gdf_sensors

def fetch_nyc_boundaries():
    """
    Retrieves the official NYC borough boundaries as a GeoJSON directly 
    from NYC Open Data for use as a basemap.

    Returns:
        gpd.GeoDataFrame: Polygons of NYC boroughs.
    """
    nyc_geojson_url = "https://raw.githubusercontent.com/codeforgermany/click_that_hood/main/public/data/new-york-city-boroughs.geojson"
    nyc_gdf = gpd.read_file(nyc_geojson_url)
    return nyc_gdf

def create_publication_figure(gdf_sensors, gdf_basemap, output_path):
    """
    Generates and saves a publication-ready map figure.

    Args:
        gdf_sensors (gpd.GeoDataFrame): Point data for flood sensors.
        gdf_basemap (gpd.GeoDataFrame): Polygon data for the basemap.
        output_path (str): File path to save the resulting figure.
    """
    # For accurate spatial representation and plotting, reproject both 
    # datasets to the local projected coordinate system (NAD83 / New York Long Island)
    local_crs = "EPSG:2263"
    gdf_sensors = gdf_sensors.to_crs(local_crs)
    gdf_basemap = gdf_basemap.to_crs(local_crs)

    # Initialize the matplotlib figure and axis
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    
    # Set background color to a light water-blue for contrast
    ax.set_facecolor('#e6f0f9')

    # Plot the NYC basemap (Landmass)
    gdf_basemap.plot(
        ax=ax, 
        color='#f4f4f4', 
        edgecolor='#cccccc', 
        linewidth=0.8,
        zorder=1
    )

    # Plot the sensor data
    # Marker size is scaled by depth_inches for visual emphasis
    scatter = gdf_sensors.plot(
        ax=ax,
        column='depth_inches',
        cmap='YlGnBu',
        markersize=gdf_sensors['depth_inches'] * 15, 
        edgecolor='black',
        linewidth=0.5,
        alpha=0.85,
        legend=True,
        legend_kwds={
            'label': 'Maximum Recorded Flood Depth (inches)',
            'orientation': 'vertical',
            'shrink': 0.6,
            'pad': 0.02
        },
        zorder=2
    )

    # Configure axes formatting for publication
    ax.set_xlabel('Easting (ft)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Northing (ft)', fontsize=12, fontweight='bold', labelpad=10)
    
    # Format tick marks cleanly
    ax.tick_params(axis='both', which='major', labelsize=10, direction='out')
    
    # Add a grid for spatial reference
    ax.grid(True, linestyle='--', color='gray', alpha=0.3, zorder=0)

    # Note: Deliberately omitting plt.title() per publication requirements.
    
    # Adjust layout to fit everything tightly
    plt.tight_layout()

    # Save the figure to the specified path
    plt.savefig(output_path, bbox_inches='tight', format='png')
    print(f"Figure successfully saved to: {output_path}")
    
    # Show the plot in the interactive window
    plt.show()

def main():
    # Define file paths
    # Assumes the merged dataset from your Jupyter Notebook is saved in your working directory
    input_csv = "full_dataset.csv" 
    
    # Saving directly to the Mac desktop
    output_png = "floodnet_max_depth_fig.png"
    
    print("Processing sensor flood data...")
    gdf_sensors = process_flood_data(input_csv)
    
    print("Fetching NYC borough boundaries...")
    gdf_basemap = fetch_nyc_boundaries()
    
    print("Generating publication figure...")
    create_publication_figure(gdf_sensors, gdf_basemap, output_png)

    # Print the required figure caption to the console so it can be copied into the report
    print("\n" + "="*80)
    print("FIGURE CAPTION FOR REPORT:")
    print("Figure 1. Spatial distribution of maximum recorded flood depths (inches) across "
          "New York City. Data points represent stationary Floodnet gage locations. "
          "Marker size and color intensity scale proportionally with the maximum water "
          "level recorded during the observation period. Coordinates are projected in "
          "NAD83 / New York Long Island (EPSG:2263).")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()