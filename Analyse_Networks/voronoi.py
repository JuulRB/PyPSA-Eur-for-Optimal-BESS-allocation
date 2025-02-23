import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pypsa
import geopandas as gpd

# Load the network
network = pypsa.Network(r"results\2023_Run\networks\base_s_37_elec_lvopt_.nc")
print(network)

# Load the Voronoi cell data you found
voronoi_cells = gpd.read_file(r"resources\2023_Run\regions_onshore_base_s_37.geojson")

# If `name` in `voronoi_cells` matches a column in `network.buses` (e.g., 'bus_id' or 'country')
# Adjust the column name as needed to match correctly
merged_data = voronoi_cells.merge(
    network.buses, left_on="name", right_on="Bus", how="inner"
)

# Display to verify the merge
print(merged_data.head())
merged_data.explore(
    column="v_nom",  # Color by voltage level or choose another attribute like 'carrier' or 'control'
    cmap="viridis",  # Choose a colormap
    legend=True,
    tooltip=["name", "v_nom", "country"],  # Show relevant attributes on hover
)


# Extract bus data
buses = network.buses
bus_coords = buses[["x", "y"]]
bus_names = buses.index

# Filter only Dutch buses
dutch_buses = bus_names[bus_names.str.startswith("NL0")]
dutch_bus_coords = bus_coords.loc[dutch_buses]

# Remove "NL0" and "H2" from bus names
cleaned_bus_names = dutch_buses.str.replace("NL0", "").str.replace(
    "H2", "", regex=False
)

# Create the figure and axes
fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(10, 8))

# Add natural earth features
ax.add_feature(cfeature.LAND, facecolor="lightgray")
ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
ax.add_feature(cfeature.BORDERS, edgecolor="black", linestyle="--", linewidth=0.5)
ax.add_feature(cfeature.COASTLINE, edgecolor="black")

# Plot Dutch bus locations
ax.scatter(
    dutch_bus_coords["x"],
    dutch_bus_coords["y"],
    color="blue",
    label="Dutch Buses",
    s=20,
    zorder=5,
)

# Annotate Dutch buses with cleaned names
for bus, cleaned_name in zip(dutch_bus_coords.index, cleaned_bus_names):
    ax.text(
        dutch_bus_coords.loc[bus, "x"],
        dutch_bus_coords.loc[bus, "y"],
        cleaned_name,
        fontsize=16,
        color="black",
        zorder=16,
    )

# Set the map boundaries to focus on the Netherlands
ax.set_extent([3.2, 7.2, 50.7, 53.7], crs=ccrs.PlateCarree())

# Add gridlines
gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
gl.top_labels = False
gl.right_labels = False

# Add a title and legend
plt.title("Dutch Bus Map Numbering", fontsize=16)
plt.legend(loc="lower left", fontsize=10)

# Show the plot
plt.show()
