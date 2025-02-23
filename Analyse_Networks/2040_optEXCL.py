# this script is used to analyse the results of the 2040 simulation run after the optimization for the EXCL scenario

import pypsa
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd
import geopandas as gpd
import matplotlib.cm as cm

# Load the network
network = pypsa.Network(r"results\2040_Run\networks\EXCL2040_optimized_.nc")
print(network)

#! 1. Creating the optimal BESS location map and capacities for the Netherlands
# Check battery capacities after optimization
battery_data = network.storage_units[
    (network.storage_units.carrier == "battery") & (network.storage_units.p_nom_opt > 0)
]
# Filter to only include Dutch storage units based on bus names
land_costs = {
    f"NL0 {i}": 0 for i in range(18)
}  # Define Dutch regions if not already defined
battery_data_dutch = battery_data[battery_data.bus.isin(land_costs.keys())]

# Get capacities and locations for Dutch batteries
battery_capacities_dutch = battery_data_dutch.p_nom_opt
battery_locations_dutch = battery_data_dutch.bus
coords_dutch = network.buses.loc[battery_locations_dutch, ["x", "y"]]

# Create a list of all Dutch buses
all_dutch_buses = pd.DataFrame({"bus": [f"NL0 {i}" for i in range(18)]})

# Merge battery data with all Dutch buses to ensure buses with 0 capacity are included
battery_capacity_table = pd.merge(
    all_dutch_buses, battery_data[["bus", "p_nom_opt"]], on="bus", how="left"
).fillna(0)

# Sort the buses numerically from NL0 0 to NL0 17
battery_capacity_table_sorted = battery_capacity_table.sort_values(
    by="bus", key=lambda x: x.str.extract(r"NL0 (\d+)")[0].astype(int)
)

# Print the battery capacities for each bus
print("Battery Capacities per NL Bus (Ranked)")
print(battery_capacity_table_sorted)

# Plot the network with Dutch storage units only
fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(10, 10))
network.plot(ax=ax, bus_sizes=0.002, line_widths=0.2, bus_colors="gray")

# Scale factor for storage markers
scaling_factor = 0.5
scatter = ax.scatter(
    coords_dutch["x"],
    coords_dutch["y"],
    s=battery_capacities_dutch * scaling_factor,
    c=battery_capacities_dutch,
    cmap="plasma",
    alpha=0.7,
    edgecolor="k",
    zorder=5,
)

# Color bar for capacity
cbar = plt.colorbar(scatter, ax=ax, orientation="horizontal", pad=0.05)
cbar.set_label("Battery Capacity (MW)")

# Set plot limits for the Netherlands
plt.xlim([3, 8])  # Adjust x-axis to zoom into the Netherlands
plt.ylim([50, 55])  # Adjust y-axis to zoom into the Netherlands
plt.title(
    "Optimal Battery Locations and Capacities for 2040 in the Netherlands (Dutch Buses Only)"
)
plt.show()

# Print network information
total_battery_capacity = network.storage_units.loc[
    network.storage_units.carrier == "battery", "p_nom_opt"
].sum()
print(f"Total System Battery Capacity: {total_battery_capacity} MW")

# Calculate total battery capacity in the Netherlands
dutch_battery_capacity = battery_data_dutch["p_nom_opt"].sum()
print(f"Dutch Total Battery Capacity: {dutch_battery_capacity} MW")


# Create a capacity table
def create_capacity_table(network):
    generator_data = network.generators.copy()
    generator_data["country"] = generator_data["bus"].str[:2]
    capacity_summary = (
        generator_data.groupby(["carrier", "country"])["p_nom_opt"]
        .sum()
        .unstack()
        .fillna(0)
        / 1e3  # Convert MW to GW
    )
    return capacity_summary


capacity_table = create_capacity_table(network)
print(capacity_table)

# Analyze battery capex
battery_data["capex"] = battery_data["p_nom_opt"] * battery_data["capital_cost"]

# Display the capex for all batteries
print("Battery Capex (EUR):")
print(battery_data[["bus", "p_nom_opt", "capital_cost", "capex"]])

# Calculate total battery capex for the system
total_battery_capex = battery_data["capex"].sum()
print(f"\nTotal Battery Capex for the System: {total_battery_capex:.2f} EUR")

# Calculate total battery capex for the Netherlands
dutch_battery_data = battery_data[battery_data.bus.str.startswith("NL")]
total_dutch_battery_capex = dutch_battery_data["capex"].sum()
print(f"Total Battery Capex for the Netherlands: {total_dutch_battery_capex:.2f} EUR")


#! 2. System Cost Breakdown
def calculate_costs_by_carrier(components, filter_func=None):
    if filter_func:
        components = components[filter_func(components)]
    return (
        (components.p_nom_opt * components.capital_cost)
        .groupby(components.carrier)
        .sum()
    )


def plot_cost_breakdown(costs, title):
    costs_df = costs.to_frame(name="Cost").T
    fig, ax = plt.subplots(figsize=(12, 8))
    costs_df.plot(kind="bar", stacked=True, ax=ax, color=plt.cm.tab20.colors)
    ax.set_title(title)
    ax.set_ylabel("System Costs (EUR/year)")
    ax.legend(title="Technology", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


# Total and Dutch system costs
total_costs = (
    pd.concat(
        [
            calculate_costs_by_carrier(network.generators),
            calculate_costs_by_carrier(network.storage_units),
            calculate_costs_by_carrier(network.links),
        ]
    )
    .groupby(level=0)
    .sum()
)


def dutch_filter(df):
    return (
        df.bus.str.startswith("NL")
        if "bus" in df.columns
        else df.bus0.str.startswith("NL") | df.bus1.str.startswith("NL")
    )


dutch_costs = (
    pd.concat(
        [
            calculate_costs_by_carrier(network.generators, filter_func=dutch_filter),
            calculate_costs_by_carrier(network.storage_units, filter_func=dutch_filter),
            calculate_costs_by_carrier(network.links, filter_func=dutch_filter),
        ]
    )
    .groupby(level=0)
    .sum()
)

# Plot costs
plot_cost_breakdown(total_costs, "Total System Cost Breakdown by Technology (Stacked)")
plot_cost_breakdown(dutch_costs, "Dutch System Cost Breakdown by Technology (Stacked)")


### Load Curve Per Technology
def prepare_time_series(network, region_filter=None):
    if region_filter:
        gen_time_series = (
            network.generators_t.p[region_filter(network.generators).index]
            .groupby(network.generators.carrier, axis=1)
            .sum()
        )
    else:
        gen_time_series = network.generators_t.p.groupby(
            network.generators.carrier, axis=1
        ).sum()
    return gen_time_series / 1e6  # Convert MW to TW


def plot_load_curve(time_series, title):
    fig, ax = plt.subplots(figsize=(15, 8))
    time_series.plot.area(ax=ax, stacked=True, alpha=0.8, colormap="tab20")
    ax.set_title(title)
    ax.set_ylabel("Power (TW)")
    ax.legend(title="Technology", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


# Total and Dutch load curves
total_time_series = prepare_time_series(network)
dutch_time_series = prepare_time_series(network, region_filter=dutch_filter)
plot_load_curve(total_time_series, "Total Load Curve by Technology")
plot_load_curve(dutch_time_series, "Dutch Load Curve by Technology")


#! 3. Line Loading Analysis
# 1. Calculate Average Line Loading (as a fraction of capacity)
average_line_loading = (
    network.lines_t.p0.abs().mean().sort_index()
    / (network.lines.s_nom_opt * network.lines.s_max_pu).sort_index()
).fillna(0)

# 2. Calculate Frequency of Peak Loading
line_loading_fraction = (
    network.lines_t.p0.abs() / (network.lines.s_nom_opt * network.lines.s_max_pu)
).fillna(0)
frequency_of_peak_loading = (line_loading_fraction == 1.0).sum() / len(
    network.snapshots
)

# Filter lines and buses in the Netherlands
netherlands_lines = network.lines[
    network.lines.bus0.str.startswith("NL") & network.lines.bus1.str.startswith("NL")
]
average_line_loading_nl = average_line_loading[netherlands_lines.index]
frequency_of_peak_loading_nl = frequency_of_peak_loading[netherlands_lines.index]

# 3. Plot Side-by-Side with Zoom
fig, axs = plt.subplots(
    1, 2, subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(20, 10)
)

# Average Line Loading Plot
network.plot(
    ax=axs[0],
    branch_components=["Line"],
    line_widths=netherlands_lines.s_nom_opt / 3e3,
    line_colors=average_line_loading_nl,
    line_cmap=cm.viridis,
    bus_sizes=0.002,
    bus_colors="gray",
    color_geomap=True,
)
axs[0].set_title("Average Line Loading (Fraction of Max Capacity)")
axs[0].axis("off")
axs[0].set_xlim(3, 8)  # Adjust x-axis for Netherlands
axs[0].set_ylim(50, 55)  # Adjust y-axis for Netherlands

# Add colorbar for Average Line Loading
cb1 = fig.colorbar(
    cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(0, 1)), ax=axs[0]
)
cb1.set_label("Average Line Loading (Fraction of Max Capacity)")

# Frequency of Peak Loading Plot
network.plot(
    ax=axs[1],
    branch_components=["Line"],
    line_widths=netherlands_lines.s_nom_opt / 3e3,
    line_colors=frequency_of_peak_loading_nl,
    line_cmap=cm.plasma,
    bus_sizes=0.002,
    bus_colors="gray",
    color_geomap=True,
)
axs[1].set_title("Frequency of Peak Loading (Fraction of Time)")
axs[1].axis("off")
axs[1].set_xlim(3, 8)  # Adjust x-axis for Netherlands
axs[1].set_ylim(50, 55)  # Adjust y-axis for Netherlands

# Add colorbar for Frequency of Peak Loading
cb2 = fig.colorbar(
    cm.ScalarMappable(cmap=cm.plasma, norm=plt.Normalize(0, 1)), ax=axs[1]
)
cb2.set_label("Frequency of Peak Loading (Fraction of Time)")

plt.tight_layout()
plt.show()

# Curtailment Analysis
curtailed_energy = (network.generators_t.p_max_pu - network.generators_t.p).clip(
    lower=0
)
total_curtailment = curtailed_energy.sum().sum()
print(f"Total Curtailed Energy (MWh): {total_curtailment}")
