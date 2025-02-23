# this script is used to analyse the results of the 2023 simulation run after the optimization for the BASE scenario

import pypsa
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd
import geopandas as gpd
import matplotlib.cm as cm

# Load the network
network = pypsa.Network(r"results\2023_Run\networks\BASE_optimized_.nc")
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
    "Optimal Battery Locations and Capacities for 2023 in the Netherlands (Dutch Buses Only)"
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
def calculate_cost_breakdown(network_file):
    # Load the network
    network = pypsa.Network(network_file)

    # Calculate capital costs
    generator_capital_costs = (
        network.generators.p_nom_opt * network.generators.capital_cost
    ).sum()
    storage_capital_costs = (
        network.storage_units.p_nom_opt * network.storage_units.capital_cost
    ).sum()
    line_capital_costs = (network.lines.s_nom_opt * network.lines.capital_cost).sum()

    # Calculate marginal (variable operating) costs
    generator_marginal_costs = (
        network.generators_t.p.multiply(network.generators.marginal_cost).sum().sum()
    )
    storage_marginal_costs = (
        network.storage_units_t.p.multiply(network.storage_units.marginal_cost)
        .sum()
        .sum()
    )

    # Calculate line expansion costs
    line_expansion_costs = (network.links.p_nom_opt * network.links.capital_cost).sum()

    # Summarize costs
    cost_breakdown = {
        "Generator Capital Costs": generator_capital_costs,
        "Storage Capital Costs": storage_capital_costs,
        "Marginal Costs": generator_marginal_costs + storage_marginal_costs,
        "Line Expansion Costs": line_expansion_costs,
        "Line Capital Costs": line_capital_costs,
    }

    return cost_breakdown


def plot_cost_breakdown(network_file, title):
    # Get cost breakdown
    cost_breakdown = calculate_cost_breakdown(network_file)

    # Prepare data for pie chart
    categories = list(cost_breakdown.keys())
    values = list(cost_breakdown.values())
    colors = ["skyblue", "lightcoral", "gold", "lightgreen", "orange"]
    explode = [
        0.1 if "Generator" in category else 0 for category in categories
    ]  # Emphasize Generator Capital Costs

    # Plot pie chart
    plt.figure(figsize=(10, 7))
    wedges, texts, autotexts = plt.pie(
        values,
        labels=None,  # Hide labels for cleaner pie chart
        autopct=lambda p: f"{p:.1f}%",  # Format percentages
        colors=colors,
        startangle=90,
        explode=explode,
        textprops={"fontsize": 12},
        pctdistance=1.1,  # Move percentages further outside the pie
    )

    # Draw a white circle in the middle to make it a donut chart
    centre_circle = plt.Circle((0, 0), 0.70, fc="white")
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    # Create legend with monetary values
    legend_labels = [f"{cat}: €{value:,.0f}" for cat, value in zip(categories, values)]
    plt.legend(
        labels=legend_labels, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=12
    )

    # Add title and show plot
    plt.title(f"Cost Breakdown for {title}", fontsize=14)
    plt.tight_layout()
    plt.show()


# Example: Analyze one network and plot its breakdown
network_file_2023_base = "results\\2023_Run\\networks\\BASE_optimized_.nc"
plot_cost_breakdown(network_file_2023_base, "2023 Base Scenario")

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
