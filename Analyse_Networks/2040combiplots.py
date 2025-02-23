# This script contains the code for creating the combined plots for the final report
# This script is a merge of different plot functions from the previous scripts, therefore it still contains overlap and repetition.


import pypsa
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd
import geopandas as gpd
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import numpy as np
import cartopy.feature as cfeature


#! 1. Plotting the Network with Battery Capacities
# Function to generate a map plot for a specific network (scatter plot is chosen here)
def plot_network(network_path, title, ax):
    network = pypsa.Network(network_path)

    # Filter battery data
    battery_data = network.storage_units[
        (network.storage_units.carrier == "battery")
        & (network.storage_units.p_nom_opt > 0)
    ]
    land_costs = {f"NL0 {i}": 0 for i in range(18)}
    battery_data_dutch = battery_data[battery_data.bus.isin(land_costs.keys())]

    battery_capacities_dutch = battery_data_dutch.p_nom_opt
    battery_locations_dutch = battery_data_dutch.bus
    coords_dutch = network.buses.loc[battery_locations_dutch, ["x", "y"]]

    # Plot the network
    network.plot(ax=ax, bus_sizes=0.002, line_widths=0.2, bus_colors="gray")

    # Scatter plot for storage markers
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

    ax.set_xlim([3, 8])  # Adjust x-axis to zoom into the Netherlands
    ax.set_ylim([50, 55])  # Adjust y-axis to zoom into the Netherlands
    ax.set_title(title)

    return scatter


# Define network paths and titles for 2040 (paths may have changed based on your directory structure)
networks_2040 = [
    ("results\\2040_Run\\networks\\2040BASE_optimized_.nc", "BASE Scenario (2040)"),
    ("results\\2040_Run\\networks\\COL2040_optimized_.nc", "COL Scenario (2040)"),
    ("results\\2040_Run\\networks\\EXCL2040_optimized_.nc", "EXCL Scenario (2040)"),
]

# Create the figure and axes
fig, axes = plt.subplots(
    len(networks_2040),
    1,
    subplot_kw={"projection": ccrs.PlateCarree()},
    figsize=(10, 15),
)

# Generate plots for each network
scatter = None
for ax, (network_path, title) in zip(axes, networks_2040):
    scatter = plot_network(network_path, title, ax)

# Add the color bar using a dedicated axis
fig.subplots_adjust(right=0.85)  # Make space on the right for the color bar
cbar_ax = fig.add_axes([0.86, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
cbar = fig.colorbar(scatter, cax=cbar_ax, orientation="vertical")
cbar.set_label("Battery Capacity (MW)")

# Adjust layout
plt.tight_layout()
plt.show()

# Define network files for each year
year_2023_files = [
    "results\\2023_Run\\networks\\BASE_optimized_.nc",
    "results\\2023_Run\\networks\\COL_optimized_.nc",
    "results\\2023_Run\\networks\\EXCL_optimized_.nc",
]

year_2040_files = [
    "results\\2040_Run\\networks\\2040BASE_optimized_.nc",
    "results\\2040_Run\\networks\\COL2040_optimized_.nc",
    "results\\2040_Run\\networks\\EXCL2040_optimized_.nc",
]


# Function to calculate net energy flows and return a DataFrame
def calculate_energy_flows(network_files, year_label):
    all_flows = []

    for i, network_file in enumerate(network_files):
        # Load the network
        network = pypsa.Network(network_file)

        # Extract interconnector links
        interconnectors = network.links[
            network.links.bus0.str[:2] != network.links.bus1.str[:2]
        ]
        link_flows = network.links_t.p0[interconnectors.index]  # Power flows (MW)

        # Check if flows are instantaneous power (MW) or cumulative energy (MWh)
        is_power = True  # Assume power (MW) unless proven otherwise

        # Calculate net energy flows (in TWh)
        if is_power:
            # Flows are in MW -> Convert to TWh
            total_energy_flows = link_flows.mean(axis=0) * 8760 / 1e6
        else:
            # Flows are already in MWh -> Convert to TWh
            total_energy_flows = link_flows.sum(axis=0) / 1e6

        # Prepare data for the table
        for link_id in interconnectors.index:
            bus0_country = interconnectors.loc[link_id].bus0[:2]
            bus1_country = interconnectors.loc[link_id].bus1[:2]
            net_flow = total_energy_flows[link_id]  # Net flow in TWh
            abs_flow = abs(net_flow)
            flow_direction = "Export" if net_flow > 0 else "Import"

            if abs_flow >= 0.1:  # Only include significant flows
                all_flows.append(
                    {
                        "Simulation": f"{year_label} Simulation {i + 1}",
                        "From": bus0_country
                        if flow_direction == "Export"
                        else bus1_country,
                        "To": bus1_country
                        if flow_direction == "Export"
                        else bus0_country,
                        "Flow (TWh)": round(abs_flow, 2),
                        "Direction": flow_direction,
                    }
                )

    return pd.DataFrame(all_flows)


# Calculate energy flows for 2023 and 2040
flows_2023 = calculate_energy_flows(year_2023_files, "2023")
flows_2040 = calculate_energy_flows(year_2040_files, "2040")

# Save or display the tables
flows_2023.to_csv("2023_energy_flows.csv", index=False)
flows_2040.to_csv("2040_energy_flows.csv", index=False)

print("2023 Energy Flows Table:")
print(flows_2023)

print("\n2040 Energy Flows Table:")
print(flows_2040)


# Function to plot net energy flows for a given set of networks
def plot_net_energy_flows_vertically(network_files, year_label):
    simulation_names = [
        "Base Simulation",
        "COL Simulation",
        "EXCL Simulation",
    ]  # Map simulation names

    fig, axes = plt.subplots(
        nrows=len(network_files),  # One row per network
        ncols=1,  # Single column
        subplot_kw={"projection": ccrs.PlateCarree()},
        figsize=(10, 15),  # Adjust size for vertical alignment
    )

    # Ensure axes is iterable, even if there's only one network
    if len(network_files) == 1:
        axes = [axes]

    for i, network_file in enumerate(network_files):
        # Load the network
        network = pypsa.Network(network_file)

        # Extract interconnector links
        interconnectors = network.links[
            network.links.bus0.str[:2] != network.links.bus1.str[:2]
        ]
        link_flows = network.links_t.p0[interconnectors.index]

        # Calculate total energy flows in TWh
        total_energy_flows = (
            link_flows.mean(axis=0)
            * 8760
            / 1e6  # Convert MW to TWh by multiplying by 8760 and dividing by 1e6
        )

        # Initialize plot for this simulation
        ax = axes[i]
        network.plot(ax=ax, bus_sizes=0.002, line_widths=0.2, color_geomap=True)

        # Plot arrows and annotate net flows
        scaling_factor = 0.1  # Arrow size scaling factor
        for link_id in interconnectors.index:
            bus0_coords = network.buses.loc[
                interconnectors.loc[link_id].bus0, ["x", "y"]
            ]
            bus1_coords = network.buses.loc[
                interconnectors.loc[link_id].bus1, ["x", "y"]
            ]

            net_flow = total_energy_flows[link_id]  # Net flow in TWh
            abs_flow = abs(net_flow)

            # Skip flows less than 0.1 TWh
            if abs_flow < 0.1:
                continue

            # Arrow properties
            arrow_color = (
                "blue" if net_flow > 0 else "red"
            )  # Export (blue), Import (red)
            arrow_start = bus1_coords if net_flow > 0 else bus0_coords
            arrow_end = bus0_coords if net_flow > 0 else bus1_coords

            # Draw the arrow
            ax.arrow(
                arrow_start["x"],
                arrow_start["y"],
                (arrow_end["x"] - arrow_start["x"]) * scaling_factor,
                (arrow_end["y"] - arrow_start["y"]) * scaling_factor,
                color=arrow_color,
                alpha=0.7,
                head_width=0.2,
                linewidth=1,
                length_includes_head=True,
            )

            # Annotate capacity near the arrow
            mid_x = (arrow_start["x"] + arrow_end["x"]) / 2
            mid_y = (arrow_start["y"] + arrow_end["y"]) / 2
            ax.text(
                mid_x,
                mid_y,
                f"{abs_flow:.1f} TWh",
                color="black",
                fontsize=8,
                ha="center",
                va="center",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
            )

        # Set title for each subplot with specific simulation name
        ax.set_title(f"{year_label} {simulation_names[i]}")

    # Add overall title and legend
    fig.suptitle(f"Country-to-Country Net Energy Flows ({year_label})", fontsize=16)
    legend_elements = [
        plt.Line2D([0], [0], color="blue", lw=2, label="Net Export (TWh)"),
        plt.Line2D([0], [0], color="red", lw=2, label="Net Import (TWh)"),
    ]
    axes[-1].legend(
        handles=legend_elements, loc="lower left", fontsize=10, frameon=True
    )

    plt.tight_layout()
    plt.show()


# Plot for 2023
plot_net_energy_flows_vertically(year_2023_files, "2023")

# Plot for 2040
plot_net_energy_flows_vertically(year_2040_files, "2040")


#! 2. line loading analysis
# Function to calculate global min and max for all scenarios
def calculate_global_min_max(network_files):
    all_average_loading = []
    all_peak_frequencies = []

    for network_file in network_files:
        # Load the network
        network = pypsa.Network(network_file)

        # Calculate Average Line Loading
        average_line_loading = (
            network.lines_t.p0.abs().mean().sort_index()
            / (network.lines.s_nom_opt * network.lines.s_max_pu).sort_index()
        ).fillna(0)

        # Calculate Frequency of Peak Loading
        line_loading_fraction = (
            network.lines_t.p0.abs()
            / (network.lines.s_nom_opt * network.lines.s_max_pu)
        ).fillna(0)
        frequency_of_peak_loading = (line_loading_fraction == 1.0).sum() / len(
            network.snapshots
        )

        # Append results to global lists
        all_average_loading.extend(average_line_loading.values)
        all_peak_frequencies.extend(frequency_of_peak_loading.values)

    return (min(all_average_loading), max(all_average_loading)), (
        min(all_peak_frequencies),
        max(all_peak_frequencies),
    )


# Function to plot line loading with consistent color scales
def plot_line_loading_with_consistent_scale(
    network_files, year_label, global_avg_scale, global_peak_scale
):
    fig, axes = plt.subplots(
        nrows=len(network_files),  # One row per scenario
        ncols=2,  # Two plots per row: Average Line Loading and Frequency of Peak Loading
        subplot_kw={"projection": ccrs.PlateCarree()},
        figsize=(15, len(network_files) * 7),
    )

    for i, network_file in enumerate(network_files):
        # Load the network
        network = pypsa.Network(network_file)

        # Calculate Average Line Loading
        average_line_loading = (
            network.lines_t.p0.abs().mean().sort_index()
            / (network.lines.s_nom_opt * network.lines.s_max_pu).sort_index()
        ).fillna(0)

        # Calculate Frequency of Peak Loading
        line_loading_fraction = (
            network.lines_t.p0.abs()
            / (network.lines.s_nom_opt * network.lines.s_max_pu)
        ).fillna(0)
        frequency_of_peak_loading = (line_loading_fraction == 1.0).sum() / len(
            network.snapshots
        )

        # Filter lines and buses in the Netherlands
        netherlands_lines = network.lines[
            network.lines.bus0.str.startswith("NL")
            & network.lines.bus1.str.startswith("NL")
        ]
        average_line_loading_nl = average_line_loading[netherlands_lines.index]
        frequency_of_peak_loading_nl = frequency_of_peak_loading[
            netherlands_lines.index
        ]

        # Use global scales for normalization
        norm_avg = Normalize(vmin=global_avg_scale[0], vmax=global_avg_scale[1])
        norm_peak = Normalize(vmin=global_peak_scale[0], vmax=global_peak_scale[1])

        # 3. Plot Average Line Loading
        ax_avg = axes[i, 0]  # First column: Average Line Loading
        network.plot(
            ax=ax_avg,
            branch_components=["Line"],
            line_widths=netherlands_lines.s_nom_opt / 3e3,
            line_colors=norm_avg(average_line_loading_nl),
            line_cmap=cm.viridis,
            bus_sizes=0.002,
            bus_colors="gray",
            color_geomap=True,
        )
        ax_avg.set_title(
            f"{year_label} {['Base', 'COL', 'EXCL'][i]}: Average Line Loading"
        )
        ax_avg.axis("off")
        ax_avg.set_xlim(3, 8)  # Adjust x-axis for Netherlands
        ax_avg.set_ylim(50, 55)  # Adjust y-axis for Netherlands

        # Add colorbar for Average Line Loading
        cb_avg = fig.colorbar(
            cm.ScalarMappable(cmap=cm.viridis, norm=norm_avg), ax=ax_avg
        )
        cb_avg.set_label("Average Line Loading (Fraction of Max Capacity)")

        # 4. Plot Frequency of Peak Loading
        ax_peak = axes[i, 1]  # Second column: Frequency of Peak Loading
        network.plot(
            ax=ax_peak,
            branch_components=["Line"],
            line_widths=netherlands_lines.s_nom_opt / 3e3,
            line_colors=norm_peak(frequency_of_peak_loading_nl),
            line_cmap=cm.plasma,
            bus_sizes=0.002,
            bus_colors="gray",
            color_geomap=True,
        )
        ax_peak.set_title(
            f"{year_label} {['Base', 'COL', 'EXCL'][i]}: Frequency of Peak Loading"
        )
        ax_peak.axis("off")
        ax_peak.set_xlim(3, 8)  # Adjust x-axis for Netherlands
        ax_peak.set_ylim(50, 55)  # Adjust y-axis for Netherlands

        # Add colorbar for Frequency of Peak Loading
        cb_peak = fig.colorbar(
            cm.ScalarMappable(cmap=cm.plasma, norm=norm_peak), ax=ax_peak
        )
        cb_peak.set_label("Frequency of Peak Loading (Fraction of Time)")

    plt.tight_layout()
    plt.show()


# Network files for 2023 and 2040
year_2023_files = [
    "results\\2023_Run\\networks\\BASE_optimized_.nc",
    "results\\2023_Run\\networks\\COL_optimized_.nc",
    "results\\2023_Run\\networks\\EXCL_optimized_.nc",
]

year_2040_files = [
    "results\\2040_Run\\networks\\2040BASE_optimized_.nc",
    "results\\2040_Run\\networks\\COL2040_optimized_.nc",
    "results\\2040_Run\\networks\\EXCL2040_optimized_.nc",
]

# Calculate global min and max for 2023 and 2040
global_avg_scale_2023, global_peak_scale_2023 = calculate_global_min_max(
    year_2023_files
)
global_avg_scale_2040, global_peak_scale_2040 = calculate_global_min_max(
    year_2040_files
)

# Plot for 2023
plot_line_loading_with_consistent_scale(
    year_2023_files, "2023", global_avg_scale_2023, global_peak_scale_2023
)

# Plot for 2040
plot_line_loading_with_consistent_scale(
    year_2040_files, "2040", global_avg_scale_2040, global_peak_scale_2040
)


#! N-1 analysis for 90% limit
# Function to calculate global min and max for all scenarios
def calculate_global_min_max(network_files, peak_threshold):
    all_average_loading = []
    all_peak_frequencies = []

    for network_file in network_files:
        # Load the network
        network = pypsa.Network(network_file)

        # Calculate Average Line Loading
        average_line_loading = (
            network.lines_t.p0.abs().mean().sort_index()
            / (network.lines.s_nom_opt * network.lines.s_max_pu).sort_index()
        ).fillna(0)

        # Calculate Frequency of Peak Loading (with threshold for peak)
        line_loading_fraction = (
            network.lines_t.p0.abs()
            / (network.lines.s_nom_opt * network.lines.s_max_pu)
        ).fillna(0)
        frequency_of_peak_loading = (
            line_loading_fraction >= peak_threshold
        ).sum() / len(network.snapshots)

        # Append results to global lists
        all_average_loading.extend(average_line_loading.values)
        all_peak_frequencies.extend(frequency_of_peak_loading.values)

    return (min(all_average_loading), max(all_average_loading)), (
        min(all_peak_frequencies),
        max(all_peak_frequencies),
    )


# Function to plot line loading with consistent color scales
def plot_line_loading_with_peak_threshold(
    network_files, year_label, global_avg_scale, global_peak_scale, peak_threshold
):
    fig, axes = plt.subplots(
        nrows=len(network_files),  # One row per scenario
        ncols=2,  # Two plots per row: Average Line Loading and Frequency of Peak Loading
        subplot_kw={"projection": ccrs.PlateCarree()},
        figsize=(15, len(network_files) * 7),
    )

    for i, network_file in enumerate(network_files):
        # Load the network
        network = pypsa.Network(network_file)

        # Calculate Average Line Loading
        average_line_loading = (
            network.lines_t.p0.abs().mean().sort_index()
            / (network.lines.s_nom_opt * network.lines.s_max_pu).sort_index()
        ).fillna(0)

        # Calculate Frequency of Peak Loading (with threshold for peak)
        line_loading_fraction = (
            network.lines_t.p0.abs()
            / (network.lines.s_nom_opt * network.lines.s_max_pu)
        ).fillna(0)
        frequency_of_peak_loading = (
            line_loading_fraction >= peak_threshold
        ).sum() / len(network.snapshots)

        # Filter lines and buses in the Netherlands
        netherlands_lines = network.lines[
            network.lines.bus0.str.startswith("NL")
            & network.lines.bus1.str.startswith("NL")
        ]
        average_line_loading_nl = average_line_loading[netherlands_lines.index]
        frequency_of_peak_loading_nl = frequency_of_peak_loading[
            netherlands_lines.index
        ]

        # Use global scales for normalization
        norm_avg = Normalize(vmin=global_avg_scale[0], vmax=global_avg_scale[1])
        norm_peak = Normalize(vmin=global_peak_scale[0], vmax=global_peak_scale[1])

        # 3. Plot Average Line Loading
        ax_avg = axes[i, 0]  # First column: Average Line Loading
        network.plot(
            ax=ax_avg,
            branch_components=["Line"],
            line_widths=netherlands_lines.s_nom_opt / 3e3,
            line_colors=norm_avg(average_line_loading_nl),
            line_cmap=cm.viridis,
            bus_sizes=0.002,
            bus_colors="gray",
            color_geomap=True,
        )
        ax_avg.set_title(
            f"{year_label} {['Base', 'COL', 'EXCL'][i]}: Average Line Loading"
        )
        ax_avg.axis("off")
        ax_avg.set_xlim(3, 8)  # Adjust x-axis for Netherlands
        ax_avg.set_ylim(50, 55)  # Adjust y-axis for Netherlands

        # Add colorbar for Average Line Loading
        cb_avg = fig.colorbar(
            cm.ScalarMappable(cmap=cm.viridis, norm=norm_avg), ax=ax_avg
        )
        cb_avg.set_label("Average Line Loading (Fraction of Max Capacity)")

        # 4. Plot Frequency of Peak Loading
        ax_peak = axes[i, 1]  # Second column: Frequency of Peak Loading
        network.plot(
            ax=ax_peak,
            branch_components=["Line"],
            line_widths=netherlands_lines.s_nom_opt / 3e3,
            line_colors=norm_peak(frequency_of_peak_loading_nl),
            line_cmap=cm.plasma,
            bus_sizes=0.002,
            bus_colors="gray",
            color_geomap=True,
        )
        ax_peak.set_title(
            f"{year_label} {['Base', 'COL', 'EXCL'][i]}: Frequency of Peak Loading (Threshold {int(peak_threshold * 100)}%)"
        )
        ax_peak.axis("off")
        ax_peak.set_xlim(3, 8)  # Adjust x-axis for Netherlands
        ax_peak.set_ylim(50, 55)  # Adjust y-axis for Netherlands

        # Add colorbar for Frequency of Peak Loading
        cb_peak = fig.colorbar(
            cm.ScalarMappable(cmap=cm.plasma, norm=norm_peak), ax=ax_peak
        )
        cb_peak.set_label(
            f"Frequency of Peak Loading (≥ {int(peak_threshold * 100)}% of Capacity)"
        )

    plt.tight_layout()
    plt.show()


# Define the threshold for peak loading
peak_threshold = 0.9  # 90% of capacity

# Calculate global min and max for 2023 and 2040
global_avg_scale_2023, global_peak_scale_2023 = calculate_global_min_max(
    year_2023_files, peak_threshold
)
global_avg_scale_2040, global_peak_scale_2040 = calculate_global_min_max(
    year_2040_files, peak_threshold
)

# Plot for 2023
plot_line_loading_with_peak_threshold(
    year_2023_files,
    "2023",
    global_avg_scale_2023,
    global_peak_scale_2023,
    peak_threshold,
)

# Plot for 2040
plot_line_loading_with_peak_threshold(
    year_2040_files,
    "2040",
    global_avg_scale_2040,
    global_peak_scale_2040,
    peak_threshold,
)


#! 3. system cost (delta) analysis
# Function to calculate selected cost components (Line Capital, Line Expansion, Battery Investment)
def calculate_selected_costs(network_file):
    network = pypsa.Network(network_file)

    # Calculate relevant cost components
    battery_investment_costs = (
        network.storage_units.p_nom_opt * network.storage_units.capital_cost
    ).sum()
    line_expansion_costs = (network.links.p_nom_opt * network.links.capital_cost).sum()
    line_capital_costs = (network.lines.s_nom_opt * network.lines.capital_cost).sum()

    # Sum the selected costs
    selected_total_cost = (
        battery_investment_costs + line_expansion_costs + line_capital_costs
    )

    return selected_total_cost


# Define network files for 2023 and 2040
network_files = {
    "2023_BASE": "results\\2023_Run\\networks\\BASE_optimized_.nc",
    "2023_COL": "results\\2023_Run\\networks\\COL_optimized_.nc",
    "2023_EXCL": "results\\2023_Run\\networks\\EXCL_optimized_.nc",
    "2040_BASE": "results\\2040_Run\\networks\\2040BASE_optimized_.nc",
    "2040_COL": "results\\2040_Run\\networks\\COL2040_optimized_.nc",
    "2040_EXCL": "results\\2040_Run\\networks\\EXCL2040_optimized_.nc",
}

# Calculate selected costs for each scenario
selected_costs = {
    scenario: calculate_selected_costs(file) for scenario, file in network_files.items()
}

# Convert to DataFrame and display
df_selected_costs = pd.DataFrame.from_dict(
    selected_costs, orient="index", columns=["Selected Total Costs (€)"]
)

# Save to CSV (optional)
df_selected_costs.to_csv("selected_total_costs.csv")

# Display results
print(df_selected_costs)


# Function to calculate cost breakdown
def calculate_cost_breakdown(network_file):
    network = pypsa.Network(network_file)

    # Calculate costs
    battery_investment_costs = (
        network.storage_units.p_nom_opt * network.storage_units.capital_cost
    ).sum()
    line_expansion_costs = (network.links.p_nom_opt * network.links.capital_cost).sum()
    variable_operating_costs = (
        (network.generators_t.p * network.generators.marginal_cost).sum().sum()
    )
    line_capital_costs = (network.lines.s_nom_opt * network.lines.capital_cost).sum()

    # Calculate total calculated costs
    calculated_total = (
        battery_investment_costs
        + line_expansion_costs
        + variable_operating_costs
        + line_capital_costs
    )

    # "Other" category: remainder difference
    other_costs = network.objective - calculated_total

    return {
        "Battery Investment Costs": battery_investment_costs,
        "HVDC and Converter Capital Costs": line_expansion_costs,
        "Variable Operating Costs": variable_operating_costs,
        "AC Transmission Line Capital Costs": line_capital_costs,
        "Other Costs": other_costs,
    }


# Function to plot waterfall diagram
def plot_deltas_only(deltas, title):
    scenarios = ["COL Scenario", "EXCL Scenario"]
    categories = deltas.columns
    num_categories = len(categories)

    # Initialize the figure and axes
    fig, ax = plt.subplots(figsize=(14, 7))
    bar_width = 0.35
    x = np.arange(num_categories + 1)  # Add +1 for the "Total" column
    hatches = ["//", "\\\\"]  # Patterns for scenarios

    for i, scenario in enumerate(scenarios):
        values = deltas.loc[scenario].values
        total_value = values.sum()
        values_with_total = np.append(values, total_value)

        # Plot bars
        colors = ["green" if v < 0 else "red" for v in values_with_total]
        bars = ax.bar(
            x + i * bar_width,
            values_with_total,
            width=bar_width,
            color=colors,
            edgecolor="black",
            hatch=hatches[i],  # Add pattern
        )

        # Annotate bars
        for j, v in enumerate(values_with_total):
            ax.text(
                x[j] + i * bar_width,
                v + (0.02 * max(abs(values_with_total)))
                if v > 0
                else v - (0.02 * max(abs(values_with_total))),
                f"{v:+,.0f} €",
                ha="center",
                va="bottom" if v > 0 else "top",
                fontsize=10,
                color="black",
            )

    # Add zero line
    ax.axhline(0, color="black", linewidth=0.8)

    # Set labels
    ax.set_xticks(x + bar_width / 2)
    ax.set_xticklabels(list(categories) + ["Total"], rotation=45, ha="right")
    ax.set_ylabel("Cost Delta (€)", fontsize=12)
    ax.set_title(title, fontsize=14)

    # Add legend with patterns
    legend_elements = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            edgecolor="black",
            facecolor="white",
            hatch="//",
            label="COL Scenario",
        ),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            edgecolor="black",
            facecolor="white",
            hatch="\\\\",
            label="EXCL Scenario",
        ),
    ]
    ax.legend(
        handles=legend_elements,
        title="Scenarios",
        loc="upper left",
        fontsize=10,
        frameon=True,
    )

    plt.tight_layout()
    plt.show()


# Load networks for 2023
networks_2023 = [
    ("results\\2023_Run\\networks\\BASE_optimized_.nc", "BASE Scenario"),
    ("results\\2023_Run\\networks\\COL_optimized_.nc", "COL Scenario"),
    ("results\\2023_Run\\networks\\EXCL_optimized_.nc", "EXCL Scenario"),
]

# Calculate base values and deltas for 2023
base_values_2023 = calculate_cost_breakdown(networks_2023[0][0])
deltas_2023 = pd.DataFrame(
    {
        scenario: pd.Series(calculate_cost_breakdown(file))
        - pd.Series(base_values_2023)
        for file, scenario in networks_2023[1:]
    }
).T

# Plot waterfall diagram for 2023
plot_deltas_only(deltas_2023, "Cost Deltas for COL and EXCL Scenarios (2023)")

# Load networks for 2040
networks_2040 = [
    ("results\\2040_Run\\networks\\2040BASE_optimized_.nc", "BASE Scenario"),
    ("results\\2040_Run\\networks\\COL2040_optimized_.nc", "COL Scenario"),
    ("results\\2040_Run\\networks\\EXCL2040_optimized_.nc", "EXCL Scenario"),
]

# Calculate base values and deltas for 2040
base_values_2040 = calculate_cost_breakdown(networks_2040[0][0])
deltas_2040 = pd.DataFrame(
    {
        scenario: pd.Series(calculate_cost_breakdown(file))
        - pd.Series(base_values_2040)
        for file, scenario in networks_2040[1:]
    }
).T

# Plot waterfall diagram for 2040
plot_deltas_only(deltas_2040, "Cost Deltas for COL and EXCL Scenarios (2040)")


#! 4. Line expansion plot


# Define network paths for 2040
networks_2040 = [
    ("results\\2040_Run\\networks\\2040BASE_optimized_.nc", "BASE (2040)"),
    ("results\\2040_Run\\networks\\COL2040_optimized_.nc", "COL (2040)"),
    ("results\\2040_Run\\networks\\EXCL2040_optimized_.nc", "EXCL (2040)"),
]


# Function to compute net grid expansion/reduction
def compute_grid_changes(base_network, scenario_network):
    # Load networks
    base = pypsa.Network(base_network)
    scenario = pypsa.Network(scenario_network)

    # Extract line and link data
    base_lines = base.lines[["bus0", "bus1", "s_nom_opt"]].copy()
    scenario_lines = scenario.lines[["bus0", "bus1", "s_nom_opt"]].copy()

    base_links = base.links[["bus0", "bus1", "p_nom_opt"]].copy()
    scenario_links = scenario.links[["bus0", "bus1", "p_nom_opt"]].copy()

    # Merge to compare changes (ensuring each line appears only once)
    lines_comparison = pd.merge(
        base_lines,
        scenario_lines,
        on=["bus0", "bus1"],
        suffixes=("_base", "_scenario"),
        how="outer",
    ).fillna(0)

    links_comparison = pd.merge(
        base_links,
        scenario_links,
        on=["bus0", "bus1"],
        suffixes=("_base", "_scenario"),
        how="outer",
    ).fillna(0)

    # Compute net capacity change
    lines_comparison["net_change"] = (
        lines_comparison["s_nom_opt_scenario"] - lines_comparison["s_nom_opt_base"]
    )
    links_comparison["net_change"] = (
        links_comparison["p_nom_opt_scenario"] - links_comparison["p_nom_opt_base"]
    )

    # Remove duplicate rows where expansion and reduction were recorded separately
    lines_comparison = lines_comparison.groupby(["bus0", "bus1"]).sum().reset_index()
    links_comparison = links_comparison.groupby(["bus0", "bus1"]).sum().reset_index()

    # Assign type and determine category
    lines_comparison["type"] = "AC"
    links_comparison["type"] = "DC"

    # Classify changes properly
    lines_comparison["change_category"] = np.where(
        lines_comparison["net_change"] > 0, "Expansion", "Reduction"
    )
    links_comparison["change_category"] = np.where(
        links_comparison["net_change"] > 0, "Expansion", "Reduction"
    )

    # Combine both tables
    grid_changes = pd.concat([lines_comparison, links_comparison], ignore_index=True)

    return grid_changes[["bus0", "bus1", "type", "change_category", "net_change"]]


# Compute changes for COL and EXCL vs BASE (2040)
grid_changes_COL_2040 = compute_grid_changes(networks_2040[0][0], networks_2040[1][0])
grid_changes_EXCL_2040 = compute_grid_changes(networks_2040[0][0], networks_2040[2][0])

# Display results
print("Grid Expansion/Reduction for COL (2040):")
print(grid_changes_COL_2040)

print("\nGrid Expansion/Reduction for EXCL (2040):")
print(grid_changes_EXCL_2040)

# Save to CSV for further analysis
grid_changes_COL_2040.to_csv("grid_expansion_COL_2040.csv", index=False)
grid_changes_EXCL_2040.to_csv("grid_expansion_EXCL_2040.csv", index=False)

# Load base network for bus coordinates
base_network = pypsa.Network(networks_2040[0][0])

# Extract bus coordinates
bus_coords = base_network.buses[["x", "y"]]

# Load grid changes
grid_changes_COL_2040 = pd.read_csv("grid_expansion_COL_2040.csv")
grid_changes_EXCL_2040 = pd.read_csv("grid_expansion_EXCL_2040.csv")


# Define function to plot grid changes with improved zoom and focus
def plot_grid_changes(grid_changes, title, ax):
    # Add natural earth features for geographic context
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
    ax.add_feature(cfeature.BORDERS, edgecolor="black", linestyle="--", linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, edgecolor="black")

    for _, row in grid_changes.iterrows():
        bus0, bus1, line_type, change_category, net_change = row

        # Get coordinates
        if bus0 in bus_coords.index and bus1 in bus_coords.index:
            x0, y0 = bus_coords.loc[bus0, ["x", "y"]]
            x1, y1 = bus_coords.loc[bus1, ["x", "y"]]

            # Define color based on change type
            color_map = {
                ("AC", "Expansion"): "green",
                ("AC", "Reduction"): "red",
                ("DC", "Expansion"): "blue",
                ("DC", "Reduction"): "purple",
            }
            color = color_map.get((line_type, change_category), "black")

            # Scale line thickness
            line_width = max(0.5, abs(net_change) / 500)  # Adjust for visibility

            # Plot the line
            ax.plot([x0, x1], [y0, y1], color=color, linewidth=line_width, alpha=0.8)

    # Set title and map settings
    ax.set_title(title, fontsize=14)
    ax.set_xlim([-5, 10])  # Zooming into the Netherlands and nearby regions
    ax.set_ylim([48, 55])  # Cropping to exclude unnecessary southern areas
    ax.grid(True)


# Create a larger figure for better visibility
fig, axes = plt.subplots(
    1, 2, figsize=(14, 8), subplot_kw={"projection": ccrs.PlateCarree()}
)

# Plot changes for COL and EXCL scenarios
plot_grid_changes(grid_changes_COL_2040, "Grid Changes: COL vs BASE (2040)", axes[0])
plot_grid_changes(grid_changes_EXCL_2040, "Grid Changes: EXCL vs BASE (2040)", axes[1])

# Add a well-structured legend
legend_elements = [
    plt.Line2D([0], [0], color="green", lw=2, label="AC Expansion"),
    plt.Line2D([0], [0], color="red", lw=2, label="AC Reduction"),
    plt.Line2D([0], [0], color="blue", lw=2, label="DC Expansion"),
    plt.Line2D([0], [0], color="purple", lw=2, label="DC Reduction"),
]
axes[1].legend(handles=legend_elements, loc="lower left", fontsize=10)

# Show improved map
plt.tight_layout()
plt.show()

#! grid expansion plot
# Define network paths for 2040
base_network_2040 = "results/2040_Run/networks/2040BASE_optimized_.nc"
col_network_2040 = "results/2040_Run/networks/COL2040_optimized_.nc"
excl_network_2040 = "results/2040_Run/networks/EXCL2040_optimized_.nc"

# Load networks
network_base = pypsa.Network(base_network_2040)
network_col = pypsa.Network(col_network_2040)
network_excl = pypsa.Network(excl_network_2040)


# Function to compute both line and link expansions
def compute_grid_expansions(network_base, network_scenario):
    expansions = []

    # Process AC Transmission Lines
    for line_id in network_base.lines.index:
        if line_id in network_scenario.lines.index:
            base_capacity = network_base.lines.at[line_id, "s_nom_opt"]
            scenario_capacity = network_scenario.lines.at[line_id, "s_nom_opt"]
            net_change = scenario_capacity - base_capacity

            if net_change != 0:
                bus0 = network_base.lines.at[line_id, "bus0"]
                bus1 = network_base.lines.at[line_id, "bus1"]
                line_type = "AC"
                change_category = "Expansion" if net_change > 0 else "Reduction"

                expansions.append([bus0, bus1, line_type, change_category, net_change])

    # Process DC Interconnectors (Links)
    for link_id in network_base.links.index:
        if link_id in network_scenario.links.index:
            base_capacity = network_base.links.at[link_id, "p_nom_opt"]
            scenario_capacity = network_scenario.links.at[link_id, "p_nom_opt"]
            net_change = scenario_capacity - base_capacity

            if net_change != 0:
                bus0 = network_base.links.at[link_id, "bus0"]
                bus1 = network_base.links.at[link_id, "bus1"]
                line_type = "DC"
                change_category = "Expansion" if net_change > 0 else "Reduction"

                expansions.append([bus0, bus1, line_type, change_category, net_change])

    return pd.DataFrame(
        expansions,
        columns=["Bus 0", "Bus 1", "line_type", "change_category", "net_change"],
    )


# Compute expansions for COL and EXCL scenarios
grid_changes_COL_2040 = compute_grid_expansions(network_base, network_col)
grid_changes_EXCL_2040 = compute_grid_expansions(network_base, network_excl)

# Save to CSV
grid_changes_COL_2040.to_csv("grid_expansion_COL_2040.csv", index=False)
grid_changes_EXCL_2040.to_csv("grid_expansion_EXCL_2040.csv", index=False)

print("CSV files saved: grid_expansion_COL_2040.csv and grid_expansion_EXCL_2040.csv")

# Load base network for bus coordinates
bus_coords = network_base.buses[["x", "y"]]


# Function to visualize AC and DC grid changes
def plot_grid_changes(grid_changes, title, ax):
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
    ax.add_feature(cfeature.BORDERS, edgecolor="black", linestyle="--", linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, edgecolor="black")

    for _, row in grid_changes.iterrows():
        bus0, bus1, line_type, change_category, net_change = row

        # Get coordinates
        if bus0 in bus_coords.index and bus1 in bus_coords.index:
            x0, y0 = bus_coords.loc[bus0, ["x", "y"]]
            x1, y1 = bus_coords.loc[bus1, ["x", "y"]]

            # Define color based on change type
            color_map = {
                ("AC", "Expansion"): "green",
                ("AC", "Reduction"): "red",
                ("DC", "Expansion"): "blue",
                ("DC", "Reduction"): "purple",
            }
            color = color_map.get((line_type, change_category), "black")

            # Scale line thickness appropriately
            line_width = max(0.5, abs(net_change) / 500)

            # Plot the line
            ax.plot([x0, x1], [y0, y1], color=color, linewidth=line_width, alpha=0.8)

    ax.set_title(title, fontsize=14)
    ax.set_xlim([-10, 15])  # Keep the same scale as before
    ax.set_ylim([50, 60])  # Show full European region
    ax.grid(True)


# Create figure with **consistent Europe-wide view**
fig, axes = plt.subplots(
    1, 2, figsize=(14, 8), subplot_kw={"projection": ccrs.PlateCarree()}
)

# Plot changes for COL and EXCL scenarios
plot_grid_changes(grid_changes_COL_2040, "Grid Changes: COL vs BASE (2040)", axes[0])
plot_grid_changes(grid_changes_EXCL_2040, "Grid Changes: EXCL vs BASE (2040)", axes[1])

# Add **legend** to match colors across scenarios
legend_elements = [
    plt.Line2D([0], [0], color="green", lw=2, label="AC Expansion"),
    plt.Line2D([0], [0], color="red", lw=2, label="AC Reduction"),
    plt.Line2D([0], [0], color="blue", lw=2, label="DC Expansion"),
    plt.Line2D([0], [0], color="purple", lw=2, label="DC Reduction"),
]
axes[1].legend(handles=legend_elements, loc="lower left", fontsize=10)

# Display final map
plt.tight_layout()
plt.show()
