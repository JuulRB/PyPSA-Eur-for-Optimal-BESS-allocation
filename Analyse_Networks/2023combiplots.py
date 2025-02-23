import pypsa
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import cartopy.crs as ccrs
import pandas as pd


# Function to generate a map plot for a specific network
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


# Define network paths and titles
networks = [
    ("results\\2023_Run\\networks\\BASE_optimized_.nc", "BASE Scenario (2023)"),
    ("results\\2023_Run\\networks\\COL_optimized_.nc", "COL Scenario (2023)"),
    ("results\\2023_Run\\networks\\EXCL_optimized_.nc", "EXCL Scenario (2023)"),
]

# Create the figure and axes
fig, axes = plt.subplots(
    len(networks), 1, subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(10, 15)
)

# Generate plots for each network
scatter = None
for ax, (network_path, title) in zip(axes, networks):
    scatter = plot_network(network_path, title, ax)

# Add the color bar using a dedicated axis outside the subplots
fig.subplots_adjust(right=0.85)  # Make space on the right for the color bar
cbar_ax = fig.add_axes([0.83, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
cbar = fig.colorbar(scatter, cax=cbar_ax, orientation="vertical")
cbar.set_label("Battery Capacity (MW)")

# Adjust layout
plt.tight_layout()
plt.show()

#! Calculate loading of lines for visualization
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


def plot_bundled_net_energy_flows(network_files, year_label):
    num_simulations = len(network_files)
    fig, axes = plt.subplots(
        nrows=1,
        ncols=num_simulations,
        subplot_kw={"projection": ccrs.PlateCarree()},
        figsize=(6 * num_simulations, 10),
    )

    for i, network_file in enumerate(network_files):
        # Load the network
        network = pypsa.Network(network_file)

        # Extract interconnector links
        interconnectors = network.links[
            network.links.bus0.str[:2] != network.links.bus1.str[:2]
        ]
        link_flows = network.links_t.p0[interconnectors.index]  # Power flows (MW)

        # Calculate net energy flows (in TWh)
        total_energy_flows = link_flows.sum(axis=0) / 1e6  # Convert to TWh

        # Get the correct axis for this simulation
        ax = axes[i] if num_simulations > 1 else axes
        network.plot(ax=ax, bus_sizes=0.002, line_widths=0.2, color_geomap=True)

        # Plot arrows and annotate net flows
        scaling_factor = 0.05  # Adjust arrow size
        for link_id in interconnectors.index:
            bus0_coords = network.buses.loc[
                interconnectors.loc[link_id].bus0, ["x", "y"]
            ]
            bus1_coords = network.buses.loc[
                interconnectors.loc[link_id].bus1, ["x", "y"]
            ]

            net_flow = total_energy_flows[link_id]  # Net flow in TWh
            abs_flow = abs(net_flow)

            # Only include significant flows
            if abs_flow < 0.1:
                continue

            # Define arrow properties
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
                alpha=0.8,
                head_width=0.3,
                linewidth=1.5,
                length_includes_head=True,
            )

            # Annotate capacity near the arrow
            mid_x = (arrow_start["x"] + arrow_end["x"]) / 2
            mid_y = (arrow_start["y"] + arrow_end["y"]) / 2
            offset_x = (
                arrow_end["y"] - arrow_start["y"]
            ) * 0.01  # Offset to avoid overlap
            offset_y = (arrow_start["x"] - arrow_end["x"]) * 0.01

            ax.text(
                mid_x + offset_x,
                mid_y + offset_y,
                f"{abs_flow:.2f} TWh",
                color="black",
                fontsize=8,
                ha="center",
                va="center",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
            )

        # Set title for each subplot
        ax.set_title(f"{year_label} Simulation {i + 1}")

    # Add overall title and legend
    fig.suptitle(f"Country-to-Country Net Energy Flows ({year_label})", fontsize=16)
    legend_elements = [
        plt.Line2D([0], [0], color="blue", lw=2, label="Net Export (TWh)"),
        plt.Line2D([0], [0], color="red", lw=2, label="Net Import (TWh)"),
    ]
    axes[0].legend(handles=legend_elements, loc="lower left", fontsize=10, frameon=True)

    # Adjust layout
    plt.tight_layout()
    plt.show()


# Generate bundled plots for 2023 and 2040
plot_bundled_net_energy_flows(year_2023_files, "2023")
plot_bundled_net_energy_flows(year_2040_files, "2040")

#!system cost bar plot

# Define network files for 2023
network_files_2023 = [
    "results\\2023_Run\\networks\\BASE_optimized_.nc",
    "results\\2023_Run\\networks\\COL_optimized_.nc",
    "results\\2023_Run\\networks\\EXCL_optimized_.nc",
]

# Retrieve system costs from networks
scenarios = ["2023BASE", "2023COL", "2023EXCL"]
total_costs = []
for network_file in network_files_2023:
    network = pypsa.Network(network_file)  # Load the network
    total_costs.append(round(network.objective))  # Get the objective value and round it

# Calculate deltas compared to the base case
delta_costs = [0] + [round(cost - total_costs[0]) for cost in total_costs[1:]]

# Create the bar chart
x = np.arange(len(scenarios))
width = 0.35

fig, ax1 = plt.subplots(figsize=(10, 6))
bars1 = ax1.bar(x, total_costs, width, label="Total System Cost (€)", color="blue")

# Overlay Delta Costs
ax2 = ax1.twinx()
bars2 = ax2.bar(x + width, delta_costs, width, label="Delta Costs (€)", color="orange")

# Annotate bars
for bar in bars1:
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"€{bar.get_height():,}",
        ha="center",
        va="bottom",
        fontsize=9,
    )
for bar in bars2:
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"€{bar.get_height():,}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

# Formatting
ax1.set_ylabel("Total System Cost (€)")
ax2.set_ylabel("Delta Costs (€)")
ax1.set_title("System Costs Comparison Across Scenarios")
ax1.set_xticks(x + width / 2)
ax1.set_xticklabels(scenarios)

# Adjust legend placement
fig.legend(
    loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=2, fontsize=10, frameon=True
)

plt.tight_layout()
plt.show()
