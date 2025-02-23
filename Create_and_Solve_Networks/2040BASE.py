import pypsa
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd
import geopandas as gpd

# Load the network
network = pypsa.Network(r"results/2040_Run/networks/base_s_37_elec_lvopt_.nc")
print(network)


# Define technology aggregation and apply
technology_map = {
    "offwind-ac": "offshore wind",
    "offwind-dc": "offshore wind",
    "offwind-float": "offshore wind",
    "solar": "solar",
    "solar-hsat": "solar",
    "OCGT": "gas",
    "CCGT": "gas",
    "coal": "coal",
    "lignite": "coal",
}
network.generators["carrier"] = network.generators["carrier"].replace(technology_map)
network.storage_units["carrier"] = network.storage_units["carrier"].replace(
    technology_map
)
network.links["carrier"] = network.links["carrier"].replace(technology_map)


#!hydro data not available in the network
# Add hydro generation for Norway
hydro_capacity = 37400  # MW
norway_buses = network.buses[network.buses.index.str.startswith("NO")].index

# Check if hydro already exists
if "hydro" not in network.carriers.index:
    network.add("Carrier", "hydro")

# Add hydro generators
for bus in norway_buses:
    network.add(
        "Generator",
        f"hydro_{bus}",
        bus=bus,
        p_nom=hydro_capacity / len(norway_buses),  # Divide capacity evenly among buses
        p_nom_extendable=False,  # Make it non-extendable
        carrier="hydro",
    )

# Set capital cost for hydro generators
HYDRO_CAPEX = 25000  # €/MW/year
network.generators.loc[network.generators.carrier == "hydro", "capital_cost"] = (
    HYDRO_CAPEX
)

# Adjust capital cost for each battery based on land costs in Dutch regions
land_costs = {f"NL0 {i}": 0 for i in range(13)}
for bus, cost in land_costs.items():
    dutch_storage_units = network.storage_units[
        (network.storage_units.bus == bus)
        & (network.storage_units.carrier == "battery")
    ]
    network.storage_units.loc[dutch_storage_units.index, "capital_cost"] += cost


# Function to fix generator capacities
def fix_generator_capacities_with_config(network, fixed_capacities):
    for tech, country_data in fixed_capacities.items():
        for country, capacity in country_data.items():
            gen_selector = (
                network.generators.carrier.isin(
                    ["offwind-ac", "offwind-dc", "offwind-float"]
                )
                if tech == "offshore_wind"
                else (network.generators.carrier == tech)
            ) & network.generators.bus.str.startswith(country)

            current_capacity = network.generators.loc[gen_selector, "p_nom"].sum()

            if current_capacity > 0:
                scale_factor = capacity / current_capacity
                network.generators.loc[gen_selector, "p_nom"] *= scale_factor
            elif capacity > 0 and len(network.generators[gen_selector]) > 0:
                print(
                    f"Info: No existing {tech} generators for {country}, adding placeholders."
                )
                network.generators.loc[gen_selector, "p_nom"] = capacity / len(
                    network.generators[gen_selector]
                )
                network.generators.loc[gen_selector, "p_nom_extendable"] = True
            elif capacity > 0:
                print(
                    f"Warning: No generators of type {tech} found for {country}, skipping adjustment."
                )
            else:
                print(f"Info: No {tech} generators needed for {country}.")

            network.generators.loc[gen_selector, "p_nom_extendable"] = False
    return network


# Fixed Capacities
fixed_capacities = {
    "nuclear": {"DE": 0, "IE": 0, "NL": 1500, "BE": 0, "NO": 0, "DK": 0, "GB": 13236},
    "offshore wind": {
        "DE": 64723,
        "IE": 11096,
        "NL": 41500,
        "BE": 6560,
        "NO": 2000,
        "DK": 10722,
        "GB": 95158,
    },
    "onshore wind": {
        "DE": 158878,
        "IE": 9686,
        "NL": 15100,
        "BE": 18412,
        "NO": 2475,
        "DK": 4925,
        "GB": 42040,
    },
    "gas": {
        "DE": 244256,
        "IE": 4398,
        "NL": 11734,
        "BE": 18412,
        "NO": 0,
        "DK": 1500,
        "GB": 61131,
    },
    "coal": {  # Bundled lignite and coal
        "DE": 40791,
        "IE": 1096,
        "NL": 6901,
        "BE": 0,
        "NO": 0,
        "DK": 1000,
        "GB": 5834,
    },
    "oil": {"DE": 786, "IE": 208, "NL": 0, "BE": 127, "NO": 0, "DK": 0, "GB": 0},
    "solar": {
        "DE": 365875,
        "IE": 13162,
        "NL": 122700,
        "BE": 26285,
        "NO": 2500,
        "DK": 36419,
        "GB": 59287,
    },
    "biomass": {
        "DE": 8012,
        "IE": 68,
        "NL": 220,
        "BE": 6251,
        "NO": 642,
        "DK": 21644,
        "GB": 3000,
    },
}


network = fix_generator_capacities_with_config(network, fixed_capacities)


#! Optimize the network
network.optimize(solver_name="gurobi")

# Save the optimized network to a NetCDF file
network.export_to_netcdf(r"results\2040_Run\networks\2040BASE_optimized_.nc")
