import pypsa
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.geometry import Point

# Load the network
network = pypsa.Network(r"results\2023_Run\networks\base_s_37_elec_lvopt_.nc")
print(network)

#! Land exclusion
# Load solar availability matrix
solar_availability = xr.open_dataset(
    r"resources\2023_Run\availability_matrix_37_solar.nc"
)

# Extract dimensions and data
x_coords = solar_availability["x"].values  # Longitude
y_coords = solar_availability["y"].values  # Latitude
availability = solar_availability["__xarray_dataarray_variable__"]  # Availability

# Convert bus coordinates to GeoDataFrame
bus_coords = gpd.GeoDataFrame(
    network.buses, geometry=gpd.points_from_xy(network.buses.x, network.buses.y)
)
bus_coords.set_crs("EPSG:4326", inplace=True)

# Filter for NL buses only
nl_buses = bus_coords[bus_coords.index.str.startswith("NL")]

# Check land availability for NL buses
for bus_id in nl_buses.index:
    bus_row = nl_buses.loc[bus_id]
    bus_geometry = bus_row.geometry
    available = False

    for lat in y_coords:
        for lon in x_coords:
            point = Point(lon, lat)
            distance = point.distance(bus_geometry)
            if distance <= 2:  # Check if within 2 km radius
                bus_index = list(nl_buses.index).index(bus_id)
                availability_value = availability[
                    bus_index, list(y_coords).index(lat), list(x_coords).index(lon)
                ]
                if availability_value > 0:
                    available = True
                    break
        if available:
            break

    # Set battery capacity to 0 if no availability within 2km
    if not available:
        batteries_at_bus = network.storage_units[
            (network.storage_units.bus == bus_id)
            & (network.storage_units.carrier == "battery")
        ]
        for battery in batteries_at_bus.index:
            network.storage_units.at[battery, "p_nom_min"] = 0
            network.storage_units.at[battery, "p_nom_max"] = 0

#! cost of land
# Define land costs per bus for Dutch regions (in euros per MW)
land_costs = {
    "NL0 0": 4710,
    "NL0 1": 4230,
    "NL0 2": 4668,
    "NL0 3": 4680,
    "NL0 4": 4848,
    "NL0 5": 4578,
    "NL0 6": 5166,
    "NL0 7": 4710,
    "NL0 8": 4848,
    "NL0 9": 4476,
    "NL0 10": 10962,
    "NL0 11": 3648,
    "NL0 12": 5424,
    "NL0 13": 4476,
    "NL0 14": 4668,
    "NL0 15": 4230,
    "NL0 16": 4848,
    "NL0 17": 4710,
}

# Define parameters
opportunity_rate = 0.08  # 8% annual rate
battery_lifetime = 25  # 25 years

# Calculate annuity factor
annuity_factor = opportunity_rate / (1 - (1 + opportunity_rate) ** -battery_lifetime)

# Annualize land costs for each node
annualized_land_costs = {bus: cost * annuity_factor for bus, cost in land_costs.items()}

# Print annualized land costs
print("Annualized Land Costs (€/MW):")
for bus, annual_cost in annualized_land_costs.items():
    print(f"{bus}: €{annual_cost:.2f}")

# Adjust capital cost for each battery based on annualized land costs
for bus, annual_cost in annualized_land_costs.items():
    # Find storage units associated with each Dutch bus
    dutch_storage_units = network.storage_units[
        (network.storage_units.bus == bus)
        & (network.storage_units.carrier == "battery")
    ]

    # Add annualized land cost to the capital cost
    network.storage_units.loc[dutch_storage_units.index, "capital_cost"] += annual_cost

# Define technology aggregation and apply
technology_map = {
    "offwind-ac": "offshore wind",
    "offwind-dc": "offshore wind",
    "offwind-float": "offshore wind",
    "solar": "solar",
    "solar-hsat": "solar",
    "OCGT": "gas",  # Bundling OCGT and CCGT under "gas"
    "CCGT": "gas",
    "coal": "coal",
    "lignite": "coal",  # Bundling lignite and coal under "coal"
}
network.generators["carrier"] = network.generators["carrier"].replace(technology_map)
network.storage_units["carrier"] = network.storage_units["carrier"].replace(
    technology_map
)
network.links["carrier"] = network.links["carrier"].replace(technology_map)


#!hydro data not available in the network
# Add hydro generation for Norway
hydro_capacity = 34139  # MW
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
    "nuclear": {"DE": 0, "IE": 0, "NL": 482, "BE": 3908, "NO": 0, "DK": 0, "GB": 5883},
    "offshore_wind": {
        "DE": 8500,
        "IE": 200,
        "NL": 5269,
        "BE": 2263,
        "NO": 96,
        "DK": 2343,
        "GB": 14741,
    },
    "onshore_wind": {
        "DE": 58000,
        "IE": 4100,
        "NL": 4500,
        "BE": 2500,
        "NO": 5073,
        "DK": 6100,
        "GB": 14000,
    },
    "gas": {
        "DE": 29600,
        "IE": 4652,
        "NL": 20000,
        "BE": 6000,
        "NO": 0,
        "DK": 1500,
        "GB": 30000,
    },
    "coal": {  # Bundled lignite and coal
        "DE": 36800,
        "IE": 855,
        "NL": 3500,
        "BE": 1800,
        "NO": 0,
        "DK": 1000,
        "GB": 1000,
    },
    "oil": {"DE": 0, "IE": 292, "NL": 500, "BE": 1000, "NO": 0, "DK": 500, "GB": 500},
    "solar": {
        "DE": 66000,
        "IE": 720,
        "NL": 15000,
        "BE": 5000,
        "NO": 299,
        "DK": 2500,
        "GB": 14000,
    },
    "biomass": {
        "DE": 9000,
        "IE": 42,
        "NL": 2000,
        "BE": 800,
        "NO": 642,
        "DK": 1000,
        "GB": 3000,
    },
}

network = fix_generator_capacities_with_config(network, fixed_capacities)

# Adjust BESS capital cost for ancillary income
BESS_ANCILLARY_INCOME_RATIO = 0.3762
network.storage_units.loc[
    network.storage_units.carrier == "battery", "capital_cost"
] *= 1 - BESS_ANCILLARY_INCOME_RATIO

# Recompute capex for batteries
network.storage_units.loc[network.storage_units.carrier == "battery", "capex"] = (
    network.storage_units.loc[network.storage_units.carrier == "battery", "p_nom_opt"]
    * network.storage_units.loc[
        network.storage_units.carrier == "battery", "capital_cost"
    ]
)

# Verify the updated capex
battery_data = network.storage_units[network.storage_units.carrier == "battery"]
print("Updated Battery Capex (EUR):")
print(battery_data[["bus", "p_nom_opt", "capital_cost", "capex"]])

# Calculate total capex
total_battery_capex = battery_data["capex"].sum()
print(f"\nTotal Battery Capex for the System: {total_battery_capex:.2f} EUR")

# Calculate total battery capex for the Netherlands
dutch_battery_data = battery_data[battery_data.bus.str.startswith("NL")]
total_dutch_battery_capex = dutch_battery_data["capex"].sum()
print(f"Total Battery Capex for the Netherlands: {total_dutch_battery_capex:.2f} EUR")

#! Optimize the network
network.optimize(solver_name="gurobi")

# Save the optimized network to a NetCDF file
network.export_to_netcdf(r"results\2023_Run\networks\EXCL_optimized_.nc")
