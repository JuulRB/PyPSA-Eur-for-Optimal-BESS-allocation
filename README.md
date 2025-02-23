# PyPSA-Eur-for-Optimal-BESS-allocation
This model is a version of PyPSA-Eur focusing on optimal BESS allocation in the Netherlands

This repository contains the PyPSA-EUR code developed for a master’s thesis investigating the optimal placement of battery storage systems in the Netherlands and its neighboring regions. The study aims to provide a comprehensive analysis of future energy system configurations by simulating two distinct temporal scenarios—2023 and 2040—across a network comprising 37 nodes. Such analysis is crucial for understanding how strategic battery deployment can enhance grid stability, support renewable energy integration, and inform long-term energy planning.

The simulations are organized into six primary runs, each representing different grid operation and planning assumptions:

2023 Simulations:

BASE Configuration: Represents the baseline scenario, stored in results/2023_Run/networks/BASE_optimized_.nc.
COL Configuration: Reflects a scenario with collaborative or integrated planning measures, stored in results/2023_Run/networks/COL_optimized_.nc.
EXCL Configuration: Corresponds to an isolated or exclusive planning approach, stored in results/2023_Run/networks/EXCL_optimized_.nc.
2040 Simulations:

2040BASE Configuration: The future baseline scenario, stored in results/2040_Run/networks/2040BASE_optimized_.nc.
COL2040 Configuration: Represents a future integrated planning scenario, stored in results/2040_Run/networks/COL2040_optimized_.nc.
EXCL2040 Configuration: Denotes a future isolated planning scenario, stored in results/2040_Run/networks/EXCL2040_optimized_.nc.

The PyPSA-EUR framework serves as a robust tool for simulating large-scale energy systems, enabling detailed assessment of the network's performance under varying operational conditions. The code provided here encompasses modules for data preprocessing, network optimization, and results post-processing, ensuring both reproducibility and flexibility for extended research.

Users are encouraged to review the in-line documentation and comments within the code for further details on implementation specifics and simulation parameters. Feedback and suggestions for improvement are welcomed to enhance the utility of this research tool.
