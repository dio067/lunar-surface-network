# Lunar Communication Network Analysis

## Overview

This project builds a realistic communication network model for the Moon using real geographic and mission data. It combines lunar craters, Apollo and Luna landing sites, and a set of real lunar satellites to create a unified communication graph.
The script computes distances, builds the network, performs several graph-theory analyses, and generates both 2D and 3D visualizations. All results are saved to CSV files for reuse.

---

## Features

### Real Lunar Data

The model uses real data from multiple sources, including:

- Well-known lunar craters
- Apollo and Luna landing locations
- Actual lunar orbiters such as LRO, Chandrayaan-2, and KPLO

### Distance and Coverage Modeling

The project includes:

- Surface-to-surface distances using the Haversine formula
- Satellite-to-surface distances using altitude and geometry
- Satellite coverage calculations
- Inter-satellite link ranges

### Network Construction

The network includes:

- Proximity-based links between surface sites
- Satellite-to-surface communication links
- Links between satellites within range
- Automatic pruning of low-degree nodes for clarity

### Graph Analysis

The script computes:

- Degree, betweenness, closeness, and eigenvector centrality
- Louvain community detection
- Modularity
- Von Neumann entropy
- Clustering coefficient
- Shortest path lengths and graph diameter
- Degree distribution

### Visualization

The project generates a multi-panel figure that includes:

- A 3D model of the Moon with nodes and edges
- A 2D projection of the network
- Community distribution
- Centrality comparison charts
- Degree histogram
- A summary panel with the main statistics

---

## Output Files

| File                           | Description                                   |
| ------------------------------ | --------------------------------------------- |
| lunar_landing_sites.csv        | Surface locations (craters and landing sites) |
| lunar_satellites.csv           | Orbital satellite data                        |
| lunar_network_edges_pruned.csv | Final edges after pruning                     |
| lunar_node_metrics_pruned.csv  | Centrality and community metrics              |
| enhanced_lunar_network_3d.png  | Visualization figure                          |

---

## Installation

Install required packages:

```
pip install pandas numpy networkx matplotlib python-louvain
```

---

## How to Run

Run the main script using:

```
python main.py
```

or execute the code within Google Colab.

The script will:

1. Generate the lunar dataset
2. Build and prune the network
3. Compute all metrics
4. Save all CSV output files
5. Generate the visualization
6. Print a detailed summary of the analysis

---

## Purpose

The goal of this project is to explore how real lunar surface locations and satellites could form an early communication network.
It demonstrates how distance, geography, and orbital placement influence the structure and behavior of a lunar communication system.

---

## References

1. **NASA Planetary Data System (PDS)**
   [https://pds.nasa.gov](https://pds.nasa.gov)
   Used for official LRO crater catalogs and lunar topography data.

2. **Lunar and Planetary Institute (LPI) – Lunar Nomenclature**
   [https://www.lpi.usra.edu/resources/lunar_orbiter/](https://www.lpi.usra.edu/resources/lunar_orbiter/)
   Provides official names and coordinates of lunar surface features.

3. **NASA – Apollo Landing Sites Archive**
   [https://www.nasa.gov/history/apollo](https://www.nasa.gov/history/apollo)
   Source of the official coordinates for all Apollo landing sites.

4. **ISRO – Chandrayaan-2 Mission Overview**
   [https://www.isro.gov.in/Chandrayaan2.html](https://www.isro.gov.in/Chandrayaan2.html)
   Provides orbital information and mission details for the Chandrayaan-2 orbiter.

5. **KARI – KPLO (Danuri) Mission Page**
   [https://www.kari.re.kr/eng/sub03_01_02.do](https://www.kari.re.kr/eng/sub03_01_02.do)
   Official mission profile for the Korea Pathfinder Lunar Orbiter.

6. **NASA – Lunar Reconnaissance Orbiter (LRO) Mission Page**
   [https://www.nasa.gov/missions/lro](https://www.nasa.gov/missions/lro)
   Mission description and background for NASA’s LRO satellite.

7. **NASA NTRS – 3GPP Lunar Communications Report**
   [https://ntrs.nasa.gov](https://ntrs.nasa.gov)
   Contains NASA’s official 3GPP study on lunar and deep-space communication systems.

8. **IEEE Xplore – Hylton et al. (2024)**
   [https://ieeexplore.ieee.org](https://ieeexplore.ieee.org)
   (Search: "A Survey of Mathematical Structures for Lunar Networks")
   Research on mathematical and graph-theoretic structures applied to lunar communication systems.

9. **Medium – “Dune: A Hidden Network”**
   [https://medium.com](https://medium.com)
   (Search: "Dune: A Hidden Network")
   Example of converting terrain into a graph/network representation.
