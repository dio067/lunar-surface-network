import pandas as pd;
import numpy as np;
import networkx as nx;
import matplotlib.pyplot as plt;
import community as community_louvain
import psycopg2





# Load lunar surface site data and satellite data
sites = pd.read_csv('../data/surface_sites.csv');
sats = pd.read_csv('../data/satellite_data.csv');

    

# Create a graph
G = nx.Graph();


# Add Surface Sites Nodes
for _, row in sites.iterrows():
   G.add_node(row['name'], type=row['type'], lat=row['lat'], lon=row['lon'])


# Add Satellites Nodes
for _, row in sats.iterrows():
    G.add_node(row['name'], pos=(row['lon'], row['lat']), type='satellite');

# Add edges Surface 
  for i, site1 in sites.iterrows():
    for j, site2 in sites.iterrows():
        dist = np.sqrt((site1.lat - site2.lat)**2 + (site1.lon - site2.lon)**2)
          if dist < 80:
            G.add_edge(site1.name, site2.name, weight=dist)


# Add edges Satellite 
for _, sat1 in sats.iterrows():
    for _, site in sites.iterrows():
        dist = np.sqrt((sat.altitude_km)**2 + (site.lat - site.lat)**2 + (site.lon - site.lon)**2)
        if dist <= sat.coverage_radius_km:
            G.add_edge(sat.name, site.name, weight=dist)

# Draw the graph
colors = ['skyblue' if G.nodes[n]['type']=='surface' else 'orange' for n in G.nodes()]
nx.draw(G, with_labels=True, node_color=colors, node_size=2000)
plt.show()

# Centerality Analysis
deg = nx.degree_centrality(G)
eig = nx.eigenvector_centrality(G)

print("Degree Centrality:", deg)
print("Eigenvector Centrality:", eig)

# Modularity (Communities)
partition = community_louvain.best_partition(G)
print("Communities:", partition)


#Von Neumann Entropy
A = nx.to_numpy_array(G)
L = np.diag(np.sum(A, axis=1)) - A
vals = np.linalg.eigvalsh(L)
vals = vals[vals > 1e-12]
p = vals / np.sum(vals)
HvN = -np.sum(p * np.log2(p))
print("Von Neumann Entropy:", HvN)

