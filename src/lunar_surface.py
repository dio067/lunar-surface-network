import pandas as pd;
import numpy as np;
import networkx as nx;
import matplotlib.pyplot as plt;
import community as community_louvain
import psycopg2


# Database Connection
def get_db_connection():
    conn = psycopg2.connect(
        host="db.lueuzoseflstoiilozqt.supabase.co",
        dbname="postgres",
        user="readonly_user",
        password="StrongPassword123",
        port="5432"
    )
    return conn


# Load Data from DB
def load_data(conn):
    landing_sites = pd.read_sql("SELECT * FROM landing_sites;", conn)
    satellites = pd.read_sql("SELECT * satellites;", conn)

    return landing_sites, satellites



# Create a graph
def build_graph(landing_sites, satellites, surface_threshold=80):
    G = nx.Graph()

    # Add landing site nodes
    for _, row in landing_sites.iterrows():
        G.add_node(row['mission'], type='surface', lat=row['latitude'], lon=row['longitude'])

    # Add satellite nodes
    for _, row in satellites.iterrows():
        G.add_node(row['name'], type='satellite', lat=row['latitude'], lon=row['longitude'])

    # Surface ↔ Surface edges
    for i, site1 in landing_sites.iterrows():
        for j, site2 in landing_sites.iterrows():
            if i < j:
                dist = np.sqrt((site1.latitude - site2.latitude)**2 + (site1.longitude - site2.longitude)**2)
                if dist < surface_threshold:
                    G.add_edge(site1.mission, site2.mission, weight=dist)

    # Satellite ↔ Surface edges
    for _, sat in satellites.iterrows():
        for _, site in landing_sites.iterrows():
            dist = np.sqrt((sat.altitude_km)**2 + (sat.latitude - site.latitude)**2 + (sat.longitude - site.longitude)**2)
            if dist <= sat.coverage_radius_km:
                G.add_edge(sat.name, site.mission, weight=dist)

    return G

    
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

