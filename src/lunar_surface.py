import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
from community import community_louvain

# # Database Connection
# def get_db_connection():
#     conn = psycopg2.connect(
#         host="db.lueuzoseflstoiilozqt.supabase.co",
#         dbname="postgres",
#         user="readonly_user",
#         password="StrongPassword123",
#         port="5432"
#     )
#     return conn

# # Load Data from DB
# def load_data(conn):
#     landing_sites = pd.read_sql("SELECT * FROM landing_sites;", conn)
#     satellites = pd.read_sql("SELECT * FROM satellites;", conn)
#     return landing_sites, satellites

# Create a graph
def create_graph(landing, sats, surface_threshold_km=900, max_isl_range=2500):
    G = nx.Graph()
    edges_data = []

    # --- ADD SURFACE NODES ---
    for _, row in landing.iterrows():
        G.add_node(row["name"],
                   type="surface",
                   lat=row["lat"],
                   lon=row["lon"],
                   mission_type=row["mission_type"])

    # --- ADD SATELLITES ---
    for _, row in sats.iterrows():
        G.add_node(row["name"],
                   type="satellite",
                   lat=row["lat"],
                   lon=row["lon"],
                   altitude=row["altitude_km"],
                   coverage=row["coverage_radius_km"],
                   mission_type=row["mission_type"])

    # --- SURFACE ↔ SURFACE ---
    for i in range(len(landing)):
        for j in range(i+1, len(landing)):
            s1 = landing.iloc[i]
            s2 = landing.iloc[j]
            dist = haversine(s1.lat, s1.lon, s2.lat, s2.lon)

            if dist <= surface_threshold_km:
                G.add_edge(s1['name'], s2['name'],
                           weight=dist,
                           edge_type='surface_link')
                edges_data.append({
                    'source': s1['name'],
                    'target': s2['name'],
                    'weight': dist,
                    'edge_type': 'surface_link'
                })

    # --- SATELLITE ↔ SURFACE ---
    for _, sat in sats.iterrows():
        for _, site in landing.iterrows():
            surf_dist = haversine(site.lat, site.lon, sat.lat, sat.lon)
            total_dist = np.sqrt(surf_dist**2 + sat.altitude_km**2)

            if total_dist <= sat.coverage_radius_km:
                G.add_edge(sat['name'], site['name'],
                           weight=total_dist,
                           edge_type='comm_link')
                edges_data.append({
                    'source': sat['name'],
                    'target': site['name'],
                    'weight': total_dist,
                    'edge_type': 'comm_link'
                })

    # --- SATELLITE ↔ SATELLITE ---
    for i in range(len(sats)):
        for j in range(i+1, len(sats)):
            sat1 = sats.iloc[i]
            sat2 = sats.iloc[j]

            surf_dist = haversine(sat1.lat, sat1.lon, sat2.lat, sat2.lon)
            alt_diff = abs(sat1.altitude_km - sat2.altitude_km)
            sat_dist = np.sqrt(surf_dist**2 + alt_diff**2)

            if sat_dist <= max_isl_range:
                G.add_edge(sat1['name'], sat2['name'],
                           weight=sat_dist,
                           edge_type='inter_sat_link')

                edges_data.append({
                    'source': sat1['name'],
                    'target': sat2['name'],
                    'weight': sat_dist,
                    'edge_type': 'inter_sat_link'
                })

    edges_df = pd.DataFrame(edges_data)
    edges_df.to_csv("lunar_network_edges_pruned.csv", index=False)

    # --- PRUNE nodes with degree <= 1 ---
    remove_list = [n for n, deg in G.degree() if deg <= 1]
    G_pruned = G.copy()
    G_pruned.remove_nodes_from(remove_list)

    landing_pruned = landing[landing['name'].isin(G_pruned.nodes())]
    sats_pruned = sats[sats['name'].isin(G_pruned.nodes())]

    print("\nPRUNED GRAPH")
    print(f"Nodes: {G_pruned.number_of_nodes()}")
    print(f"Edges: {G_pruned.number_of_edges()}")

    return G_pruned, edges_df, landing_pruned, sats_pruned


# Compute Metrics
def compute_metrics(G):
    print("\nComputing metrics...")

    degree_cent = nx.degree_centrality(G)
    bet_cent = nx.betweenness_centrality(G, weight='weight')
    close_cent = nx.closeness_centrality(G, distance='weight')

    # eigenvector (handle disconnected)
    try:
        eigen_cent = nx.eigenvector_centrality(G, max_iter=900, weight='weight')
    except:
        biggest = max(nx.connected_components(G), key=len)
        G2 = G.subgraph(biggest).copy()
        eigen_cent = nx.eigenvector_centrality(G2, max_iter=900, weight='weight')
        for node in G.nodes():
            if node not in eigen_cent:
                eigen_cent[node] = 0.0

    # Louvain modularity
    communities = community_louvain.best_partition(G, weight='weight')
    modularity = community_louvain.modularity(communities, G, weight='weight')

    # Von Neumann Entropy
    A = nx.to_numpy_array(G)
    degs = np.sum(A, axis=1)
    degs[degs == 0] = 1
    D_inv = np.diag(1.0 / np.sqrt(degs))
    Lnorm = np.eye(len(G)) - D_inv @ A @ D_inv

    vals = np.linalg.eigvalsh(Lnorm)
    vals = vals[vals > 1e-12]
    vals = vals / np.sum(vals)
    vne = -np.sum(vals * np.log2(vals + 1e-10))

    # clustering / path
    avg_clustering = nx.average_clustering(G)
    if nx.is_connected(G):
        avg_path = nx.average_shortest_path_length(G, weight='weight')
        diameter = nx.diameter(G)
    else:
        comp = max(nx.connected_components(G), key=len)
        G2 = G.subgraph(comp)
        avg_path = nx.average_shortest_path_length(G2, weight='weight')
        diameter = nx.diameter(G2)

    return {
        'degree_centrality': degree_cent,
        'betweenness_centrality': bet_cent,
        'closeness_centrality': close_cent,
        'eigenvector_centrality': eigen_cent,
        'communities': communities,
        'modularity': modularity,
        'von_neumann_entropy': vne,
        'avg_clustering': avg_clustering,
        'avg_path_length': avg_path,
        'diameter': diameter
    }

# Visualize Graph
def visualize_graph(G):
    colors = ['skyblue' if G.nodes[n]['type']=='surface' else 'orange' for n in G.nodes()]
    nx.draw(G, with_labels=True, node_color=colors, node_size=1500)
    plt.show()

# Main
def main():
    conn = get_db_connection()
    landing_sites, satellites = load_data(conn)
    G = create_graph(landing_sites, satellites)
    deg, eig, partition, HvN = compute_metrics(G)
    visualize_graph(G)

    print("Degree Centrality:", deg)
    print("Eigenvector Centrality:", eig)
    print("Communities:", partition)
    print("Von Neumann Entropy:", HvN)

if __name__ == "__main__":
    main()
