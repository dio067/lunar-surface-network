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
    
# Helpers
def lat_lon_to_3d(lat, lon, radius=MOON_RADIUS_KM):
    """Convert lat/lon to 3D Cartesian coordinates"""
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    return x, y, z
    


def plot_3d_moon_sphere(ax, alpha=0.05):
    """Draw a wireframe sphere representing the Moon"""
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = MOON_RADIUS_KM * np.outer(np.cos(u), np.sin(v))
    y = MOON_RADIUS_KM * np.outer(np.sin(u), np.sin(v))
    z = MOON_RADIUS_KM * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x, y, z, color='gray', alpha=alpha, linewidth=0.3)


# Visualize Graph
def visualize_3d_network(G, landing, sats, metrics, save_fig=True):
    """Create 3D + 2D visualizations and metric plots"""
    communities = metrics['communities']

    fig = plt.figure(figsize=(20, 15))

    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    plot_3d_moon_sphere(ax1, alpha=0.08)

    # 3D positions
    pos_3d = {}
    for node in G.nodes():
        node_data = G.nodes[node]
        if node_data['type'] == 'surface':
            x, y, z = lat_lon_to_3d(node_data['lat'], node_data['lon'])
        else:
            radius = MOON_RADIUS_KM + node_data['altitude']
            x, y, z = lat_lon_to_3d(node_data['lat'], node_data['lon'], radius)
        pos_3d[node] = (x, y, z)

    edge_colors = {
        'surface_link': '#00bfff',
        'comm_link': '#ff6b6b',
        'inter_sat_link': '#4ecdc4'
    }

    # draw edges
    for u, v, data in G.edges(data=True):
        x_vals = [pos_3d[u][0], pos_3d[v][0]]
        y_vals = [pos_3d[u][1], pos_3d[v][1]]
        z_vals = [pos_3d[u][2], pos_3d[v][2]]

        etype = data.get('edge_type', 'surface_link')
        color = edge_colors.get(etype, 'gray')
        alpha = 0.6 if etype == 'comm_link' else 0.4
        lw = 0.8 if etype == 'inter_sat_link' else 0.5

        ax1.plot(x_vals, y_vals, z_vals,
                 color=color, alpha=alpha, linewidth=lw)

    # draw nodes
    for node in G.nodes():
        node_data = G.nodes[node]
        x, y, z = pos_3d[node]

        if node_data['type'] == 'surface':
            color = plt.cm.tab10(communities[node] % 10)
            size = 80 + 200 * metrics['degree_centrality'][node]
            marker = 'o'
        else:
            color = 'gold'
            size = 150 + 250 * metrics['degree_centrality'][node]
            marker = '^'

        ax1.scatter(x, y, z,
                    c=[color],
                    s=size,
                    marker=marker,
                    edgecolors='black',
                    linewidths=0.5,
                    alpha=0.9)
        ax1.text(x, y, z, node, fontsize=6)

    ax1.set_xlabel('X (km)', fontsize=10)
    ax1.set_ylabel('Y (km)', fontsize=10)
    ax1.set_zlabel('Z (km)', fontsize=10)
    ax1.set_title('3D Lunar Communication Network\n(Node size = Degree Centrality)',
                  fontsize=13, fontweight='bold', pad=20)

    # legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#00bfff', linewidth=2, label='Surface Links'),
        Line2D([0], [0], color='#ff6b6b', linewidth=2, label='Comm Links'),
        Line2D([0], [0], color='#4ecdc4', linewidth=2, label='Inter-Sat Links'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='skyblue',
               markersize=10, label='Surface Sites'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gold',
               markersize=10, label='Satellites')
    ]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=8)

    # === 2D PROJECTION ===
    ax2 = plt.subplot(2, 3, 2)
    pos_2d = {n: (G.nodes[n]["lon"], G.nodes[n]["lat"]) for n in G.nodes()}

    for edge_type, color in edge_colors.items():
        edges_of_type = [(u, v) for u, v, d in G.edges(data=True)
                         if d.get('edge_type') == edge_type]
        nx.draw_networkx_edges(
            G, pos_2d,
            edgelist=edges_of_type,
            alpha=0.4,
            width=1.2,
            edge_color=color,
            ax=ax2
        )

    surface_nodes = [n for n in G.nodes() if G.nodes[n]['type'] == 'surface']
    sat_nodes = [n for n in G.nodes() if G.nodes[n]['type'] == 'satellite']

    surface_colors = [communities[n] for n in surface_nodes]
    surface_sizes = [100 + 400 * metrics['degree_centrality'][n] for n in surface_nodes]

    nx.draw_networkx_nodes(
        G, pos_2d,
        nodelist=surface_nodes,
        node_color=surface_colors,
        node_size=surface_sizes,
        cmap='tab10',
        alpha=0.8,
        ax=ax2
    )

    sat_sizes = [200 + 500 * metrics['degree_centrality'][n] for n in sat_nodes]
    nx.draw_networkx_nodes(
        G, pos_2d,
        nodelist=sat_nodes,
        node_color='gold',
        node_size=sat_sizes,
        node_shape='^',
        alpha=0.9,
        edgecolors='black',
        ax=ax2
    )

    all_labels = {n: n for n in G.nodes()}
    nx.draw_networkx_labels(
        G, pos_2d,
        labels=all_labels,
        font_size=7,
        font_weight='bold',
        ax=ax2
    )

    ax2.set_xlabel('Longitude (°)', fontsize=10)
    ax2.set_ylabel('Latitude (°)', fontsize=10)
    ax2.set_title('2D Network Projection (Lat/Lon)',
                  fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    # === CENTRALITY COMPARISON ===
    ax3 = plt.subplot(2, 3, 3)

    top_nodes = sorted(
        metrics['degree_centrality'].items(),
        key=lambda x: x[1],
        reverse=True
    )[:12]
    names = [n for n, _ in top_nodes]

    deg_vals = [metrics['degree_centrality'][n] for n in names]
    bet_vals = [metrics['betweenness_centrality'][n] for n in names]
    eig_vals = [metrics['eigenvector_centrality'][n] for n in names]

    x = np.arange(len(names))
    width = 0.25

    ax3.barh(x - width, deg_vals, width,
             label='Degree', alpha=0.8, color='skyblue')
    ax3.barh(x, bet_vals, width,
             label='Betweenness', alpha=0.8, color='coral')
    ax3.barh(x + width, eig_vals, width,
             label='Eigenvector', alpha=0.8, color='lightgreen')

    ax3.set_yticks(x)
    ax3.set_yticklabels(names, fontsize=8)
    ax3.set_xlabel('Centrality Value', fontsize=10)
    ax3.set_title('Top Nodes: Centrality Comparison',
                  fontsize=12, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.invert_yaxis()

    # === COMMUNITY STRUCTURE ===
    ax4 = plt.subplot(2, 3, 4)
    community_counts = {}
    for node, comm in communities.items():
        community_counts[comm] = community_counts.get(comm, 0) + 1

    colors_comm = plt.cm.tab10(np.linspace(0, 1, len(community_counts)))
    ax4.bar(community_counts.keys(), community_counts.values(),
            color=colors_comm, alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Community ID', fontsize=10)
    ax4.set_ylabel('Number of Nodes', fontsize=10)
    ax4.set_title(f'Community Distribution\n(Modularity: {metrics["modularity"]:.4f})',
                  fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)

    # === DEGREE DISTRIBUTION ===
    ax5 = plt.subplot(2, 3, 5)
    degrees = [G.degree(n) for n in G.nodes()]
    ax5.hist(degrees, bins=10,
             color='steelblue', alpha=0.7, edgecolor='black')
    ax5.axvline(np.mean(degrees),
                color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(degrees):.1f}')
    ax5.set_xlabel('Node Degree', fontsize=10)
    ax5.set_ylabel('Frequency', fontsize=10)
    ax5.set_title('Degree Distribution',
                  fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)

    # === STATS PANEL ===
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    stats_text = f"""
    LUNAR NETWORK ANALYSIS REPORT
    {'='*45}

    TOPOLOGY
    • Total Nodes:              {G.number_of_nodes()}
    • Surface Sites:            {len([n for n in G.nodes() if G.nodes[n]['type']=='surface'])}
    • Satellites:               {len([n for n in G.nodes() if G.nodes[n]['type']=='satellite'])}
    • Total Edges:              {G.number_of_edges()}

    CONNECTIVITY
    • Network Density:          {nx.density(G):.4f}
    • Avg Clustering Coeff:     {metrics['avg_clustering']:.4f}
    • Avg Path Length:          {metrics['avg_path_length']:.2f}
    • Network Diameter:         {metrics['diameter']:.2f}
    • Is Connected:             {nx.is_connected(G)}

    COMPLEXITY & STRUCTURE
    • Von Neumann Entropy:      {metrics['von_neumann_entropy']:.4f}
    • Modularity:               {metrics['modularity']:.4f}
    • Number of Communities:    {len(set(communities.values()))}

    INTERPRETATION
    {('High entropy indicates diverse, complex' if metrics['von_neumann_entropy'] > 3 else 'Moderate complexity in') + ' connectivity patterns.'}
    {('Strong modularity suggests distinct' if metrics['modularity'] > 0.3 else 'Weak modularity indicates integrated') + ' operational clusters.'}
    """

    ax6.text(
        0.05, 0.95,
        stats_text,
        fontsize=9,
        family='monospace',
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3)
    )

    plt.tight_layout()

    if save_fig:
        plt.savefig('enhanced_lunar_network_3d.png',
                    dpi=300,
                    bbox_inches='tight')
        print("\n✓ Visualization saved as 'enhanced_lunar_network_3d.png'")

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
