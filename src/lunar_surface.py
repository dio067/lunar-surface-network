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

# Constants
MOON_RADIUS_KM = 1737.4

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
    LUNAR NETWORK ANALYSIS DISCRIPTION
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


# Detailed Discription
def print_detailed_discription(G, landing, sats, metrics):
    print("\n" + "="*80)
    print("LUNAR COMMUNICATION NETWORK ANALYSIS - COMPREHENSIVE DISCRIPTION(PRUNED)")
    print("Based on REAL Lunar Craters + Apollo/Luna Landing Sites + Orbital Assets")
    print("="*80)

    print("\n1. NETWORK TOPOLOGY")
    print("-" * 80)
    print(f"   Total Nodes: {G.number_of_nodes()}")
    print(f"   • Surface Landing/Craters: {len(landing)}")
    print(f"   • Orbital Assets: {len(sats)}")
    print(f"   Total Edges: {G.number_of_edges()}")
    print(f"   Network Density: {nx.density(G):.4f}")

    deg_cent = sorted(metrics['degree_centrality'].items(), key=lambda x: x[1], reverse=True)[:8]
    bet_cent = sorted(metrics['betweenness_centrality'].items(), key=lambda x: x[1], reverse=True)[:8]
    eig_cent = sorted(metrics['eigenvector_centrality'].items(), key=lambda x: x[1], reverse=True)[:8]

    print("\n2. DEGREE CENTRALITY (Top 8 - Most Connected Nodes)")
    print("-" * 80)
    for i, (node, cent) in enumerate(deg_cent, 1):
        node_type = G.nodes[node]['type']
        mission = G.nodes[node].get('mission_type', 'N/A')
        print(f"   {i}. {node:30s} ({node_type:9s} | {mission:10s}): {cent:.4f}")

    print("\n3. BETWEENNESS CENTRALITY (Top 8 - Critical Bridge Nodes)")
    print("-" * 80)
    for i, (node, cent) in enumerate(bet_cent, 1):
        node_type = G.nodes[node]['type']
        mission = G.nodes[node].get('mission_type', 'N/A')
        print(f"   {i}. {node:30s} ({node_type:9s} | {mission:10s}): {cent:.4f}")

    print("\n4. EIGENVECTOR CENTRALITY (Top 8 - Influence/Cluster Hubs)")
    print("-" * 80)
    for i, (node, cent) in enumerate(eig_cent, 1):
        node_type = G.nodes[node]['type']
        mission = G.nodes[node].get('mission_type', 'N/A')
        print(f"   {i}. {node:30s} ({node_type:9s} | {mission:10s}): {cent:.4f}")

    print("\n5. COMMUNITY DETECTION & MODULARITY")
    print("-" * 80)
    print(f"   Modularity Score: {metrics['modularity']:.4f}")
    num_communities = len(set(metrics['communities'].values()))
    print(f"   Number of Communities: {num_communities}")
    print("\n   Community Membership:")

    comm_groups = {}
    for node, comm_id in metrics['communities'].items():
        comm_groups.setdefault(comm_id, []).append(node)

    for comm_id, nodes in sorted(comm_groups.items()):
        print(f"\n   Community {comm_id} ({len(nodes)} nodes):")
        for node in nodes[:5]:
            node_type = G.nodes[node]['type']
            print(f"      • {node} ({node_type})")
        if len(nodes) > 5:
            print(f"      ... and {len(nodes)-5} more")

    print("\n6. ENTROPY & COMPLEXITY")
    print("-" * 80)
    print(f"   Von Neumann Entropy: {metrics['von_neumann_entropy']:.4f}")
    print("   Interpretation: ", end="")
    if metrics['von_neumann_entropy'] > 3.5:
        print("High complexity - diverse connectivity patterns")
    elif metrics['von_neumann_entropy'] > 2.5:
        print("Moderate complexity - balanced network structure")
    else:
        print("Low complexity - more regular structure")

    print("\n7. NETWORK EFFICIENCY")
    print("-" * 80)
    print(f"   Average Clustering Coefficient: {metrics['avg_clustering']:.4f}")
    print(f"   Average Path Length: {metrics['avg_path_length']:.2f}")
    print(f"   Network Diameter: {metrics['diameter']:.2f}")
    print(f"   Connected Components: {nx.number_connected_components(G)}")

    print("\n8. ROBUSTNESS ANALYSIS")
    print("-" * 80)
    critical_nodes = [n for n, c in metrics['betweenness_centrality'].items() if c > 0.1]
    print(f"   Critical Nodes (betweenness > 0.1): {len(critical_nodes)}")
    print(f"   Top Critical: {', '.join([n for n, _ in bet_cent[:3]])}")

    if nx.is_connected(G):
        edge_conn = nx.edge_connectivity(G)
        node_conn = nx.node_connectivity(G)
        print(f"   Edge Connectivity: {edge_conn}")
        print(f"   Node Connectivity: {node_conn}")
    else:
        print(f"   Network is disconnected - {nx.number_connected_components(G)} components")

    print("\n9. MISSION-BASED ANALYSIS")
    print("-" * 80)
    mission_counts = {}
    for node in G.nodes():
        mission = G.nodes[node].get('mission_type', 'Unknown')
        mission_counts[mission] = mission_counts.get(mission, 0) + 1

    print("   Nodes by Mission Program:")
    for mission, count in sorted(mission_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"      {mission:15s}: {count:2d} nodes")

    print("\n10. EDGE TYPE DISTRIBUTION")
    print("-" * 80)
    edge_types = {}
    total_weight = 0
    for u, v, data in G.edges(data=True):
        etype = data.get('edge_type', 'unknown')
        edge_types[etype] = edge_types.get(etype, 0) + 1
        total_weight += data.get('weight', 0)

    print(f"   Edge Types:")
    for etype, count in edge_types.items():
        print(f"      {etype:20s}: {count:3d} edges")
    if G.number_of_edges() > 0:
        print(f"   Total Network Weight (distance): {total_weight:.2f} km")
        print(f"   Average Edge Weight: {total_weight/G.number_of_edges():.2f} km")

    print("\n" + "="*80)
    print("END OF DISCRIPTION")
    print("="*80 + "\n")


# Lunar Data Craters + Landing Sites + Satellites
def generate_enhanced_nasa_data():
    """
    Generate REAL lunar network data using:
    - Famous lunar craters from LRO / PDS
    - Real Apollo & Luna landing sites
    - A small set of real lunar orbiters
    Kept small on purpose to reduce nodes & edges and improve visualization.
    """

    # ==========================
    # REAL LUNAR CRATERS (LRO/PDS)
    # ==========================
    landing_craters = pd.DataFrame({
        'name': [
            'Tycho Crater',
            'Copernicus Crater',
            'Aristarchus Crater',
            'Plato Crater',
            'Kepler Crater',
            'Clavius Crater',
            'Langrenus Crater',
            'Humboldt Crater',
            'Gassendi Crater'
        ],
        'lat': [
            -43.3,   # Tycho
            9.62,    # Copernicus
            23.7,    # Aristarchus
            51.6,    # Plato
            8.1,     # Kepler
            -58.8,   # Clavius
            -8.9,    # Langrenus
            -27.0,   # Humboldt
            -17.5    # Gassendi
        ],
        'lon': [
            -11.2,   # Tycho
            -20.08,  # Copernicus
            -47.4,   # Aristarchus
            -9.3,    # Plato
            -38.0,   # Kepler
            -14.0,   # Clavius
            61.0,    # Langrenus
            80.9,    # Humboldt
            -39.2    # Gassendi
        ],
        'type': ['surface'] * 9,
        'mission_type': ['Crater'] * 9
    })

    # ==========================
    # REAL LANDING SITES (Apollo + Luna)
    # ==========================
    landing_sites_real = pd.DataFrame({
        'name': [
            'Apollo 11 Landing Site',
            'Apollo 12 Landing Site',
            'Apollo 14 Landing Site',
            'Apollo 15 Landing Site',
            'Luna 9 Landing Site',
            'Luna 16 Landing Site'
        ],
        'lat': [
            0.674,   # Apollo 11
            -3.012,  # Apollo 12
            -3.645,  # Apollo 14
            26.132,  # Apollo 15
            7.08,    # Luna 9
            -0.68    # Luna 16
        ],
        'lon': [
            23.473,   # Apollo 11 (E)
            -23.421,  # Apollo 12 (W)
            -17.471,  # Apollo 14 (W)
            3.633,    # Apollo 15 (E)
            -64.37,   # Luna 9 (W)
            56.3      # Luna 16 (E)
        ],
        'type': ['surface'] * 6,
        'mission_type': [
            'Apollo', 'Apollo', 'Apollo', 'Apollo',
            'Luna', 'Luna'
        ]
    })

    # MERGE craters + landing sites ⇒ all surface nodes
    landing_sites = pd.concat([landing_craters, landing_sites_real],
                              ignore_index=True)

    # ==========================
    # SMALL REAL SATELLITE SET
    # ==========================
    satellites = pd.DataFrame({
        'name': [
            'LRO',                     # Lunar Reconnaissance Orbiter (NASA)
            'Chandrayaan-2 Orbiter',   # ISRO
            'KPLO (Danuri)'            # KARI
        ],
        'lat': [
            0.0,
            -25.0,
            10.0
        ],
        'lon': [
            0.0,
            45.0,
            -90.0
        ],
        'altitude_km': [
            50,     # low polar
            100,    # rough mean
            100
        ],
        'coverage_radius_km': [
            800,
            900,
            900
        ],
        'type': ['satellite'] * 3,
        'mission_type': [
            'NASA',
            'ISRO',
            'KARI'
        ]
    })

    return landing_sites, satellites


# Main Excution
def main():
    def main():
    print("\n" + "="*80)
    print("LUNAR COMMUNICATION NETWORK ANALYSIS")
    print("Real Dataset: Famous Craters + Apollo/Luna Sites + 3 Orbiters")
    print("="*80 + "\n")

    # Generate REAL data
    print("Generating REAL lunar data (craters + landing sites + orbiters)...")
    landing_sites_orig, satellites_orig = generate_enhanced_nasa_data()
    print(f"Surface sites (craters + landings): {len(landing_sites_orig)}")
    print(f"Satellites: {len(satellites_orig)}")

    # Build and prune graph
    print("\nBuilding and pruning lunar network graph...")
    G, edges_df, landing_sites, satellites = create_graph(
        landing_sites_orig,
        satellites_orig,
        surface_threshold_km=900,   # smaller threshold for fewer edges
        max_isl_range=2500          # moderate sat-to-sat links
    )

    # Save original node data to CSV
    save_to_csv(landing_sites_orig, satellites_orig)

    # Compute metrics
    metrics = compute_metrics(G)
    print("✓ Metrics computed")

    # Print detailed terminal discription
    print_detailed_discription(G, landing_sites, satellites, metrics)

    # Visualize
    print("\nCreating visualizations...")
    visualize_3d_network(G, landing_sites, satellites, metrics, save_fig=True)

    # Export centrality data
    centrality_df = pd.DataFrame({
        'node': list(G.nodes()),
        'type': [G.nodes[n]['type'] for n in G.nodes()],
        'mission': [G.nodes[n].get('mission_type', 'N/A') for n in G.nodes()],
        'degree_centrality': [metrics['degree_centrality'][n] for n in G.nodes()],
        'betweenness_centrality': [metrics['betweenness_centrality'][n] for n in G.nodes()],
        'closeness_centrality': [metrics['closeness_centrality'][n] for n in G.nodes()],
        'eigenvector_centrality': [metrics['eigenvector_centrality'][n] for n in G.nodes()],
        'community': [metrics['communities'][n] for n in G.nodes()]
    })

    centrality_df = centrality_df.sort_values('degree_centrality',
                                              ascending=False)
    centrality_df.to_csv('lunar_node_metrics_pruned.csv', index=False)
    print("Node metrics saved to: lunar_node_metrics_pruned.csv")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated Files:")
    print(f"  Total Nodes in Final Graph: {G.number_of_nodes()}")
    print("  1. lunar_landing_sites.csv        - Real surface locations (craters + landings)")
    print("  2. lunar_satellites.csv           - Real orbital assets")
    print("  3. lunar_network_edges_pruned.csv - Edge list (after thresholds)")
    print("  4. lunar_node_metrics_pruned.csv  - Centrality + community metrics")
    print("  5. enhanced_lunar_network_3d.png  - Full visualization figure")

    print("\nTOP 10 NODES BY DEGREE CENTRALITY (PRUNED):")
    print("-" * 80)
    top_10 = centrality_df.head(10)
    for idx, row in top_10.iterrows():
        print(f"  {row['node']:30s} ({row['type']:9s} | {row['mission']:10s}) "
              f"Degree: {row['degree_centrality']:.4f}")



if __name__ == "__main__":
    main()
