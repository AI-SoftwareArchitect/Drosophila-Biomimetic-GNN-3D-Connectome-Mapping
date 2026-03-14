"""
GNN 3D Visualization Map (Connectome Topology)
----------------------------------------------
Role      : Senior Software Architect
Task      : Visualize the EXACT directed biological graph (the 262 nodes and their true synaptic connections) 
            that is used by the Directed Biomimetic GNN classifier.

This script uses PyVista to render the biological hardware our AI uses.
"""

import ast
import numpy as np
import pandas as pd
import pyvista as pv

def parse_location(loc_str):
    try:
        return np.array(ast.literal_eval(loc_str), dtype=np.float32)
    except Exception:
        return None

def main():
    print("[INFO] Biyomimetik GNN'in kullandığı gerçek ağ verileri yükleniyor...")
    
    # 1. Nöronları Yükle
    nodes_df = pd.read_csv("al_neurons_metadata.csv")
    nodes_df['pos'] = nodes_df['somaLocation'].apply(parse_location)
    
    # GNN modelindeki gibi sadece koordinatı olanları filtrele (262 Nöron)
    nodes_df = nodes_df.dropna(subset=['pos']).reset_index(drop=True)
    num_nodes = len(nodes_df)
    
    # 2. Bağlantıları (Edges) Yükle
    edges_df = pd.read_csv("al_connectome_small.csv")
    
    # GNN modelindeki haritalama mantığı (0'dan başlayan indekslere)
    idx_map = {body_id: i for i, body_id in enumerate(nodes_df['bodyId'])}
    
    edges_df['pre_idx'] = edges_df['bodyId_pre'].map(idx_map)
    edges_df['post_idx'] = edges_df['bodyId_post'].map(idx_map)
    
    # Sadece 262 nöron arasındaki bağlantıları (edges) sakla
    edges_df = edges_df.dropna(subset=['pre_idx', 'post_idx'])
    edges_df['pre_idx'] = edges_df['pre_idx'].astype(int)
    edges_df['post_idx'] = edges_df['post_idx'].astype(int)
    
    # Tekrarlayan bağlantıları topla (GNN'in kullandığı gerçek edge yapısı)
    grouped_edges = edges_df.groupby(['pre_idx', 'post_idx'])['weight'].sum().reset_index()
    
    print(f"[INFO] GNN İskeleti Hazır - Nöron: {num_nodes}, Bağlantı: {len(grouped_edges)}")
    
    # 3. PyVista 3D Ortamını Başlat
    plotter = pv.Plotter(title="Directed Biomimetic GNN - 3D Connectome Map")
    plotter.set_background("black")
    
    # Koordinatları Al
    coords = np.stack(nodes_df['pos'].values)
    cloud = pv.PolyData(coords)
    
    # Nöron büyüklüklerini ayarla
    cloud["neuron_size"] = nodes_df['size'].fillna(100).values
    
    # Nöronları (Düğümleri) Çiz
    nodes_mesh = cloud.glyph(scale=False, orient=False, factor=50, geom=pv.Sphere(theta_resolution=10, phi_resolution=10))
    plotter.add_mesh(
        nodes_mesh, 
        cmap="plasma",   
        ambient=0.5, 
        specular=1.0, 
        opacity=0.9,
        scalar_bar_args={'title': 'Neuron Size (Bio-Mass)'}
    )
    
    # Bağlantıları (Çizgileri) Oluştur
    lines = []
    weights = []
    
    for _, row in grouped_edges.iterrows():
        lines.append(2)  # 2 nokta (pre ve post)
        lines.append(row['pre_idx'])
        lines.append(row['post_idx'])
        weights.append(row['weight'])
        
    if lines:
        lines_array = np.array(lines)
        lines_poly = pv.PolyData()
        lines_poly.points = coords
        lines_poly.lines = lines_array
        lines_poly.cell_data["synapse_weight"] = weights

        # Yönlü grafın ağırlıklarını (sinaps güçlerini) renderlıyoruz
        plotter.add_mesh(
            lines_poly,
            scalars="synapse_weight",
            cmap="viridis",
            opacity=0.15,  # Binlerce çizgi olacağı için şeffaflık ekliyoruz
            render_lines_as_tubes=True,
            line_width=1,
            scalar_bar_args={'title': 'Synaptic Strength (Edge Weight)'}
        )

    plotter.show_grid(color='#333333')
    print("[INFO] 3D Ekran Açılıyor...")
    
    plotter.camera_position = 'iso'
    plotter.show()

if __name__ == "__main__":
    main()
