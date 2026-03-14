import pandas as pd
import pyvista as pv
import numpy as np
import ast

# 1. Verileri Yükle
nodes_df = pd.read_csv("al_neurons_metadata.csv")
edges_df = pd.read_csv("al_connectome_small.csv")

# 2. Koordinatları Temizle
def parse_location(loc_str):
    try:
        # String "[x, y, z]" formatını float array'e çevir
        return np.array(ast.literal_eval(loc_str), dtype=np.float32)
    except:
        return None

nodes_df['pos'] = nodes_df['somaLocation'].apply(parse_location)
nodes_df = nodes_df.dropna(subset=['pos']).reset_index(drop=True)

# Edge'leri topla (aynı pre-post arasındaki farklı ROI ağırlıklarını birleştir)
edges_df = edges_df.groupby(['bodyId_pre', 'bodyId_post'])['weight'].sum().reset_index()

# Sadece güçlü bağlantıları alalım (görsel karmaşayı azaltmak için)
min_weight = 10
edges_df = edges_df[edges_df['weight'] >= min_weight]

# 3. Görselleştirme Ayarları
plotter = pv.Plotter(title="Drosophila AL Connectome 3D Map")
plotter.set_background("black")

# Koordinatları stackle
coords = np.stack(nodes_df['pos'].values)
cloud = pv.PolyData(coords)

# Nöron büyüklüklerini (size) veri setine ekle. NaN değerleri 100 ile doldur
cloud["neuron_size"] = nodes_df['size'].fillna(100).values

# Düğümleri (Nöronları) Çiz
# scale=False yaptık ve factor'ü 50 gibi küçük sabit bir değere çektik.
nodes_mesh = cloud.glyph(scale=False, orient=False, factor=50, geom=pv.Sphere(theta_resolution=10, phi_resolution=10))
plotter.add_mesh(
    nodes_mesh, 
    cmap="inferno",   
    ambient=0.6, 
    specular=1.0, 
    opacity=0.9,
    scalar_bar_args={'title': 'Neuron Size'}
)

# 4. Bağlantıları (Edges) Çiz
print(f"Toplam {len(edges_df)} adet güçlü bağlantı ({min_weight}+ sinaps) hesaplanıyor...")
idx_map = {row['bodyId']: i for i, row in nodes_df.iterrows()}

lines = []
weights = []

for _, row in edges_df.iterrows():
    idx_pre = idx_map.get(row['bodyId_pre'])
    idx_post = idx_map.get(row['bodyId_post'])
    
    # İki düğüm de koordinat listemizde varsa çizgiyi ekle
    if idx_pre is not None and idx_post is not None:
        lines.append(2)          # Çizgi 2 noktadan oluşur
        lines.append(idx_pre)
        lines.append(idx_post)
        weights.append(row['weight'])

if lines:
    lines_array = np.array(lines)
    # Çizgilerin (hücrelerin) verisi PolyData içine kaydedilir
    lines_poly = pv.PolyData()
    lines_poly.points = coords
    lines_poly.lines = lines_array
    lines_poly.cell_data["weight"] = weights

    # Bağlantıları tüp/çizgi şeklinde çiz
    plotter.add_mesh(
        lines_poly,
        scalars="weight",
        cmap="coolwarm",
        opacity=0.3,
        render_lines_as_tubes=True,
        line_width=2,
        scalar_bar_args={'title': 'Synapse Weight'}
    )

plotter.show_grid(color='#555555')

print(f"Toplam {len(nodes_df)} nöron ve {len(weights)} bağlantı 3D uzayda çiziliyor...")
plotter.camera_position = 'iso'
plotter.show()