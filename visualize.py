import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# 1. Verileri Yükle
print("Veriler yükleniyor...")
nodes_df = pd.read_csv("al_neurons_metadata.csv")
edges_df = pd.read_csv("al_connectome_small.csv")

# 2. Graph Nesnesini Oluştur (Directed Graph)
G = nx.DiGraph()

# Sadece en güçlü bağlantıları gösterelim (Gürültüyü azaltmak için)
# Ağırlığı (synapse count) 5'ten büyük olanları filtreleyelim
min_weight = 5
filtered_edges = edges_df[edges_df['weight'] >= min_weight]

print(f"Toplam {len(filtered_edges)} güçlü bağlantı işleniyor...")

for _, row in filtered_edges.iterrows():
    G.add_edge(row['bodyId_pre'], row['bodyId_post'], weight=row['weight'])

# 3. Analiz: En "Merkezi" Nöronları Bul (Degree Centrality)
# Bu, hangi nöronun trafik merkezi olduğunu belirler
centrality = nx.degree_centrality(G)
node_sizes = [v * 5000 for v in centrality.values()]

# 4. Görselleştirme Ayarları
plt.figure(figsize=(15, 10))
pos = nx.spring_layout(G, k=0.15, iterations=50) # Fizik tabanlı yerleşim

# Çizgileri (Edges) çiz
nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='gray', arrows=True)

# Düğümleri (Nodes) çiz
nodes = nx.draw_networkx_nodes(G, pos, 
                               node_size=node_sizes, 
                               node_color=list(centrality.values()), 
                               cmap=plt.cm.plasma,
                               alpha=0.8)

plt.colorbar(nodes, label='Bağlantı Yoğunluğu (Centrality)')
plt.title(f"Antennal Lobe Sinir Ağı (Min Synapse: {min_weight})", fontsize=15)
plt.axis('off')

# Kaydet ve Göster
print("Görselleştirme oluşturuluyor...")
plt.savefig("neuron_graph.png", dpi=300, bbox_inches='tight')
plt.show()