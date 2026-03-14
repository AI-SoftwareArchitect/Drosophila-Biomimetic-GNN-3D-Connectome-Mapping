"""
Directed Biomimetic GNN Classifier
----------------------------------
Role      : Senior Software Architect / Deep Learning Expert
Task      : Map a biological topology (Drosophila Antennal Lobe) into a PyTorch Geometric Graph 
            using ACTUAL EVOLUTIONARY SYNAPTIC CONNECTIONS (Directed Graph with Edge Weights).
Architecture:
  1. Graph Construction : Directed Edge list and weights extracted directly from the true connectome.
  2. Linear Encoder     : Feature expanding from tabular domain to biological network dimensional space.
  3. GNN Propagation    : Graph Neural Network layers passing signals through the TRUE biological topology.
  4. Classifier Head    : Global pooling & Feed Forward outputs.
"""

import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

class DataProcessor:
    """Handles parsing and preprocessing of biological data and classification datasets."""
    
    @staticmethod
    def parse_location(loc_str):
        try:
            return np.array(ast.literal_eval(loc_str), dtype=np.float32)
        except Exception:
            return None

    @staticmethod
    def build_true_connectome_graph(nodes_csv="al_neurons_metadata.csv", edges_csv="al_connectome_small.csv"):
        """
        Builds a biological graph using true evolutionary synaptic connections and weights.
        Returns edge_index and edge_weight in PyTorch Geometric format.
        """
        print("[INFO] Gerçek biyolojik ağ (Connectome) inşa ediliyor...")
        try:
            nodes_df = pd.read_csv(nodes_csv)
            edges_df = pd.read_csv(edges_csv)
        except Exception as e:
            raise RuntimeError(f"CSV dosyaları okunamadı: {e}")
            
        nodes_df['pos'] = nodes_df['somaLocation'].apply(DataProcessor.parse_location)
        nodes_df = nodes_df.dropna(subset=['pos']).reset_index(drop=True)
        num_nodes = len(nodes_df)
        
        if nodes_df.empty:
            raise ValueError("Geçerli koordinat verisi bulunamadı.")
            
        # Nöron 'bodyId' değerlerini 0'dan başlayan indekslere haritalandır (Map)
        idx_map = {body_id: i for i, body_id in enumerate(nodes_df['bodyId'])}
        
        # Sadece haritamızdaki (pos verisi eksik olmayan) nöronlar arasındaki bağlantıları filtrele
        edges_df['pre_idx'] = edges_df['bodyId_pre'].map(idx_map)
        edges_df['post_idx'] = edges_df['bodyId_post'].map(idx_map)
        
        edges_df = edges_df.dropna(subset=['pre_idx', 'post_idx'])
        edges_df['pre_idx'] = edges_df['pre_idx'].astype(int)
        edges_df['post_idx'] = edges_df['post_idx'].astype(int)
        
        # Farklı ROI bölgeleri için olan ağırlıkları pre-post çifti bazında topla
        grouped_edges = edges_df.groupby(['pre_idx', 'post_idx'])['weight'].sum().reset_index()
        
        # PyTorch Geometric formatı: [2, num_edges] tipi edge_index
        source_nodes = grouped_edges['pre_idx'].values
        target_nodes = grouped_edges['post_idx'].values
        weights = grouped_edges['weight'].values
        
        edge_index = torch.tensor(np.vstack((source_nodes, target_nodes)), dtype=torch.long)
        edge_weight = torch.tensor(weights, dtype=torch.float32)
        
        # Ağırlıkları normalize edelim (Büyük sinaps ağırlıklarının patlamasını engellemek için)
        if edge_weight.max() > 0:
            edge_weight = edge_weight / edge_weight.max()
            
        print(f"[INFO] Gerçek Evrimsel Graf kuruldu! Nöron Sayısı: {num_nodes}, Bağlantı (Edge) Sayısı: {edge_index.shape[1]}")
        return edge_index, edge_weight, num_nodes

    @staticmethod
    def get_kaggle_dataset():
        print("[INFO] Sınıflandırma veriseti (Breast Cancer) yükleniyor...")
        data = load_breast_cancer()
        X = data.data
        y = data.target
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        num_features = X.shape[1]
        num_classes = len(np.unique(y))
        
        return train_loader, test_loader, num_features, num_classes


class TrueBiomimeticGNN(nn.Module):
    """Hibrit GNN Modeli: Gerçek ağırlıklı biyolojik yapı üzerinden geçerek sınıflandırma yapar."""
    def __init__(self, num_features, num_nodes_bio, num_classes, edge_index, edge_weight):
        super(TrueBiomimeticGNN, self).__init__()
        self.num_nodes_bio = num_nodes_bio
        
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_weight', edge_weight)
        
        self.encoder = nn.Linear(num_features, num_nodes_bio)
        
        self.conv1 = GCNConv(1, 16)
        self.conv2 = GCNConv(16, 32)
        
        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, num_classes)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        node_signals = self.encoder(x)
        h = node_signals.view(-1, 1)
        
        # Dinamik Edge offset hesaplama (Batched Graphs)
        offset = torch.arange(0, batch_size * self.num_nodes_bio, self.num_nodes_bio, device=x.device).view(1, -1)
        batched_edge_index = self.edge_index.unsqueeze(2) + offset.unsqueeze(0)
        batched_edge_index = batched_edge_index.permute(0, 2, 1).reshape(2, -1)
        
        # Edge weight'leri batch boyutunda kopyalama
        batched_edge_weight = self.edge_weight.unsqueeze(1).repeat(1, batch_size).reshape(-1)
        
        # 2) GNN Propagation (Kanser sinyalini gerçek beyin yollarında ve sinaps güçlerinde aktarır)
        h = F.relu(self.conv1(h, batched_edge_index, edge_weight=batched_edge_weight))
        h = F.relu(self.conv2(h, batched_edge_index, edge_weight=batched_edge_weight))
        
        # 3) Global Mean Pooling
        h = h.view(batch_size, self.num_nodes_bio, -1)
        pooled_out = h.mean(dim=1)
        
        # 4) Sınıflandırma
        out = self.classifier(pooled_out)
        return out


def train_model(model, train_loader, test_loader, epochs=25, lr=0.005, device='cpu'):
    print(f"\n[INFO] Model eğitimi başlıyor... (Cihaz: {device})")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct_train, total_train = 0, 0, 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * X_batch.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct_train += (preds == y_batch).sum().item()
            total_train += y_batch.size(0)
            
        train_acc = correct_train / total_train
        train_loss = total_loss / total_train
        
        model.eval()
        correct_test, total_test, val_loss = 0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                val_loss += loss.item() * X_batch.size(0)
                preds = torch.argmax(outputs, dim=1)
                correct_test += (preds == y_batch).sum().item()
                total_test += y_batch.size(0)
                
        test_acc = correct_test / total_test
        val_loss = val_loss / total_test
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} - Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} - Acc: {test_acc:.4f}")
            
    print("\n[INFO] Eğitim tamamlandı. GERÇEK Biyomimetik Connectome GNN başarıyla test edildi.")
    return model

def evaluate_and_predict(model, test_loader, device='cpu'):
    """Modelin doğruluğunu (Precision, Recall, F1) ve gerçek örneklerini analiz eder."""
    from sklearn.metrics import classification_report, confusion_matrix
    import random
    
    print("\n[INFO] Detaylı Test ve Doğrulama Yapılıyor...")
    model.eval()
    all_preds = []
    all_labels = []
    
    # Tüm test verisini modelden geçir
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            
    # 1. Classification Report (F1, Precision, Recall)
    target_names = ['Malignant (Kötü Huylu - 0)', 'Benign (İyi Huylu - 1)']
    print("\n" + "="*40)
    print("      Sınıflandırma Raporu (Test Verisi)")
    print("="*40)
    print(classification_report(all_labels, all_preds, target_names=target_names))
    
    # 2. Örnek Birkaç Gerçek Tahmin Gösterimi
    print("\n[INFO] Rastgele 5 Hastanın Gerçek vs Tahmin Edilen Durumu:")
    sample_indices = random.sample(range(len(all_labels)), 5)
    
    for i in sample_indices:
        gercek_etiket = target_names[all_labels[i]]
        tahmin = target_names[all_preds[i]]
        is_correct = "✅ DOĞRU" if all_labels[i] == all_preds[i] else "❌ YANLIŞ"
        print(f"  - Biyolojik GNN Tahmini: {tahmin: <28} | Gerçek Değer: {gercek_etiket: <28} | Sonuç: {is_correct}")
        

def main():
    try:
        # 1. GERÇEK Biyolojik İskeleti ve Ağırlıklı Sinapsları İnşa Et
        edge_index, edge_weight, num_nodes_bio = DataProcessor.build_true_connectome_graph(
            "al_neurons_metadata.csv", "al_connectome_small.csv"
        )
        
        # 2. Kaggle Model Verilerini Yükle
        train_loader, test_loader, num_features, num_classes = DataProcessor.get_kaggle_dataset()
        
        # 3. Model Mimarisini Başlat
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = TrueBiomimeticGNN(
            num_features=num_features,
            num_nodes_bio=num_nodes_bio, 
            num_classes=num_classes,
            edge_index=edge_index,
            edge_weight=edge_weight
        )
        
        # 4. Modeli Eğit ve Raporla
        trained_model = train_model(model, train_loader, test_loader, epochs=25, lr=0.005, device=device)
        
        # 5. Modelin Gerçekten İşe Yaradığını Test Et (Doğrulama)
        evaluate_and_predict(trained_model, test_loader, device=device)
        
    except Exception as e:
        print(f"\n[FATAL ERROR] Sistemde kritik bir hata oluştu:\n{e}")

if __name__ == "__main__":
    main()
