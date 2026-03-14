"""
Biomimetic GNN Classifier
-------------------------
Role      : Senior Software Architect / Deep Learning Expert
Task      : Map a biological topology (Drosophila Antennal Lobe) into a PyTorch Geometric Graph 
            and use it to classify standard tabular data (e.g. Breast Cancer).
Architecture:
  1. Graph Construction : KNN Graph extracted from 3D Soma locations.
  2. Linear Encoder     : Feature expanding from tabular domain to biological network dimensional space.
  3. GNN Propagation    : GCN layers passing signals through the structural biological topology.
  4. Classifier Head    : Global pooling & Feed Forward outputs.

Note: Requires torch, torch-geometric, scikit-learn, pandas, numpy
"""

import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.neighbors import kneighbors_graph
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

class DataProcessor:
    """Handles parsing and preprocessing of both biological data and classification datasets."""
    
    @staticmethod
    def parse_location(loc_str):
        """Parses a string representation of 3D coordinates into a numpy array."""
        try:
            return np.array(ast.literal_eval(loc_str), dtype=np.float32)
        except Exception:
            return None

    @staticmethod
    def build_biological_graph(csv_path="al_neurons_metadata.csv", k=5):
        """Builds a KNN biological graph from soma locations."""
        print("[INFO] Biyolojik altyapı (skelet) oluşturuluyor...")
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            raise RuntimeError(f"CSV dosyası okunamadı: {e}")
            
        df['pos'] = df['somaLocation'].apply(DataProcessor.parse_location)
        df = df.dropna(subset=['pos']).reset_index(drop=True)
        
        if df.empty:
            raise ValueError("Geçerli koordinat verisi bulunamadı.")
            
        coords = np.stack(df['pos'].values)
        num_nodes = coords.shape[0]
        
        # K-Nearest Neighbors ile biyolojik bağlantı iskeleti (graf) kuruyoruz
        knn_graph = kneighbors_graph(coords, n_neighbors=k, mode='connectivity', include_self=False)
        coo = knn_graph.tocoo()
        
        # PyTorch Geometric formatı: [2, num_edges] tipi edge_index
        edge_index = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long)
        
        print(f"[INFO] Biyolojik graf kuruldu. Nöron Sayısı: {num_nodes}, Edge Sayısı: {edge_index.shape[1]}")
        return edge_index, num_nodes

    @staticmethod
    def get_kaggle_dataset():
        """Scikit-learn üzerinden Breast Cancer (Kaggle muadili) tabular dataseti yükler."""
        print("[INFO] Sınıflandırma veriseti (Breast Cancer) yükleniyor...")
        data = load_breast_cancer() # Girdi özellikleri: 30 boyutlu
        X = data.data
        y = data.target
        
        # Normalization
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train-Test Split (%80 eğitim, %20 test)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Tensor'e çevrim
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

        # DataLoader
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        num_features = X.shape[1]
        num_classes = len(np.unique(y))
        
        print(f"[INFO] Veriseti hazır. Özellik: {num_features}, Sınıf: {num_classes}")
        return train_loader, test_loader, num_features, num_classes


class BiomimeticGNN(nn.Module):
    """Hibrit GNN Modeli: Tablo verisini biyolojik iskelet üzerinde geçirerek sınıflandırır."""
    def __init__(self, num_features, num_nodes_bio, num_classes, edge_index):
        super(BiomimeticGNN, self).__init__()
        self.num_nodes_bio = num_nodes_bio
        
        # edge_index'i register ediyoruz böylece model device'a (.cuda()) alındığında o da taşınır.
        self.register_buffer('edge_index', edge_index)
        
        # 1. Linear Encoder: Girdi verisini biyofiziksel boyuta (biyolojik graftaki nöron sayısına) uyarla
        # Her girdi özelliği 1500 nöron düğümüne bir sinyal pompalar.
        self.encoder = nn.Linear(num_features, num_nodes_bio)
        
        # 2. Graph Convolutional Layers (Biyolojik Yapı Üzerinde Sinyal İşleme)
        # Biyolojik düğümlerdeki sinyal ilk başta 1 skaler uyarım olarak alınır.
        self.conv1 = GCNConv(1, 16)
        self.conv2 = GCNConv(16, 32)
        
        # 3. Classifier Head (Sınıflandırma Kafası)
        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, num_classes)
        )
        
    def forward(self, x):
        """İleri yayılım (Forward pass)"""
        batch_size = x.size(0)
        
        # 1) Encoder: [batch_size, num_features] -> [batch_size, num_nodes_bio]
        node_signals = self.encoder(x)
        
        # PyTorch Geometric GCN, veriyi (graph içerisindeki node sayısı, features) olarak bekler.
        # Batched input vermek adına tensorü [batch_size * num_nodes_bio, 1] düzlemine açıyoruz.
        h = node_signals.view(-1, 1)
        
        # Sinyal işleme esnasında bağımsız mini-batch grafları üretmek için dinamik edge_index offsetliyoruz.
        offset = torch.arange(0, batch_size * self.num_nodes_bio, self.num_nodes_bio, device=x.device).view(1, -1)
        batched_edge_index = self.edge_index.unsqueeze(2) + offset.unsqueeze(0)  # Shape: [2, num_edges, batch_size]
        batched_edge_index = batched_edge_index.permute(0, 2, 1).reshape(2, -1)  # Shape: [2, num_edges * batch_size]
        
        # 2) GNN Propagation
        h = F.relu(self.conv1(h, batched_edge_index))
        h = F.relu(self.conv2(h, batched_edge_index))
        
        # 3) Global Mean Pooling (Tüm biyolojik iskeletteki nöronlardan çıkan sinyallerin ortalaması)
        h = h.view(batch_size, self.num_nodes_bio, -1) # Shape: [batch_size, num_nodes_bio, 32]
        pooled_out = h.mean(dim=1)  # Shape: [batch_size, 32]
        
        # 4) Sınıflandırma
        out = self.classifier(pooled_out)
        return out


def train_model(model, train_loader, test_loader, epochs=20, lr=0.001, device='cpu'):
    """Hibrit modeli eğitme ve test döngüsü."""
    print(f"\n[INFO] Model eğitimi başlıyor... (Cihaz: {device})")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(1, epochs + 1):
        # --- Training Phase ---
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
        
        # --- Validation Phase ---
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
        
        # Her 5 epoch'ta veya ilk epoch'ta log ver.
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} - Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} - Acc: {test_acc:.4f}")
            
    print("\n[INFO] Eğitim tamamlandı. Biyomimetik GNN başarıyla test edildi.")

def main():
    try:
        # 1. Biyolojik İskeleti İnşa Et (Meyve Sineği Koordinatlarından KNN)
        edge_index, num_nodes_bio = DataProcessor.build_biological_graph("al_neurons_metadata.csv", k=5)
        
        # 2. Kaggle Model Verilerini (Breast Cancer) Yükle
        train_loader, test_loader, num_features, num_classes = DataProcessor.get_kaggle_dataset()
        
        # 3. Model Mimarisini Başlat
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = BiomimeticGNN(
            num_features=num_features,
            num_nodes_bio=num_nodes_bio, 
            num_classes=num_classes,
            edge_index=edge_index
        )
        
        # 4. Modeli Eğit ve Raporla
        train_model(model, train_loader, test_loader, epochs=25, lr=0.005, device=device)
        
    except Exception as e:
        print(f"\n[FATAL ERROR] Sistemde kritik bir hata oluştu:\n{e}")

if __name__ == "__main__":
    main()
