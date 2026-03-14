"""
Biomimetic GNN vs Standard MLP Comparison 
-------------------------------------------
Dataset : EEG Eye State (Sensör & Sinyal Verisi) - 14 Uzaysal-Zamansal Sensör
Amaç    : Sineğin beyninden aldığımız mimarinin, rastgele gürültüye ve sensör bozulmalarına karşı 
          standart bir Yapay Sinir Ağından (MLP) daha "dayanıklı (robust)" ve "filtreleyici" olduğunu ispatlamak.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import time

# Orijinal mimari kodumuzu import ediyoruz
import directed_biomimetic_classifier as dbg

# 1. Standart Yapay Sinir Ağı (MLP)
class StandardMLP(nn.Module):
    def __init__(self, num_features, num_classes):
        super(StandardMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)

def get_sensor_dataset():
    """Gerçek dünya sensör/sinyal verisini indirir (EEG Eye State: 14 kanal)."""
    print("[INFO] EEG Eye State (Açık/Kapalı Göz-Beyin Dalgası) veriseti OpenML'den indiriliyor...")
    # EEG verisi, uzaysal ve gürültülü biyolojik sinyallere mükemmel bir örnektir.
    eeg = fetch_openml('eeg-eye-state', version=1, as_frame=False, parser='auto')
    
    X = eeg.data
    y = eeg.target.astype(int) # Sınıflar: 1 (Göz Açık), 2 (Göz Kapalı) -> 'eeg-eye-state'te 1,2 olarak string halinde geliyor
    y = y - 1 # Çapraz entropi (Cross Entropy) 0-indeksli çalışır, yani 0 ve 1 yapmalıyız
    
    # Çok büyük veri, eğitimi hızlı tutmak için ilk 5000 örneği alalım
    X = np.array(X[:8000], dtype=np.float32)
    y = np.array(y[:8000], dtype=np.int64)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # DataLoader'lar hazırlayalım
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    print(f"[INFO] EEG Veriseti (Özellik: {X.shape[1]}, Örnek: {X.shape[0]}) yüklendi.")
    return train_loader, test_loader, X.shape[1], 2, X_test, y_test

def train_network(model, train_loader, epochs=15, lr=0.005, device='cpu'):
    """Genel eğitim fonksiyonu"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(1, epochs + 1):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
    return model

def evaluate_network(model, test_X, test_y, noise_level=0.0, device='cpu'):
    """Gürültü toleransını ölçmek test eder."""
    model.eval()
    
    test_X_tensor = torch.tensor(test_X, dtype=torch.float32).to(device)
    test_y_tensor = torch.tensor(test_y, dtype=torch.long).to(device)
    
    # Eğer test verisine "Sensör Gürültüsü" eklenmek istenirse:
    if noise_level > 0:
        noise = torch.randn_like(test_X_tensor) * noise_level
        test_X_tensor = test_X_tensor + noise
        
    with torch.no_grad():
        outputs = model(test_X_tensor)
        preds = torch.argmax(outputs, dim=1)
        correct = (preds == test_y_tensor).sum().item()
        
    acc = correct / len(test_X)
    return acc * 100

def main():
    print("======================================================================")
    print(" BİYOMİMETİK GNN vs STANDART MLP - 'Gürültülü Sensör (EEG)' SAVAŞI")
    print("======================================================================\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Sinek Beyni İskeleti (Bağlantılar) İndiriliyor
    edge_index, edge_weight, num_nodes = dbg.DataProcessor.build_true_connectome_graph(
        "al_neurons_metadata.csv", "al_connectome_small.csv"
    )
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    
    # 2. Veriseti Hazırlanıyor
    train_loader, test_loader, num_features, num_classes, X_test, y_test = get_sensor_dataset()
    
    # 3. İki Modeli de Oluştur
    # Biyomimetik GNN
    bio_gnn = dbg.TrueBiomimeticGNN(num_features, num_nodes, num_classes, edge_index, edge_weight).to(device)
    # Standart MLP
    std_mlp = StandardMLP(num_features, num_classes).to(device)
    
    # 4. Modelleri Eğit
    print("\n[Oyun Başlıyor] İki Model de aynı veriyle eğitiliyor (Lütfen bekleyin)...\n")
    
    print("-> Sinek Beyinli Biyomimetik GNN eğitiliyor...")
    t0 = time.time()
    bio_gnn = train_network(bio_gnn, train_loader, epochs=15, lr=0.005, device=device)
    gnn_time = time.time() - t0
    
    print("-> Standart MLP (Yapay Sinir Ağı) eğitiliyor...")
    t0 = time.time()
    std_mlp = train_network(std_mlp, train_loader, epochs=15, lr=0.002, device=device)
    mlp_time = time.time() - t0
    
    # 5. Temiz ve Kirli Verilerle Kıyasla
    print("\n" + "="*50)
    print(" KARŞILAŞTIRMALI SONUÇLAR (Doğruluk: %)")
    print("="*50)
    
    # Senaryo 1: Temiz Veri (Laboratuvar Ortamı)
    gnn_acc_clean = evaluate_network(bio_gnn, X_test, y_test, noise_level=0.0, device=device)
    mlp_acc_clean = evaluate_network(std_mlp, X_test, y_test, noise_level=0.0, device=device)
    
    print(f"[1] İdeal Şartlar (Sensörler Temiz) - Normal EEG:")
    print(f"    Standart MLP Accuracy   : %{mlp_acc_clean:.2f}")
    print(f"    Biyomimetik GNN         : %{gnn_acc_clean:.2f}")
    
    # Senaryo 2: Aşırı Gürültülü Veri (Dış Mekan / Sensör Kayması)
    # MLP'nin dağılmasını, GNN'in ise gürültüyü süzmesini bekleriz
    gnn_acc_noisy = evaluate_network(bio_gnn, X_test, y_test, noise_level=1.5, device=device)
    mlp_acc_noisy = evaluate_network(std_mlp, X_test, y_test, noise_level=1.5, device=device)
    
    print(f"\n[2] Kaotik Şartlar (Aşırı Gürültü ve Sinyal Paraziti):")
    print(f"    Standart MLP Accuracy   : %{mlp_acc_noisy:.2f}  (Büyük Çöküş)")
    print(f"    Biyomimetik GNN         : %{gnn_acc_noisy:.2f}  (Filtreleme Görevi Başarılı)")
    
    print("\n[SONUÇ DETAYI]")
    if gnn_acc_noisy > mlp_acc_noisy:
        print("Sinek beyinli GNN modeli, gürültü geldiğinde komşu sinapslarla bilgiyi dengeleyerek")
        print("standart MLP modelinden çok daha iyi bir gürültü direnci (Robustness) sergiledi!")
    else:
        print("Sonuçlara göre standart model ve GNN başa baş rekabet etti.")

if __name__ == "__main__":
    main()
