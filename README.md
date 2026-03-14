# Drosophila Biomimetic GNN & 3D Connectome Mapping

Bu proje, biyolojik bir sinir ağı mimarisini (Meyve Sineği - Drosophila Antennal Lobe) hem 3D uzayda interaktif olarak görselleştirmeyi hem de bu **biyolojik iskeleti matematiksel bir referans alarak** standart veri setleri (Örn: Breast Cancer) üzerinde çalışan hibrit bir Graph Neural Network (GNN) modeline dönüştürmeyi amaçlamaktadır.

## Projenin Temel İçerikleri

Proje temel olarak 4 ana adımdan oluşmaktadır:

1. **`main.py` (Veri Toplama):** Janelia Research Campus'ün `hemibrain` veri setinden `neuprint-python` kütüphanesi yardımıyla Meyve Sineği Antennal Lobe nöronlarını ve aralarındaki gerçek sinaptik bağlantıları (connectome) indirir. Verileri `.csv` formatında kaydeder.
2. **`vista.py` (3D İnteraktif Görselleştirme):** `PyVista` kullanarak beynin 3D uzaydaki yapısını interaktif bir şekilde modellediğimiz kısımdır. Nöronlar karmaşıklığına ve boyutuna göre kürelerle, bağlantıları ise sinaps gücüne (*weight*) göre kalınlaşan tüplerle temsil edilir.
3. **`visualize.py` (2D Ağ Grafiği):** `NetworkX` ve `Matplotlib` kullanarak en güçlü bağlantıların (sinaps akışının) 2D düzlemdeki topolojik haritasını ve Degree Centrality analizini çıkarır.
4. **`biomimetic_classifier.py` (Biomimetic GNN Modeli):** Biyolojik nöron topolojisini alıp, bir KNN grafına dönüştürür. PyTorch ve PyTorch Geometric (*PyG*) kullanarak tablo tipi (Kaggle Breast Cancer) verilerini bu doğal sinir ağı hattından geçirir (GCNConv). Biyomimetik bir yaklaşım test edilir ve **%98+** Accuracy ile başarılı sonuçlar elde edilmiştir.

## Kurulum ve Gereksinimler

Proje sanal ortam (virtual environment) kullanılarak birbirinden farklı bilimsel kütüphanelerin uyumlu çalışacağı şekilde inşa edilmiştir. Terminalinizde (veya `venv` içerisinde) şu bağımlılıkları yükleyin:

```bash
# Temel veri yükleme ve analiz
pip install pandas numpy scikit-learn matplotlib networkx

# 3D Uzamsal Görselleştirme
pip install pyvista

# Nöro-Veri Çekme (Hemibrain API)
pip install neuprint-python

# Biyomimetik GNN (Deep Learning)
pip install torch torchvision torchaudio
pip install torch_geometric
```

## Kullanım Adımları

### 1- Bağlantı ve Nöron Meta-Verisini İndirme
Öncelikle nöron koordinat ve bağlantılarının (`al_connectome_small.csv` ve `al_neurons_metadata.csv`) elde edilmesi gerekir. (Not: Bu aşamada `TOKEN` bilgisinin geçerli/hesabınıza ait olduğundan emin olun).
```bash
python main.py
```

### 2- 3D Uzayda Nöral Ağı İncelemek (Connectome Map)
Drosophila Antennal Lobe mimarisini tamamen interaktif (Döndürülebilir, zoom yapılabilir) bir karanlık tema haritası üzerinde inceleyin:
```bash
python vista.py
```

### 3- 2B Graf Analizi Yapmak
Nöronların merkeziliğini (hangi nöronun biyolojik network için trafiğin odağında olduğunu) 2B bir network diagramında görün:
```bash
python visualize.py
```

### 4- Biyomimetik Yapay Zeka Modelini Eğitmek
Gerçek biyolojik bağlanabilirlik ağını ve uzamsal organizasyonu esas alan sinir ağını (BiomimeticGNN), standart bir Kaggle veri setinde kanser hücrelerini sınıflandırmak için eğitmek:
```bash
python biomimetic_classifier.py
```
**Örnek Çıktı Performansı**:
```text
[INFO] Model eğitimi başlıyor... (Cihaz: cpu)
Epoch 01/25 | Train Loss: 0.6870 - Acc: 0.5011 | Val Loss: 0.5302 - Acc: 0.9649
...
Epoch 25/25 | Train Loss: 0.0469 - Acc: 0.9912 | Val Loss: 0.1116 - Acc: 0.9825
[INFO] Eğitim tamamlandı. Biyomimetik GNN başarıyla test edildi.
```

## Model Mimarisi: Biyomimetik GNN (BiomimeticGNN)

`biomimetic_classifier.py` içindeki mimarinin yapısal tasarımı:
* **Spatial Transformation:** $x,y,z$ biyolojik koordinatlar 5 komşulu KNN grafiği (`k=5`) ile PyTorch Tensörüne dönüştürülür.
* **Linear Encoder:** 30 girişli (Tablo Verisi) bilgiyi, Meyve Sineği genetiğinde bulunan 1500 limitli nöron ağı yapısına paralel bir şekilde pompalar.
* **GCN Propagation:** Tabular sinyal, Meyve Sineği biyolojik şablonundan esinlenmiş Graph Convolutional (GCNConv) katmanlarından geçirilerek süzülür.
* **Global Pooling:** Gelen veriler toplanır, ortalaması alınır ve `Classifier Head` tarafından sonuç/tahmin üretilir.

---
*Geliştirilmiş olan bu proje "Biomimetic AI Architect" konseptine dayalı, verimli, ölçeklenebilir ve mühendislik prensipleriyle optimize edilmiş bir repodur.*
