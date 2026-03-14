from neuprint import Client, fetch_neurons, fetch_adjacencies
import pandas as pd
import time
import os
TOKEN=''
c = Client('neuprint.janelia.org', dataset='hemibrain:v1.2.1', token=TOKEN)

try:
    print("Nöron listesi indiriliyor...")
    results = fetch_neurons()
    all_neurons = results[0] if isinstance(results, tuple) else results

    # roiInfo genelde bir dict formatındadır. 
    # İçinde 'AL(R)' bölgesi olanları daha güvenli bir yöntemle bulalım:
    print("Antennal Lobe (AL) bölgesi filtreleniyor...")
    
    def is_in_al(roi_info):
        if isinstance(roi_info, dict):
            return 'AL(R)' in roi_info
        return False

    al_mask = all_neurons['roiInfo'].apply(is_in_al)
    al_neurons = all_neurons[al_mask].head(1500)

    if al_neurons.empty:
        print("Hata: Seçilen bölgede nöron bulunamadı. Lütfen ROI ismini kontrol edin.")
    else:
        body_ids = al_neurons['bodyId'].tolist()
        print(f"{len(body_ids)} adet koku merkezi nöronu seçildi.")

        # 2. Bağlantıları çek
        print("Bağlantılar çekiliyor (bu işlem kısa sürecektir)...")
        adj_results = fetch_adjacencies(body_ids, body_ids)
        conn_df = adj_results[0] if isinstance(adj_results, tuple) else adj_results

        # 3. Kaydet
        al_neurons.to_csv("al_neurons_metadata.csv", index=False)
        conn_df.to_csv("al_connectome_small.csv", index=False)
        
        meta_size = os.path.getsize("al_neurons_metadata.csv") / (1024*1024)
        conn_size = os.path.getsize("al_connectome_small.csv") / (1024*1024)
        
        print(f"\nBAŞARILI! Toplam boyut: {meta_size + conn_size:.2f} MB")
        print(f"Toplam Bağlantı Sayısı: {len(conn_df)}")

except Exception as e:
    print(f"Hata: {e}")