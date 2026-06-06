# Explainable Automata for Time Series Analysis
**Yazılım Geliştirme Laboratuvarı II — Proje II**

Bu projede zaman serisi verilerinde anomali tespiti için iki yaklaşımı karşılaştırdık: sembolik otomata modeli ve derin öğrenme (LSTM, GRU, 1D-CNN). Deneyleri SKAB ve BATADAL veri setlerinde, orijinal veri + gürültü + unseen pattern senaryolarında koştuk.

---

## Veri setleri ve protokol

### SKAB
- `data/SKAB/valve1` ve `valve2` altındaki tüm csv dosyalarını `concat` ile birleştirdik.
- `source_group` ve `source_file` sütunlarını ekledik (sadece takip ve bölme için, modele girmiyor).
- Hedef: `anomaly`
- Modele girmeyen sütunlar: `datetime`, `changepoint`, `source_group`, `source_file`
- Bölme: `source_file` üzerinden StratifiedGroupKFold, 5 fold. Aynı dosya train ve testte birlikte olmuyor.
- Normalizasyon, PCA (PC1) ve SAX sözlüğü sadece train fold üzerinde fit ediliyor.

### BATADAL
- Sadece `Training_Dataset_2.csv` kullandık.
- Hedef sütun: `ATT_FLAG` (`-999` → 0 normal, `2` → 1 saldırı)
- `DATETIME`, `Date`, `Time` modele girmiyor.
- Bölme: kronolojik %60 train / %20 validation / %20 test.
- Otomata hattında PCA→PC1 ve train-fitted SAX var.

### DL eğitim ayarları
Hepsi `config/config.yaml` içinde:
- max epoch: 50, batch: 32, early stopping patience: 5
- seed'ler: 42, 123, 2026, 7, 999

| Veri seti | Train | Validation | Test |
|---|---|---|---|
| SKAB (fold başına) | Train fold'un %90'ı | Train fold'un %10'u | Test fold |
| BATADAL | İlk %60 | Sonraki %20 | Son %20 |

Epoch sayısı ve early stopping bilgisi `outputs/logs/experiments.jsonl` dosyasına, özet ise `outputs/logs/run_summary.json` dosyasına yazılıyor.

---

## 1. Model karşılaştırması

Otomata tarafında akış: PAA → SAX → sliding window → her pattern bir state, geçiş olasılıkları frekansla öğreniliyor. Düşük path probability gelirse anomali diyoruz. Her adımda state, pattern ve geçişler loglanabiliyor.

DL modelleri çok değişkenli girdiyle çalışıyor; kararın nedenini doğrudan okuyamıyoruz ama SKAB'da F1 daha yüksek çıktı.

![SKAB senaryo karşılaştırması](outputs/figures/model_comparison_skab.png)

![BATADAL senaryo karşılaştırması](outputs/figures/model_comparison_batadal.png)

**SKAB (original):** LSTM/GRU/CNN1D F1 yaklaşık 0.83 bandında. Otomata F1 ~0.46; recall ~0.63, precision ~0.37. Yani otomata daha çok anomali yakalıyor ama yanlış alarm da fazla.

**BATADAL (original):** Burada sonuçlar veri setine göre çok değişti. Otomata hiç saldırı yakalayamadı (recall 0, F1 0). LSTM de neredeyse hiç alarm üretmedi (recall ~0.01). En iyi DL tarafı GRU oldu (F1 ~0.35).

Detaylı sayılar aşağıdaki tablolarda.

---

