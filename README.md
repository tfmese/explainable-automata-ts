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

## 2. Veri setleri arası farklar

SKAB valf sensör verisi; otomata için PC1'e indirgedik. BATADAL su şebekesi saldırı verisi, sınıf dengesizliği daha belirgin. Aynı model SKAB'da işe yararken BATADAL'da otomata tarafı zayıf kaldı — muhtemelen PC1'e sıkışınca saldırı sinyali kayboluyor.

Aşağıda original senaryodan örnek ROC/PR grafikleri var. Diğer modeller ve senaryolar outputs/figures/ altında, isim formatı: {dataset}_{model}_{scenario}_roc_curve.png

*SKAB — otomata*
![SKAB otomata ROC](outputs/figures/skab_automata_original_roc_curve.png)
![SKAB otomata PR](outputs/figures/skab_automata_original_precision_recall.png)

*SKAB — LSTM*
![SKAB LSTM ROC](outputs/figures/skab_lstm_original_roc_curve.png)
![SKAB LSTM PR](outputs/figures/skab_lstm_original_precision_recall.png)

*BATADAL — otomata*
![BATADAL otomata ROC](outputs/figures/batadal_automata_original_roc_curve.png)
![BATADAL otomata PR](outputs/figures/batadal_automata_original_precision_recall.png)

---

## 3. Gürültü etkisi

Test setine mean=0, std=0.05 Gaussian noise ekledik.

*SKAB:* DL modellerinin F1'i ~0.83'ten ~0.33–0.51 aralığına düştü. Otomata F1 ~0.52'de kaldı (recall 1.0'a çıktı, precision düştü). SAX küçük dalgalanmaları yuttuğu için otomata gürültüde daha stabil görünüyor.

*BATADAL:* Otomata F1 ~0.06, recall ~0.20. DL tarafında GRU hâlâ en iyi (~0.38 F1 gürültüde).

![SKAB otomata — orijinal](outputs/figures/skab_automata_original_confusion_matrix.png)
![SKAB otomata — gürültülü](outputs/figures/skab_automata_gaussian_noise_confusion_matrix.png)

![BATADAL otomata — orijinal](outputs/figures/batadal_automata_original_confusion_matrix.png)
![BATADAL otomata — gürültülü](outputs/figures/batadal_automata_gaussian_noise_confusion_matrix.png)

---

## 4. Unseen pattern

Testte eğitimde görmediğimiz bir SAX pattern gelirse Levenshtein ile en yakın state'e map ediyoruz. Unseen senaryoda bunu kasıtlı tetiklemek için test dizisine sözlükte olmayan bir pattern enjekte ettik.

DL modelleri SAX kullanmadığı için unseen senaryoda girdi değişmiyor; onların metrikleri original ile aynı.

![SKAB otomata unseen](outputs/figures/skab_automata_unseen_confusion_matrix.png)
![BATADAL otomata unseen](outputs/figures/batadal_automata_unseen_confusion_matrix.png)

Açıklama çıktıları outputs/explanations/ klasöründe. Kısa örnek: [docs/sample_explanation.json](docs/sample_explanation.json)

json
{
  "time_step": 5,
  "state": "abc",
  "pattern": "adc",
  "status": "unseen",
  "mapped_to": "abc",
  "mapping_distance": 1,
  "transitions": [{"from": "aab", "to": "abc", "probability": 0.72}],
  "probability": 0.108,
  "decision": "anomaly",
  "confidence": 0.108
}


---
## 5. Parametre etkileri

Model karşılaştırması için window_size=4, alphabet_size=3 sabitledik. Sonra window ve alphabet'i 3–6 arasında taradık; state sayısı, geçiş yoğunluğu ve F1'e bakıldı. Parametre taraması da 5 seed ile koşuldu.

SKAB'da alphabet büyüdükçe F1 biraz artıyor ama state sayısı patlıyor. BATADAL'da w=4, α=3 kombinasyonunda F1 sıfır çıktı; bazı kombinasyonlarda (ör. w=3, α=5) F1 ~0.26'ya çıkabiliyor.

![SKAB parametre duyarlılığı](outputs/figures/parameter_sensitivity.png)
![SKAB state duyarlılığı](outputs/figures/state_sensitivity.png)

![BATADAL parametre duyarlılığı](outputs/figures/batadal_parameter_sensitivity.png)
![BATADAL state duyarlılığı](outputs/figures/batadal_state_sensitivity.png)

---

## 6. Otomata görselleri

Original senaryo için state diagram ve transition heatmap:

![SKAB state diagram](outputs/figures/skab_automata_original_state_diagram.png)
![SKAB transition heatmap](outputs/figures/skab_automata_original_transition_heatmap.png)

![BATADAL state diagram](outputs/figures/batadal_automata_original_state_diagram.png)
![BATADAL transition heatmap](outputs/figures/batadal_automata_original_transition_heatmap.png)

Gürültü ve unseen senaryolarının grafikleri de outputs/figures/ içinde aynı isimlendirmeyle var.

---

## 7. Olasılıksal yorumlama

Prefix üzerindeki geçiş olasılıklarını çarpıyoruz; sonuç yüksekse normal, düşükse anomali. Güven skoru olarak da aynı path probability değerini kullanıyoruz. Eşik train setindeki path olasılıklarının alt yüzdelik diliminden geliyor.

---
## 8. Kod yapısı ve çalıştırma


config/config.yaml          → tüm parametreler
src/pipeline.py             → otomata pipeline
src/models/automata.py      → geçiş olasılıkları, Levenshtein
src/models/deep_learning*   → LSTM, GRU, 1D-CNN
src/data/                   → veri yükleme, bölme, ön işleme
src/experiments/            → senaryolar, parametre taraması
src/explainability/         → açıklama modülü
run_experiments.py          → ana deney scripti


bash
pip install -r requirements.txt
python3 run_experiments.py          # tam deney (uzun süreli)
python3 run_experiments.py --fast   # kısa smoke test
python3 -m pytest
python3 scripts/generate_report_metrics.py   # alttaki tabloları günceller


SKAB'da Wilcoxon ve McNemar testleri otomata ile LSTM/GRU arasındaki F1 farkının anlamlı olduğunu gösterdi (p < 0.05). Sayılar ek tabloda.

Tüm confusion matrix, ROC, PR ve otomata grafikleri: outputs/figures/

---

<!-- AUTO_METRICS_START -->
## Ek: Sayısal sonuçlar

Özet tablolar outputs/results/*.json dosyalarından üretilir. Deneyleri yeniden koştuktan sonra
python3 scripts/generate_report_metrics.py ile bu bölüm güncellenyior.

### SKAB — original (5 fold ort., 5 seed)

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| AUTOMATA | 0.5000 ± 0.0055 | 0.3721 ± 0.0052 | 0.6251 ± 0.0149 | 0.4612 ± 0.0047 |
| LSTM | 0.8977 ± 0.0049 | 0.9405 ± 0.0156 | 0.7604 ± 0.0103 | 0.8323 ± 0.0078 |
| GRU | 0.8986 ± 0.0053 | 0.9395 ± 0.0162 | 0.7661 ± 0.0046 | 0.8359 ± 0.0050 |
| CNN1D | 0.8975 ± 0.0033 | 0.9444 ± 0.0105 | 0.7590 ± 0.0077 | 0.8317 ± 0.0037 |

### BATADAL — original (%20 test, 5 seed)

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| AUTOMATA | 0.7311 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| LSTM | 0.9028 ± 0.0008 | 0.1000 ± 0.2000 | 0.0100 ± 0.0200 | 0.0182 ± 0.0364 |
| GRU | 0.9246 ± 0.0190 | 0.5187 ± 0.4248 | 0.2625 ± 0.2184 | 0.3481 ± 0.2878 |
| CNN1D | 0.9155 ± 0.0154 | 0.3149 ± 0.3859 | 0.1675 ± 0.2277 | 0.2107 ± 0.2709 |

### SKAB — gaussian_noise (5 fold ort., 5 seed)

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| AUTOMATA | 0.3597 ± 0.0004 | 0.3525 ± 0.0003 | 1.0000 ± 0.0000 | 0.5212 ± 0.0003 |
| LSTM | 0.6167 ± 0.0251 | 0.5118 ± 0.0716 | 0.3809 ± 0.0308 | 0.3662 ± 0.0204 |
| GRU | 0.6166 ± 0.0128 | 0.5529 ± 0.0491 | 0.3486 ± 0.0602 | 0.3340 ± 0.0428 |
| CNN1D | 0.5638 ± 0.0301 | 0.4801 ± 0.0607 | 0.6919 ± 0.0884 | 0.5071 ± 0.0441 |

### BATADAL — gaussian_noise (%20 test, 5 seed)

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| AUTOMATA | 0.5335 ± 0.0447 | 0.0363 ± 0.0624 | 0.2025 ± 0.3571 | 0.0616 ± 0.1063 |
| LSTM | 0.8972 ± 0.0050 | 0.1287 ± 0.1578 | 0.0275 ± 0.0436 | 0.0423 ± 0.0640 |
| GRU | 0.9037 ± 0.0400 | 0.4554 ± 0.3269 | 0.3300 ± 0.2254 | 0.3774 ± 0.2600 |
| CNN1D | 0.9143 ± 0.0148 | 0.3100 ± 0.3800 | 0.1625 ± 0.2222 | 0.2048 ± 0.2641 |

### SKAB — unseen (5 fold ort., 5 seed)

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| AUTOMATA | 0.4988 ± 0.0044 | 0.3719 ± 0.0029 | 0.6317 ± 0.0134 | 0.4637 ± 0.0051 |
| LSTM | 0.8977 ± 0.0049 | 0.9405 ± 0.0156 | 0.7604 ± 0.0103 | 0.8323 ± 0.0078 |
| GRU | 0.8986 ± 0.0053 | 0.9395 ± 0.0162 | 0.7661 ± 0.0046 | 0.8359 ± 0.0050 |
| CNN1D | 0.8975 ± 0.0033 | 0.9444 ± 0.0105 | 0.7590 ± 0.0077 | 0.8317 ± 0.0037 |

### BATADAL — unseen (%20 test, 5 seed)

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| AUTOMATA | 0.6435 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| LSTM | 0.9028 ± 0.0008 | 0.1000 ± 0.2000 | 0.0100 ± 0.0200 | 0.0182 ± 0.0364 |
| GRU | 0.9246 ± 0.0190 | 0.5187 ± 0.4248 | 0.2625 ± 0.2184 | 0.3481 ± 0.2878 |
| CNN1D | 0.9155 ± 0.0154 | 0.3149 ± 0.3859 | 0.1675 ± 0.2277 | 0.2107 ± 0.2709 |

### SKAB fold sonuçları (seed=42, original)

| Fold | AUTOMATA F1 | LSTM F1 | GRU F1 | CNN1D F1 |
|---:|---:|---:|---:|---:|
| 0 | 0.5069 | 0.8632 | 0.8280 | 0.8554 |
| 1 | 0.3898 | 0.8606 | 0.8426 | 0.8569 |
| 2 | 0.4088 | 0.7581 | 0.7482 | 0.7179 |
| 3 | 0.4591 | 0.8733 | 0.8750 | 0.8738 |
| 4 | 0.5201 | 0.8780 | 0.8807 | 0.8809 |

### SKAB fold detayı (seed=42, original)

#### AUTOMATA

| Fold | Acc | Prec | Rec | F1 |
|---:|---:|---:|---:|---:|
| 0 | 0.5258 | 0.3882 | 0.7304 | 0.5069 |
| 1 | 0.5260 | 0.3564 | 0.4300 | 0.3898 |
| 2 | 0.4955 | 0.3466 | 0.4981 | 0.4088 |
| 3 | 0.4490 | 0.3500 | 0.6669 | 0.4591 |
| 4 | 0.5065 | 0.3974 | 0.7523 | 0.5201 |

| Fold | State | Density |
|---:|---:|---:|
| 0 | 66 | 0.0312 |
| 1 | 70 | 0.0300 |
| 2 | 72 | 0.0293 |
| 3 | 59 | 0.0345 |
| 4 | 52 | 0.0381 |

#### LSTM

| Fold | Acc | Prec | Rec | F1 |
|---:|---:|---:|---:|---:|
| 0 | 0.9112 | 0.8879 | 0.8399 | 0.8632 |
| 1 | 0.9062 | 0.9025 | 0.8224 | 0.8606 |
| 2 | 0.8566 | 0.9258 | 0.6418 | 0.7581 |
| 3 | 0.9211 | 0.9989 | 0.7757 | 0.8733 |
| 4 | 0.9226 | 0.9976 | 0.7840 | 0.8780 |

#### GRU

| Fold | Acc | Prec | Rec | F1 |
|---:|---:|---:|---:|---:|
| 0 | 0.8924 | 0.8879 | 0.7757 | 0.8280 |
| 1 | 0.8897 | 0.8464 | 0.8389 | 0.8426 |
| 2 | 0.8575 | 0.9807 | 0.6048 | 0.7482 |
| 3 | 0.9220 | 0.9979 | 0.7791 | 0.8750 |
| 4 | 0.9239 | 0.9938 | 0.7908 | 0.8807 |

#### CNN1D

| Fold | Acc | Prec | Rec | F1 |
|---:|---:|---:|---:|---:|
| 0 | 0.9024 | 0.8455 | 0.8655 | 0.8554 |
| 1 | 0.9024 | 0.8855 | 0.8301 | 0.8569 |
| 2 | 0.8452 | 0.9912 | 0.5627 | 0.7179 |
| 3 | 0.9214 | 0.9989 | 0.7766 | 0.8738 |
| 4 | 0.9237 | 0.9892 | 0.7939 | 0.8809 |

### SKAB parametre taraması

| W | α | F1 | State | Density |
|---:|---:|---:|---:|---:|
| 3 | 3 | 0.4557 ± 0.0428 | 26.4800 | 0.0951 |
| 3 | 4 | 0.4753 ± 0.0486 | 48.3600 | 0.0611 |
| 3 | 5 | 0.4954 ± 0.0465 | 78.0000 | 0.0395 |
| 3 | 6 | 0.5157 ± 0.0355 | 124.1200 | 0.0258 |
| 4 | 3 | 0.4612 ± 0.0478 | 64.2800 | 0.0325 |
| 4 | 4 | 0.4850 ± 0.0465 | 139.8800 | 0.0173 |
| 4 | 5 | 0.5111 ± 0.0497 | 236.0800 | 0.0111 |
| 4 | 6 | 0.5244 ± 0.0359 | 393.6000 | 0.0067 |
| 5 | 3 | 0.4683 ± 0.0539 | 131.5200 | 0.0149 |
| 5 | 4 | 0.4866 ± 0.0418 | 335.6000 | 0.0062 |
| 5 | 5 | 0.5090 ± 0.0460 | 615.2800 | 0.0036 |
| 5 | 6 | 0.5309 ± 0.0304 | 1039.3200 | 0.0021 |
| 6 | 3 | 0.4749 ± 0.0539 | 251.7600 | 0.0074 |
| 6 | 4 | 0.4927 ± 0.0309 | 700.7200 | 0.0027 |
| 6 | 5 | 0.5260 ± 0.0325 | 1360.6400 | 0.0014 |
| 6 | 6 | 0.5197 ± 0.0295 | 2287.0400 | 0.0008 |

### BATADAL parametre taraması

| W | α | F1 | State | Density |
|---:|---:|---:|---:|---:|
| 3 | 3 | 0.0000 ± 0.0000 | 26.0000 | 0.1108 |
| 3 | 4 | 0.0000 ± 0.0000 | 59.0000 | 0.0511 |
| 3 | 5 | 0.2575 ± 0.0000 | 105.0000 | 0.0310 |
| 3 | 6 | 0.2173 ± 0.0000 | 166.0000 | 0.0191 |
| 4 | 3 | 0.0000 ± 0.0000 | 72.0000 | 0.0331 |
| 4 | 4 | 0.2288 ± 0.0000 | 175.0000 | 0.0135 |
| 4 | 5 | 0.2070 ± 0.0000 | 339.0000 | 0.0067 |
| 4 | 6 | 0.2015 ± 0.0000 | 522.0000 | 0.0039 |
| 5 | 3 | 0.0000 ± 0.0000 | 169.0000 | 0.0111 |
| 5 | 4 | 0.2100 ± 0.0000 | 412.0000 | 0.0044 |
| 5 | 5 | 0.1954 ± 0.0000 | 765.0000 | 0.0022 |
| 5 | 6 | 0.1937 ± 0.0000 | 1051.0000 | 0.0014 |
| 6 | 3 | 0.2466 ± 0.0000 | 316.0000 | 0.0051 |
| 6 | 4 | 0.2018 ± 0.0000 | 745.0000 | 0.0020 |
| 6 | 5 | 0.1905 ± 0.0000 | 1268.0000 | 0.0011 |
| 6 | 6 | 0.1876 ± 0.0000 | 1588.0000 | 0.0008 |



<!-- AUTO_METRICS_END -->

---

## Kaynaklar

## Kaynaklar

- SAX (Symbolic Aggregate approXimation), 2003
- PAA (Piecewise Aggregate Approximation), 2001
- LSTM, 1997
- GRU, 2014
- Levenshtein edit distance, 1966
- SKAB veri seti: https://github.com/waico/SKAB
- BATADAL veri seti: https://www.batadal.net/
- scikit-learn: https://scikit-learn.org/
- PyTorch: https://pytorch.org/
