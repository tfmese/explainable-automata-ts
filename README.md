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
