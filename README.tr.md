# WBC Analyzer — Türkçe Dokümantasyon

Bu belge, proje ana README dosyasındaki içeriğin Türkçe çevirisidir (üst rozetler/banner çıkarıldı). Projenin amaçları, hızlı başlatma talimatları, değerlendirme ve API örnekleri İngilizce README ile uyumludur.

---

<p align="center">
  <a href="https://emirhanyildirim.me/wbc-analyzer/" target="_blank" rel="noopener">Canlı demoyu aç →</a>
</p>

## Genel Bakış

WBC Analyzer, periferik kan yayma görüntülerinde beyaz kan hücresi (WBC) sınıflandırması için uçtan uca bir uygulamadır. Sistem; ön işleme (MEF), özel dikkat blokları (WBCAttention), öğrenilebilir aktivasyon (MedSwish) ve çıkarım zamanı alan adaptasyonu mekanizmalarını birleştirir. Üretim için Flask tabanlı bir REST API içerir.

## Öne Çıkan Katkılar

- Medical Enhanced Filter (MEF): Cihazlar arası parlaklık ve renk farklılıklarını azaltan 5 adımlı ön işleme.
- WBCAttention & MedSwish: Parametre verimli dikkat bloğu ve öğrenilebilir aktivasyon.
- XAIFocusMonitor: Eğitim sırasında Grad-CAM odak oranını izleyen Keras callback'i.
- Çıkarım katmanında LLM tabanlı açıklama (Grad-CAM + özerk raporlama).

## Kısa Performans Özeti

- TestA (İN-D): %98.53
- TestB (OOD): %89.05 (temel modele göre ~+32 puan kazanç)
- Birleşik: %95.42

Detaylı sınıf bazlı metrikler, omurga karşılaştırmaları ve ablation analizleri İngilizce README'de yer almaktadır.

## Proje Yapısı

Önemli dosyalar ve klasörler:

- `app.py` — Flask API ve LLM entegrasyonu
- `train_main_model.py` — İki aşamalı eğitim + XAI izleme
- `eval_final.py` — TTA + ikili yönlendirme + Reinhard değerlendirme
- `src/` — `custom_layers.py`, `custom_losses.py`, `preprocessing.py`
- `data/models/` — Üretim ağırlıklarının konulacağı klasör
- `outputs/` — Değerlendirme çıktıları

## Hızlı Başlangıç

1. Depoyu klonlayın ve bağımlılıkları yükleyin:

```bash
git clone https://github.com/yildirimemirhan/wbc-analyzer.git
cd wbc-analyzer
pip install -r requirements.txt
```

2. Üretim modelini indirin ve şu konuma yerleştirin:

```
data/models/wbc_final_model_densenet.keras
```

3. Proje kök dizinine bir `.env` dosyası ekleyin:

```
GITHUB_TOKEN=your_github_models_token
GEMINI_API_KEY=your_gemini_api_key
```

4. Sunucuyu başlatın:

```bash
python app.py
```

Sunucu varsayılan olarak `http://localhost:5000` üzerinde çalışır. `/predict` uç noktasına WBC görüntüleri göndererek sınıflandırma, Grad-CAM görseli ve LLM açıklaması alabilirsiniz.

> Not: Windows üzerinde GPU hızlandırma gerekiyorsa WSL2 + CUDA kullanılması önerilir.

## Değerlendirme ve Eğitim Örnekleri

Final değerlendirmeyi şu komutla çalıştırın:

```bash
python eval_final.py \
  --model-path data/models/wbc_final_model_densenet.keras \
  --data-root data/raabin-wbc-data \
  --output-dir outputs/final_model_results \
  --testb-binary-mode main \
  --tta light \
  --color-normalization reinhard \
  --preprocessing v1
```

Ana modeli eğitmek için:

```bash
python train_main_model.py \
  --data-root data/raabin-wbc-data \
  --phase1-epochs 15 \
  --phase2-epochs 15 \
  --main-loss cce \
  --label-smoothing 0.1 \
  --crop-prob 0.2 \
  --bg-randomization-prob 0.15 \
  --stain-jitter-prob 0.3 \
  --aux-loss-weight 1.0 \
  --xai-focus-threshold 0.55 \
  --xai-every-n-epochs 2 \
  --model-path data/models/wbc_final_model_densenet.keras
```

## Kaggle Notebook

`notebooks/wbc_analyzer_kaggle_quickstart.ipynb` kurulum gerektirmeyen bir başlangıç defteridir:
özel katmanları, ön işleme filtresini ve beş örnek görüntüyü içinde taşır; modeli yükler, tahmin ve
Grad-CAM üretir, kendi görüntü klasörünüzü sınıflandırır ve isteğe bağlı olarak Raabin-WBC üzerinde
değerlendirme yapar. `src/` veya `docs/samples/` değişirse `python notebooks/build_kaggle_notebook.py`
ile yeniden üretin (`tests/test_notebook.py` eski kalırsa başarısız olur).

## API Örneği

`POST /predict` form alanı:

- `file`: Görüntü (JPG, PNG, BMP, TIFF, WebP)

Başarılı yanıt örneği:

```json
{
    "class": "Neutrophil",
    "confidence": 0.977,
    "all_probabilities": {
        "Basophil": 0.001,
        "Eosinophil": 0.002,
        "Lymphocyte": 0.012,
        "Monocyte": 0.008,
        "Neutrophil": 0.977
    },
    "gradcam_image": "data:image/png;base64,iVBORw0KGgo...",
    "llm_report": "Grad-CAM doğrulama raporu: Model odaklanma çekirdek lobasyonları ve sitoplazmik granülasyonu gösteriyor. Arka plan kısayolu tespit edilmedi."
}
```

Hata kodları: `400` (geçersiz görüntü), `415` (desteklenmeyen format), `500` (sunucu/model hatası).

## Atıf

```bibtex
@article{yildirim2026wbc,
  title={Achieving Robust Out-of-Distribution Generalization in Peripheral Blood Smears via Custom Attention Mechanisms, Medical Enhanced Filtering, and Inference-Time Domain Adaptation},
  author={Yildirim, Emirhan},
  journal={arXiv preprint arXiv:2605.XXXXX},
  year={2026}
}
```

---

Yardımcı olmamı istediğiniz başka bir kısım varsa söyleyin; örneğin: örnek istek/cevap curl komutu, .env için `.gitignore` eklenmesi ya da Türkçe içerikte daha fazla ayrıntı eklemek gibi.
