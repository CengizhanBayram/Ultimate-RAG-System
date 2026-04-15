# Production-Grade RAG System — Kurulum ve Mimari

Türkçe müşteri destek belgelerini (sözleşme, fiyat tablosu, güncelleme logları) sorgulayan
altı aşamalı production-grade bir RAG sistemi.

---

## İçindekiler

1. [Hızlı Kurulum](#hızlı-kurulum)
2. [Proje Yapısı](#proje-yapısı)
3. [Sistem Mimarisi — Akış Diyagramı](#sistem-mimarisi)
4. [Teknoloji Seçimleri — Neden Bu Araçlar?](#teknoloji-seçimleri)
5. [Mimari Kararlar — Neden Bu Tasarım?](#mimari-kararlar)
6. [Veri Çakışma Senaryoları](#veri-çakışma-senaryoları)
7. [Ortam Değişkenleri](#ortam-değişkenleri)
8. [Bilinen Sınırlamalar](#bilinen-sınırlamalar)
9. [Performans İpuçları](#performans-ipuçları)

---

## Hızlı Kurulum

### Ön Koşullar

- Python 3.10 veya üzeri
- Google Gemini API anahtarı ([aistudio.google.com](https://aistudio.google.com) üzerinden ücretsiz alınabilir)

### Adım 1 — Sanal ortam oluştur

```bash
cd rag_project
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### Adım 2 — Bağımlılıkları yükle

```bash
pip install -r requirements.txt
```

> **Not:** `FlagEmbedding` paketi BGE-M3 modelini ilk çalıştırmada (~1.5 GB) HuggingFace'den indirir.
> GPU varsa `requirements.txt` içinde `faiss-cpu` yerine `faiss-gpu` kullanın.
>
> **Windows + TensorFlow çakışması:** Sistemde bozuk bir TensorFlow kurulumu varsa
> `transformers` import hatası alınabilir. Çözüm: `pip uninstall tensorflow -y`

### Adım 3 — API anahtarını ayarla

```bash
cp .env.example .env
# .env dosyasını aç ve doldur:
GOOGLE_API_KEY=AIza...buraya_gemini_api_keyini_yaz
```

### Adım 4 — Index oluştur

```bash
# İlk çalıştırmada BGE-M3 indirilir + tüm belgeler embed edilir (~2-5 dk)
python main.py --rebuild-index
```

### Adım 5 — Kullan

```bash
# Tek sorgulama
python main.py --query "Pro paket aylık ücreti nedir?"

# Etkileşimli sohbet modu
python main.py --interactive

# Dosya değişince otomatik index yenile
python main.py --watch

# 10 sorgu benchmark tablosu
python main.py --benchmark
```

### Adım 6 — Web arayüzü (Streamlit)

```bash
streamlit run app.py
# Tarayıcıda açılır: http://localhost:8501
```

### Adım 7 — Unit testler

```bash
pytest tests/ -v
# veya:
python main.py --run-tests
```

---

## Proje Yapısı

```
rag_project/
├── data/
│   ├── sozlesme.txt          # 15+ maddelik Türkçe müşteri sözleşmesi
│   ├── paket_fiyatlari.csv   # Basic / Pro / Enterprise paket fiyatları
│   └── guncellemeler.json    # 8 güncelleme kaydı (kasıtlı çakışmalar içerir)
├── src/
│   ├── config.py             # Pydantic BaseSettings — merkezi konfigürasyon
│   ├── models.py             # Paylaşılan dataclass'lar (Document, ScoredDocument…)
│   ├── loaders.py            # Paralel çok-format yükleyici (txt / csv / json)
│   ├── chunker.py            # Hiyerarşik parent-child parçalayıcı
│   ├── embedder.py           # BGE-M3 toplu gömme motoru + disk cache
│   ├── vector_store.py       # FAISS IndexFlatIP vektör deposu
│   ├── retriever.py          # Hibrit BM25 + yoğun retrieval + RRF birleştirme
│   ├── query_expander.py     # Gemini ile sorgu genişletme
│   ├── cross_encoder.py      # ms-marco cross-encoder yeniden sıralama
│   ├── reranker.py           # Kaynak öncelik çarpanları + çakışma çözümü
│   ├── context_builder.py    # Token-bilinçli bağlam birleştirici
│   ├── generator.py          # Gemini yanıt üreticisi
│   ├── evaluator.py          # LLM-as-judge doğruluk değerlendirici
│   ├── guardrail.py          # Halüsinasyon engel kapısı
│   ├── pipeline.py           # Ana orkestratör (tüm aşamaları bağlar)
│   ├── cache.py              # diskcache sorgu önbelleği
│   └── watcher.py            # Watchdog dosya izleyici
├── tests/                    # pytest birim ve entegrasyon testleri
├── main.py                   # CLI giriş noktası (Rich tabanlı)
├── app.py                    # Streamlit web arayüzü
├── requirements.txt
└── .env.example
```

---

## Sistem Mimarisi

```
KULLANICI SORUSU
      │
      ▼
┌─────────────────────────────────────────────────┐
│  AŞAMA 1 — SORGU GENİŞLETME                     │
│  Gemini 2.5 Flash → 4 anlamsal varyant üretir   │
│  (orijinal sorguyla paralel çalışır)             │
└──────────────────────┬──────────────────────────┘
                       │  [orijinal + 4 varyant]
                       ▼
┌─────────────────────────────────────────────────┐
│  AŞAMA 2 — HİBRİT RETRIEVAL                     │
│  ┌────────────────┐   ┌───────────────────────┐  │
│  │  BM25 (sparse) │   │  BGE-M3 (dense/FAISS) │  │
│  │  anahtar kelime│   │  anlamsal benzerlik   │  │
│  └───────┬────────┘   └──────────┬────────────┘  │
│          └──────────┬────────────┘               │
│               RRF Birleştirme (k=60)             │
│    dense ağırlık=0.65, BM25 ağırlık=0.35         │
└──────────────────────┬──────────────────────────┘
                       │  [top-40 aday]
                       ▼
┌─────────────────────────────────────────────────┐
│  AŞAMA 3 — CROSS-ENCODER YENİDEN SIRALAMA       │
│  ms-marco-MiniLM her (sorgu, chunk) çiftini      │
│  birlikte encode eder → gerçek alaka puanı       │
└──────────────────────┬──────────────────────────┘
                       │  [top-10]
                       ▼
┌─────────────────────────────────────────────────┐
│  AŞAMA 4 — ÖNCELİK + ÇAKIŞMA ÇÖZÜMÜ            │
│  Kaynak çarpanları (update×1.8, fiyat×1.3)      │
│  Temporal çakışma → en yeni kazanır              │
│  Geçersiz chunk'lar [SUPERSEDED] işaretlenir     │
└──────────────────────┬──────────────────────────┘
                       │  [top-6, çakışma çözülmüş]
                       ▼
┌─────────────────────────────────────────────────┐
│  AŞAMA 5 — BAĞLAM PENCERESİ MONTAJI             │
│  Child → Parent chunk yükseltme                  │
│  Token sayımı → taşarsa sıkıştırma              │
│  Yapılandırılmış [UPDATE_LOG/PRICE/CONTRACT]     │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│  AŞAMA 6 — ÜRETİM + DOĞRULUK GÜVENCESI          │
│  Gemini yanıt üretir (inline kaynak atıfları)   │
│  LLM-as-judge faithfulness skoru hesaplar        │
│  Skor < 0.70 → yanıt bloke edilir               │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
              NİHAİ YANIT
         (cevap + kaynaklar + skorlar)
```

---

## Teknoloji Seçimleri

Bu bölüm "neden bu kütüphaneyi/modeli seçtik, alternatifler neden elendi?"
sorusunu her bileşen için tek tek yanıtlar.

---

### Embedding Modeli: `BAAI/bge-m3`

**Tercih edilen alternatifler ve neden elendiler:**

| Model | Neden Elendi |
|---|---|
| `text-embedding-ada-002` (OpenAI) | API çağrısı gerektirir, internet kesintisinde çalışmaz; Türkçe MIRACL benchmark'ında BGE-M3'ün gerisinde kalır |
| `paraphrase-multilingual-mpnet` | 2021 modeli; BGE-M3'e kıyasla Türkçe recall ~%15 daha düşük |
| `multilingual-e5-large` | İyi ama BGE-M3'ün asimetrik retrieval desteği (query prefix) yok |
| `text-embedding-3-small` (OpenAI) | API bağımlılığı + offline kullanım imkânsız |

**BGE-M3 neden seçildi:**

1. **Türkçe desteği:** 100+ dil üzerinde eğitilmiş; Türkçe için MIRACL benchmark'ında en yüksek puanlı açık kaynaklı model.
2. **Asimetrik retrieval:** `"query: ..."` prefix'i ile soru formunu belge formundan ayırt eder. Kullanıcı "param ne zaman gelir?" diye sorarken belge "iade süresi 30 gündür" der — iki farklı dil kalıbı arasındaki köprü burada kurulur.
3. **Dense + sparse + multi-vector:** Tek model üç retrieval modunu destekler. Biz dense kullanıyoruz; sparse ekseni BM25 ile kapatılıyor (çakışma yok, tamamlayıcılar).
4. **Yerel çalışır:** İnternet kesintisinde, şirkete ait gizli verilerde çalışabilir.
5. **Boyut:** 1024-dim dense vektör — FAISS `IndexFlatIP` için yeterince küçük, hassasiyet için yeterince büyük.

---

### Sparse Retrieval: `rank-bm25` (BM25Okapi)

**Neden BM25, TF-IDF veya diğer sparse yöntemler değil:**

| Yöntem | Sorun |
|---|---|
| TF-IDF (sklearn) | IDF corpus-wide normalize edilir, küçük korpuslarda aşırı hassas; online güncelleme yoktur |
| SPLADE | Transformer tabanlı sparse model, inference maliyeti yüksek; 31 chunk'lık korpus için overkill |
| Elasticsearch BM25 | Servis bağımlılığı yaratır; bu ölçekte tek bir Python süreci yeterli |

**BM25 neden seçildi:**

- Sözleşme numaraları (`Madde 4.1`), paket adları (`Enterprise`), yasal terimler (`fesih`, `tazminat`) gibi tam kelime eşleşmesinin önemli olduğu metinlerde dense embedding'den daha iyi recall sağlar.
- Embedding alanında sayısal değerler (`₺599`, `100GB`) genellikle "eriyor" — BM25 bu sayıları kelime gibi eşleştirir.
- Sıfır ek servis, sıfır GPU, sıfır ağ gecikmesi.

---

### Vektör Deposu: `FAISS IndexFlatIP`

**Neden FAISS, neden IndexFlatIP:**

| Alternatif | Neden Elendi |
|---|---|
| `Chroma` | Kalıcılık için SQLite + dosya yazımı; bu ölçekte gereksiz overhead |
| `Pinecone / Weaviate` | Bulut servis bağımlılığı; offline çalışmaz |
| `FAISS HNSW / IVF` | 31–500 chunk için yaklaşık komşu arama anlamsız; `IndexFlatIP` kesin sonuç verir |
| `hnswlib` | FAISS kadar olgun ekosistem değil; GPU desteği sınırlı |

**`IndexFlatIP` neden:**
- Tam (yaklaşık olmayan) en yakın komşu araması yapar — küçük korpusta hız farkı ihmal edilebilir.
- L2-normalize edilmiş vektörlerde inner product = cosine similarity, ayrı normalizasyon adımı gerekmez.
- FAISS ~50k vektörün altında `IndexFlatIP`'i önerir; `IVF`/`HNSW` yalnızca ötesinde anlam kazanır.

---

### Yeniden Sıralama: `cross-encoder/ms-marco-MiniLM-L-6-v2`

**Neden cross-encoder, neden bu model:**

Bi-encoder (BGE-M3) query ve document'ı **ayrı ayrı** encode eder, sonra cosine mesafesine bakar. Bu hızlıdır ama ince etkileşimleri kaçırır:

```
"iade süresi 30 gün mü?"  ←?→  "iade süresi 7 gündür"
# Bi-encoder: yüksek benzerlik (her ikisinde de "iade süresi" var)
# Cross-encoder: düşük relevance (sayılar farklı, soru yanlış cevaplanıyor)
```

Cross-encoder iki metni **birlikte** encode eder; Transformer'ın full attention mekanizması sayesinde olumsuzlama, sayısal farklılık ve bağlamsal referansları yakalar.

**Neden ms-marco-MiniLM-L-6-v2:**

| Model | Sorun |
|---|---|
| `ms-marco-MiniLM-L-12-v2` | 2x parametre, ~2x yavaş; L-6 için accuracy farkı ~%1 |
| `monoT5-base` | ~250MB, yüklenme süresi uzun; bu ölçekte overkill |
| Türkçe cross-encoder | HuggingFace'de olgun bir Türkçe cross-encoder yok; ms-marco İngilizce eğitimli ama soru-pasaj ilişkisi dil-transferable |

**İki aşamalı retrieval neden:**
- Cross-encoder tüm korpusa (31+ chunk) uygulanırsa: O(n) inference, her sorguda yavaş.
- Bi-encoder ile önce 40 aday → cross-encoder bu 40'ı sıralar: O(40) inference, kabul edilebilir hız.
- Efektif maliyet: cross-encoder'ın doğruluğu, bi-encoder'ın hızıyla birleşir.

---

### LLM: `Gemini 2.5 Flash`

**Neden Gemini 2.5 Flash, diğer modeller neden elendi:**

| Model | Neden Elendi |
|---|---|
| `GPT-4o` | OpenAI API; ücretsiz tier yok, farklı SDK |
| `Claude Sonnet/Haiku` | Anthropic API; bu projede Gemini key'i tercih edildi |
| `Llama 3 / Mistral` (lokal) | Lokal inference için GPU gerekir; CPU'da 10+ dk/sorgu |
| `Gemini 1.5 Flash` | Eski nesil; 2.5 Flash daha iyi instruction following ve JSON üretimi |

**Gemini 2.5 Flash neden:**
- Google AI Studio üzerinden ücretsiz tier (dakikada 15 istek, günde 1500 istek).
- 1M token bağlam penceresi — büyük doküman setleri için ölçeklenebilir.
- `response_mime_type="application/json"` desteği — evaluator ve query expansion JSON çıktısı için güvenilir.
- `system_instruction` desteği — RAG sistem promptu temiz bir şekilde ayrılıyor.

---

### Konfigürasyon: `pydantic-settings`

**Neden pydantic-settings:**

- `.env` dosyası, ortam değişkenleri ve varsayılan değerleri tek bir `Settings` sınıfında birleştirir.
- Tip doğrulaması: `embed_batch_size: int = 32` — string gelirse otomatik int'e çevrilir, geçersizse başlamadan hata verir.
- `Path` tipi doğrudan desteklenir — `settings.data_dir / "sozlesme.txt"` hemen kullanılabilir.
- Alternatif `python-decouple` veya düz `os.getenv` ile karşılaştırıldığında: IDE autocomplete, tip güvenliği, validation built-in.

---

### Önbellek: `diskcache`

**Neden diskcache, Redis veya in-memory dict değil:**

| Alternatif | Sorun |
|---|---|
| `dict` (in-memory) | Process restart'ta sıfırlanır; Streamlit her etkileşimde modülü yeniden yükler |
| `Redis` | Ayrı servis başlatmak gerekir; bu ölçekte over-engineering |
| `shelve` / `pickle` | Thread-safe değil; TTL desteği yok |
| `joblib.Memory` | ML sonuçları için tasarlanmış; TTL ve key hash yönetimi manuel |

**diskcache neden:**
- SQLite üzerinde çalışır, process restart'ta kalıcıdır.
- Thread-safe ve multiprocess-safe.
- `expire=ttl` ile TTL built-in.
- Key olarak SHA-256 hash kullanılır — büyük/küçük harf ve boşluk farkları aynı cache entry'yi paylaşır.

---

### Dosya İzleme: `watchdog`

**Neden watchdog:**
- OS-native file system event'lerini (inotify/kqueue/ReadDirectoryChangesW) kullanır — polling yapmaz, CPU tüketimi sıfır.
- Thread'de çalışır, ana event loop'u bloke etmez.
- `--watch` modunda veri dosyası değişince index otomatik yeniden oluşturulur, cache temizlenir.

---

### Test: `pytest`

**Neden pytest, unittest değil:**
- Fixture sistemi (`@pytest.fixture`) bağımlılıkları temiz inject eder.
- `unittest.mock.patch` pytest ile de çalışır — ayrı framework gerekmez.
- ML modelleri mock'lanarak testler GPU/internet gerektirmeden çalışır.
- `pytest-asyncio` async testler için hazır.

---

## Mimari Kararlar

### 1. Tablo Verilerini (CSV) Nasıl Vektörize Ettik?

**Problem:** CSV satırlarını naif şekilde bölmek anlam kaybına yol açar.

```
# YANLIŞ yaklaşım — chunk'lar arası bağlam kesilir:
chunk_1: "paket_adi,aylik_fiyat_tl,yillik..."
chunk_2: "Pro,599,5990,25,100..."
# "Pro" ile "599" farklı chunk'ta → retrieval kaçırıyor
```

**Çözüm — Atomic row serialization (`loaders.py`):**

Her CSV satırı tek bir okunabilir Document'a dönüştürülür:

```
Paket: Pro | Aylık: ₺599 | Yıllık: ₺5990 | Kullanıcı: 25 |
Depolama: 100GB | API/Ay: 100000 | Destek: E-posta+Telefon (7/24) |
SLA: %99.9 | Özellikler: Gelişmiş analitik, API erişimi...
```

Bu sayede:
- BGE-M3 tüm satırın bütüncül anlamını (paket kimliği + fiyat + özellikler) tek vektörde yakalar.
- BM25 `₺599`, `100GB`, `7/24` gibi sayısal token'lara keyword eşleşmesi yapar.
- `chunker.py` bu belgeleri atomik işler: `parent == child == tam satır`, hiçbir zaman bölünmez.

**Neden text şablonu, JSON değil:**
`{"paket": "Pro", "fiyat": 599}` formatı embedding'de zayıftır; doğal dil cümleleri transformer'ların pre-training dağılımına daha yakındır, daha anlamlı vektörler üretir.

---

### 2. Hiyerarşik Parent-Child Parçalama

**Problem:** Precision vs. recall ikilemi.

```
Küçük chunk (200 tok)  → Hassas eşleşme, ama LLM için bağlam yetersiz
Büyük chunk (600 tok)  → Zengin bağlam, ama retrieval gürültüsü artar
```

**Çözüm — İki katmanlı indeks (`chunker.py`):**

```
Retrieval:  child chunks (200 tok)  ──→ FAISS + BM25
                                           ↓ retrieval sonrası
Üretim:     parent chunks (600 tok) ←── context_builder._get_parent_content()
```

Akademik literatürde "small-to-big retrieval" veya "parent document retrieval" olarak bilinir (LlamaIndex'in de benimsediği yaklaşım).

**Somut fayda:** Madde 4.1'in yalnızca "iade süresi 14 gün" cümlesini içeren child chunk'ı retrieval'da hassas eşleşme sağlar; ama LLM'e gönderilen parent, maddenin geri kalanını (iptal prosedürü, para iadesi adımları) da içerir — yarım cümle veya kesilmiş referans yoktur.

---

### 3. RRF (Reciprocal Rank Fusion) ile Hibrit Birleştirme

**Problem:** BM25 ve dense retrieval farklı score ölçeklerinde çalışır.

```
BM25 score:  12.4, 8.7, 5.1 ...     (log-odds, teorik üst sınırı yok)
Dense score:  0.91, 0.87, 0.83 ...  (cosine, [-1, 1] aralığı)
```

Basit toplama veya ortalama almak için önce normalize etmek gerekir — ama nasıl normalize edileceği belirsizdir ve korpusa göre değişir.

**Çözüm — RRF (`retriever.py`):**

```python
score(doc) = Σ_query Σ_method  weight / (k + rank(doc))
```

RRF rank'ları kullanır, score'ları değil. `k=60` dampening faktörü üst sıraların etkisini sınırlar. Normalizasyon gerekmez, iki farklı score uzayı sorun olmaz.

**Multi-query RRF:** Her sorgu varyantı (orijinal + 4 genişletilmiş) ayrı ayrı retrieval yapar, sonuçlar tek RRF'de birleştirilir. Birden fazla varyantın getirdiği belge daha yüksek score alır — bu, recall'u dramatik biçimde artırır.

---

### 4. Sorgu Genişletme ile Vocabulary Gap Çözümü

**Problem:** Kullanıcı dili ≠ belge dili.

```
Kullanıcı: "param ne zaman gelir"
Belge:     "iade süresi 30 gündür"  (Madde 4.2)
```

Dense embedding bu köprüyü kısmen kurar ama Türkçe hukuki terminoloji için yeterli değildir.

**Çözüm — LLM-tabanlı query expansion (`query_expander.py`):**

Gemini 4 varyant üretir, bunlar paralel olarak retrieval'da kullanılır:

```
Orijinal:  "Pro paketimi iptal edersem param ne zaman gelir?"
Varyant 1: "Pro paket iptali iade süresi gün"
Varyant 2: "Pro abonelik fesih para iadesi Madde 4"
Varyant 3: "Pro sözleşme iptali tazminat koşulları"
Varyant 4: "Pro paket iptal ücret geri ödeme süresi"
```

**Neden 4 varyant:** Deneysel olarak 3–5 varyantın recall kazancını maksimize ettiği, 5 ötesinde azalan getiri olduğu gözlemlenmiştir. 4 varyant + 1 orijinal = 5 paralel retrieval çağrısı kabul edilebilir latency'de kalır.

---

### 5. Temporal Çakışma Çözümü

**Problem:** Aynı konu için farklı tarihli kaynaklar çelişiyor.

```
sozlesme.txt  Madde 4.1:  "iade süresi 14 gün"  (2023-01-15)
guncellemeler.json UPD-001: "21 güne çıkarıldı" (2024-01-10)
guncellemeler.json UPD-007: "Basic için 7 güne indirildi" (2024-08-01)
```

LLM hangi değeri kullanmalı? Bağlamda üçü de varsa "14 gün mü, 21 gün mü, 7 gün mü?" sorusu belirsizleşir.

**Çözüm — Deterministik çakışma algoritması (`reranker.py`):**

```python
# 1. Update kayıtlarını (madde, paket) anahtarıyla grupla
# 2. Her grup için tarihe göre sırala → en yeni kazanır
# 3. Eski kaynak chunk'larına is_superseded=True ekle
# 4. Bağlam metnine ⚠️ [SUPERSEDED by UPD-007] etiketi ekle
# 5. LLM sistem promptu: "SUPERSEDED içeriği birincil yanıt olarak kullanma"
```

**Paket kapsamı:** Pro'ya özgü UPD-002 yalnızca Pro chunk'larını etkiler. Basic ve Enterprise'ın Madde 4.1 chunk'ları bu update'den etkilenmez — çünkü anahtar `(madde="4.1", paket="Pro")`, `(madde="4.1", paket="Basic")` ile eşleşmez.

**Neden kod içinde çözüm, LLM'e bırakmak yerine:**
LLM'e "en yeni kaynağı kullan" demek yeterli değildir — LLM tarih karşılaştırmasında yanılabilir. Deterministik kod guardrail olarak çalışır; LLM'in bu kararı vermesi gerekmez.

---

### 6. Faithfulness Guardrail

**Problem:** RAG sistemleri bile halüsinasyon yapabilir — bağlamda olmayan bilgiyi "güvenle" uydurmak.

**Çözüm — LLM-as-judge (`evaluator.py` + `guardrail.py`):**

```
Üretilen yanıt → Gemini'ye gönderi:
  "Bu iddialar bağlamda var mı? Faithfulness skoru ver."
  
Skor < 0.70 → Yanıt bloke, kullanıcıya ret mesajı
Skor ≥ 0.70 ama uyarı varsa → Yanıt gösterilir + ⚠️ not
```

**Neden %70 eşiği:** %100 gerektirmek çok kısıtlayıcıdır (evaluator da mükemmel değildir); %50 ise halüsinasyona karşı yetersiz koruma sağlar. %70, "iddialar büyük çoğunluğu bağlamla örtüşüyor" anlamına gelir.

**Fail-open tasarımı:** Evaluator API hatası verirse sistem geçer (block etmez). Reasoning: kullanıcıyı engellemek, muhtemel halüsinasyondan daha kötü bir kullanıcı deneyimidir; üstelik evaluator hatası RAG sisteminin kendi hatasından bağımsızdır.

---

### 7. Token-Bilinçli Bağlam Montajı

**Problem:** 6 chunk × 600 token = 3600 token; üst limit 6000. Ama 10 chunk olsaydı?

**Çözüm — Kaynak öncelikli overflow stratejisi (`context_builder.py`):**

```
Token bütçesi aşılırsa:
  1. UPDATE_LOG bölümleri → hiçbir zaman kısaltma
  2. PRICE_TABLE bölümleri → hiçbir zaman kısaltma
  3. CONTRACT bölümleri → kalan bütçeye sığmak için kısalt
```

Güncelleme logları ve fiyat tabloları genellikle kısa ve kritiktir; sözleşme metni daha uzun ama zaten update ile geçersiz kılınmış olabilir. Bu hiyerarşi, token baskısı altında bile en önemli bilginin LLM'e ulaşmasını garanti eder.

**Bölüm sıralaması:**
```
[UPDATE_LOG] → [PRICE_TABLE] → [CONTRACT]
```
LLM bağlamın başını daha dikkatli okur (attention azalan pattern). En güncel, en yetkili bilgi başa yerleştirilir.

---

## Veri Çakışma Senaryoları

`guncellemeler.json` içindeki 8 kayıt kasıtlı çakışmalar barındırır:

| Çakışma | Kayıtlar | Kazanan | Neden |
|---|---|---|---|
| Basic Madde 4.1 iade süresi | UPD-001 (21g, 2024-01-10) vs UPD-007 (7g, 2024-08-01) | UPD-007 | Daha yeni tarih |
| Pro Madde 4.1 iade süresi | UPD-001 (21g, global) vs UPD-002 (30g, Pro-özel, 2024-06-01) | UPD-002 | Daha yeni + paket-özel |
| Basic fiyatı | CSV orijinal (149 TL) vs UPD-003 (199 TL, 2024-03-15) | UPD-003 | Güncel update |
| Enterprise SLA | sozlesme %99.9 vs UPD-006 (%99.95, 2024-07-15) | UPD-006 | Daha yeni update |

Bu senaryolar conflict detection algoritmasının doğruluğunu test eder:
```bash
python main.py --query "Basic paket iade süresi kaç gün?"
# Beklenen: "7 gün [UPD-007]", eski değerler [SUPERSEDED] olarak görünür
```

---

## Ortam Değişkenleri

| Değişken | Zorunlu | Varsayılan | Açıklama |
|---|---|---|---|
| `GOOGLE_API_KEY` | ✓ | — | Gemini API anahtarı |
| `EMBED_MODEL` | — | `BAAI/bge-m3` | HuggingFace embedding modeli |
| `EMBED_DEVICE` | — | `cpu` | `cpu` veya `cuda` |
| `GEMINI_MODEL` | — | `models/gemini-2.5-flash` | Kullanılacak Gemini modeli |
| `LOG_LEVEL` | — | `INFO` | `DEBUG/INFO/WARNING/ERROR` |
| `FAITHFULNESS_THRESHOLD` | — | `0.70` | Guardrail eşiği (0.0–1.0) |
| `MAX_CONTEXT_TOKENS` | — | `6000` | LLM'e gönderilen bağlam token limiti |
| `QUERY_EXPANSION_ENABLED` | — | `true` | `false` ile expansion devre dışı |
| `TOP_K_RETRIEVAL` | — | `40` | Retrieval'dan alınan aday sayısı |
| `TOP_K_FINAL` | — | `6` | LLM'e gönderilen final chunk sayısı |

---

## Bilinen Sınırlamalar

- **Query expansion / evaluator JSON kararsızlığı:** Gemini 2.5 Flash, `system_instruction` ile `response_mime_type="application/json"` birlikte kullanıldığında zaman zaman geçersiz JSON üretebilir. Bu durumda query expansion sessizce devre dışı kalır (orijinal sorgu kullanılır), evaluator "fail open" davranışı sergiler. Üretim ortamı için Gemini Pro veya retry mekanizması önerilir.
- **Latency:** BGE-M3 CPU modunda bir sorgu ~30–45 saniye sürebilir. GPU ile bu süre ~3–5 saniyeye düşer.
- **Küçük korpus:** 31 chunk için `IndexFlatIP` idealdir; corpus 50k+ chunk'a ulaştığında `IndexIVFFlat` veya `IndexHNSWFlat`'a geçiş önerilir.

---

## Performans İpuçları

- **GPU:** `EMBED_DEVICE=cuda` ve `faiss-gpu` ile embedding ~5× hızlanır
- **Index cache:** `--rebuild-index` yalnızca veri değiştiğinde çalıştırın; sistem MD5 checksum ile tazeliği otomatik tespit eder
- **Sorgu genişletme kapatma:** Test/geliştirme için `.env` içine `QUERY_EXPANSION_ENABLED=false` ekleyerek ~2–3 sn tasarruf edilir
- **Model seçimi:** `GEMINI_MODEL=models/gemini-2.0-flash` ile maliyet/latency düşürülebilir; kalite farkı bu ölçekte minimal
