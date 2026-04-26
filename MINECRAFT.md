# Minecraft Redstone Implementation Plan

## Hedef

Eğitilmiş transformer modelini **pure redstone** devresi olarak Minecraft Java Edition'da çalıştırmak. Inference sırasında oyuncu bir kategori seçer, redstone devresi 64 token üretir ve 16×16 binary pixel art görüntü çıktı verir.

---

## Mevcut Model (Referans)

| Parametre | Değer |
|---|---|
| `d_model` | 64 |
| `n_heads` | 4 |
| `n_layers` | 4 |
| `d_ff` | 128 |
| `vocab_size` | 16 |
| `seq_len` | 65 (1 kategori + 64 image token) |
| Aktivasyon | ReLU |
| Parametre sayısı | ~157K |
| Int8 ağırlık boyutu | ~157KB |
| Çıktı | Binary (0 veya 15) — her piksel ya boş ya dolu |

Greedy argmax decoding — deterministik, her çalıştırmada aynı çıktı.

---

## Inference Adımları (Redstone'a Çevrilecek)

```
1. Kategori index (0-344) → category embedding lookup   [345 × 64 ROM]
2. Token 0 olarak embedding vektörü beslenir
3. 64 kez tekrarla:
   a. Tüm sekansı transformer'dan geçir
      - Embedding + 2D positional encoding
      - 4 × TransformerBlock:
          LayerNorm → CausalAttention → LayerNorm → FFN(ReLU)
      - Final LayerNorm
      - Output head (64 → 16 logit)
   b. Son pozisyonun logitlerinde argmax → next token
   c. Token sekansa eklenir
4. 64 token → 8×8 patch grid → 16×16 binary görüntü
```

---

## Redstone Implementation Araçları

### Geliştirme Ortamı
- **Minecraft Java Edition** — redstone davranışı en tutarlı ve dokümante edilmiş versiyon
- **Mac M1 16GB** — Java Edition native ARM destekli, 4-6GB RAM ayır

### Block Yerleştirme (Elle Değil)
- **Amulet** — Python tabanlı external world editor. Ağırlıkları okuyup blok koordinatlarına çeviren script yazılabilir.
- **WorldEdit mod** — Minecraft içinde schematic import/export
- **Pipeline**: `export.py → int8 weights → Python ROM builder → .schem → WorldEdit import`

### Devre Tasarımı
- **Logisim Evolution** veya **Digital** — logic gate simülatörü. Minecraft'a girmeden önce her komponenti burada tasarla ve test et. Redstone'a 1:1 çevrilebilir.

---

## Redstone Komponentleri

### Yapılması Gereken Devreler

| Komponent | Zorluk | Açıklama |
|---|---|---|
| ROM (weight storage) | Orta | Int8 ağırlıkları okumak için address decoder + data lines |
| 8-bit çarpıcı | Orta | Matris çarpımı için temel birim |
| Akümülatör | Kolay | Matris çarpım sonuçlarını toplar |
| ReLU | Trivial | Negatif değerleri sıfırla |
| Argmax | Kolay | 16 logit arasında en büyüğü bul (comparator zinciri) |
| LayerNorm | **Zor** | Ortalama + varyans + karekök gerektirir |
| Causal Attention | Zor | QKV çarpımları + softmax |
| Softmax | **Çok Zor** | e^x için lookup table gerektirir |

### LayerNorm Alternatifleri
LayerNorm redstone'da en pahalı operasyon. Seçenekler:
1. **Koru** — karekök + bölme devresi gerekir, büyük ama doğru
2. **Kaldır, sıfırdan eğit** — training daha unstable ama devre basitleşir
3. **RMSNorm ile değiştir** — ortalama hesabı yok, sadece RMS + scale. Llama gibi modeller kullanıyor, iyi tradeoff.

---

## Gelecekte Denenebilecekler

### Model Tarafı

| Deneme | Motivasyon | Durum |
|---|---|---|
| Binary tokenizer (vocab=16) | Greedy için 16-way sınıflandırma çok daha kolay | ✅ Tamamlandı |
| Dataset curation | Kategori içi varyansı azalt, greedy'yi güçlendir | Planlandı |
| d_model=128 | Daha iyi model kalitesi, 4x parametre artışı | Planlandı |
| RMSNorm | LayerNorm yerine redstone dostu alternatif | Planlandı |
| LayerNorm'u kaldır | Devre sadeliği için, kalite kaybıyla | Değerlendirilecek |

### Redstone Tarafı

| Deneme | Açıklama |
|---|---|
| Sequential vs parallel | Sequential = küçük devre ama yavaş. Parallel = hızlı ama büyük |
| ROM optimizasyonu | Ağırlıkları kompakt depolamak için encoding stratejisi |
| Clock hızı optimizasyonu | Tick başına işlem sayısını artır |
| Lookup table e^x | Softmax için gerekli, ~6000 blok tahmin |

---

## Hız Tahmini (Sequential Implementasyon)

Redstone clock ~1 tick = 0.1 saniye varsayımıyla:

| d_model | 1 token süresi | Tam görüntü (64 token) |
|---|---|---|
| 64 | ~10 saniye | ~10.7 dakika |
| 128 | ~40 saniye | ~43 dakika |

*Not: Bu çok kaba bir tahmin. Gerçek süre redstone CPU tasarımına göre değişir.*

---

## Öneri: Geliştirme Sırası

1. **Python inference'ı bitir** — greedy çalışıyor mu doğrula
2. **`export.py` yaz** — int8 ağırlıkları dışa aktar
3. **Logisim'de 8-bit çarpıcı tasarla** — temel blok
4. **Logisim'de küçük matris çarpımı test et** — 4×4 örneğiyle başla
5. **Amulet ile ROM builder yaz** — Python'dan Minecraft'a weight transfer
6. **Tek bir TransformerBlock'u redstone'da yap** — 4 katmanı sonra ekle
7. **End-to-end test** — bir kategori için 64 token üret

---

## Referanslar

- Toplulukta redstone'da hesap makinesi ve basit CPU örnekleri mevcut — sıfırdan başlamak yerine mevcut 8-bit ALU tasarımlarını adapte et
- YouTube: "Minecraft redstone CPU", "Minecraft neural network" aramaları ile benzer projeler bulunabilir
