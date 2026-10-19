---
name: extract-coeffs
description: OpenCFD case'inin postProcessing/forceCoeffs1 çıktısından son N iterasyonun ortalama Cl ve Cd değerini (± std) hesaplayıp veri seti CSV'sine (data/dataset.csv) satır ekler veya günceller. Kullanıcı katsayıları çıkarmak, sonucu CSV'ye kaydetmek veya Cl/Cd değerini öğrenmek istediğinde kullan.
argument-hint: "aoa=<deg> h=<h/c> flap=<deg> [last=300] [csv=data/dataset.csv]"
---

# /extract-coeffs

Argümanlar: `$ARGUMENTS`

## Ne yapar

`pipeline/extract_coeffs.py`:
- `postProcessing/forceCoeffs1/*/coefficient.dat` dosyalarını başlık satırındaki sütun adlarına göre okur (yeniden başlatılan çözümlerde birden fazla zaman klasörünü birleştirir)
- Son N iterasyonun Cl, Cd ortalamasını ve standart sapmasını hesaplar
- `params.json`, `case_setup.json`, `run_status.json`, `mesh_quality.json` dosyalarından meta veriyi ekler
- CSV'ye yazar; aynı `case_id` varsa satırı **günceller** (yinelenen satır olmaz)

CSV sütunları: `case_id, geometry_id, aoa_deg, h_over_c, flap_deg, cl, cd, cl_std, cd_std, l_over_d, n_avg, iterations, status, cells, u_inf, re_c, timestamp`

## Adımlar

1. Argümanları çöz (`aoa`, `h`, `flap` veya `--case`). `last` verilmezse 300 kullanılır.
2. Çalıştır:
   ```
   .venv/Scripts/python.exe -m pipeline.extract_coeffs --aoa <AOA> --h <H> --flap <FLAP> [--last 300] [--csv data/dataset.csv]
   ```
3. Çıkış kodları:
   - **0**: Cl, Cd, L/D ve std değerlerini raporla. `cl_std/|cl|` %0.5'ten büyükse sonucun güvenilirliğinin düşük olduğunu belirt.
   - **3**: `run_status.json` durumu `diverged` veya `not_converged`, satır yazılmadı. Kullanıcıya sor; açıkça isterse `--allow-unconverged` ile yaz.
   - **1**: Dosya bulunamadı (çözüm yok veya `/run-and-check` çalıştırılmamış).
4. `run_status.json` yoksa durum `unchecked` olarak yazılır; bu durumda önce `/run-and-check ... check-only` önermek iyi olur.

## Notlar

- `geometry_id` = h/c + flap açısı konfigürasyonu (`config.geometry_id`). `/train-surrogate` bu sütuna göre grup bazlı bölme yapar.
- Cl < 0 = bastırma kuvveti (liftDir = +y). Veri setinde işaret değiştirilmez.
- Katsayılar 2D (birim açıklık başına): Aref = c · span, lRef = c.
