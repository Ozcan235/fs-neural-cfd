---
name: setup-case
description: OpenCFD şablon case'ini (templates/case) mesh'lenmiş run klasörüne kopyalar; U, p, k, omega, nut sınır koşullarını ve controlDict içindeki forceCoeffs ayarlarını (magUInf, lRef, Aref, yamalar) doldurur. Kullanıcı bir simülasyonu kurmak, sınır koşullarını hazırlamak veya case'i yeniden ayarlamak istediğinde kullan.
argument-hint: "aoa=<deg> h=<h/c> flap=<deg> [u=20] [np=6] [endTime=3000] [robust]"
---

# /setup-case

Argümanlar: `$ARGUMENTS`

## Ne yapar

`pipeline/setup_case.py`, `templates/case/{0,constant,system}` dosyalarındaki `@@...@@` yer tutucularını doldurarak `runs/<case_id>/` içine yazar. Eski zaman klasörlerini, `processor*` ve `postProcessing` klasörlerini temizler; `constant/polyMesh` korunur. Kullanılan değerler `runs/<case_id>/case_setup.json` dosyasına kaydedilir.

| Alan | Değer |
|---|---|
| inlet | U sabit (U∞,0,0); k = 1.5(U∞·I)², ω = k/(ν·νt/ν oranı) |
| outlet | p = 0, U/k/ω inletOutlet |
| ground | hareketli zemin: U = (U∞,0,0), duvar fonksiyonları |
| top | slip |
| wing_main, wing_flap | noSlip, kqRWallFunction, omegaWallFunction, nutUSpaldingWallFunction |
| frontAndBack | empty (2D) |
| forceCoeffs1 | patches (wing_main wing_flap), lRef = c, Aref = c·span, liftDir (0 1 0), dragDir (1 0 0) |

Varsayılanlar `pipeline/config.py` içindedir (U∞ = 20 m/s, ν = 1.5e-5, I = %0.5, νt/ν = 10, c = 0.25 m, span = 0.01 m).

## Adımlar

1. Argümanları çöz (`aoa`, `h`, `flap` zorunlu; ya da `--case runs/<klasör>`). Eksikse sor.
2. Önce mesh'in var olduğunu doğrula: `runs/<case_id>/constant/polyMesh/boundary`. Yoksa kullanıcıya `/mesh-wing` ile başlamayı öner (veya kullanıcı isterse önce onu çalıştır). `case_id` biçimi `config.case_id()` ile üretilir, örn. `ap04.00_h0.200_fp20.00`.
3. Çalıştır:
   ```
   .venv/Scripts/python.exe -m pipeline.setup_case --aoa <AOA> --h <H> --flap <FLAP> [--u 20] [--np 6] [--end-time 3000] [--robust] [--intensity 0.005]
   ```
   - `--np`: paralel çözüm için işlemci sayısı (makinede 12 çekirdek, WSL'de ~6 GB RAM; 4–6 iyi bir değer).
   - `--robust`: düşük relaxation (p 0.3, U 0.5, k/ω 0.4). Iraksayan case'ler için.
4. Betiğin çıktısındaki Re_c, k, ω, Aref ve relaxation modunu kullanıcıya kısaca bildir. Sıradaki adım olarak `/run-and-check` öner.

## Notlar

- Şablonu kalıcı olarak değiştirmek için `templates/case/` altındaki dosyaları düzenle; yeni bir `@@ANAHTAR@@` eklersen `setup_case.build_values()` içine değerini de ekle (eksik anahtar hata verir).
- Cl < 0 bastırma kuvveti demektir (liftDir = +y).
