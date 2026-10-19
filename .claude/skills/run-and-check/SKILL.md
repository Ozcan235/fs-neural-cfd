---
name: run-and-check
description: OpenCFD case'inde simpleFoam'u (WSL/OpenFOAM v2606) çalıştırır, log'dan residual'ları ve Cl/Cd kararlılığını okuyarak yakınsamayı değerlendirir; ıraksarsa sebebini raporlar. Kullanıcı simülasyonu çalıştırmak, yakınsamayı kontrol etmek veya bir çözümün neden ıraksadığını anlamak istediğinde kullan.
argument-hint: "aoa=<deg> h=<h/c> flap=<deg> [np=6] [check-only]"
---

# /run-and-check

Argümanlar: `$ARGUMENTS`

## Ne yapar

`pipeline/run_and_check.py`:
1. `np > 1` ise `decomposePar` → `mpirun simpleFoam -parallel` → `reconstructPar -latestTime`, değilse seri `simpleFoam`. Log: `runs/<case_id>/log.simpleFoam`
2. Log'u iterasyon iterasyon ayrıştırır (Ux, Uy, p, k, omega initial residual'ları, bounding uyarıları, süreklilik hatası, FOAM FATAL / NaN)
3. `postProcessing/forceCoeffs1` içinden son `CONV_WINDOW` (300) iterasyonda Cl/Cd'nin bağıl standart sapmasını ve kaymasını hesaplar
4. Durumu belirler ve `runs/<case_id>/run_status.json` dosyasına yazar

| Durum | Anlam | Çıkış kodu |
|---|---|---|
| `converged` | residualControl (1e-5) sağlandı | 0 |
| `converged_coeffs` | residual eşiğine inmedi ama Cl/Cd kararlı (rel_std < %0.5) — veri için kabul | 0 |
| `not_converged` | endTime'a ulaştı, kriterler sağlanmadı | 3 |
| `diverged` | NaN / floating point exception / FOAM FATAL | 4 |

## Adımlar

1. Argümanları çöz; `runs/<case_id>/0` yoksa önce `/setup-case` gerektiğini söyle.
2. Çözüm dakikalar sürer (~44k hücre, 3000 iterasyon; np=6 ile yaklaşık 5–10 dk). Bash aracında **`run_in_background: true`** ile başlat:
   ```
   .venv/Scripts/python.exe -m pipeline.run_and_check --aoa <AOA> --h <H> --flap <FLAP> [--np 6]
   ```
   Beklerken kullanıcıya ilerlemeyi göstermek istersen `log.simpleFoam` içindeki `^Time =` satırlarını say veya `postProcessing/forceCoeffs1/0/coefficient.dat` son satırlarına bak. Polling yapma; bitince bildirim gelir.
3. Kullanıcı yalnızca analiz isterse (`check-only`, "sadece kontrol et") çözücüyü çalıştırmadan:
   ```
   .venv/Scripts/python.exe -m pipeline.run_and_check --aoa <AOA> --h <H> --flap <FLAP> --no-run
   ```
4. Raporla: durum, iterasyon sayısı, son residual'lar, son pencerede Cl/Cd ortalaması ± std, ve betiğin listelediği sebepler.
5. Iraksama/yakınsamama durumunda sebebe göre somut öneri ver:
   - İlk ~50 iterasyonda çöktü → sınır koşulları/başlangıç alanı veya mesh; `log.checkMesh` ve `mesh_quality.json` incele
   - Sonradan çöktü / residual'lar artıyor → `/setup-case ... --robust` ile yeniden kur ve çalıştır
   - Çok sayıda `bounding omega/k` → kötü hücreler (genellikle flap boşluğu veya firar kenarı); `/mesh-wing` ile `gap`/`blThk` ayarla
   - Cl/Cd periyodik salınıyor → yüksek flap açısı veya çok düşük h'de akış ayrılması; steady RANS sınırında. Ortalamayı daha uzun pencereyle almayı (`/extract-coeffs last=1000`) veya case'i veri setinden çıkarmayı öner
   - endTime'a ulaştı ama hâlâ iniyor → `/setup-case ... --end-time 5000` ile yeniden kur ve çalıştır (yeniden kurulum çözümü sıfırdan başlatır)
   Gerekirse `log.simpleFoam` dosyasının ilgili kısmını kendin oku ve kanıtla birlikte açıkla.
6. Başarılıysa sıradaki adım olarak `/extract-coeffs` öner.
