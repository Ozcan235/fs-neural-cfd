---
name: sweep
description: OpenCFD veri seti üretimi — bir parametre ızgarasını (grid) veya Sobol/LHS örneklemesini döngüyle mesh-wing → setup-case → run-and-check → extract-coeffs adımlarından geçirir, kaldığı yerden devam eder ve sonuçları data/dataset.csv'ye yazar. Kullanıcı açıkça /sweep çağırdığında kullan.
argument-hint: "grid aoa=0:12:4 h=0.1,0.2,0.3 flap=10:30:3 | sobol n=32 aoa-values=0:12:7 | lhs n=100 [np=6] [dry-run]"
disable-model-invocation: true
---

# /sweep

Argümanlar: `$ARGUMENTS`

## Ne yapar

`pipeline/sweep.py` planı üretir (`data/sweep_plan_<method>.csv`), sonra her case için:
1. `mesh_wing` — kalite eşiğini geçemezse case atlanır (`--allow-bad-mesh` ile zorlanabilir)
2. `setup_case` → `run_and_check`; **ıraksarsa bir kez `--robust` relaxation ile yeniden dener**
3. `extract_coeffs` → `data/dataset.csv` (yakınsamayanlar yazılmaz, `--keep-unconverged` hariç)
4. Her adımın sonucu ve süresi `data/sweep_log.csv` dosyasına eklenir

`dataset.csv` içinde durumu `converged`/`converged_coeffs` olan case'ler atlanır; yani yarıda kalan sweep aynı komutla devam ettirilir.

## Adımlar

1. Argümanları komuta çevir:
   - Yöntem: `grid`, `sobol` veya `lhs` (verilmezse sor; veri seti için README Sobol/LHS önerir).
   - grid ekseni: `a:b:n` (n eşit aralıklı nokta) veya virgüllü liste. Verilmeyen eksen `config.PARAM_RANGES` aralığında 3 noktadır.
   - sobol/lhs: `--n`, `--seed`; aralıklar `--aoa 0:12` biçiminde daraltılabilir. Sobol için n'in 2'nin kuvveti olması önerilir.
   - **Surrogate eğitimi için veri üretiliyorsa `--aoa-values` kullan** (ör. `sobol --n 32 --aoa-values 0:12:7`): (h, flap) konfigürasyonları örneklenir, her biri aynı AoA listesiyle taranır. `/train-surrogate` train/test ayrımını `geometry_id` (= h + flap) grubuna göre yapar; `--aoa-values` olmadan her konfigürasyonda tek case olur ve grup bazlı bölme anlamını yitirir. Bu durumda kullanıcıyı uyar.
   - Diğer: `--np`, `--end-time`, `--last`, `--csv`, `--set gap=0.025` (tüm case'lerin geometrisi), `--max-cases`, `--cleanup`.
2. **Önce her zaman dry-run** çalıştır ve planı kullanıcıya göster:
   ```
   .venv/Scripts/python.exe -m pipeline.sweep <method> [seçenekler] --dry-run
   ```
   Toplam case sayısını, kaç tanesinin zaten tamamlandığını ve tahmini süreyi bildir (case başına ~5–10 dk, np=6; ilk case'in gerçek süresi daha iyi tahmin verir). Kullanıcı dry-run istediyse burada dur. Aksi halde, büyük sweep'lerde (> ~10 case) başlatmadan önce onay al.
3. Gerçek çalıştırmayı Bash aracında **`run_in_background: true`** ile başlat (saatler sürebilir):
   ```
   .venv/Scripts/python.exe -m pipeline.sweep <method> [seçenekler]
   ```
   Kullanıcıya log dosyalarını (`data/sweep_log.csv`, arka plan çıktısı) ve kesilirse aynı komutla devam edileceğini söyle. Polling yapma; bitince bildirim gelir.
4. Bittiğinde özetle: başarılı / yakınsamayan / ıraksayan / mesh hatası sayıları, `dataset.csv` satır sayısı. Başarısız case'leri `sweep_log.csv`'den listele ve ortak bir örüntü varsa (ör. hepsi yüksek flap + düşük h) belirt. Tek bir case'i incelemek için `/run-and-check ... check-only` öner.

## Notlar

- Çok uzun bir sweep'i parçalara bölmek için `--max-cases 20` kullanılabilir; sonraki çağrı kalanlardan devam eder.
- Disk: her case ~44k hücre; `purgeWrite 2` son iki zaman klasörünü tutar. `--cleanup` paralel `processor*` klasörlerini siler.
- Parametre aralıkları ve kalite/yakınsama eşikleri: `pipeline/config.py`.
