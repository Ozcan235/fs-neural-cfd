---
name: mesh-wing
description: OpenCFD ön kanat için Gmsh .geo parametrelerini (hücum açısı, yerden yükseklik h/c, flap açısı) alır, mesh üretir, gmshToFoam + checkMesh çalıştırır ve kalite eşiklerini kontrol eder. Kullanıcı mesh üretmek, ağ oluşturmak veya mesh kalitesini kontrol etmek istediğinde kullan.
argument-hint: "aoa=<deg> h=<h/c> flap=<deg> [gap=0.025 sWing=0.003 ...]"
---

# /mesh-wing

Argümanlar: `$ARGUMENTS`

## Ne yapar

`pipeline/mesh_wing.py` şunları sırayla yapar:
1. `geometry/wing.geo` dosyasını `-setnumber` ile parametreleyip `.venv` içindeki gmsh ile 3D (tek hücre kalınlığında) mesh üretir → `runs/<case_id>/wing.msh`
2. WSL (Ubuntu-24.04, OpenFOAM v2606) içinde `gmshToFoam` çalıştırır
3. `constant/polyMesh/boundary` içinde tipleri düzeltir: `frontAndBack → empty`, `ground / wing_main / wing_flap → wall`; yedi yamanın da var ve dolu olduğunu doğrular
4. `checkMesh -allGeometry -allTopology` çalıştırır, sonuçları `pipeline/config.py → MESH_LIMITS` eşikleriyle karşılaştırır
5. Sonucu `runs/<case_id>/mesh_quality.json` dosyasına yazar

## Adımlar

1. Argümanları çöz. Kullanıcı `aoa=4 h=0.2 flap=20`, `4 0.2 20` veya doğal dilde ("5 derece, h/c 0.15, flap 25") yazabilir. Üçünden biri eksikse kullanıcıya sor; tahmin etme.
   - `h` her zaman h/c (boyutsuz) olarak yorumlanır. Kullanıcı mm/m verirse veter 0.25 m'ye (`config.CHORD`) bölerek çevir ve bunu söyle.
   - Ek wing.geo sabitleri (`gap`, `ovl`, `cf`, `sWing`, `yFirst`, `blThk`, `Npts` vb.) `--set ad=değer` ile geçilir.
2. Proje kökünden (C:\OpenCFD) çalıştır:
   ```
   .venv/Scripts/python.exe -m pipeline.mesh_wing --aoa <AOA> --h <H> --flap <FLAP> [--set gap=0.025 ...]
   ```
   Parametre `config.PARAM_RANGES` dışındaysa betik reddeder; kullanıcı açıkça isterse `--force` ekle.
3. Çıkış koduna göre raporla:
   - **0**: Mesh geçerli. Hücre sayısını, max non-orthogonality, max skewness, max aspect ratio ve varsa uyarıları kısa bir tabloyla ver. Sıradaki adım olarak `/setup-case` öner.
   - **2**: Kalite eşiği aşıldı. Hangi metriğin aştığını söyle ve olası çözümü öner:
     - yüksek skewness/non-ortho flap boşluğunda → `--set gap=0.03` veya `blThk` küçült
     - aspect ratio → `sWing` küçült veya `yFirst` büyüt
     - checkMesh hatası (negatif hacim, açık hücre) → genellikle profiller kesişiyor; `gap`/`ovl`/flap açısını kontrol et
   - **1**: gmsh veya OpenFOAM hatası; stderr'deki mesajı özetle. `runs/<case_id>/log.gmsh`, `log.gmshToFoam`, `log.checkMesh` dosyalarına bak.
4. "small interpolation weight" ve "small volume ratio" uyarıları firar kenarı fan bölgesinde normaldir; tek başına başarısızlık sayılmaz.

## Notlar

- Geometri: ters çevrilmiş NACA 4412 ana eleman (c = 0.25 m) + %40 veterli flap. Hücum açısı ana elemanın; flap açısı ana elemana göredir. Açılar firar kenarını yukarı kaldırır (pozitif = daha fazla bastırma).
- `h`, zeminden kanadın en alt noktasına olan mesafedir (/c).
- Mesh'i görsel olarak kontrol etmek gerekirse `runs/<case_id>/wing.msh` dosyasını gmsh ile açmayı öner: `.venv/Scripts/gmsh.bat runs/<case_id>/wing.msh`
- Mesh mevcut bir case klasörünü yeniden oluşturursa yalnızca `wing.msh` ve `constant/polyMesh` değişir; önceki çözüm klasörleri `/setup-case` ile temizlenir.
