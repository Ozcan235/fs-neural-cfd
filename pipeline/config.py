"""Pipeline için ortak ayarlar: yollar, OpenFOAM ortamı, akış koşulları, kalite eşikleri.

Varsayılanları değiştirmek için bu dosyayı düzenleyin; skill'ler ve betikler buradan okur.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GEO_FILE = ROOT / "geometry" / "wing.geo"
TEMPLATE_CASE = ROOT / "templates" / "case"
RUNS_DIR = ROOT / "runs"
DATA_DIR = ROOT / "data"
DATASET_CSV = DATA_DIR / "dataset.csv"

# OpenFOAM WSL içinde çalışır
WSL_DISTRO = "Ubuntu-24.04"
FOAM_BASHRC = "/usr/lib/openfoam/openfoam2606/etc/bashrc"

# Geometri (wing.geo içindeki DefineConstant ile aynı olmalı; mesh'e -setnumber ile aktarılır)
CHORD = 0.25        # ana eleman veteri [m] (referans uzunluk)
SPAN = 0.01         # ekstrüzyon kalınlığı [m]; Aref = CHORD * SPAN

# Akış koşulları
U_INF = 20.0        # serbest akış hızı [m/s]
NU = 1.5e-5         # kinematik viskozite [m^2/s]
RHO_INF = 1.225     # yoğunluk [kg/m^3] (kinematik basınçta katsayıları etkilemez)
TURB_INTENSITY = 0.005
TURB_VISC_RATIO = 10.0

# Çözücü
END_TIME = 3000     # simpleFoam maksimum iterasyon
WRITE_INTERVAL = 500
N_PROCS = 1         # >1 ise decomposePar + mpirun

# Parametre aralıkları (sweep ve doğrulama için)
PARAM_RANGES = {
    "aoa": (0.0, 12.0),     # derece
    "h": (0.05, 0.50),      # h/c
    "flap": (5.0, 40.0),    # derece
}

# checkMesh kalite eşikleri
MESH_LIMITS = {
    "max_non_orthogonality": 70.0,
    "max_skewness": 4.0,
    "max_aspect_ratio": 1000.0,
    "min_determinant": 0.001,
}

# Yakınsama kriterleri
CONV_RESIDUAL = 1e-5          # son residual'lar (p, U, k, omega) bu değerin altında
CONV_WINDOW = 300             # Cl/Cd kararlılığı için son N iterasyon
CONV_COEFF_REL_STD = 0.005    # pencere içinde std/|ortalama| bu değerin altında

# extract-coeffs varsayılan ortalama penceresi
AVG_LAST_N = 300


def case_id(aoa: float, h: float, flap: float) -> str:
    """Parametrelerden deterministik klasör adı üretir."""
    return f"a{aoa:+06.2f}_h{h:.3f}_f{flap:+06.2f}".replace("+", "p").replace("-", "m")


def geometry_id(h: float, flap: float) -> str:
    """Konfigürasyon kimliği (h/c + flap). Aynı konfigürasyonun farklı AoA'ları aynı gruptadır;
    surrogate eğitiminde train/test ayrımı bu kimliğe göre yapılır."""
    return f"h{h:.3f}_f{flap:+06.2f}".replace("+", "p").replace("-", "m")
