"""Windows'tan WSL içindeki OpenFOAM komutlarını çalıştırma yardımcıları."""
import subprocess
from pathlib import Path, PureWindowsPath

from . import config


def to_wsl(path: Path) -> str:
    """C:\\OpenCFD\\runs -> /mnt/c/OpenCFD/runs"""
    p = PureWindowsPath(Path(path).resolve())
    drive = p.drive.rstrip(":").lower()
    return f"/mnt/{drive}/" + "/".join(p.parts[1:])


def run(case_dir: Path, command: str, log_name: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
    """OpenFOAM ortamı yüklenmiş bash içinde case klasöründe komut çalıştırır.

    log_name verilirse çıktı case klasöründe log.<log_name> dosyasına yazılır.
    """
    cd = to_wsl(case_dir)
    script = f"source {config.FOAM_BASHRC} >/dev/null 2>&1 && cd '{cd}' && {command}"
    if log_name:
        script += f" > log.{log_name} 2>&1"
    proc = subprocess.run(
        ["wsl.exe", "-d", config.WSL_DISTRO, "--exec", "bash", "-c", script],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if check and proc.returncode != 0:
        tail = ""
        if log_name and (Path(case_dir) / f"log.{log_name}").exists():
            tail = "\n".join((Path(case_dir) / f"log.{log_name}").read_text(errors="replace").splitlines()[-30:])
        raise RuntimeError(f"'{command}' başarısız (kod {proc.returncode}).\n{proc.stdout}\n{proc.stderr}\n{tail}")
    return proc
