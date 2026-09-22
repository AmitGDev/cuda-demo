"""Project-specific CI environment setup.

Installs the CUDA Toolkit. This used to be the `Jimver/cuda-toolkit`
step in action.yml; it moved here because CUDA is a project-specific
dependency, not something every consumer of the shared C++ setup
action needs. Reproduces the 'network' install method on both
Windows and Linux:

- Windows: downloads NVIDIA's small network installer exe and runs
  it silently (this is what `method: network` did).
- Linux: registers NVIDIA's apt repo via the `cuda-keyring` package
  and installs through apt, since Linux's network method never went
  through a downloaded installer binary in the first place.

CUDA_PATH is exported the same way the action did: written to
GITHUB_ENV, with its `bin` directory appended to GITHUB_PATH, so
later steps can call `nvcc` without knowing the install layout.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

# Single source of truth for the CUDA version now that it no longer
# lives as an action.yml `with:` input.
CUDA_VERSION = "13.3.1"

_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def _major_minor(version: str) -> tuple[str, str]:
    match = _VERSION_RE.match(version)
    if match is None:
        raise ValueError(f"Unexpected CUDA version format: {version!r}")
    return match.group(1), match.group(2)


def _append_github_path(entry: str) -> None:
    # Same effect as `echo ... >> $GITHUB_PATH` in the old step: makes
    # nvcc usable in subsequent steps without them knowing where CUDA
    # landed.
    github_path = os.environ.get("GITHUB_PATH")
    if not github_path:
        return
    with open(github_path, "a", encoding="utf-8") as handle:
        handle.write(entry + "\n")


def _append_github_env(name: str, value: str) -> None:
    github_env = os.environ.get("GITHUB_ENV")
    if not github_env:
        return
    with open(github_env, "a", encoding="utf-8") as handle:
        handle.write(f"{name}={value}\n")


def _download(url: str, destination: Path) -> None:
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, destination)


def install_cuda_windows(version: str, temp_dir: Path) -> None:
    """Install CUDA via NVIDIA's network installer.

    `-s` with no component list mirrors the old action's default
    (no `sub-packages` given): it silently installs NVIDIA's default
    component set instead of prompting or requiring a reboot.
    """
    url = (
        f"https://developer.download.nvidia.com/compute/cuda/{version}"
        f"/network_installers/cuda_{version}_windows_network.exe"
    )
    installer = temp_dir / f"cuda_{version}_windows_network.exe"
    _download(url, installer)

    subprocess.run([str(installer), "-s"], check=True)

    major_minor = version.rsplit(".", 1)[0]
    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    cuda_path = Path(program_files) / "NVIDIA GPU Computing Toolkit" / "CUDA" / f"v{major_minor}"

    _append_github_env("CUDA_PATH", str(cuda_path))
    _append_github_path(str(cuda_path / "bin"))


def install_cuda_linux(version: str, temp_dir: Path) -> None:
    """Install CUDA from NVIDIA's apt repo.

    This is the only method that works on Linux in the first place
    (the old action's `local` method never applied here), so there is
    no local-installer fallback to preserve.
    """
    major, minor = _major_minor(version)

    ubuntu_version = subprocess.run(
        ["lsb_release", "-sr"], check=True, capture_output=True, text=True
    ).stdout.strip()
    ubuntu_version_no_dot = ubuntu_version.replace(".", "")

    # NVIDIA publishes a separate repo tree for server-class arm64
    # (sbsa) rather than folding it into the generic arm64 tree.
    machine = platform.machine().lower()
    arch = "sbsa" if machine in {"aarch64", "arm64"} else "x86_64"

    https_repo_base = (
        f"https://developer.download.nvidia.com/compute/cuda/repos/"
        f"ubuntu{ubuntu_version_no_dot}/{arch}"
    )
    # The apt source line itself is published over plain http; only
    # the keyring and pin downloads use https.
    http_repo_base = (
        f"http://developer.download.nvidia.com/compute/cuda/repos/"
        f"ubuntu{ubuntu_version_no_dot}/{arch}"
    )

    keyring_deb = temp_dir / "cuda-keyring.deb"
    _download(f"{https_repo_base}/cuda-keyring_1.1-1_all.deb", keyring_deb)
    subprocess.run(["sudo", "dpkg", "-i", str(keyring_deb)], check=True)

    pin_filename = f"cuda-ubuntu{ubuntu_version_no_dot}.pin"
    pin_file = temp_dir / pin_filename
    _download(f"{https_repo_base}/{pin_filename}", pin_file)
    subprocess.run(
        ["sudo", "mv", str(pin_file), "/etc/apt/preferences.d/cuda-repository-pin-600"],
        check=True,
    )
    subprocess.run(
        ["sudo", "add-apt-repository", f"deb {http_repo_base}/ /"],
        check=True,
    )
    subprocess.run(["sudo", "apt-get", "update"], check=True)

    # No sub-packages were configured on the old step, so install the
    # full "cuda" meta-package rather than filtering to a subset.
    package_name = f"cuda-{major}-{minor}"
    subprocess.run(["sudo", "apt-get", "-y", "install", package_name], check=True)

    cuda_path = Path(f"/usr/local/cuda-{major}.{minor}")
    _append_github_env("CUDA_PATH", str(cuda_path))
    _append_github_path(str(cuda_path / "bin"))


def main() -> int:
    temp_dir = Path(os.environ.get("RUNNER_TEMP", "/tmp"))
    system = platform.system()

    if system == "Windows":
        install_cuda_windows(CUDA_VERSION, temp_dir)
    elif system == "Linux":
        install_cuda_linux(CUDA_VERSION, temp_dir)
    else:
        raise RuntimeError(f"Unsupported platform for CUDA setup: {system}")

    print(f"CUDA {CUDA_VERSION} installed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
