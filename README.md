# Dataset-2026

## Quick Start

1. Install nix package manager:

```bash
sh <(curl --proto '=https' --tlsv1.2 -L https://nixos.org/nix/install) --daemon
```
2. Enable nix flake:

Edit `/etc/nix/nix.conf` and add line: `experimental-features = nix-command flakes`

3. Clone repository and setup nix develop:

```bash
git clone --depth 1 https://github.com/Tao-Boy/dataset-2026.git
cd dataset-2026
nix develop
```

4. Edit `config.py`

5. Run python:

```bash
python generate_dataset.py
```


