# =========================
# Install Miniconda (Linux/macOS)
# =========================

# Download installer
curl -LO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh 

# For Apple Silicon Mac (M1/M2/M3), use:
# curl -LO https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh

# Run installer
bash Miniconda3-latest-*.sh -u

# Reload shell
source ~/.bashrc
# or
source ~/.zshrc

# Verify install
conda --version


# =========================
# Create Conda Environment
# =========================

# Create environment with Python 3.10
conda create -n myenv python=3.10 -y

# Activate environment
conda activate myenv

