# CCE_NLI


To run pruning: Upload model_best.pth to directory CCE_NLI and snli_1.0 to CCE_NLI/data (both uploaded here: https://drive.google.com/drive/folders/1D9onWZBu8aJWnRABIkyIPmeIYHUb50Ky?usp=sharing)

```
git clone https://github.com/rishikasrinivas/CCE_NLI.git
git switch cofi_merging
mkdir DataLoaders
cd CCE_NLI

# if conda is not installed
curl -LO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  (for linux)
curl -LO https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh (for Mac)

bash Miniconda3-latest-*.sh -u
source ~/.bashrc or source ~/.zshrc

# if conda installed
conda create -n py310 python=3.10 -y
conda activate py310 

pip install -r requirements.txt
./downloads.sh
pip install -U llm2vec


# for llama cofi
wget (llama_student)
./scripts/runCofi.sh llama

```
