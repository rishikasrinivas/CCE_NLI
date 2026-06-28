# CCE_NLI


To run pruning: Upload model_best.pth to directory CCE_NLI and snli_1.0 to CCE_NLI/data (both uploaded here: https://drive.google.com/drive/folders/1D9onWZBu8aJWnRABIkyIPmeIYHUb50Ky?usp=sharing)

```
git clone https://github.com/rishikasrinivas/CCE_NLI.git
git switch cofi_merging
mkdir DataLoaders
cd CCE_NLI
./install_conda_env.sh
pip install -r requirements.txt
./downloads.sh

./scripts/runCoFi llama

```

To run lottery ticket and wanda pruning
./scripts/run_[model].sh

To run CoFi
```
# On bert, llama:
./code/cofi/iter_prune.sh

# On bowman
./code/cofi/iter_prune-bowman.sh
```


To ONLY run explanations
Load BERT folder (https://drive.google.com/drive/folders/1HYT3ZqGOi2c_DsvKN9vVf9ImNv7urxrn?usp=sharing) in CCE_NLI/
```
git clone https://github.com/rishikasrinivas/CCE_NLI.git
git switch cofi_merging
cd CCE_NLI

# use python3.10 either locally or through virtual env (ex:)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh 
bash Miniconda3-latest-*.sh
close shell and reopen
source ~./bashrc
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create -n py310 python=3.10 -y
conda activate py310 

#install dependencies
pip install -r requirements.txt
./downloads.sh

python3 code/analyze.py --directory [directory_of_masks] --model_type [bert, llama, bowman] --pruning_method [lottery_ticket, wanda, CoFi] --filename Run0.25_5
```

