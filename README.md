# CCE_NLI


```
git clone https://github.com/rishikasrinivas/CCE_NLI.git
git switch cofi_merging
mkdir DataLoaders
wget https://huggingface.co/ccenli/llama/resolve/main/dataloaders.tar.gz
tar -xvzf dataloaders.tar.gz

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
#load the student model 
wget https://huggingface.co/ccenli/llama/resolve/main/llama_fullprec_student.tar.gz
tar -xvzf llama_fullprec_student.tar.gz

#load the teacher model 
wget https://huggingface.co/ccenli/llama/resolve/main/llama_dense.tar.gz
tar -xvzf llama_dense.tar.gz -C ../

./scripts/runCofi.sh llama <starting sparsity> <device number> 
#device number = 0 if on 1st gpu, 1 if on 2nd, etc
#starting sparsity options are 0.25, 0.4375, 0.57812, 0.68359, 0.7627. if it crashes specify which sparsity to restart from

```
