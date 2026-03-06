#!/bin/bash
#SBATCH --job-name=sample
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -A pnlp
#SBATCH -t 11:00:00


glue_low=(MRPC RTE STSB CoLA)
glue_high=(MNLI QQP QNLI SST2 SNLI)

proj_dir="${10^^}"

code_dir=./code/cofi/
path_to_pretrained="${10^^}/models/pretrained/"
echo "path to pretrained model ${path_to_pretrained}"


# task and data
task_name=$1

# pretrain model



# logging & saving
logging_steps=100
save_steps=0


# train parameters
max_seq_length=128
batch_size=32
learning_rate=2e-5
reg_learning_rate=0.01
epochs=5

# seed
seed=57 #_5 57 seed  _6 42 seed _7 48

#debug
debug=100

# pruning and distillation
pruning_type=$4
target_sparsity=$5
distill_layer_loss_alpha=$6
distill_ce_loss_alpha=$7
distill_temp=4
# 2: fix hidden layers, 3: min distance matching without restriction, 4: min distance matching with restriction
layer_distill_version=${8} 

scheduler_type=linear


# output dir
ex_name_suffix=$2
ex_name=${task_name}_${ex_name_suffix}
ex_cate=$3
iteration=${12}
teacher_model_dir=${proj_dir}/models/CoFi/Run0.25_5
output_dir=${teacher_model_dir}/${iteration}_Pruning_Iter


if [[ " ${glue_low[*]} " =~ ${task_name} ]]; then
    eval_steps=50
    epochs=100
    start_saving_best_epochs=50
    prepruning_finetune_epochs=4
    lagrangian_warmup_epochs=20
fi

if [[ " ${glue_high[*]} " =~ ${task_name} ]]; then
    eval_steps=500
    prepruning_finetune_epochs=1
    lagrangian_warmup_epochs=2
fi

pretrained_pruned_model=None
using_untrained_student=True
# FT after pruning
if [[ $pruning_type == None ]]; then
  pretrained_pruned_model=${9}
  learning_rate=${10}
  scheduler_type=none
  output_dir=$pretrained_pruned_model/FT-lr${learning_rate}
  epochs=20
  batch_size=64
fi


mkdir -p "$output_dir"
python3 ${code_dir}/run.py \
        --model_name ${10}\
        --using_untrained_student ${using_untrained_student} \
        --path_to_pretrained ${path_to_pretrained}\
        --teacher_model_dir ${teacher_model_dir} \
       --data_debug ${debug}\
	   --output_dir ${output_dir} \
	   --logging_steps ${logging_steps} \
	   --task_name ${task_name} \
	   --model_name_or_path ${11} \
	   --ex_name ${ex_name} \
	   --do_train \
	   --do_eval \
	   --max_seq_length ${max_seq_length} \
	   --per_device_train_batch_size ${batch_size} \
	   --per_device_eval_batch_size 32 \
	   --learning_rate ${learning_rate} \
	   --reg_learning_rate ${reg_learning_rate} \
	   --num_train_epochs ${epochs} \
	   --overwrite_output_dir \
	   --save_steps ${save_steps} \
	   --eval_steps ${eval_steps} \
	   --evaluation_strategy steps \
	   --seed ${seed} \
	   --pruning_type ${pruning_type} \
     --pretrained_pruned_model ${pretrained_pruned_model} \
     --target_sparsity $target_sparsity \
     --freeze_embeddings \
     --do_distill \
     --do_layer_distill \
     --distill_ce_loss_alpha $distill_ce_loss_alpha \
     --distill_loss_alpha $distill_layer_loss_alpha \
     --distill_temp $distill_temp \
     --scheduler_type $scheduler_type \
     --layer_distill_version $layer_distill_version \
     --prepruning_finetune_epochs $prepruning_finetune_epochs \
     --lagrangian_warmup_epochs $lagrangian_warmup_epochs 2>&1 | tee ${output_dir}/log.txt
