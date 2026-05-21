TASK=SNLI

EX_CATE=CoFi
PRUNING_TYPE=${4}
SPARSITY=${3}
SUFFIX=sparsity${SPARSITY}
DISTILL_LAYER_LOSS_ALPHA=0.1 #0.3
DISTILL_CE_LOSS_ALPHA=0.05
LAYER_DISTILL_VERSION=4 #4
SPARSITY_EPSILON=0.01
model_name=${1}
model_name_or_path=${2}
pruning_iteration=${5}
start=$(date +%s)
bash ./code/cofi/scripts/run_CoFi.sh $TASK $SUFFIX $EX_CATE $PRUNING_TYPE $SPARSITY $DISTILL_LAYER_LOSS_ALPHA $DISTILL_CE_LOSS_ALPHA $LAYER_DISTILL_VERSION $SPARSITY_EPSILON $model_name $model_name_or_path $pruning_iteration 
end=$(date +%s)
elapsed=$(( end - start ))

echo "Elapsed time: $elapsed seconds"
