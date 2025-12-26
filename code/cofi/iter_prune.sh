#!/bin/bash

sparsities="0.0 0.25 0.4375 0.57812 0.68359 0.7627"
model_name=llama #llama #bert
encoder_name=knowledgator/Llama-encoder-1.0B  #knowledgator/Llama-encoder-1.0B #bert-base-uncased   #knowledgator/Llama-encoder-1.0B #bert-base-uncased 

pruning_iter=0
for sparsity in ${sparsities}
do
    #./run.sh ${model_name} ${encoder_name} ${ckpt} ${sparsity} "llm" "structured_heads+structured_mlp+hidden+layer"
    echo "Running pruning for $sparsity" 
    ./code/cofi/run.sh ${model_name} ${encoder_name} ${sparsity} "structured_heads+structured_mlp+hidden+layer+final_mlp_hidden" ${pruning_iter} #for bert and llama
    #./code/cofi/run.sh ${model_name} ${encoder_name} ${sparsity} "final_mlp_hidden" ${pruning_iter}#for bwoman
    ((pruning_iter++))
done
