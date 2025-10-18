#!/bin/bash

sparsities="0.4375 0.57812 0.68359 0.7627"
model_name=bowman #llama #bert
encoder_name=bowman #bert-base-uncased   #knowledgator/Llama-encoder-1.0B #bert-base-uncased 


for sparsity in ${sparsities}
do
    #./run.sh ${model_name} ${encoder_name} ${ckpt} ${sparsity} "llm" "structured_heads+structured_mlp+hidden+layer"
    echo "Running pruning for $sparsity" 
    #./run.sh ${model_name} ${encoder_name} ${sparsity} "structured_heads+structured_mlp+hidden+layer+final_mlp_hidden" #for bert and llama
    ./run.sh ${model_name} ${encoder_name} ${sparsity} "final_mlp_hidden" #for bwoman
done
