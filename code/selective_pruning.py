import models
import torch
import snli_eval
import train_utils
import pandas as pd
import csv
NEURONS_TO_PRUNE=[330, 492, 1010]
RANDOM_TO_PRUNE=[325]
def get_model(args, train):

    # ==== BUILD MODEL ====
    model = train_utils.build_model(vocab_size=len(train.stoi), model_type=args.model_type, vocab={'stoi': train.stoi, 'itos': train.itos}, embedding_dim=300, hidden_dim=512, is_cofi=args.pruning_method=='cofi')
   

    
    
    def fill_inputs_with_zs(zs, inputs):
        for key in zs:
            inputs[key] = zs[key]
        return inputs
    return model

def prune_neurons(model, ckpt, neurons_to_prune):
    ckpt = torch.load(ckpt, map_location='cpu')['state_dict']
    for neuron in neurons_to_prune:
        ckpt['mlp.0.weight'][neuron] = torch.zeros_like(ckpt['mlp.0.weight'][neuron])
        
        #print(f"Pruned neuron {neuron}")
    
    #assert (torch.sum(ckpt['mlp.0.weight'][neuron])==0 for neuron in neurons_to_prune)
    model.load_state_dict(ckpt)
    return model


    

def parse_args():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--data",
        default="test.txt",
        help="Data to eval interactively (pairs of sentences); use - for stdin",
    )
   
    parser.add_argument("--ckpt", default="BERT/models/lottery_ticket/Run0.25_3/0_Pruning_Iter/model_best.pth")
    parser.add_argument("--model_type", default="bert", choices=["bowman", "bert", "llama"])
    parser.add_argument("--pruning_method", default="lottery_ticket", choices=["lottery_ticket", "wanda", "cofi"])
    return parser.parse_args()
pruning neurons absed on top 5 foundational concepts

if __name__ == "__main__":
    args = parse_args()
    train,_,dataloaders=train_utils.create_dataloaders(max_data=10000, model_type=args.model_type, pruning_method=args.pruning_method)
    val_loader = dataloaders['val']
    model = get_model(args, train)
    model=  prune_neurons(model, args.ckpt, neurons_to_prune=[])
    model.eval()
    results={}
    original_val_acc = train_utils.run_eval(model, val_loader, args.model_type, args.pruning_method)
    print(f"Original Accuracy (Pruning out Nothing) | Accuracy = {original_val_acc}")
    for c in range(1,4):
        for unit in pd.read_csv(f'BERT/exp/lottery_ticket/Run0.25_3/Expls/68.359%Pruned/Cluster{c}IOUS1024N.csv')['unit']:
            is_found = unit not in [330, 492, 1010, 308]

            specifically_pruned_model = prune_neurons(model, args.ckpt, neurons_to_prune=[unit])
            specifically_pruned_model.eval()
            specifically_pruned_model_val_acc = train_utils.run_eval(specifically_pruned_model, val_loader, args.model_type, args.pruning_method)

            '''random_pruned_model = prune_neurons(model, args.ckpt, neurons_to_prune=RANDOM_TO_PRUNE)
            random_pruned_model.eval()
            random_pruned_model_val_acc = train_utils.run_eval(random_pruned_model, val_loader, args.model_type, args.pruning_method)'''

            print(f"Targetted pruning (Pruning out {unit}) | Accuracy = {specifically_pruned_model_val_acc}")
            #print(f"Random pruning (Pruning out {RANDOM_TO_PRUNE}) | Accuracy = {random_pruned_model_val_acc}")
            results[unit] = {
                "isFoundational": is_found,
                "acc": specifically_pruned_model_val_acc,
                "orig_acc": original_val_acc,
                'change_in_acc': round(specifically_pruned_model_val_acc-original_val_acc,3)
            }
    
    CSV_OUTPUT = './code/lottery_ticket_foundational_vs_acc_bert.csv'
    with open(CSV_OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["unit", "isFoundational", "acc", "orig_acc", 'change_in_acc'])

        for unit, data in sorted(results.items()):
            writer.writerow([
                unit,
                data["isFoundational"],
                data["acc"],
                data["orig_acc"],
                data['change_in_acc']
            ])

print(f"\nCSV written to: {CSV_OUTPUT}")
    
    
    
"""
Bowman: 377,595 are the only 2 neurons learning concepts (in theory as per our alg) in the 3rd cluster and they explain foundational cps so removing these neuron should drop the accuracy, and it does so slightly
        - that could mean that bowman doesnt depend so much on these high activations 
        
386,683 are the only 2 neurons that don't explain any foundational concepts (in C2 but do in C1) so removing these as per the hypothesis shouldn't change the accuracy, and it doesn't
    -if you remove only 683 u see an increase in performance, but only 386 decreased performance and both 386 & 683 no change in performance
    - 683 has 1 foundational cp in C1 and 386 has 1 in C1 so 
    - 683 fires 1376 samples in C1 386 fires for 3284 samples in C1 
            so removing 683 helping maybe means that it wasnt positively contrivuting 
    
If you remove 750, 26 which are 2 neurons explaining foundational concepts in the 2nd cluster, there is a slight performance drop
        - this is keeping all other neurons so these 2 neurons made a diff but the ones tht don't explain foundational doesn't 
        
Removing 0,12 which explain foundationals in Cluster 1 had no effect on performance, but 815,213 slightly does
        - 0,12 cover a lot of samples, but 815 and 213 don't yet that latter pair impacted accuracy. These both are at Cluster 1 (213 is activated at C1 for the most samples of the 4 neurons)
        - 750 and 377 cover 90% of the dataset and removing these causes slight drop in accuracy
so acc will drop if you remove the neurons that fire a lot (regardless of the cluster) irresective of

Removing neurons that fire a lot hurts acc (regardless of the cluster they fire at) but removing neurons that dont fire much doesnt (147 -low reach vs 172-high reach in C1)
also removing neurons that not expl foundational at a given cluster helps the acc r hurts dependng on the reach of that neuron (683 had low reach in C2 but 386 had high even tho the foundational wasnt there)


but if i remove (at 67%) the 3 non foundational cps neurons fro C1, acc dropps ([330, 492, 1010]) and thses 3 arent expld anywhere else in that iter so that means that 
    its mot that foundational concepts are the only contributors bc otherwise remving these wuld have no effect
    and removing all C3 neurons hurt the acc a lot (71.7 to 71.1) so means it does depend on these?
 
 
overall theres no significant inc/decresase in the performanve baed on pruning out neurons. meaning if you prune out all C3 neurons you'll see some change
but in terms of remoing neurons that don't define foundational I'm seeing that theres no fluctuation from if u were to remive indiv neurons at random
but if you remove non foundatonal neurons some of them result in performance improvement

like if u remove all neurons that are expainabe at a Cluster 1, the performacne drops. meaning that even tho these neurins arent explainable at the 3rd cluster they still are crtiical to the underlying knowledhe  
but if u remove C1 expl neurons that are not expl at C3, performance drop is not as bad as if you remove c1 that are expl at c3
if u remove localized expl nurons (neurons only expld in c1 or c2 or c3) perf drop not significant but if u remove cross clusteer neurons it is (c1,c2, or c1,c2,c3 or c2,c3)

so theres light arg that the model can be learning founational and thats what helps it retain acc (bc if u prune out some non foundational neurons you see a perf increase) but 
overall there isnt a significant difference.  like removing an arbitrary neuron that does expl foundational cps can cause a performance slight increase or decrease and pruning a non foundational can also result in a performane decrease
"""