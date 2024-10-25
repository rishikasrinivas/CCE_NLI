import collections,torch,settings
import numpy as np
class Pruner:
    def __init__(self,model):
        self.model=model
        self.pruning_percents=settings.PRUNE
        
        self.pruning_masks=collections.defaultdict()
        for name, weights in self.model.named_parameters():
            self.pruning_masks[name]=torch.ones(weights.shape)
    
    def prune_by_percent_once(self,percent, mask, final_weight, reverse=False):
            # Put the weights that aren't masked out in sorted order.
            
            mask=mask.cpu()
            if reverse:
                sorted_weights = np.sort(np.abs(final_weight[mask == 1]))[::-1]
            else:
                print(mask)
                sorted_weights = np.sort(np.abs(final_weight[mask == 1]))

            # Determine the cutoff for weights to be pruned.

            cutoff_index = np.round(percent * sorted_weights.size).astype(int)
            cutoff = sorted_weights[cutoff_index - 1] 
            print(cutoff)
            # Prune all weights below the cutoff
            if reverse:
                new_mask = torch.where(torch.abs(torch.tensor(final_weight)) >= cutoff, torch.zeros(mask.shape), mask)
                new_weights= torch.where(torch.abs(torch.tensor(final_weight)) >= cutoff, torch.zeros(final_weight.shape), torch.tensor(final_weight))
            else:
                new_mask = torch.where(torch.abs(torch.tensor(final_weight)) <= cutoff, torch.zeros(mask.shape), mask)
                new_weights= torch.where(torch.abs(torch.tensor(final_weight)) <= cutoff, torch.zeros(final_weight.shape), torch.tensor(final_weight))
            return new_mask, new_weights
    
    def prune(self):
        for layername in self.model.get_layer_names():
            if self.pruning_percents[layername] == 0.0:
                continue
            layer=self.model.get_layer(layername)
            shape=layer.pruning_mask.shape
            print(layer.pruning_mask.flatten().shape)
            new_mask, new_weights = self.prune_by_percent_once(self.pruning_percents[layername], layer.pruning_mask.flatten(), layer.weights.reshape(-1), reverse=False)
            print("Pruned ", layername, ",  ", torch.where(new_weights==0,1,0).sum()/(layer.weights.shape[0] * layer.weights.shape[1]))
            self.model.get_layer(layername).pruning_mask=new_mask
            new_weights=new_weights.reshape(shape)
            self.model.get_layer(layername).layer.weights=new_weights
            self.model.update_layer_weights(new_mask, layername, new_weights)
         
        for la in self.model.get_layer_names():
            la=self.model.get_layer(la)
            print("Pruned at end ", la, ",  ", torch.where(la.weights==0,1,0).sum()/(layer.weights.shape[0] * layer.weights.shape[1]))
            
        return self.model
    def get_mask(self,layer):
        return self.pruning_masks[layer]
        
