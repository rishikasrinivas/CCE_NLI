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
            if layername not in self.pruning_percents or self.pruning_percents[layername] == 0.0:
                continue
            layer=self.model.get_layer(layername)
            shape=layer.pruning_mask.shape
            new_mask, new_weights = self.prune_by_percent_once(self.pruning_percents[layername], layer.pruning_mask.flatten(), layer.weights.detach().cpu().reshape(-1), reverse=False)
            
            new_weights=new_weights.reshape(shape)
            new_mask=new_mask.reshape(shape)
            self.model.update_layer_weights(new_mask, layername, new_weights)
            vweights=self.model.get_layer(layername).weights
            
        #====Logging=======
            
        for la in self.model.get_layer_names():
            la=self.model.get_layer(la)
            if len(list(la.weights.shape)) <= 1:
                continue
            print("Pruned at end ", la.name, ",  ", torch.where(la.pruning_mask==0,1,0).sum()/(la.weights.shape[0] * la.weights.shape[1]))
            
        return self.model
    def get_mask(self,layer):
        return self.pruning_masks[layer]
        
