import collections,torch,settings
import numpy as np
import torch.nn as nn

class Pruner_:
    def __init__(self,model):
        self.model=model
        self.prunable_layers = collections.defaultdict()
        for layername, module in self.find_layers(self.model, self.model).items():
            if layername == 'encoder.rnn':
                layername += '.weight_ih_l0'
            else:
                layername += ".weight"
            if 'mlp.3' not  in layername:
                self.prunable_layers[layername] = 0.2

                
                
            
            
            
    def find_layers(self, model, module, layers=[nn.Linear, nn.LSTM], name=''):
    
        """
        Recursively find the layers of a certain type in a module.

        Args:
            module (nn.Module): PyTorch module.
            layers (list): List of layer types to find.
            name (str): Name of the module.

        Returns:
            dict: Dictionary of layers of the given type(s) within the module.
        """
        if type(module) in layers:
            return {name: module}
        res = {}
        for name1, child in module.named_children():
            res.update(self.find_layers(
                model, child, layers=layers, name=name + '.' + name1 if name != '' else name1
            ))
        return res
    
    def prune_by_percent_once(self,percent, mask, final_weight, reverse=False):
            # Put the weights that aren't masked out in sorted order.
            
            mask=mask.cpu()
            if reverse:
                sorted_weights = np.sort(np.abs(final_weight[mask == 1]))[::-1]
            else:
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
        for layername in self.prunable_layers:
            if layername  not in self.model.get_layer_names() or self.prunable_layers[layername] == 0.0:
                print("Skipped ", layername, " because not in ", self.model.get_layer_names())
                continue
                
            layer=self.model.get_layer(layername)
            shape=layer.pruning_mask.shape
            new_mask, new_weights = self.prune_by_percent_once(self.prunable_layers[layername], layer.pruning_mask.flatten(), layer.weights.detach().cpu().reshape(-1), reverse=False)
            
            new_weights=new_weights.reshape(shape)
            new_mask=new_mask.reshape(shape)
            self.model.update_layer_weights(new_mask, layername, new_weights)
            vweights=self.model.get_layer(layername).weights
            
        #====Logging=======
            
        '''for la in self.model.get_layer_names():
            la=self.model.get_layer(la)
            if len(list(la.weights.shape)) <= 1:
                continue
            print("Pruned at end ", la.name, ",  ", torch.where(la.pruning_mask==0,1,0).sum()/(la.weights.shape[0] * la.weights.shape[1]))'''
            
        return self.model
        
