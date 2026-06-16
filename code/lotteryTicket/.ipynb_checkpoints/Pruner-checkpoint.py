import collections,torch,settings
import numpy as np
import torch.nn as nn

class Pruner_:
    def __init__(self,model):
        self.model=model
        self.prunable_layers = collections.defaultdict()
        for layername, module in self.find_layers(self.model, self.model).items():
            if layername == 'encoder.rnn':
                continue #don't prune rnn
            else:
                layername += ".weight"
                if 'mlp.3' not in layername:
                    self.prunable_layers[layername] = 0.25
        print(self.prunable_layers)
       
        

                
                
            
            
            
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
            sorted_weights = np.sort(np.abs(final_weight[mask == 1]))
            

            # Determine the cutoff for weights to be pruned.

            cutoff_index = np.round(percent * sorted_weights.size).astype(int)
            cutoff = sorted_weights[cutoff_index] 
            
            
            # Prune all weights below the cutoff
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
            self.model.update_layer_weights(new_mask, layername, new_weights, pruned=True)
         
        return self.model
    '''
            
    
    def prune_by_percent_once(self,percent, mask, final_weight, reverse=False):
            
        # Since the weights are pruned by their magnitude, the absolute values of the weights are retrieved
        final_weight = torch.abs(final_weight)  # pylint: disable=no-member

        # Sorts the weights ascending by their magnitude, this makes it easy to prune weights with the smallest magnitude, because the indices of
        # the weights with the smallest magnitude are at the beginning of the this array
        sorted_indices = torch.argsort(final_weight)  # pylint: disable=no-member

        # Determines the number of weights that should be pruned, since pruning is an iterative process of training, pruning, re-training, it
        # could be that the specified model was already sparsified previously, because the pruning strategy employed here is to sort the elements
        # by magnitude and then take the smallest n%, the same weights that are already zero would be pruned in any consecutive pruning,
        # therefore, the number of zeros in the layer are added to the number of pruned weights, otherwise no further pruning would occur
        number_of_zero_weights = final_weight.numel() - final_weight.nonzero().size(0)
        number_of_pruned_weights = int(percent * (len(sorted_indices) - number_of_zero_weights)) + number_of_zero_weights
        #print(f"Cutoff is {final_weight[sorted_indices[number_of_pruned_weights-1]]}")

        # Creates the pruning mask which is 1 for all weights that are not pruned and
        pruning_mask = torch.zeros_like(final_weight, dtype=torch.uint8)  # pylint: disable=no-member
        pruning_mask[sorted_indices[:number_of_pruned_weights]] = 0
        pruning_mask[sorted_indices[number_of_pruned_weights:]] = 1
        return pruning_mask
    
    def prune(self):
        for layername in self.prunable_layers:
            if layername  not in self.model.get_layer_names() or self.prunable_layers[layername] == 0.0:
                print("Skipped ", layername, " because not in ", self.model.get_layer_names())
                continue
                
            layer=self.model.get_layer(layername)
            shape=layer.pruning_mask.shape
            
            new_mask = self.prune_by_percent_once(self.prunable_layers[layername], layer.pruning_mask.flatten(), layer.weights.detach().cpu().reshape(-1), reverse=False)
            
            
        
            new_mask=new_mask.reshape(shape)
            new_weights = layer.weights.cpu() *new_mask.cpu()
            new_weights=new_weights.reshape(shape)
            
            
            self.model.update_layer_weights(new_mask, layername, new_weights, pruned=True)
            
        #torch.save(self.model.state_dict(), f'newlth.pth')
        #====Logging=======
       
        return self.model'''
        
