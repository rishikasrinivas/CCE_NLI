import sys
sys.path.append('code/')
import train_utils 
import settings
def get_model(model_name, ckpt):
    train,val,test,dataloaders=train_utils.create_dataloaders(max_data=10000)

    model = train_utils.load_model(10000, model_name, train, ckpt=ckpt, device='cuda')
 

    return model, dataloaders
