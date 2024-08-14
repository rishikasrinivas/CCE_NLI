import wandb

def wandb_init(proj_name, exp_name):
    wandb.login()
    wandb_=wandb.init(
      # Set the project where this run will be logged
      project=proj_name, 
      # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
      name=exp_name, 
      # Track hyperparameters and run metadata
      config={
          "architecture": "LSTM",
          "dataset": "SNLI",
          "epochs": 10,
          "prune_iters": 20
        }
    )
    
    
    wandb_.define_metric("prune_iter")
    # define which metrics will be plotted against it
    wandb_.define_metric("accuracy_*", step_metric="prune_iter")
    return wandb_