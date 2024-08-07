import wandb

def wandb_init(proj_name, exp_name, x_axis):
    wandb.login()
    wandb_=wandb.init(
      # Set the project where this run will be logged
      project=proj_name, 
      # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
      name=exp_name, 
      # Track hyperparameters and run metadata
      config={
          "learning_rate": 0.02,
          "architecture": "CNN",
          "dataset": "CIFAR-100",
          "epochs": 10,
        }
    )
    
    
    wandb_.define_metric("prune iter")
    # define which metrics will be plotted against it
    wandb_.define_metric("accuracy_*", step_metric="prune iter")
    return wandb_