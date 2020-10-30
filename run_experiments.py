import pdb
import optuna
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback

from Data.GM12878_DataModule import GM12878Module
from VAE_Module import VAE_Model
import argparse

class MetricsCallback(Callback):
    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)

def objective(trial):
    #params

    #DS paramgs
    batch_size       = trial.suggest_int('batch_size', 1,80)

    #Model Params
    kld_weight       = trial.suggest_uniform('kld_weight',0,1)
    lr               = trial.suggest_loguniform('lr',0.00001,0.1)
    gamma            = trial.suggest_uniform('gamma', 0.8, 1.0)
    latent_dim       = trial.suggest_int('latent_dim', 50, 500) #step =1
    
    metrics_callback = MetricsCallback()
    dm               = GM12878Module(batch_size=batch_size)
    model            = VAE_Model(
                                kld_weight=kld_weight,
                                lr=lr,
                                gamma=gamma,
                                latent_dim=latent_dim,
                                )
    trainer          = Trainer(gpus=1,
                              max_epochs=30,
                              callbacks=[metrics_callback])
    trainer.fit(model, dm)
    return metrics_callback.metrics[-1]["recon_loss"].item()

study_name  = "simple_exp"
study       =  optuna.create_study(study_name=study_name)
study.optimize(objective, n_trials=5)
