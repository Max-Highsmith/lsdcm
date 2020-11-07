import pdb
import optuna
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, EarlyStopping
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
    #lr               = trial.suggest_loguniform('lr',0.00001,0.1)
    pargs = {'batch_size': 128,
        'condensed_latent': 3,
        'gamma': 1.0,
        'kld_weight': 0,
        'kld_weight_inc': 0,
        'latent_dim': 110,
        'lr': 0.00001,
        'pre_latent': 4608}
    
    metrics_callback = MetricsCallback()
    dm               = GM12878Module(batch_size=pargs['batch_size'])

    model    = VAE_Model(batch_size=pargs['batch_size'],
                    condensed_latent=pargs['condensed_latent'],
                    gamma=pargs['gamma'],
                    kld_weight_inc=pargs['kld_weight_inc'],
                    latent_dim=pargs['latent_dim'],
                    lr=pargs['lr'],
                    pre_latent=pargs['pre_latent'])



    trainer          = Trainer(gpus=1,
                              max_epochs=500,
                              callbacks=[metrics_callback])
    trainer.fit(model, dm)
    return metrics_callback.metrics[-1]["train_loss"].item()

study_name  = "simple_exp"
study       =  optuna.create_study(study_name=study_name)
study.optimize(objective, n_trials=100)
