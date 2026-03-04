#!/gpfs/mskmind_ess/aukermaa/usr/bin/python3.9

# Basic imports
import sys, os
import numpy as np
from tqdm import tqdm
import pandas as pd
from signal import signal, SIGPIPE, SIG_DFL  
import pdb
#signal(SIGPIPE,SIG_DFL) 

# Torch
import math
import torch
import torch.nn    as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Sklearn
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.metrics         import roc_auc_score, precision_score, recall_score, f1_score

# Custom Defs
from datasets import CombinedDataset, Normalizer, CombinedDatasetPETComponetBag, load_patient_data
from model    import BoneInstanceRadiomicsModel
from utils    import plot_layer_grad_summary

import wandb
from wandb import agent, init, config

model_name     = 'BlobInstanceModelV1-LBP'
source_ds      = 'DISCOVERY__NB_CT_SKELETAL_MASK_WITH_PROGRESSION_1.50mm_v2'

from utils import elevate

#wandb.login(key='b8c197b4370972203de057f3b56350587c3305cb')

user_config = {
   'method': 'random',
   'name': 'loss',
   'metric': {
      'goal': 'maximize',
      'name': 'valid_auc'
   },
   'parameters': {
    "lr": {
         'distribution': 'uniform',
            'min': 0,
            'max': 0.1
      },
    "kappa": {'values': [5, 10, 15]},
    "alpha": {'values': [0.00001,0.001]},
    "optim": {'values': ["Adam","SGD"]},
    "radiomics":{'values': ["lbp-3D-m2", "original","gradient"]},
    "epochs": {'values': [10]},
   "batch_size": {
               'distribution': 'q_log_uniform',
               'q': 1,
               'min': math.log(32),
               'max': math.log(256),
               }
   }
}

#wandb.init(project="mad1/my_project")

sweep_id = wandb.sweep(user_config, project="my_project", entity="mad1")
print(sweep_id)

print('wandb_config: ', wandb.config)

def train():
    print ("GPU/CUDA:", torch.cuda.is_available(), torch.cuda.device_count())
    
    data_file = f'/gpfs/mskmind_ess/mad1/neuroblastoma/NB-16-1335/tables/{source_ds}'
    print('wandb_config: ', wandb.config)
    patients = load_patient_data(data_file=data_file)
#    pdb.set_trace()
    kf = ShuffleSplit(n_splits=20, test_size=0.2, random_state=0)
    
    with wandb.init() as run:
        print ("Started run", run)
        valid_aucs = []

        wandb_config = wandb.config
        print("wandb_config=", wandb_config)
        
#        pdb.set_trace()
        for fold, (train, test) in enumerate(list(kf.split(patients))):
 #           print('patients train: ',patients[train])
  #          print('patients test: ',patients[test])
            auc = train_test(fold + 1, patients[train], patients[test], wandb_config)
            valid_aucs.append(auc)
            print('valid_aucs: ',valid_aucs)
        hypo_result = {"valid_auc": np.mean(valid_aucs), "valid_auc_stdev": np.std(valid_aucs)}
        print ("hypo_result=", hypo_result)
        
        wandb.log(hypo_result)

def train_test(fold, train_patients, valid_patients, config):

    print(config)
    hypo_lr        = config['lr']
    hypo_kappa     = config['kappa']
    hypo_alpha     = config['alpha']
    hypo_optim     = config['optim']
    hypo_radiomics = config['radiomics']
    hypo_epochs    = config['epochs']
    radiomics_string = '_'.join(hypo_radiomics)

    torch.manual_seed(0)

    train_ds = CombinedDatasetPETComponetBag(train_patients, hu_cutoff=225, layer0="/gpfs/mskmind_ess/mad1/neuroblastoma/NB_16-1335/BoneCache", layer1=source_ds)
    valid_ds = CombinedDatasetPETComponetBag(valid_patients, hu_cutoff=225, layer0="/gpfs/mskmind_ess/mad1/neuroblastoma/NB_16-1335/BoneCache", layer1=source_ds)

    train_loader = DataLoader(train_ds, batch_size=None, shuffle=True, num_workers=16, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=None, shuffle=True, num_workers=16, pin_memory=True)

    print (f"Fold [{fold}]", "Train/test:", len(train_loader), len(valid_loader))

    normalizer = Normalizer(radiomics_subset=hypo_radiomics)

    for input_data, target in tqdm(train_loader):
        normalizer.add_instance (input_data)

    normalizer.fit()
    normalizer.print()
    print("finished normalizing")
    network    = BoneInstanceRadiomicsModel(n_features=normalizer.n_features, init_temp = 2.5, kappa = hypo_kappa)
    criterion  = nn.BCEWithLogitsLoss()

    if hypo_optim=="Adam":
        optimizer  = optim.Adam(network.parameters(), lr=hypo_lr)
    elif hypo_optim=="SGD":
        optimizer  = optim.SGD(network.parameters(), lr=hypo_lr)
    network.cuda()

    all_out_metrics = []

    for epoch in range(hypo_epochs + 1):
        out_metrics = {}
        out_metrics['model_name'] = model_name
        out_metrics['fold']       = fold
        out_metrics['epoch']      = epoch
        out_metrics['n_epochs']   = hypo_epochs
        out_metrics['source_ds']  = source_ds
        out_metrics['lr']         = hypo_lr
        out_metrics['kappa']      = hypo_kappa
        out_metrics['alpha']      = hypo_alpha
        out_metrics['radiomics']  = radiomics_string
        out_metrics['optimizer']  = hypo_optim

        train_total_loss   = 0
        train_kld_loss     = 0
        train_labels = []
        train_scores = []
         
        print("------Starting to train------")
        for input_data, target in tqdm(train_loader):
            input_data = normalizer.transform (input_data)
            input_data, target = input_data.cuda(), target.cuda()

            optimizer.zero_grad()

            attn, output, kld = network(input_data)

            train_labels.append(target.cpu().item())
            train_scores.append(output.cpu().item())
            
            
            loss = criterion(output, target) + hypo_alpha * kld 

            if epoch > 0:
                loss.backward()
                optimizer.step()

            train_total_loss  += loss.item()
            train_kld_loss += kld.item()
            
            print("Train loss: ", train_total_loss)

        out_metrics['train_auc']   = roc_auc_score(train_labels, train_scores)
        out_metrics['train_total_loss']   = train_total_loss  / len(train_loader)
        out_metrics['train_kld_loss']     = train_kld_loss / len(train_loader)
        out_metrics['train_score_mean'] = np.mean(train_scores)
        out_metrics['train_score_std']  = np.std (train_scores)
        out_metrics['current_temp']  = network.temp.item()

        valid_total_loss = 0
        valid_kld_loss    = 0
        valid_labels = []
        valid_scores = []
        for input_data, target in tqdm(valid_loader):
            input_data = normalizer.transform (input_data)
            input_data, target = input_data.cuda(), target.cuda()

            with torch.no_grad():
                attn, output, kld = network(input_data)

            valid_labels.append(target.cpu().item())
            valid_scores.append(output.cpu().item())

            loss = criterion(output, target) + hypo_alpha * kld 
            valid_total_loss  += loss.item()
            valid_kld_loss += kld.item()
            print("Validation_loss: ",valid_total_loss)
        out_metrics['valid_auc']   = roc_auc_score(valid_labels, valid_scores)
        out_metrics['valid_total_loss']   = valid_total_loss  / len(valid_loader)
        out_metrics['valid_kld_loss']     = valid_kld_loss / len(valid_loader)
        out_metrics['valid_score_mean'] = np.mean(valid_scores)
        out_metrics['valid_score_std']  = np.std (valid_scores)

        print (out_metrics)
        all_out_metrics.append(out_metrics)

    save_dir = f"/gpfs/mskmind_ess/mad1/neuroblastoma/neuro-ana/results/m{model_name}.n{hypo_epochs}.r{radiomics_string}.s{source_ds}.l{hypo_lr}.o{hypo_optim}.k{hypo_kappa}.a{hypo_alpha}"
    os.makedirs(save_dir, exist_ok=True)

    pd.DataFrame(all_out_metrics).to_parquet(f"{save_dir}/results.f{fold}.parquet", index=False)
    
    return roc_auc_score(valid_labels, valid_scores)

if __name__=="__main__":
    #wandb.agent('8envcv16', project="neuroblastoma", function=train, count=1)
    wandb.agent(sweep_id, project="my_project", function=train, count=20)
