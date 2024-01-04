#This script makes the database given PDB files and their corrosponding classes/labels/values
import argparse
import torch
import wandb
import yaml

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from pathlib import Path
from dataloader import get_data
from models import DGNN_classification


# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

'''
##################################################################################################
#                                         Parser                                                 #
##################################################################################################
'''

def parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="train/eval a D-GNN model.")
    parser.add_argument("--session", type=Path, required=True, help="Session directory")
    parser.add_argument("--Adjacency", type=str, required=True, help="Pickle file with data.")

    parser.add_argument("--logger", type=str, default="tensorboard", help="Logger.", choices=["tensorboard", "wandb"])

    #parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--nodes", type=int, default=1)

    parser.add_argument("--lr", help="Learning rate", type=float, default=1e-3)
    parser.add_argument("--Layers", type=str, default='H-100-50-100', help="Hidden layer dims in different GraphConv layers.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--train_batchsize", type=int, default=32)
    parser.add_argument("--test_batchsize", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser



'''
##################################################################################################
#                                          Main                                                  #
##################################################################################################
'''

def main():
    args = parser().parse_args()
    print ("### Input arguments:", args)

    # Setting the seed
    pl.seed_everything(args.seed)

    # Input config file
    with open('config.yaml', 'r') as file:
        yaml_input = yaml.safe_load(file)

    #To keep layers categorical in the wandb
    new_Layers = []
    for i in args.Layers.split('-'):
        try:
            new_Layers.append( int(i) )
        except:
            None
    args.Layers = new_Layers

    checkpoints = args.session / 'checkpoints'
    checkpoints.mkdir(exist_ok=True, parents=True)

    # Load the datasets
    train_loader, valid_loader, num_classes, num_features = get_data('pickles/'+args.Adjacency+'.pickle', args.train_batchsize, args.test_batchsize, 'classification', args.seed)

    # Seeds shouldn't exceed 5 digits
    checkpoint = pl.callbacks.ModelCheckpoint(dirpath=checkpoints,
            filename="d-gnn_{epoch:04d}-{val_f1:0.3f}",
            monitor="val_loss",
            save_top_k=10,
            save_last=True,
            verbose=True)

    #Early stopping (MENTION DETAILS IN THE PAPER)
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=int(args.epochs/10), verbose=False, mode="min")

    if args.logger == "wandb":
        wandb.init(entity=yaml_input['wandb_entity'], project=yaml_input['project name'])
        logger = WandbLogger(project=yaml_input['project name'])

        # Register everything except seeds
        wandb.log( {"parameter_grouping": args.Adjacency+'_'+ '-'.join([str(i) for i in args.Layers]) +'_'+str(args.train_batchsize)+'_'+str(args.test_batchsize)} )

    else:
        logger = pl.loggers.TensorBoardLogger('./tensorboard')
    
    
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(gpus=torch.cuda.device_count(),
                         num_nodes=args.nodes,
                         logger=logger,
                         max_epochs=args.epochs,
                         gradient_clip_val=0.1,
                         gradient_clip_algorithm="norm",
                         callbacks=[LearningRateMonitor("epoch"), checkpoint, early_stop_callback ]
                         )

    #model = DGNN(hidden=args.Layers,lr=args.lr)
    model = DGNN_classification(num_features, num_classes, hidden=args.Layers, lr=args.lr)

    trainer.fit(model, train_loader, valid_loader)
    return True

if(__name__=='__main__'):
    main()