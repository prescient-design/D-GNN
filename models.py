
import wandb

import torch
from torch.nn import Linear, MSELoss, HuberLoss
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Dropout

from torch_geometric.nn import Sequential, global_mean_pool, GCNConv
import pytorch_lightning as pl
from torchmetrics.functional import precision_recall, accuracy




'''
##################################################################################################
#                                  1. Single Value Regression                                    #
##################################################################################################
'''

class DGNN_regression(pl.LightningModule):
    """NOT COMPLETE!! USE AT YOUR OWN RISK.
    """

    def __init__(self, num_features, hidden=[100,100,100], lr=3e-4):
        super(DGNN_regression, self).__init__()

        self.num_features = num_features
        self.num_classes = 1
        self.lr = lr
        self.hidden = hidden
        self.Best_Train_Loss = float('Inf')
        self.Best_Val_Loss = float('Inf')

        layers = []
        #hidden
        for numi,i in enumerate(hidden):
            if(numi==0):
                layers.append( ( GCNConv( self.num_features, i ), 'H, A -> H1' ) )
            else:
                layers.append( ( GCNConv( hidden[numi-1], i ), 'H'+str(numi)+', A -> H'+str(numi+1) ) )
        #pooling and final
        layers.append( (global_mean_pool, 'H'+str(numi+1)+', batch_index -> H'+str(numi+2) ) )
        layers.append( (Linear( hidden[-1], self.num_classes ), 'H'+str(numi+2)+' -> x_out') )

        self.model = Sequential('H, A, batch_index', layers)
    
    def forward(self, H, A, batch_index):
        #https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html
        #https://www.exxactcorp.com/blog/Deep-Learning/gnn-demo-using-pytorch-lightning-and-pytorch-geometric

        x_out = self.model( H.float(), A, batch_index)
        return x_out
    
    def training_step(self, batch):
        batch_index = batch.batch
        
        x_out = self.forward(batch.x, batch.edge_index, batch_index)

        mse_loss = HuberLoss(reduction='mean')
        loss = mse_loss(x_out,batch.y)
        
        # logging
        self.log("train_loss", loss)

        if(loss<self.Best_Train_Loss):
            data =  [[X,Y] for (X,Y) in zip(x_out.T[0], batch.y)]
            table = wandb.Table(data=data, columns = ["pKd (Predicted)", "pKd (True)"])
            try:
                wandb.log({"Training Step" : wandb.plot.scatter(table, "pKd (True)", "pKd (Predicted)")})
            except:
                None
            self.Best_Train_Loss = loss
        
        return loss
    
    def validation_step(self, batch, batch_index):
        batch_index = batch.batch
        
        x_out = self.forward(batch.x, batch.edge_index, batch_index)

        mse_loss = HuberLoss(reduction='mean')
        loss = mse_loss(x_out,batch.y)

        # logging 
        self.log("val_loss", loss)

        if(loss<self.Best_Val_Loss):
            data =  [[X,Y] for (X,Y) in zip(x_out.T[0], batch.y)]
            table = wandb.Table(data=data, columns = ["pKd (Predicted)", "pKd (True)"])
            try:
                wandb.log({"Validation Step" : wandb.plot.scatter(table, "pKd (True)", "pKd (Predicted)")})
            except:
                None
            self.Best_Val_Loss = loss
        
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = self.lr)

    
'''
##################################################################################################
#                                      2. DGNN_classification                                    #
##################################################################################################
'''

class DGNN_classification(pl.LightningModule):
    # Semi-Supervised Classification with Graph Convolutional Networks
    def __init__(self, num_features, num_classes, hidden=[100,100,100], lr=3e-4):
        super(DGNN_classification, self).__init__()

        self.num_features = num_features
        self.num_classes = num_classes

        self.lr = lr
        self.hidden = hidden

        layers = []

        track_numi = 0
        #hidden
        for numi,i in enumerate(hidden):
            if(numi==0):
                layers.append( (GCNConv( self.num_features, i ), 'H, A -> H1') )
                layers.append( (ReLU(), 'H1 -> H1a') )
            
            else:
                layers.append( ( GCNConv( hidden[numi-1], i ), 'H'+str(numi)+'a, A -> H'+str(numi+1) ) )
                layers.append( (ReLU(), 'H'+str(numi+1)+' -> H'+str(numi+1)+'a') )
            
            track_numi = numi

        
        #pooling and final
        layers.append( (global_mean_pool, 'H'+str(track_numi+1)+'a, batch_index -> H'+str(track_numi+2) ) )
        layers.append( (Linear( hidden[track_numi], self.num_classes ), 'H'+str(track_numi+2)+' -> x_out') )

        self.model = Sequential('H, A, batch_index', layers)


    def forward(self, H, A, batch_index):
        #https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html
        #https://www.exxactcorp.com/blog/Deep-Learning/gnn-demo-using-pytorch-lightning-and-pytorch-geometric

        x_out = self.model( H.float(), A, batch_index)
        return x_out
    
    def training_step(self, batch):
        batch_index = batch.batch
        x_out = self.forward(batch.x, batch.edge_index, batch_index)
        
        loss = F.cross_entropy( x_out, batch.y )

        # train metrics here
        pred = x_out.argmax(-1)
        label = batch.y
        
        accuracy__ = (pred == label).sum() / pred.shape[0]

        # logging
        self.log("train_loss", loss)
        self.log("train_accuracy", accuracy__)

        # Average macro assures the equal importance to all the classes (for binary, it is equal to micro)
        precision, recall =  precision_recall( pred, label, num_classes = self.num_classes, average='macro' )

        self.log("train_precision", precision)
        self.log("train_recall", recall)
        self.log("train_f1", (2*precision*recall)/(precision+recall) )
        return loss
    
    def validation_step(self, batch, batch_index):
        
        batch_index = batch.batch
        
        x_out = self.forward(batch.x, batch.edge_index, batch_index)

        loss = F.cross_entropy( x_out, batch.y )

        # val metrics here 
        pred = x_out.argmax(-1)
        label = batch.y

        accuracy__ = (pred == label).sum() / pred.shape[0]

        # logging 
        self.log("val_loss", loss)
        self.log("val_accuracy", accuracy__)

        # Average macro assures the equal importance to all the classes (for binary, it is equal to micro)
        precision, recall =  precision_recall( pred, label, num_classes = self.num_classes, average='macro' )

        self.log("val_precision", precision)
        self.log("val_recall", recall)
        self.log("val_f1", (2*precision*recall)/(precision+recall))

        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = self.lr)