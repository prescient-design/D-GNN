import torch
import pickle
import numpy
from sklearn.preprocessing import normalize


from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from torch_geometric.data import Data
from torch_geometric.nn.models.explainer import to_captum
from torch_geometric.nn import Explainer

from captum.attr import Saliency, Occlusion, IntegratedGradients

from matplotlib import pyplot as plt

# The model
from models import DGNN_classification

# Variables
device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")

class classification(Dataset):
    '''
    In case of classification, we are treating 'y' as an integer.
    '''
    def __init__(self, pckl):
        super().__init__()
        self.data = pckl
        self.pdb_list = list(pckl.keys())
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        pdb_id = self.pdb_list[index]

        # extract data for the selected PPDB
        H = self.data[pdb_id]['NodeFeatures']
        A = self.data[pdb_id]['edge_list']
        Y = int(self.data[pdb_id]['Y'])

        data = Data( x = torch.from_numpy(H), edge_index = torch.from_numpy(A) ,y = torch.tensor([Y]) )
        return data

class regression(Dataset):
    '''
    In case of regression, we are treating 'y' as an float (default).
    '''
    def __init__(self, pckl):
        super().__init__()
        self.data = pckl
        self.pdb_list = list(pckl.keys())
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        pdb_id = self.pdb_list[index]

        # extract data for the selected PPDB
        H = self.data[pdb_id]['NodeFeatures']
        A = self.data[pdb_id]['edge_list']
        Y = float(self.data[pdb_id]['Y'])

        data = Data( x = torch.from_numpy(H), edge_index = torch.from_numpy(A) ,y = torch.tensor([Y]) )
        return data
    

def generate_kidera_importance(ig_attr_node_1,ig_attr_node_0):
    # Scale attributions to [0, 1]:
    ig_attr_node_1 = ig_attr_node_1.squeeze(0).abs().sum(dim=0)
    ig_attr_node_1 /= ig_attr_node_1.max()

    ig_attr_node_0 = ig_attr_node_0.squeeze(0).abs().sum(dim=0)
    ig_attr_node_0 /= ig_attr_node_0.max()


    fig, (ax1, ax2) = plt.subplots(2,1)
    fig.set_size_inches(10,5)
    ax1.set_title('Label 1')
    ax1.stem(ig_attr_node_1)
    ax1.set_xticks( range(0,len(ig_attr_node_1)), [ 'KF'+str(i+1) for i in range(0,len(ig_attr_node_1)) ] )

    ax2.set_title('Label 0')
    ax2.stem(ig_attr_node_0)
    ax2.set_xticks( range(0,len(ig_attr_node_0)), [ 'KF'+str(i+1) for i in range(0,len(ig_attr_node_0)) ] )

    plt.tight_layout()
    plt.savefig('kidera.png')
    return True


def explain_classification():
    pckl = pickle.load( open('pickles/DT_5.0.pickle','rb') )

    # Optional
    keys = list(pckl.keys())
    data = classification(pckl)

    all_data_x = tuple( data[i].x.unsqueeze(0) for i in range(0,len(data)) )
    all_data_edge_index = tuple( data[i].edge_index for i in range(0,len(data)) )
    all_data_batch = tuple( data[i].batch for i in range(0,len(data)) )
    
    # Load the model
    model = DGNN_classification(10, 2).to(device)
    path_to_model = 'test_1/checkpoints/d-gnn_epoch=0546-val_f1=0.763.ckpt'
    checkpoint = torch.load( path_to_model )
    model.load_state_dict(checkpoint['state_dict'])

    # Print the model structure
    #print(model.eval())

    # Saliency
    captum_model = to_captum(model=model, mask_type="node")
    
    current_data = data[22]
    #current_data = DataLoader(classification(pckl), batch_size=32, shuffle=True, drop_last=True )

    node_mask = torch.ones( (1, current_data.x.shape[0],current_data.x.shape[1] ), requires_grad=False, device=device)

    #print(node_mask.shape, current_data.x.unsqueeze(0).shape )
    
    ig = IntegratedGradients(captum_model)
    ig_attr_node_1 = ig.attribute( node_mask , target=1, additional_forward_args=( current_data.edge_index, current_data.batch ), internal_batch_size=1 )
    ig_attr_node_0 = ig.attribute( node_mask , target=0, additional_forward_args=( current_data.edge_index, current_data.batch ), internal_batch_size=1 )
    
    # Make them 2D
    ig_attr_node_1 = ig_attr_node_1.squeeze(0).numpy()
    ig_attr_node_0 = ig_attr_node_0.squeeze(0).numpy()

    #ig_attr_node_1 = normalize( ig_attr_node_1, norm='l1', )

    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow( ig_attr_node_1, cmap=plt.get_cmap('viridis'),aspect='auto' )
    ax.set_xticks( [i for i in range(0,10)], ['KF'+str(i) for i in range(1,11)] )
    ax.set_xlabel('Kidera Factors')
    ax.set_ylabel('Chothia Index')
    
    fig.colorbar(im)
    plt.savefig('imprint.png',dpi=300)
    return True


if(__name__=='__main__'):
    explain_classification()