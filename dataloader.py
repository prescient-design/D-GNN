import torch
import random
import pickle

from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from torch_geometric.data import Data

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


def get_data(pickle_file_path, train_batchsize, validation_batchsize, problem_type, seed):
    '''
    '''
    pckl = pickle.load(open(pickle_file_path, 'rb'))

    #collate_padd = PaddCollator()
    pdb_list = list(pckl.keys())
    n = len(pdb_list)

    # shuffle the list
    random.shuffle(pdb_list)

    #80 / 20 train validation division
    train_pdb_list = pdb_list[:int(0.8*n)]
    valid_pdb_list = pdb_list[int(0.8*n):]

    train_pckl = {i:pckl[i] for i in train_pdb_list}
    valid_pckl = {i:pckl[i] for i in valid_pdb_list}

    #Uncomments parts of next lines to add collate (applicable to non geometric data)

    fh = open('meta/split.txt','a')
    fh.write('Seed : '+str(seed)+'\n')
    fh.write('train : '+ ','.join(train_pdb_list)+'\n')
    fh.write('validation : '+ ','.join(valid_pdb_list)+'\n\n')
    fh.flush()
    fh.close()

    #num_features = [i['NodeFeatures'].shape[1] for i in pckl.values()][0]
    num_features = list(pckl.values())[0]['NodeFeatures'].shape[1]

    if(problem_type=='classification'):
        train_loader = DataLoader(classification(train_pckl), batch_size=train_batchsize, shuffle=True, drop_last=True )
        valid_loader = DataLoader(classification(valid_pckl), batch_size=validation_batchsize, shuffle=False, drop_last=True )
        num_classes = len(set([i['Y'] for i in pckl.values()]))
        return train_loader, valid_loader, num_classes, num_features
        
    elif(problem_type=='regression'):
        train_loader = DataLoader(regression(train_pckl), batch_size=train_batchsize, shuffle=True, drop_last=True )
        valid_loader = DataLoader(regression(valid_pckl), batch_size=validation_batchsize, shuffle=False, drop_last=True )
        return train_loader, valid_loader, num_features
    else:
        print('Provide right problem_type')
        exit()

# For testing
'''
def main():
    T,V = get_data('data/DT_3.5.pickle',27,9)

    for i in V:
        print(i)
        None
    return True

if(__name__=='__main__'):
    main()
'''