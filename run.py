
import logging
import yaml
import os
import pickle
import csv
import multiprocessing
import multiprocessing.dummy as mp
import time

from tqdm import tqdm
from scripts import single_entry_process


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ab-gnn-auto-pipeline')


# Dataset generation
def generate_dataset(yaml_input):
    '''Generate the pickle files
    '''
    init = time.process_time()
    logger.info('Dataset(s) generation started '+str(multiprocessing.cpu_count())+' cores being used.' )
    for i in yaml_input['parameters']['Adjacency']['values']:
        pool = mp.Pool( multiprocessing.cpu_count() )
        lines = [j for j in csv.reader(open(yaml_input['data_file'])) ]
        
        DATA = {}
        multiprocessing_input = zip( lines, [i]*len(lines) , [yaml_input['structure_location']]*len(lines) )
        for result in tqdm(pool.imap_unordered( single_entry_process, multiprocessing_input ), total=len(lines)):
            if(result is not None):
                DATA[result[0]] = result[1]
        
        with open( 'pickles/'+i+'.pickle', 'wb') as handle:
            pickle.dump(DATA, handle, protocol=pickle.HIGHEST_PROTOCOL)

    end = time.process_time()
    logger.info('Dataset generated(s) in '+str( round( (end-init)/60,3) )+' Minutes.')

# Sweep config file
def generate_sweep(yaml_input):
    '''Generate the sweep file
    '''
    sweep_config = { 'method':'grid', 'command': [ '${env}', 'python3', '${program}', '${args}' ] }

    # Depending on the objective type, we select train.py files
    if(yaml_input['problem_type']=='classification'):
        sweep_config['program'] = 'train_classification.py'
    
    elif(yaml_input['problem_type']=='regression'):
        sweep_config['program'] = 'train_regression.py'

    sweep_config['parameters'] = yaml_input['parameters']
    sweep_config['parameters']['session'] = {'distribution': 'constant', 'value': yaml_input['project name'] }
    yaml.safe_dump( sweep_config, open('sweep.yaml','w') )
    logger.info('Sweep.yaml generated!')

# Run the sweep
import torch

torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

def run_sweep(yaml_input):
    '''
    '''
    logger.info('Sweep.yaml is registered / asked to resumed.')
    if( os.path.isfile('meta/wandb.txt') ):
        print('Check your sweep on wandb website and run the following command: python3 -m wandb agent <yoursweepid>')
    else:
        os.system('python3 -m wandb sweep sweep.yaml > meta/wandb.txt')
    return True


def main():
    '''
    '''
    # STEP 0: Load the configuration
    with open('config.yaml', 'r') as file:
        yaml_input = yaml.safe_load(file)

    # STEP 1: Make a director for pickes if there is none
    if not os.path.exists(yaml_input['structure_location']):
        os.makedirs(yaml_input['structure_location'])
    
    if not os.path.exists('pickles'):
        os.makedirs('pickles')
    
    if not os.path.exists('meta'):
        os.makedirs('meta')

    # STEP 2: Generate dataset
    generate_dataset(yaml_input)

    # STEP 3: Generate the sweep file.
    generate_sweep(yaml_input)

    # Step 4: Run the sweep
    run_sweep(yaml_input)
    return True


if(__name__=='__main__'):
    main()