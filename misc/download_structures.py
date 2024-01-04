from packman import molecule
from tqdm import tqdm

import multiprocessing
import os
import yaml

# Load config.yaml
with open('config.yaml', 'r') as file:
    yaml_input = yaml.safe_load(file)

def single_download(entry):
    try:
        molecule.download_structure(entry, yaml_input['structure_location']+entry)
    except:
        None

def main():
    if not os.path.exists(yaml_input['structure_location']):
        os.makedirs(yaml_input['structure_location'])

    multiprocessing_input = []
    for i in open(yaml_input['data_file']):
        multiprocessing_input.append(i[:4])
    
    multiprocessing_input = list(set(multiprocessing_input))

    # Multiprocessing
    pool = multiprocessing.Pool( multiprocessing.cpu_count() )
    for result in tqdm(pool.imap_unordered( single_download, multiprocessing_input ), total=len(multiprocessing_input)):
        None

if(__name__=='__main__'):
    main()