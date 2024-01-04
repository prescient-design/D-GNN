# Delaunay Graph Neural Network (D-GNN)

This repo is to create a d-gnn model as described in the [original manuscript](https://www.biorxiv.org/content/10.1101/2023.06.26.546331v1).

## How to use this repo?

0. Install the the Prerequisites.
1. Open `config.yaml` file and make changes in the fields required. If structures from PDB and not readily downloaded, use `python3 misc/download_structures.py`
2. `python3 run.py`
3. Run the wandb command suggested or save the command in a slurm file to submit it as a job.
4. (Optional) if you want to do testruns/ make changes with tensorboard instead of wandb, use: `python3 train_regression.py --logger tensorboard --session test --Adjacency DT_5.0` (Change paramers as you need)

## Prerequisites

* py-packman
* pytorch
* torch_geometric
* pytorch-lightning
* wandb

## Input file columns (CSV)

The input file should be a comma-separated (.csv) file. All the files must be downloaded and placed in a convenient and accessible location before running any scripts. The CSV columns are as follows:


| PDB ID | Model ID | Heavy Chain ID | Light Chain ID | Antigen Chain ID(s) | y |
|--------|----------|----------------|----------------|---------------------|---|
|........|..........|................|................|.....................|...|
|........|..........|................|................|.....................|...|
|........|..........|................|................|.....................|...|

NOTES: 
* y can either be a discrete or continuous variable.
* Antigen Chains should be separated by pipe ('|') eg... A|B|C. If Analysis is only on the Antibody, ALL the Ag fields should be left NA
* The fields `Heavy Chain ID`, `Light Chain ID`, and `Antigen Chain ID(s)`  are named because of their application on antibodies. However, they can be used on any protein(s) as long as mandatory fields are not empty.

## I am getting errors

1. Check the config.yaml file.
2. Check if the .csv input file is according to the format described.
3. Your progress can be resumed by following instructions given after running `run.py`
4. Run freshly cloned repo everytime you have to run the new dataset and settings.

## Citation

If you use the code and/or model, please cite:
```
@article {Khade2023.06.26.546331,
	author = {Pranav M. Khade and Michael Maser and Vladimir Gligorijevic and Andrew Watkins},
	title = {Mixed structure- and sequence-based approach for protein graph neural networks with application to antibody developability prediction},
	elocation-id = {2023.06.26.546331},
	year = {2023},
	doi = {10.1101/2023.06.26.546331},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2023/06/28/2023.06.26.546331},
	eprint = {https://www.biorxiv.org/content/early/2023/06/28/2023.06.26.546331.full.pdf},
	journal = {bioRxiv}
}

```