# wandb api key
api_key = 'd19d6d45e22348abe3f9a8d08c705711a80668ca'  # 50d21c78c93763fe3ab0c50a896d798f9a8e7d0a'

# is it a new run or continue
new_run = True
sweep_id = ''  # 'cf3vvoss'

# where should the intermedia generated scripts be saved (automatically cleaned at the start of each run)
slurm_scripts_path = './slurm_scripts/'
# where should the outputs & logs be saved (automatically cleaned at the start of each run)
log_path = './logs/'
# where should calculation nodes save their important results (e.g. best model weights)
# coffer_path = './coffer/'
coffer_path = './Checkpoints/'

# entity name (your wandb account name)
entity_name = 'jiahong'
# project name on wandb and HPC
project_name = 'diffusion-master'
# e-mail address to recieve notifications
e_mail = 'upzmb@student.kit.edu'
# conda location
conda_env = '/home/kit/tm/qg0211/miniconda3/envs/clean'

# file name of the slurm_wrapper, don't change this if you haven't write a new one
slurm_wrapper_name = './slurm_wrapper.py'
# file name of the training code
train_script_name = './Trainer_gnn.py'


# define custom sweep hyperparameters
#     - how many sweeps do you want to run in total
total_sweep = 60
#     - how many sweeps do you want to run parallelly
pool_size = 20


# define wandb sweep parameters
#     - project definition
sweep_config = {
    "project": project_name,
    'program': slurm_wrapper_name,
    "name": "diffusion-gnn-noc",
    'method': 'grid',
}
#     - metric definition
metric = {
    'name': 'best_err',
    'goal': 'minimize'
}
sweep_config['metric'] = metric
#     - parameters search range definition
parameters_dict = {
    'batch': {
        'values': [64]
    },
    'nn_lr': {
        'values': [1e-6]
    },
    'trans_lr': {
        'values': [1e-6] 
    },

    'dropout': {
        'values': [0.1]
    },
    'T': {
        'values': [400]
    },
    'ddim_step': {
        'values': [4]
    },
    'eta': {
        'values': [0]
    },
    'nheads': {
            'values': [8]  #, 16 ]
    },
    'gnn_dim': {
            'values': [32]
    },
    'emb_hidden_dim' :{
            'values': [512]
    },
    'seed': {
        'values': [1]
    },
    'fold': {
        'values': [0,1,2,3,4]
    },
    'holdout': {
        'values': [0,1,2,3]
    },
}
# parameters_dict = {
#     'batch': {
#         'values': [64, 128, 256]
#     },
#     'nn_lr': {
#         'values': [0.001, 0.001, 0.0001, 1e-5]
#     },
#     'latent_dim': {
#         'values': [256, 512]
#     },
# }
    
sweep_config['parameters'] = parameters_dict
