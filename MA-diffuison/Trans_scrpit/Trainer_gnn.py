import os
import sys
import json
import time
import myconfig1
# import offlinesolver_tr_e2e_diffusion as solver
import solver_gnn_ddim_test as solver
from datetime import datetime


def make_dir(path):
    try:
        os.mkdir(path)
    except:
        pass


def train(settings):
    result_sheet = []
    list_total, list_err = solver.training(settings=settings)
    print(list_total)
    print(list_err)
    best_err, r_squared = solver.evaluate(settings=settings)
    result_sheet.append([list_total, list_err, best_err, r_squared])
    print(result_sheet)
    # result_sheet.append([best_err, r_squared])
    # collect wandb result into file
    rtn = {
        "best_err": sum(result_sheet[0][2]) / len(result_sheet[0][2]),
        "r_squared": sum(result_sheet[0][3]) / len(result_sheet[0][3]),
        "list_total_0": result_sheet[0][0],
        "list_err_0": result_sheet[0][1],
    }
    print(rtn)
    # rtn = {
    #     "best_err": sum(result_sheet[0][0]) / len(result_sheet[0][0]),
    #     "r_squared": sum(result_sheet[0][1]) / len(result_sheet[0][1]),
    # }
    json_dump = json.dumps(rtn)
    with open(settings['agent_dir'] + f'/{job_id}.rtn', 'w') as fresult:
        fresult.write(json_dump)


if __name__ == '__main__':

    print(sys.argv)
    job_id = sys.argv[1]
    config = json.loads(sys.argv[2])
    agent_id = sys.argv[3]
    agent_dir = sys.argv[4]
    settings = {
        'agent_id': agent_id,
        'agent_dir': agent_dir,
        'origin_path': '../Datasets/Dataset_res250/',
        'debug': False,
        'bp': False,
        'debug_stage': 1,
        # 'agent_dir': 'coffer_slot/',
        
        'batch': config['batch'],
        'accumulation_steps': 1,
        'epoch': 10,
        'trans_lr':config['trans_lr'],
        'nn_lr': config['nn_lr'],
        'es_mindelta': 1.0,
        'es_endure': 15,

        'word_col': [0, 1],
        'pos_col': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        'q_col': 3,

        'embedding_dim': config['gnn_dim'],
        'feedforward_dim': 4 * config['gnn_dim'],
        'num_head': 2,
        'num_layers': 2,
        'output_hidden_layers': 2,
        'vi_dim': config['batch'],
        'dropout': 0,

        'fold': config['fold'],
        'holdout': config['holdout'],
        'lowest_rank': 1,
        'seed': config['seed'],

        'eta': config['eta'],
        'T': config['T'] ,
        'ddim_step':config['ddim_step'],
        'cdim': config['gnn_dim'],
        'beta_1': 1e-4,
        'beta_T': 0.02,
        'channel': 32,
        'tdim': config['gnn_dim'],
        'nheads': config['nheads'], 
        'k': 20,
        'emb_hidden_dim': config['emb_hidden_dim'],
        'emb_dim': config['gnn_dim'],
        'input_emb': config['gnn_dim'],
        'latent_dim': 256,
    }
    dt_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    coffer_slot = myconfig1.coffer_path + str(job_id) + '-' + dt_string + '/'
    make_dir(coffer_slot)
    settings['coffer_slot'] = coffer_slot
    # settings['coffer_slot'] = './coffer_slot'
    train(settings)