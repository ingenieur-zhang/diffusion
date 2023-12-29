import os, glob, inspect, time, math, torch, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import myconfig1
sys.path.append('..')
import Datasets.Dataloader_PEGNN as dl
import Trans_scrpit.Transformer_gnn_new as tr
import torch.optim as optim
from torch.nn import functional as F
import mpl_scatter_density
import random
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
# import Trans_scrpit.myconfig as myconfig
from datetime import datetime
import json
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import LambdaLR
from Diffusion.pecgn_test import GAT_Unet
from Diffusion.Diffusion_ddim_test import GaussianDiffusionTrainer, GaussianDiffusionSampler
from Diffusion.vae import VAE


def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results

    Arguments:
        seed {int} -- Number of the seed
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_dir(path):
    try:
        os.mkdir(path)
    except:
        pass


def save_square_img(contents, xlabel, ylabel, savename, title):
    # "Viridis-like" colormap with white background
    white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
        (0, '#ffffff'),
        (1e-20, '#440053'),
        (0.2, '#404388'),
        (0.4, '#2a788e'),
        (0.6, '#21a784'),
        (0.8, '#78d151'),
        (1, '#fde624'),
    ], N=256)
    
    plt.clf()
    plt.rcParams['font.size'] = 15
    
    max_value = max(contents[0].max(), contents[1].max())
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    fig.set_size_inches(7, 6)
    ax.set_position([0, 0, 0.8, 1])
    
    density = ax.scatter_density(contents[0], contents[1], cmap=white_viridis)
    fig.colorbar(density, label='Number of points')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout(pad=1, w_pad=1, h_pad=1)
    
    
    ax.set_xlim([0, max_value])
    ax.set_ylim([0, max_value])
    ax.plot([0, max_value], [0, max_value], color='k')
    fig.savefig("%s.png" %(savename))
    plt.close(fig)


def lr_schedule(epoch):
    num_epochs = 200000
    warmup_steps = num_epochs * 0.005
    if epoch < warmup_steps:
        return float(epoch) / float(warmup_steps)
    else:
        return 1.0 - float(epoch - warmup_steps) / float(num_epochs - warmup_steps)


def var_loss(output, targets, mu, log_var):
    # mse_loss
    recon_loss = F.mse_loss(output, targets)
    # KLD
    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0) / (5 * mu.size(1))

    loss = recon_loss + kld_loss

    return loss, recon_loss, kld_loss


def training(settings):
    seed_everything(settings['seed'])
    print('fuck')
    fold = settings['fold']
    holdout = settings['holdout']
    lowest_rank = settings['lowest_rank']

    # coffer_slot = './coffer_slot'
    # make_dir(coffer_slot)
    coffer_slot = settings['coffer_slot'] + f'{fold}/'   # coffer
    make_dir(coffer_slot)
    
    # print sweep settings
    print(json.dumps(settings, indent=2, default=str))

    # Get device setting
    if not torch.cuda.is_available(): 
        device = torch.device("cpu")
        ngpu = 0
        workers = 16
        print(f'Working on CPU')
    else:
        device = torch.device("cuda")
        ngpu = torch.cuda.device_count()
        if ngpu > 1:
            device_list = [i for i in range(ngpu)]
            workers = 16
            print(f'Working on multi-GPU {device_list}')
        else:
            workers = 10
            print(f'Working on single-GPU')

    # get standarization restore info
    with open(settings['origin_path'] + f'Folds_Info/norm_{fold}.info', 'rb') as f:
        dic_op_minmax, dic_op_meanstd = pickle.load(f)

    # build dataloader
    dataset_train = dl.IntpDataset(settings=settings, mask_distance=-1, call_name='train')
    dataloader_tr = torch.utils.data.DataLoader(dataset_train, batch_size=settings['batch'], shuffle=True, collate_fn=dl.collate_fn, num_workers=workers, prefetch_factor=32, drop_last=True)
    print('train data loaded')
    
    test_dataloaders = []
    test_trans_dataloaders = []
    
    dataset_test = dl.IntpDataset(settings=settings, mask_distance=20, call_name='test')
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=settings['batch'], shuffle=False, collate_fn=dl.collate_fn, num_workers=workers, prefetch_factor=32, drop_last=True)
    print('test data loaded')
     
    # build model
    model = tr.Transformer(settings=settings, device=device)
    model_gnn = GAT_Unet(tdim=settings['tdim'], nheads=settings['nheads'], T=settings['T'], cemb_dim=settings['embedding_dim'], emb_hidden_dim=settings['emb_hidden_dim'], emb_dim=settings['emb_dim'], input_emb=settings['input_emb'], num_features_in=1, num_features_out=1,  k=20, conv_dim=64).to(device)
    model_gnn = model_gnn.float()
    small_model = VAE(input_dim=1, latent_dim=256).to(device)
    # small_model.load_state_dict(torch.load('./coffer_slot/vae_test_64_lr_0.01_latent_dim_256'))
    print('model loaded')
    
    trainer = GaussianDiffusionTrainer(
        model_gnn, settings['beta_1'], settings['beta_T'], settings['T']).to(device)
    sampler = GaussianDiffusionSampler(
        model_gnn, settings['beta_1'], settings['beta_T'], settings['T'], settings['ddim_step'], settings['eta']).to(device)
    print('trainer and sampler loaded')
    
    loss = var_loss
    optimizer = optim.AdamW([
        {'params': model_gnn.time_embedding.parameters(), 'lr': settings['nn_lr'], 'weight_decay': 0},
        {'params': model_gnn.attentions.parameters(), 'lr': settings['trans_lr'], 'weight_decay': 0},
        {'params': model_gnn.out_att.parameters(), 'lr': settings['trans_lr'], 'weight_decay': 0},
        {'params': model_gnn.mid_att.parameters(), 'lr': settings['trans_lr'], 'weight_decay': 0},
        {'params': model_gnn.dec.parameters(), 'lr': settings['nn_lr'], 'weight_decay': 0},
        {'params': model_gnn.spenc.parameters(), 'lr': settings['nn_lr'], 'weight_decay': 0},
        {'params': model_gnn.valuelayer.parameters(), 'lr': settings['nn_lr'], 'weight_decay': 0},
        {'params': model_gnn.tif_conv.parameters(), 'lr': settings['nn_lr'], 'weight_decay': 0},
        {'params': [model.cls_token], 'lr': settings['nn_lr'], 'weight_decay': 0},
        {'params': model.s_embedding_layer.parameters(), 'lr': settings['nn_lr'], 'weight_decay': 0},
        {'params': model.p_embedding_layer.parameters(), 'lr': settings['nn_lr'], 'weight_decay': 0},
        {'params': model.transformer.parameters(), 'lr': settings['trans_lr'], 'weight_decay': 0},
        {'params': model.fc_mu.parameters(), 'lr': settings['nn_lr'], 'weight_decay': 0}, 
        {'params': model.fc_var.parameters(), 'lr': settings['nn_lr'], 'weight_decay': 0},
        {'params': model.output.parameters(), 'lr': settings['nn_lr'], 'weight_decay': 0},
    ],
        betas=(0.9, 0.999), eps=1e-8
    )
    scheduler = LambdaLR(optimizer, lr_lambda=lr_schedule)

    if ngpu > 1:
        model_gnn = torch.nn.DataParallel(model_gnn, device_ids=device_list)
        model = torch.nn.DataParallel(model, device_ids=device_list)

    
   
   
#     model_gnn.load_state_dict(torch.load( "nan_best_params_batch_64_lr_0.001_tr_lr_0.001_T_100_ddim_20_cdim_64noqtoken"))
                                          
#     model.load_state_dict(torch.load(  "nan_tr_best_params_batch_64_lr_0.001_tr_lr_0.001_T_100_ddim_20_cdim_64noqtoken"))

    # set training loop
    epochs = settings['epoch']
    batch_size = settings['batch']
    print("\nTraining to %d epochs (%d of minibatch size)" % (epochs, batch_size * settings['accumulation_steps']))

    # fire training loop
    start_time = time.time()
    list_total = []
    list_err = []
    best_err = float('inf')
    es_counter = 0
    inter_loss = 0
    iter_counter = 0
    mini_loss = 0
    loss_gragh = []
    # while True:

    for epoch in range(epochs):
        total_loss = 0
        mini_batch = 0
        # iter_counter = 0
        if es_counter >= settings['es_endure']:
            print('INFO: Early stopping')
            break
        for x_b, c_b, y_b, x_pm, q_tr, input_lenths, tif in dataloader_tr:
            model.train()
            model_gnn.train()
            # small_model.train()
            condition = model(input_lenths.to(device), q_tr.to(device))
            # x_pm, mu, var = small_model(x_pm.to(device))
            print('condition: ', torch.isnan(condition).any())
            t_loss = 0
            iter_counter += 1
            real_iter = iter_counter // settings['accumulation_steps'] + 1
            
            # q_gnn[:, :, 2:3] = y_b
            noise_list_full, target_list_full, coord_list_full, mask, batch_loss = trainer(x_pm.to(device), condition.to(device), y_b.to(device), c_b.to(device), input_lenths.to(device), tif.to(device))
           
            batch_loss =  batch_loss / settings['batch']
            batch_loss.backward()
            total_loss += batch_loss.item()
            t_loss += batch_loss.item()
            inter_loss += batch_loss.item()
            mini_loss += batch_loss.item()
            torch.nn.utils.clip_grad_norm_(model_gnn.parameters(), 0.02)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.02)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            print(f'\tEpoch {epoch}, Iter {iter_counter} - Loss: {batch_loss.item()}')
            # loss_gragh.append((iter_counter, t_loss))
            # file.write(f'{iter_counter} {t_loss}\n')
            if (iter_counter+1) % 50 ==0:
                print(f'\tEpoch {epoch}, Iter {iter_counter} - mini_loss: {mini_loss}')
                mini_loss = 0
                print(f"Epoch {epoch} - Loss: {total_loss:.3f}")           
            # if iter_counter > 300:
            #     break
         
        # Test batch
            if (iter_counter + 1) % (50 * settings['accumulation_steps']) == 0:
                model.eval()
                model_gnn.eval()
                output_list = []
                output_result = []
                target_list = []
                test_loss = 0
                i = 0
                with torch.no_grad():
                    for x_b, c_b, y_b, x_pm, q_tr, input_lenths, tif in dataloader_test:
                        condition = model(input_lenths.to(device), q_tr.to(device))
                        # x_pm, _, _ = small_model(x_pm.to(device))
                        result, mask = sampler(x_pm.to(device), condition.to(device), y_b.to(device), c_b.to(device), input_lenths.to(device), tif.to(device))

                        filtered_result = result[:, 0:1, -1]   # [mask[:, 0:1, -1].bool()]
                        filtered_target = y_b[:, 0:1, :].to(device)  # [mask[:, 0:1, 0:1].bool()]
  
                        filtered_result = filtered_result.squeeze()
                        filtered_target = filtered_target.squeeze()
        
                        output_list.append(filtered_result)
                        target_list.append(filtered_target)
                        output_result = []

                        # q_gnn[:, :, -1] = y_b.squeeze()
                        noise_list_full, target_list_full, coord_list_full, mask, batch_loss = trainer(x_pm.to(device), condition.to(device), y_b.to(device), c_b.to(device), input_lenths.to(device), tif.to(device))

                        batch_loss =  batch_loss.sum() / settings['batch']
                        test_loss += batch_loss.item()

                        # i += 1
                        # if i > 100:
                        #     break
                    output = torch.cat(output_list, 0)
                    target = torch.cat(target_list, 0)

                test_means_origin = output * dic_op_meanstd['mcpm10'][1] + dic_op_meanstd['mcpm10'][0]
                test_y_origin = target * dic_op_meanstd['mcpm10'][1] + dic_op_meanstd['mcpm10'][0]

                test_means_origin = test_means_origin.cpu()
                test_y_origin = test_y_origin.cpu()

                mae = mean_squared_error(test_y_origin, test_means_origin, squared=False)
                mae = round(mae, 2)
                r_squared = stats.pearsonr(test_y_origin, test_means_origin)
                print(f'\t\t--------\n\t\tIter: {str(real_iter)}\n\t\t--------\n')
                print(
                    f'\t\t--------\n\t\ttest_loss: {str(test_loss)}, last best test_loss: {str(best_err)}\n\t\t--------\n')
                print(f'\t\t--------\n\t\tr_squared: {str(r_squared[0])}, MSE: {str(mae)}\n\t\t--------\n')

                list_err.append(float(test_loss))
                list_total.append(float(inter_loss))
                inter_loss = 0

                title = f'Fold{fold}_holdout{holdout}_Md_all: MSE ' + str(mae) + f' R2 {round(r_squared[0], 2)}'
                if best_err - test_loss > settings['es_mindelta']:
                    best_err = test_loss
                    torch.save(model_gnn.state_dict(), coffer_slot + "/best_params"+ "_batch_" + str(settings['batch']) + "_lr_" + str(settings['nn_lr']) + "_tr_lr_" + str(settings['trans_lr']) + "_T_" + str(settings['T']) + "_ddim_" + str(settings['ddim_step']) + "_cdim_" + str(settings['embedding_dim']) + "_test")
                    torch.save(model.state_dict(), coffer_slot + "/tr_best_params"+ "_batch_" + str(settings['batch']) + "_lr_" + str(settings['nn_lr']) + "_tr_lr_" + str(settings['trans_lr']) + "_T_" + str(settings['T']) + "_ddim_" + str(settings['ddim_step']) + "_cdim_" + str(settings['embedding_dim']) + "_test")
                    save_square_img(
                        contents=[test_y_origin.numpy(), test_means_origin.numpy()],
                        xlabel='targets_ex', ylabel='output_ex',
                        savename=os.path.join(coffer_slot, f'diffusion_test_{iter_counter}' +  "_batch_" + str(settings['batch']) + "_lr_" + str(settings['nn_lr']) + "_tr_lr_" + str(settings['trans_lr']) + "_T_" + str(settings['T']) + "_ddim_" + str(settings['ddim_step']) + "_cdim_" + str(settings['embedding_dim']) + "_test"),
                        title=title
                    )
                    es_counter = 0
                else:
                    es_counter += 1
                    print(f"INFO: Early stopping counter {es_counter} of {settings['es_endure']}")
                    if es_counter >= settings['es_endure']:
                        print('INFO: Early stopping')
                        break
            
       
            # if iter_counter > settings['epoch']:
            #     break


    elapsed_time = time.time() - start_time
    print("Elapsed: " + str(elapsed_time))

    return list_total, list_err


def evaluate(settings):
    # scan the correct coffer
    fold = settings['fold']
    holdout = settings['holdout']
    lowest_rank = settings['lowest_rank']

    # coffer_slot = './coffer_slot'
    # make_dir(coffer_slot)
    coffer_slot = settings['coffer_slot'] + f'{fold}/'
    # scan the correct coffer
    coffer_dir = ''
    dirs = os.listdir(myconfig1.coffer_path)
    dirs.sort()
    # for dir in dirs:
    #     if job_id in dir:
    #         coffer_dir = myconfig.coffer_path + dir + f'/{fold}/'
    #         break

    # Get device setting
    if not torch.cuda.is_available(): 
        device = torch.device("cpu")
        ngpu = 0
        workers = 16
        print(f'Working on CPU')
    else:
        device = torch.device("cuda")
        ngpu = torch.cuda.device_count()
        if ngpu > 1:
            device_list = [i for i in range(ngpu)]
            workers = 16
            print(f'Working on multi-GPU {device_list}')
        else:
            workers = 10
            print(f'Working on single-GPU')
    
    # get standarization restore info
    with open(settings['origin_path'] + f'Folds_Info/norm_{fold}.info', 'rb') as f:
        dic_op_minmax, dic_op_meanstd = pickle.load(f)
        
     # build model
    model = tr.Transformer(settings=settings, device=device)
    model_gnn = GAT_Unet(tdim=settings['tdim'], nheads=settings['nheads'], T=settings['T'], cemb_dim=settings['embedding_dim'], emb_hidden_dim=settings['emb_hidden_dim'], emb_dim=settings['emb_dim'], input_emb=settings['input_emb'], num_features_in=1, num_features_out=1,  k=20, conv_dim=64).to(device)
    model_gnn = model_gnn.float()
    # small_model = VAE(input_dim=1, latent_dim=256).to(device)
    # small_model.load_state_dict(torch.load(coffer_dir + "idw_test"))
    # model_gnn.load_state_dict(torch.load(coffer_dir + "gnn_idw_test"))
 
    print('model loaded')
    
    trainer = GaussianDiffusionTrainer(
        model_gnn, settings['beta_1'], settings['beta_T'], settings['T']).to(device)
    sampler = GaussianDiffusionSampler(
        model_gnn, settings['beta_1'], settings['beta_T'], settings['T'], settings['ddim_step'], settings['eta']).to(device)
    print('trainer and sampler loaded')
    
    if ngpu > 1:
        model_gnn = torch.nn.DataParallel(model_gnn, device_ids=device_list)
        model = torch.nn.DataParallel(model, device_ids=device_list)       
    
    model_gnn.load_state_dict(torch.load(coffer_slot + "/best_params"+ "_batch_" + str(settings['batch']) + "_lr_" + str(settings['nn_lr']) + "_tr_lr_" + str(settings['trans_lr']) + "_T_" + str(settings['T']) + "_ddim_" + str(settings['ddim_step']) + "_cdim_" + str(settings['embedding_dim']) + "_test"))
                                          
    model.load_state_dict(torch.load(coffer_slot + "/tr_best_params"+ "_batch_" + str(settings['batch']) + "_lr_" + str(settings['nn_lr']) + "_tr_lr_" + str(settings['trans_lr']) + "_T_" + str(settings['T']) + "_ddim_" + str(settings['ddim_step']) + "_cdim_" + str(settings['embedding_dim']) + "_test"))
    
    test_dataloaders = []
    test_trans_dataloaders = []
    
    # dataset_test = dl.IntpDataset(settings=settings, mask_distance=20, call_name='test')
    # dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=settings['batch'], shuffle=False, collate_fn=dl.collate_fn, num_workers=workers, prefetch_factor=32, drop_last=True)

    
    rtn_mae_list = []
    rtn_rsq_list = []
    for mask_distance in [0, 20, 50]:
        dataset_eval = dl.IntpDataset(settings=settings, mask_distance=mask_distance, call_name='eval')
        dataloader_ev = torch.utils.data.DataLoader(dataset_eval, batch_size=settings['batch'], shuffle=False, collate_fn=dl.collate_fn, num_workers=workers, prefetch_factor=32, drop_last=True)
    

#         model_gnn.load_state_dict(torch.load("./coffer_slot/nan_best_params_batch_64_lr_0.0001_tr_lr_0.0001_T_20_ddim_20_cdim_64cheat"))
                                          
#     model.load_state_dict(torch.load("./coffer_slot/nan_tr_best_params_batch_64lr0.0001_tr_lr_0.0001_T_20_ddim_20_cdim_64cheat"))
    
    # set training loop

        batch_size = settings['batch']
        
    
        # fire training loop
        start_time = time.time()
        list_total = []
        list_err = []
        best_err = float('inf')
        es_counter = 0
        
        iter_counter = 0
        inter_loss = 0
        test_loss = 0
        mini_loss = 0
        sample_loss = 0
        loss_gragh = []
        output_list = []
        target_list = []
        with torch.no_grad():
            i = 0
            for x_b, c_b, y_b, x_pm, q_tr, input_lenths, tif in dataloader_ev:
                condition = model(input_lenths.to(device), q_tr.to(device))
                result, mask = sampler(x_pm.to(device), condition.to(device), y_b.to(device), c_b.to(device), input_lenths.to(device), tif.to(device))
                # predicted, mu, var = small_model(x_pm.to(device))
                filtered_result = result[:, 0:1, :].to(device)
                filtered_target = y_b[:, 0:1, :].to(device)
    
                filtered_result = filtered_result.squeeze()
                filtered_target = filtered_target.squeeze()
    
                output_list.append(filtered_result)
                target_list.append(filtered_target)
                output_result = []
                # i += 1
                # if i > 100:
                #     break
    
    
            output = torch.cat(output_list, 0)
            target = torch.cat(target_list, 0)
    
            test_means_origin = output * dic_op_meanstd['mcpm10'][1] + dic_op_meanstd['mcpm10'][0]
            test_y_origin = target * dic_op_meanstd['mcpm10'][1] + dic_op_meanstd['mcpm10'][0]
    
            test_means_origin = test_means_origin.cpu()
            test_y_origin = test_y_origin.cpu()
    
            mae = mean_squared_error(test_y_origin, test_means_origin, squared=False)
            mae = round(mae, 2)
            r_squared = stats.pearsonr(test_y_origin, test_means_origin)

            rtn_mae_list.append(float(mae))
            rtn_rsq_list.append(float(r_squared[0]))
          
            inter_loss = 0
    
            title = f'Fold{fold}_holdout{holdout}_Md_all: MSE ' + str(mae) + f' R2 {round(r_squared[0], 2)}'
    
            save_square_img(
                contents=[test_y_origin.numpy(), test_means_origin.numpy()],
                xlabel='targets_ex', ylabel='output_ex',
                savename=os.path.join(coffer_slot, f'result_{mask_distance}'),
                title=title
            )
            targets_ex = test_y_origin.unsqueeze(1)
            output_ex = test_means_origin.unsqueeze(1)
            diff_ex = targets_ex - output_ex
            pd_out = pd.DataFrame(
                torch.cat(
                    (targets_ex, output_ex, diff_ex), 1
                ).numpy()
            )
            pd_out.columns = ['Target', 'Output', 'Diff']
            pd_out.to_csv(os.path.join(coffer_slot, f'result_{mask_distance}.csv'), index=False)

    return rtn_mae_list, rtn_rsq_list