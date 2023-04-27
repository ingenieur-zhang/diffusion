import os, glob, inspect, time, math, torch, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import vae as v
import Dataloader as dl
import Transformer as tr
import torch.optim as optim
from torch.nn import functional as F
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from matplotlib.lines import Line2D

import gdm

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().detach().cpu().numpy())
            max_grads.append(p.grad.abs().max().detach().cpu().numpy())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    # plt.ylim(bottom=-0.001, top=0.1)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.gcf().set_size_inches(15, 10)
    plt.show()
    
    
def make_dir(path):
    try: 
        os.mkdir(path)
    except: 
        pass
    
    


    
def save_graph(contents, xlabel, ylabel, savename):
    np.save(savename, np.asarray(contents))
    plt.clf()
    plt.rcParams['font.size'] = 15
    plt.plot(contents, color='blue', linestyle="-", label="loss")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout(pad=1, w_pad=1, h_pad=1)
    plt.savefig("%s.png" %(savename))
    plt.close()
    
    





    
    
    
    
    
def save_img(contents, colors, labels, xlabel, ylabel, savename):
    plt.clf()
    plt.rcParams['font.size'] = 15
    num_cont = len(contents)
    for i in range(num_cont):
        plt.plot(contents[i], color=colors[i], linestyle="-", label=labels[i])
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout(pad=1, w_pad=1, h_pad=1)
    plt.gcf().set_size_inches(350, 10)
    plt.savefig("%s.png" %(savename))
    plt.close()


    
    
def training(settings):
    # Get device setting
    if not torch.cuda.is_available(): 
        device = torch.device("cpu")
        ngpu = 0
        print(f'Working on CPU')
    else:
        device = torch.device("cuda")
        ngpu = torch.cuda.device_count()
        if ngpu > 1:
            device_list = [i for i in range(ngpu)]
            print(f'Working on multi-GPU {device_list}')
        else:
            print(f'Working on single-GPU')
    with open(settings['origin_path'] + 'norm.info', 'rb') as f:
        dic_op_minmax = pickle.load(f)

    dataset_train = dl.IntpDataset(origin_path=settings['origin_path'], call_path=settings['call_path'], call_name='train', debug=settings['debug'])
    dataloader_tr = torch.utils.data.DataLoader(dataset_train, batch_size=settings['batch'], shuffle=True, collate_fn=dl.collate_fn, num_workers=4, prefetch_factor=4)


    dataset_test = dl.IntpDataset(origin_path=settings['origin_path'], call_path=settings['call_path'], call_name='test', debug=settings['debug'])
    dataloader_ex = torch.utils.data.DataLoader(dataset_test, batch_size=settings['batch'], shuffle=False, collate_fn=dl.collate_fn, num_workers=4, prefetch_factor=4)
    model = tr.Transformer(
        word_col=settings['word_col'], pos_col=settings['pos_col'], embedding_dim=settings['embedding_dim'], 
        feedforward_dim=settings['feedforward_dim'], output_dim=settings['output_dim'], 
        num_head=settings['num_head'], num_layers=settings['num_layers'], 
        dropout=0.1, activation="gelu", device=device
    )
    if ngpu > 1:
        model = torch.nn.DataParallel(model, device_ids=device_list)
    # model_vae = v.VAE().to(device)
    epochs = settings['epoch']
    batch_size = settings['batch']
    print("\nTraining to %d epochs (%d of minibatch size)" %(epochs, batch_size))
    make_dir(path="results")
    result_list = ["tr_resotring", ]
    for result_name in result_list: 
        make_dir(path=os.path.join("results", result_name))

    start_time = time.time()
    list_total = []

    loss = torch.nn.MSELoss()
    loss_vae = v.loss_function()
    optimizer = optim.Adam(model.parameters(), lr=settings['lr'], weight_decay=1e-4)
    # optimizer_vae = optim.Adam(model_vae.parameters(), lr=settings['lr'], weight_decay=1e-4)
    best_err = float('inf')
    for epoch in range(epochs):
        # Test batch
        output_list = []
        target_list = []
        for inputs_ex, lengths_ex, targets_ex in dataloader_ex:
            # output_vae = model_vae(inputs_ex.to(device))
            # output_ex = model(output_vae, lengths_ex.to(device))
            output_ex = model(inputs_ex.to(device), lengths_ex.to(device))
            output_list.append(output_ex.squeeze().detach().cpu())
            target_list.append(targets_ex)
        output = torch.cat(output_list, 0)
        target = torch.cat(target_list, 0)
        
        test_means_origin = output * dic_op_minmax['mcpm10'][1] + dic_op_minmax['mcpm10'][0]
        test_y_origin = target * dic_op_minmax['mcpm10'][1] + dic_op_minmax['mcpm10'][0]

        mae = mean_absolute_error(test_y_origin, test_means_origin)
        r_squared = stats.pearsonr(test_y_origin, test_means_origin)
        print(f'\t\t--------\n\t\tr_squared: {str(r_squared[0])}, MAE: {str(mae)}, last best MAE: {str(best_err)}\n\t\t--------\n')
        print(f'\t\t--------\n\t\tDiffer: {test_means_origin.max() - test_means_origin.min()}, count: {test_y_origin.size(0)}\n\t\t--------\n')
        if mae < best_err:
            best_err = mae
        
        save_img(
            contents=[test_y_origin.numpy(), test_means_origin.numpy()], 
            colors=['red', 'blue'], labels=['targets_ex', 'output_ex'], 
            xlabel='index', ylabel='output(zs)', 
            savename=os.path.join("results", "tr_resotring", "%08d" %(epoch))
        )
        targets_ex = test_y_origin.unsqueeze(1)
        output_ex = test_means_origin.unsqueeze(1)
        diff_ex = targets_ex - output_ex
        pd_out = pd.DataFrame(
            torch.cat(
                (targets_ex, output_ex, diff_ex), 1
            ).numpy()
        )
        pd_out.to_csv(os.path.join("results", "tr_resotring", "%08d.csv" %(epoch)), index=False)

        # Train batches
        total_loss = 0
        total_loss_vae = 0
        mini_batch = 0
        for inputs_tr, lengths_tr, targets_tr in dataloader_tr:
            output_tr = model(inputs_tr.to(device), lengths_tr.to(device))
            targets_tr = targets_tr.unsqueeze(1).to(device)
            # x_hat, mu, log_var = model_vae(inputs_tr)
            # batch_loss_vae = v.loss_function(x_hat, inputs_tr, mu, log_var)
            batch_loss = loss(output_tr, targets_tr)
            optimizer.zero_grad()
            # optimizer_vae.zero_grad()
            batch_loss.backward()
            # batch_loss_vae.backwards()
            # optimizer_vae.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.02)
            if mini_batch % 100 == 0:
                plot_grad_flow(model.named_parameters())
            optimizer.step()

            print(f'\tEpoch {epoch}, Iter {mini_batch} - Loss: {batch_loss.item()}')
            # print(f'\tEpoch {epoch}, Iter {mini_batch} - Loss: {batch_loss.item()}, Loss_vae: {batch_loss_vae.item()}')
            mini_batch += 1
            total_loss += batch_loss.item()
            # total_loss_vae += batch_loss_vae.item()
        print(f"Epoch {epoch} - Loss: {total_loss:.3f}")
        # print(f"Epoch {epoch} - Loss: {total_loss:.3f}, Loss_vae: {total_loss_vae:.3f}")
        list_total.append(total_loss)
        # list_total.append(total_loss_vae)
        with open(os.path.join("results", "train.log"), 'a') as f_out:
            f_out.write(f'Epoch {epoch} - Loss: {total_loss:.3f}\n')
            # f_out.write(f'Epoch {epoch} - Loss: {total_loss:.3f}, Loss_vae: {total_loss_vae:.3f}\n')

        torch.save(model.state_dict(), PACK_PATH+"/results/params")
        # torch.save(model_vae.state_dict(), PACK_PATH + "/results/params_vae")

    elapsed_time = time.time() - start_time
    print("Elapsed: "+str(elapsed_time))

    save_graph(contents=list_total, xlabel="Iteration", ylabel="Total Loss", savename=PACK_PATH+"/results/loss")

    
def test(neuralnet, dataset):
    return 0
