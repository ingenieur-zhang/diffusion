import os, glob, inspect, time, math, torch, pickle
from torch import optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from scipy import stats
import vae
from Datasets import Dataloader_new
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

settings = {
    'origin_path': '../Datasets/Dataset_res250/',
    'debug': False,
    'bp': False,

    'batch': 2,
    'accumulation_steps': 256 // 128,
    'epoch': 200000,
    'trans_lr': 1e-7,
    'nn_lr': 1e-6,
    'es_mindelta': 1.0,
    'es_endure': 50,

    'word_col': [0, 1],
    'pos_col': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'q_col': 3,

    'embedding_dim': 256,
    'feedforward_dim': 4 * 256,
    'num_head': 2,
    'num_layers': 4,
    'output_hidden_layers': 2,
    'vi_dim': 256,
    'dropout': 0.1,

    'fold': 0,
    'holdout': 0,
    'lowest_rank': 1,
}


def save_square_img(contents, xlabel, ylabel, savename, title):
    plt.clf()
    plt.rcParams['font.size'] = 15

    max_value = max(contents[0].max(), contents[1].max())
    plt.scatter(contents[0], contents[1], marker=".")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout(pad=1, w_pad=1, h_pad=1)
    plt.gcf().set_size_inches(8, 8)
    plt.axis([0, max_value, 0, max_value])
    plt.axline((0, 0), slope=1, color='k')
    plt.savefig("%s.png" % (savename))
    plt.close()


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

# get standarization restore info
# with open(settings['origin_path'] + f'Folds_Info/norm_{fold}.info', 'rb') as f:
#     dic_op_minmax, dic_op_meanstd = pickle.load(f)

# build dataloader
dataset_train = Dataloader_new.IntpDataset(settings=settings, mask_distance=-1, call_name='train')
dataloader_tr = torch.utils.data.DataLoader(dataset_train, batch_size=settings['batch'], shuffle=True,
                                            collate_fn=Dataloader_new.collate_fn_img, num_workers=10,
                                            prefetch_factor=32,
                                            drop_last=True)
dataset_test = Dataloader_new.IntpDataset(settings=settings, mask_distance=-1, call_name='test')
dataloader_ex = torch.utils.data.DataLoader(dataset_test, batch_size=settings['batch'], shuffle=True,
                                            collate_fn=Dataloader_new.collate_fn_img, num_workers=10,
                                            prefetch_factor=32,
                                            drop_last=True)

with open(settings['origin_path'] + f'Folds_Info/norm_{0}.info', 'rb') as f:
    dic_op_minmax, dic_op_meanstd = pickle.load(f)

model = vae.VAE(settings['batch'])
# loss = vae.loss_function()
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

best_loss = 1e9
best_epoch = 0
best_err = float('inf')
es_counter = 0
valid_losses = []
train_losses = []
idx = 0
for epoch in range(200):
    print(f"Epoch {epoch}")
    model.train()
    train_loss = 0.
    train_num = len(dataloader_tr)
    for q_tokens_tr, input_lenths_tr, input_img_tr, answers_tr, input_series, answer_loc in dataloader_tr:
        # for idx, (x, _) in enumerate(train_loader):
        batch = input_lenths_tr.to(((device)))
        x = input_img_tr.float()
        # x = x.reshape(settings['batch'], 1, 256, 256).to(device)

        x = x.to(device)
        x_hat, mu, logvar = model(x)
        loss, bce, kld = vae.loss_function(x_hat, x, mu, logvar)

        loss = loss + kld
        train_loss += loss.item()
        loss = loss.to(device) / batch

        optimizer.zero_grad()
        loss.backward(torch.ones_like(x_hat))
        optimizer.step()
        idx = idx + 1
        # if idx % 100 == 0:
        print(f"Training loss {loss: .3f} \t Recon {x_hat / batch: .3f} \t KL {kld / batch: .3f} in Step {idx}")
    idx = 0
    train_losses.append(train_loss / train_num)

    model.eval()
    output_list = []
    target_list = []
    test_loss = 0
    with torch.no_grad():
        for q_tokens_ex, input_lenths_ex, input_img_ex, answers_ex, input_series, answer_loc in dataloader_tr:
            batch = input_lenths_ex
            x = input_img_ex.float()
            x = x.reshape(settings['batch'], 1, 256, 256).to(device)
            x = x.to(device)
            x_hat_ex, mu_ex, logvar_ex = model(x)
            loss_ex, bce_ex, kld_ex = vae.loss_function(x_hat_ex, x, mu_ex, logvar_ex)
            test_loss += loss_ex.item()
            output_list.append(x_hat_ex.squeeze().detach().cpu())
            target_list.append(answers_ex)

        output = torch.cat(output_list, 0)
        target = torch.cat(target_list, 0)

        test_means_origin = output * dic_op_meanstd['mcpm10'][1] + dic_op_meanstd['mcpm10'][0]
        test_y_origin = target * dic_op_meanstd['mcpm10'][1] + dic_op_meanstd['mcpm10'][0]

        mae = mean_squared_error(test_y_origin, test_means_origin, squared=False)
        r_squared = stats.pearsonr(test_y_origin, test_means_origin)
        # print(f'\t\t--------\n\t\tepoch: {str(epoch)}, inter_train_loss: {inter_loss}\n\t\t--------\n')
        # print(f'\t\t--------\n\t\ttest_loss: {str(test_loss)}, last best test_loss: {str(best_err)}\n\t\t--------\n')
        print(f'\t\t--------\n\t\tr_squared: {str(r_squared[0])}, MSE: {str(mae)}\n\t\t--------\n')

        # list_err.append(float(test_loss))
        # list_total.append(float(inter_loss))
        inter_loss = 0

        title = f'Fold{0}_holdout{0}_Md_all: MSE {round(mae, 2)} R2 {round(r_squared[0], 2)}'
        save_square_img(
            contents=[test_y_origin.numpy(), test_means_origin.numpy()],
            xlabel='targets_ex', ylabel='output_ex',
            savename=os.path.join('./coffer_slot', f'test_{epoch}'),
            title=title
        )
        if best_err - test_loss > settings['es_mindelta']:
            best_err = test_loss
            torch.save(model.state_dict(),  "vae_best_params")
            es_counter = 0
        else:
            es_counter += 1
            print(f"INFO: Early stopping counter {es_counter} of {settings['es_endure']}")
            if es_counter >= settings['es_endure']:
                print('INFO: Early stopping')
                break


