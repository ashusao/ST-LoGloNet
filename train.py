import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from data.helper import create_generators
from data.dataset import STDataset

from model.st_loglonet import STLoGloNet

from utils.early_stopping import EarlyStopping
from utils.eval import valid

def train(config):

    model_name = config['model']['name']
    # create checkpoint and run dirs if not exist
    out_dir = config['model']['dir'] + model_name + '/' + config['model']['exp']
    chkpt_dir = out_dir + '/checkpoint'
    writer_dir = out_dir + '/runs'
    os.makedirs(chkpt_dir, exist_ok=True)
    os.makedirs(writer_dir, exist_ok=True)

    # set seed for reproducbility
    seed = int(config['general']['seed'])
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # create generators
    T = int(config['data']['T'])
    train_dataset = STDataset(config=config, mode='train', T=T)
    train_generator, val_generator = create_generators(config=config, mode='train', dataset=train_dataset,
                                                       val_split=float(config['data']['val_split']))

    # tensorboard writer
    writer = SummaryWriter(writer_dir)

    len_c, len_p, len_t = int(config['data']['closeness']), int(config['data']['period']), int(config['data']['trend'])
    map_h, map_w = int(config['grid']['height']),  int(config['grid']['width'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_c, X_p, X_t, Y, meta = next(iter(train_generator))
    X_c = X_c[:, :, :, :, :].type(torch.FloatTensor).to(device)  # 0 channel: volume
    X_p = X_p[:, :, :, :, :].type(torch.FloatTensor).to(device)  # 0 channel: volume
    X_t = X_t[:, :, :, :, :].type(torch.FloatTensor).to(device)  # 0 channel: volume
    TS_c = meta[0].type(torch.FloatTensor).to(device)  # 7 dow + 1 (wd/we)
    TS_p = meta[1].type(torch.FloatTensor).to(device)  # 7 dow + 1 (wd/we)
    TS_t = meta[2].type(torch.FloatTensor).to(device)  # 7 dow + 1 (wd/we)
    pois = meta[4].type(torch.FloatTensor).to(device)  # 7 dow + 1 (wd/we)
    Ts_Y_str = meta[5]

    print(Ts_Y_str)
    print(X_c.shape, X_p.shape, X_t.shape, Y.shape, TS_c.shape, TS_p.shape, TS_t.shape, pois.shape)

    n_layer_spatial, n_layer_temporal, n_head_spatial, n_head_temporal, dim_feat = \
        int(config['stloglonet']['n_layer_spatial']), \
        int(config['stloglonet']['n_layer_temporal']), \
        int(config['stloglonet']['n_head_spatial']), \
        int(config['stloglonet']['n_head_temporal']), \
        int(config['stloglonet']['dim_feat'])

    #dim_feat = max(n_head_spatial, n_head_temporal) * dim_feat_head

    model = STLoGloNet(len_conf=(len_c, len_p, len_t), n_c=3, n_poi=10, dim_feat=dim_feat, map_w=map_w, map_h=map_h,
                       dim_ts_feat=10, n_head_spatial=n_head_spatial, n_head_temporal=n_head_temporal,
                       n_layer_spatial=n_layer_spatial, n_layer_temporal=n_layer_temporal,
                       dropout=0.1, device=device)
    model.to(device)
    writer.add_graph(model, [X_c, X_p, X_t, TS_c, TS_p, TS_t, pois])

    n_epoch = int(config['train']['n_epoch'])
    lr = float(config['train']['lr'])
    patience = int(config['train']['patience'])
    alpha = float(config['train']['alpha'])
    epoch_save = [0, n_epoch - 1] + list(range(0, n_epoch, 50))  # 1*1000

    loss_fn = nn.MSELoss()
    loss_fn_ = nn.L1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    es = EarlyStopping(patience=patience, mode='min', model=model, save_path=chkpt_dir + '/model.best.pth')

    loss_fn.to(device)
    loss_fn_.to(device)

    print('==== Starting Training ====')

    for e in range(n_epoch):
        model.train()
        for i, (X_c, X_p, X_t, Y, meta) in enumerate(train_generator):

            X_c = X_c.type(torch.FloatTensor).to(device)
            X_p = X_p.type(torch.FloatTensor).to(device)
            X_t = X_t.type(torch.FloatTensor).to(device)

            TS_c = meta[0].type(torch.FloatTensor).to(device)  # 7 dow + 1 (wd/we)
            TS_p = meta[1].type(torch.FloatTensor).to(device)  # 7 dow + 1 (wd/we)
            TS_t = meta[2].type(torch.FloatTensor).to(device)  # 7 dow + 1 (wd/we)
            pois = meta[4].type(torch.FloatTensor).to(device)
            Y = Y.type(torch.FloatTensor).to(device)

            outputs = model(X_c, X_p, X_t, TS_c, TS_p, TS_t, pois)

            loss = alpha * loss_fn(outputs, Y) + (1 - alpha) * loss_fn_(outputs, Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        t_mse_vol, t_rmse_vol, t_mae_vol, \
        t_mse_inflow, t_rmse_inflow, t_mae_inflow, \
        t_mse_outflow, t_rmse_outflow, t_mae_outflow = \
            valid(model, train_generator, device, train_dataset)

        print('Epoch [{}/{}], VOL Train MSE: {:.4f} RMSE: {:.4f} MAE: {:.4f} '
              .format(e + 1, n_epoch, t_mse_vol, t_rmse_vol, t_mae_vol))
        print('Epoch [{}/{}], INFLOW Train MSE: {:.4f} RMSE: {:.4f} MAE: {:.4f} '
              .format(e + 1, n_epoch, t_mse_inflow, t_rmse_inflow, t_mae_inflow))
        print('Epoch [{}/{}], OUTFLOW Train MSE: {:.4f} RMSE: {:.4f} MAE: {:.4f} '
              .format(e + 1, n_epoch, t_mse_outflow, t_rmse_outflow, t_mae_outflow))

        v_mse_vol, v_rmse_vol, v_mae_vol,  \
        v_mse_inflow, v_rmse_inflow, v_mae_inflow, \
        v_mse_outflow, v_rmse_outflow, v_mae_outflow = \
            valid(model, val_generator, device, train_dataset)

        print('Epoch [{}/{}], VOL Val MSE: {:.4f} RMSE: {:.4f} MAE: {:.4f} '
              .format(e + 1, n_epoch, v_mse_vol, v_rmse_vol, v_mae_vol))
        print('Epoch [{}/{}], INFLOW Val MSE: {:.4f} RMSE: {:.4f} MAE: {:.4f} '
              .format(e + 1, n_epoch, v_mse_inflow, v_rmse_inflow, v_mae_inflow))
        print('Epoch [{}/{}], OUTFLOW Val MSE: {:.4f} RMSE: {:.4f} MAE: {:.4f} '
              .format(e + 1, n_epoch, v_mse_outflow, v_rmse_outflow, v_mae_outflow))

        writer.add_scalars('loss', {'train_vol': t_mse_vol, 'val_vol': v_mse_vol,
                                    'train_inflow': t_mse_inflow, 'val_inflow': v_mse_inflow,
                                    'train_outflow': t_mse_outflow, 'val_outflow': v_mse_outflow}, e)

        v_mse = np.mean([v_mse_vol, v_mse_inflow, v_mse_outflow])
        if es.step(v_mse, e):
            print('early stopped! With val loss:', v_mse)
            break

        if e in epoch_save:
            torch.save(model.state_dict(), chkpt_dir + '/%08d_model.pth' % (e))
            torch.save({
                    'optimizer': optimizer.state_dict(),
                    'epoch': e,
                }, chkpt_dir + '/%08d_optimizer.pth' % (e))

            print(chkpt_dir + '/%08d_model.pth' % (e) +' saved!')


    print('==== Training Finished ====')

    # create generators
    test_dataset = STDataset(config=config, mode='test', T=T)
    test_generator = create_generators(config=config, mode='test', dataset=test_dataset)

    best_model = chkpt_dir + '/model.best.pth'
    model.load_state_dict(torch.load(best_model, map_location=lambda storage, loc: storage))
    model.to(device)
    model.eval()

    train_mse_vol, train_rmse_vol, train_mae_vol, \
    train_mse_inflow, train_rmse_inflow, train_mae_inflow, \
    train_mse_outflow, train_rmse_outflow, train_mae_outflow = \
        valid(model, train_generator, device, train_dataset)

    test_mse_vol, test_rmse_vol, test_mae_vol, \
    test_mse_inflow, test_rmse_inflow, test_mae_inflow, \
    test_mse_outflow, test_rmse_outflow, test_mae_outflow = \
        valid(model, test_generator,  device, test_dataset)

    print('VOL Train MSE: {:.4f} Train RMSE: {:.4f} Train MAE: {:.4f} Test MSE: {:.4f} Test RMSE: {:.4f} Test MAE: {:.4f}'
          .format(train_mse_vol, train_rmse_vol, train_mae_vol, test_mse_vol, test_rmse_vol, test_mae_vol))

    print('INFLOW Train MSE: {:.4f} Train RMSE: {:.4f} Train MAE: {:.4f} Test MSE: {:.4f} Test RMSE: {:.4f} Test MAE: {:.4f}'
        .format(train_mse_inflow, train_rmse_inflow, train_mae_inflow, test_mse_inflow, test_rmse_inflow, test_mae_inflow))

    print('OUTFLOW Train MSE: {:.4f} Train RMSE: {:.4f} Train MAE: {:.4f} Test MSE: {:.4f} Test RMSE: {:.4f} Test MAE: {:.4f}'
        .format(train_mse_outflow, train_rmse_outflow, train_mae_outflow, test_mse_outflow, test_rmse_outflow, test_mae_outflow))


