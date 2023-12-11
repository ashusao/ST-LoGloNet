import numpy as np
import torch

def compute_errors(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)

    return mse.item(), rmse.item(), mae.item()


def valid(model, val_generator, device, dataset=None):
    model.eval()

    mean_mse_loss_vol = []
    mean_mae_loss_vol = []
    mean_rmse_loss_vol = []

    mean_mse_loss_inflow = []
    mean_mae_loss_inflow = []
    mean_rmse_loss_inflow = []

    mean_mse_loss_outflow = []
    mean_mae_loss_outflow = []
    mean_rmse_loss_outflow = []

    for i, (X_c, X_p, X_t, Y, meta) in enumerate(val_generator):
        X_c = X_c[:, :, :, :, :].type(torch.FloatTensor).to(device)
        X_p = X_p[:, :, :, :, :].type(torch.FloatTensor).to(device)
        X_t = X_t[:, :, :, :, :].type(torch.FloatTensor).to(device)
        TS_c = meta[0].type(torch.FloatTensor).to(device)  # 7 dow + 1 (wd/we)
        TS_p = meta[1].type(torch.FloatTensor).to(device)  # 7 dow + 1 (wd/we)
        TS_t = meta[2].type(torch.FloatTensor).to(device)  # 7 dow + 1 (wd/we)
        TS_Y = meta[3][:, :8].type(torch.FloatTensor).to(device)  # 7 dow + 1 (wd/we)
        pois = meta[4].type(torch.FloatTensor).to(device)
        Y = Y.type(torch.FloatTensor).to(device)

        # Forward pass
        pred = model(X_c, X_p, X_t, TS_c, TS_p, TS_t, pois)
        pred = pred.cpu().data.numpy()
        Y = Y.cpu().data.numpy()

        # denormalize each channel
        if dataset is not None:
            pred_vol = dataset.denormalize(pred[:, 0], 0)
            Y_vol = dataset.denormalize(Y[:, 0], 0)
            pred_inflow = dataset.denormalize(pred[:, 1], 1)
            Y_inflow = dataset.denormalize(Y[:, 1], 1)
            pred_outflow = dataset.denormalize(pred[:, 2], 2)
            Y_outflow = dataset.denormalize(Y[:, 2], 2)

        mse, rmse, mae = compute_errors(Y_vol, pred_vol)

        mean_mse_loss_vol.append(mse)
        mean_rmse_loss_vol.append(rmse)
        mean_mae_loss_vol.append(mae)

        mse, rmse, mae = compute_errors(Y_inflow, pred_inflow)

        mean_mse_loss_inflow.append(mse)
        mean_rmse_loss_inflow.append(rmse)
        mean_mae_loss_inflow.append(mae)

        mse, rmse, mae = compute_errors(Y_outflow, pred_outflow)

        mean_mse_loss_outflow.append(mse)
        mean_rmse_loss_outflow.append(rmse)
        mean_mae_loss_outflow.append(mae)

    mean_mse_loss_vol = np.mean(mean_mse_loss_vol)
    mean_rmse_loss_vol = np.mean(mean_rmse_loss_vol)
    mean_mae_loss_vol = np.mean(mean_mae_loss_vol)

    mean_mse_loss_inflow = np.mean(mean_mse_loss_inflow)
    mean_rmse_loss_inflow = np.mean(mean_rmse_loss_inflow)
    mean_mae_loss_inflow = np.mean(mean_mae_loss_inflow)

    mean_mse_loss_outflow = np.mean(mean_mse_loss_outflow)
    mean_rmse_loss_outflow = np.mean(mean_rmse_loss_outflow)
    mean_mae_loss_outflow = np.mean(mean_mae_loss_outflow)

    return mean_mse_loss_vol, mean_rmse_loss_vol, mean_mae_loss_vol, \
           mean_mse_loss_inflow, mean_rmse_loss_inflow, mean_mae_loss_inflow, \
           mean_mse_loss_outflow, mean_rmse_loss_outflow, mean_mae_loss_outflow

