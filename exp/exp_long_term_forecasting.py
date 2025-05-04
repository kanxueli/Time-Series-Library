from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single
import json
import utils.run_metrics as run_metrics
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm 
import copy
import random
import pickle

from tslearn.clustering import KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance


warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        # 设置多任务学习的参数
        self.use_multi_task = args.use_multi_task == 1 if hasattr(args, 'use_multi_task') else False
        self.mask_rate = args.mask_rate if hasattr(args, 'mask_rate') else 0.15
        
        # 如果启用多任务学习，确保mask_rate大于0
        if self.use_multi_task and self.mask_rate <= 0:
            print("Warning: use_multi_task=True but mask_rate<=0, setting mask_rate to default 0.15")
            self.mask_rate = 0.15
            
        print(f"Multi-task learning: {'Enabled' if self.use_multi_task else 'Disabled'}, Mask rate: {self.mask_rate}")

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
 

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # 使用统一的多任务学习标志
                use_multi_task = self.use_multi_task
                
                if use_multi_task:
                    # 创建掩码（只用于模型输入，验证时只关注预测性能）
                    B, T, N = batch_x.shape
                    mask = torch.rand((B, T, N)).to(self.device)
                    mask[mask <= self.mask_rate] = 0  # masked
                    mask[mask > self.mask_rate] = 1  # remained
                    
                    # 验证时使用未掩码的原始数据进行预测，只评估预测能力
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            # 使用原始未掩码数据评估预测性能
                            outputs, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, mask)
                    else:
                        outputs, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, mask)
                else:
                    # 原始单任务预测
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def apply_random_mask(self, x):
        """
        对输入应用随机掩码，用于插值任务
        
        Args:
            x (torch.Tensor): 输入张量，形状为 [B, T, N]
            
        Returns:
            tuple: (掩码后的输入, 掩码)，掩码中0表示被掩盖的位置，1表示保留的位置
        """
        B, T, N = x.shape
        mask = torch.rand((B, T, N)).to(self.device)
        mask[mask <= self.mask_rate] = 0  # masked
        mask[mask > self.mask_rate] = 1  # remained
        inp = x.masked_fill(mask == 0, 0)
        return inp, mask

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # 使用统一的多任务学习标志
        use_multi_task = self.use_multi_task

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            forecast_losses = []
            imputation_losses = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                # 准备解码器输入
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # 多任务学习模式：预测 + 重建
                if use_multi_task:
                    # 创建掩码
                    B, T, N = batch_x.shape
                    mask = torch.rand((B, T, N)).to(self.device)
                    mask[mask <= self.mask_rate] = 0  # masked
                    mask[mask > self.mask_rate] = 1  # remained
                    
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            forecast_out, imputation_out = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, mask)
                            
                            # 预测任务损失
                            f_dim = -1 if self.args.features == 'MS' else 0
                            forecast_out = forecast_out[:, -self.args.pred_len:, f_dim:]
                            batch_y_pred = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                            forecast_loss = criterion(forecast_out, batch_y_pred)
                            
                            # 重建任务损失
                            imputation_out = imputation_out[:, :, f_dim:]
                            batch_x_imp = batch_x[:, :, f_dim:]
                            mask = mask[:, :, f_dim:]
                            imputation_loss = criterion(imputation_out[mask == 0], batch_x_imp[mask == 0])
                            
                            # 总损失
                            loss = forecast_loss + self.args.mr_loss_ratio * imputation_loss
                            forecast_losses.append(forecast_loss.item())
                            imputation_losses.append(imputation_loss.item())
                    else:
                        forecast_out, imputation_out = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, mask)
                        
                        # 预测任务损失
                        f_dim = -1 if self.args.features == 'MS' else 0
                        forecast_out = forecast_out[:, -self.args.pred_len:, f_dim:]
                        batch_y_pred = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        forecast_loss = criterion(forecast_out, batch_y_pred)
                        
                        # 重建任务损失
                        imputation_out = imputation_out[:, :, f_dim:]
                        batch_x_imp = batch_x[:, :, f_dim:]
                        mask = mask[:, :, f_dim:]
                        imputation_loss = criterion(imputation_out[mask == 0], batch_x_imp[mask == 0])
                        
                        # 总损失
                        loss = forecast_loss + self.args.mr_loss_ratio * imputation_loss
                        forecast_losses.append(forecast_loss.item())
                        imputation_losses.append(imputation_loss.item())
                else:
                    # 原始单任务模式
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            f_dim = -1 if self.args.features == 'MS' else 0
                            outputs = outputs[:, -self.args.pred_len:, f_dim:]
                            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                            loss = criterion(outputs, batch_y)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    if use_multi_task and forecast_losses and imputation_losses:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}, forecast_loss: {3:.7f}, imputation_loss: {4:.7f}".format(
                            i + 1, epoch + 1, loss.item(), forecast_loss.item(), imputation_loss.item()))
                    else:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            
            # 输出多任务学习的详细损失
            if use_multi_task and forecast_losses and imputation_losses:
                avg_forecast_loss = np.average(forecast_losses)
                avg_imputation_loss = np.average(imputation_losses)
                print("Train details - forecast_loss: {:.7f}, imputation_loss: {:.7f}".format(
                    avg_forecast_loss, avg_imputation_loss))
            
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        # 保存模型和优化器状态
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        # 保存优化器状态
        optimizer_status = {
            'optimizer_state_dict': model_optim.state_dict(),
            'epoch': epoch,
            'train_loss': train_loss,
            'vali_loss': vali_loss,
            'test_loss': test_loss
        }
        torch.save(optimizer_status, os.path.join(path, 'optimizer.pt'))

        return self.model

    def save_predictions(self, preds, trues, file_path="predictions_and_trues.json"):
        """
        Save the predictions and true values to a file for later evaluation.
        If the file already exists, append the new predictions and true values.

        Parameters:
        - preds (ndarray): The predicted values from the model.
        - trues (ndarray): The ground truth values.
        - file_path (str): The file path where the data will be saved.
        """
        # Convert ndarray to list for JSON serialization
        new_data = {
            'predictions': preds,
            'ground_truths': trues
        }

        existing_data = {'predictions': [], 'ground_truths': []}

        # Append new data to existing data
        existing_data['predictions'].extend(new_data['predictions'])
        existing_data['ground_truths'].extend(new_data['ground_truths'])

        # Write the updated data back to the file
        with open(file_path, 'w') as f:
            json.dump(existing_data, f)

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # 多任务学习模式处理
                if self.use_multi_task:
                    # 创建掩码用于多任务学习，但在测试时我们只关注预测任务
                    B, T, N = batch_x.shape
                    mask = torch.rand((B, T, N)).to(self.device)
                    mask[mask <= self.mask_rate] = 0  # masked
                    mask[mask > self.mask_rate] = 1  # remained
                    
                    # 测试时使用未掩码的原始数据进行预测
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            # 使用原始未掩码数据
                            outputs, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, mask)
                    else:
                        outputs, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, mask)
                else:
                    # 原始单任务模式
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                # print(f"outputs.shape: {outputs.shape}, batch_y.shape: {batch_y.shape}")
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + self.args.model + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.save_predictions(preds.tolist(), trues.tolist(), file_path=os.path.join(folder_path, self.args.model_id + "_results.json"))
        print("\nOverall metrics:")
        mae, mse, _, _, _, auroc, auprc, acc, f1, recall, precision, npv, tn, fp, fn, tp = run_metrics.run_metric(preds, trues)

        # dtw calculation
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('auroc:{:.3f}, auprc:{:.3f}, acc:{:.3f},\n'.format(auroc, auprc, acc))
        f.write('f1:{:.3f}, recall:{:.3f}, precision:{:.3f}, npv:{:.3f},\n'.format(f1, recall, precision, npv))
        f.write('tn:{:.3f}, fp:{:.3f}, fn:{:.3f}, tp:{:.3f}\n'.format(tn, fp, fn, tp))
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        return

    def test_ttt(self, setting, test=0):
        """使用test-time training (TTT)技术对模型进行测试时微调"""
        self.args.augmentation_ratio = 0.0
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        
        if self.args.use_ttt:
            self.optimizer_status = torch.load(os.path.join('./checkpoints/' + setting, 'optimizer.pt'))
            # # Reset lr
            for param_group in self.optimizer_status['optimizer_state_dict']['param_groups']:
                param_group['lr'] = self.args.ttt_lr

            # 保存基础模型
            self.base_model = copy.deepcopy(self.model)

            # # 构建参考样本库，用于数据增强
            if not hasattr(self, 'reference_library'):
                self.build_reference_sample_bank()

            if self.args.use_amp:
                scaler = torch.cuda.amp.GradScaler()
            
        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 获取所有患者ID
        all_caseids = test_data.get_all_caseids()
        print(f"Total number of cases: {len(all_caseids)}")
        
        # 按患者逐个测试
        for case_idx, caseid in enumerate(tqdm(all_caseids, desc="Test-time Traning Processing")):
            # get test dataset
            case_data = test_data.get_case_series(caseid)
            valid_range = test_data.get_case_valid_range(caseid)
            if valid_range is None:
                continue
                
            start_idx, end_idx = valid_range
            print(f"caseid: {caseid}, valid_range: {valid_range}")
            
            # organaize testing data
            test_loader = test_data.create_case_dataloader(
                caseid=caseid,
                batch_size=self.args.ttt_test_batch_size,
                seq_len=self.args.seq_len,
                pred_len=self.args.pred_len,
                sample_step=self.args.sample_step
            )
            
            if not test_loader:
                continue
                
            case_preds = []
            case_trues = []
            
            for batch_idx, (batch_x, batch_x_mark, batch_y_mark, batch_y) in enumerate(test_loader):
                # testing 
                self.model.eval()
                with torch.no_grad():
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                    
                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    
                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    
                    outputs = outputs.detach().cpu().numpy()
                    batch_y = batch_y.detach().cpu().numpy()
                    
                    if test_data.scale and self.args.inverse:
                        shape = batch_y.shape
                        if outputs.shape[-1] != batch_y.shape[-1]:
                            outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                        outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                        batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    
                    case_preds.append(outputs[:, :, f_dim:])
                    case_trues.append(batch_y[:, :, f_dim:])
                
                # finetuning (using TTT)
                if self.args.use_ttt and batch_x.size(0) == self.args.ttt_test_batch_size:
                    last_test_idx = start_idx + batch_x.size(0)* self.args.sample_step + self.args.seq_len
                    if last_test_idx >= end_idx:
                        last_test_idx = end_idx

                    # 修改起始点,避免数据重叠
                    if start_idx > 0:
                        ttt_start_idx = start_idx - self.args.pred_len + self.args.sample_step
                    else:
                        ttt_start_idx = start_idx

                    print(f'caseid:{caseid}, true_sample:{batch_x.size(0)}, start_index:{start_idx}, end_index:{last_test_idx}')
                    
                    # 创建TTT训练数据
                    ttt_data = self._create_ttt_train_data(
                        case_data, ttt_start_idx, last_test_idx,
                        self.args.seq_len, self.args.pred_len,
                        self.args.sample_step, test_data
                    )
                    print(f"TTT train data size: {len(ttt_data)}")

                    # 创建TTT训练集
                    ttt_dataset = TTTDataset(ttt_data)
                    if len(ttt_dataset):
                        ttt_loader = DataLoader(
                            ttt_dataset,
                            batch_size=self.args.ttt_train_batch_size,
                            shuffle=True,
                            num_workers=4
                        )

                        if len(ttt_loader) > 0:
                            self.reset_ttt_environment()
                            
                            # TTT训练
                            batch_num = len(ttt_loader)
                            criterion = self._select_criterion()
                            for ttt_epoch in range(self.args.ttt_train_epochs):
                                total_loss = 0.0
                                forecast_loss_total = 0.0
                                imputation_loss_total = 0.0
                                self.model.train()
                                for tt_idx, batch in enumerate(ttt_loader):
                                    ttt_batch_x, ttt_batch_y, ttt_batch_x_mark, ttt_batch_y_mark = batch
                                    self.model_optim.zero_grad()
                                    ttt_batch_x = ttt_batch_x.float().to(self.device)
                                    ttt_batch_y = ttt_batch_y.float().to(self.device)
                                    ttt_batch_x_mark = ttt_batch_x_mark.float().to(self.device)
                                    ttt_batch_y_mark = ttt_batch_y_mark.float().to(self.device)
                                    
                                    ttt_dec_inp = torch.zeros_like(ttt_batch_y).float()
                                    ttt_dec_inp = torch.cat([ttt_batch_y[:, :self.args.label_len, :], ttt_dec_inp], dim=1).float().to(self.device)
                                    
                                    if self.args.use_amp:
                                        with torch.cuda.amp.autocast():
                                            if self.use_multi_task:
                                                # 创建掩码用于多任务学习
                                                B, T, N = ttt_batch_x.shape
                                                mask = torch.rand((B, T, N)).to(self.device)
                                                mask[mask <= self.mask_rate] = 0  # masked
                                                mask[mask > self.mask_rate] = 1  # remained
                                                
                                                forecast_out, imputation_out = self.model(ttt_batch_x, ttt_batch_x_mark, ttt_dec_inp, ttt_batch_y_mark, mask)
                                                
                                                # 预测任务损失
                                                f_dim = -1 if self.args.features == 'MS' else 0
                                                forecast_out = forecast_out[:, -self.args.pred_len:, f_dim:]
                                                batch_y_pred = ttt_batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                                                forecast_loss = criterion(forecast_out, batch_y_pred)
                                                
                                                # 重建任务损失
                                                imputation_out = imputation_out[:, :, f_dim:]
                                                batch_x_imp = ttt_batch_x[:, :, f_dim:]
                                                mask = mask[:, :, f_dim:]
                                                imputation_loss = criterion(imputation_out[mask == 0], batch_x_imp[mask == 0])
                                                
                                                # 总损失
                                                ttt_loss = forecast_loss + self.args.mr_loss_ratio * imputation_loss
                                                forecast_loss_total += forecast_loss.item()
                                                imputation_loss_total += imputation_loss.item()
                                            else:
                                                ttt_outputs = self.model(ttt_batch_x, ttt_batch_x_mark, ttt_dec_inp, ttt_batch_y_mark)
                                                ttt_loss = criterion(ttt_outputs, ttt_batch_y[:, -self.args.pred_len:, :])
                                    else:
                                        if self.use_multi_task:
                                            forecast_out, imputation_out = self.model(ttt_batch_x, ttt_batch_x_mark, ttt_dec_inp, ttt_batch_y_mark, mask)
                                            
                                            # 预测任务损失
                                            f_dim = -1 if self.args.features == 'MS' else 0
                                            forecast_out = forecast_out[:, -self.args.pred_len:, f_dim:]
                                            batch_y_pred = ttt_batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                                            forecast_loss = criterion(forecast_out, batch_y_pred)
                                            
                                            # 重建任务损失
                                            imputation_out = imputation_out[:, :, f_dim:]
                                            batch_x_imp = ttt_batch_x[:, :, f_dim:]
                                            mask = mask[:, :, f_dim:]
                                            imputation_loss = criterion(imputation_out[mask == 0], batch_x_imp[mask == 0])
                                            
                                            # 总损失
                                            ttt_loss = forecast_loss + self.args.mr_loss_ratio * imputation_loss
                                            forecast_loss_total += forecast_loss.item()
                                            imputation_loss_total += imputation_loss.item()
                                        else:
                                            ttt_outputs = self.model(ttt_batch_x, ttt_batch_x_mark, ttt_dec_inp, ttt_batch_y_mark)
                                            ttt_loss = criterion(ttt_outputs, ttt_batch_y[:, -self.args.pred_len:, :])
                                    
                                    total_loss += ttt_loss.item()
                                    
                                    if self.args.use_amp:
                                        scaler.scale(ttt_loss).backward()
                                        scaler.step(self.model_optim)
                                        scaler.update()
                                    else:
                                        ttt_loss.backward()
                                        self.model_optim.step()

                                if self.use_multi_task:
                                    print(f"TTT train epoch: {ttt_epoch+1}/{self.args.ttt_train_epochs}, loss: {total_loss / batch_num:.4f}, forecast_loss: {forecast_loss_total / batch_num:.4f}, imputation_loss: {imputation_loss_total / batch_num:.4f}")
                                else:
                                    print(f"TTT train epoch: {ttt_epoch+1}/{self.args.ttt_train_epochs}, loss: {total_loss / batch_num:.4f}")

                            self.model.eval()

                    start_idx = last_test_idx - self.args.seq_len

            # 保存该患者的预测结果
            case_preds = np.concatenate(case_preds, axis=0)
            case_trues = np.concatenate(case_trues, axis=0)
            preds.append(case_preds)
            trues.append(case_trues)
            
            # # 计算该患者的指标
            # mae, mse, rmse, mape, mspe = metric(case_preds, case_trues)
            # print(f'Case {caseid} - mse:{mse:.4f}, mae:{mae:.4f}')
        
        # 合并所有患者的预测结果
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('Final test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + self.args.model + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.save_predictions(preds.tolist(), trues.tolist(), file_path=os.path.join(folder_path, self.args.model_id + "_results.json"))
        print("\nOverall metrics:")
        mae, mse, _, _, _, auroc, auprc, acc, f1, recall, precision, npv, tn, fp, fn, tp = run_metrics.run_metric(preds, trues)

        # dtw calculation
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        print('mae:{:.4f}, mse:{:.4f}, dtw:{}'.format(mae, mse, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('auroc:{:.3f}, auprc:{:.3f}, acc:{:.3f},\n'.format(auroc, auprc, acc))
        f.write('f1:{:.3f}, recall:{:.3f}, precision:{:.3f}, npv:{:.3f},\n'.format(f1, recall, precision, npv))
        f.write('tn:{}, fp:{}, fn:{}, tp:{}\n'.format(tn, fp, fn, tp))
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        return

    def model_parameter_check(self):
        pytorch_total_params = sum(p.numel() for p in self.model.parameters())
        print(f"########## Parameters number for all {pytorch_total_params/(1024*1024):.2f} M  #########")
        # for name, param in self.model.named_parameters():
        #     print(f"Parameter: {name}, Shape: {param.size()}")
        # for name, module in self.model.named_children():
        #     print(f"Module name: {name}, Module: {module}")

    def choose_training_parts(self, full_ft=False):
        total_params = 0
        trainable_params = 0
        print("\n=== Checking trainable parameters ===")
        
        for name, param in self.model.named_parameters():
            total_params += param.numel()  # 计算总参数量
            
            if full_ft:
                param.requires_grad = True
                trainable_params += param.numel()
            else:
                trainable = any([
                    "linear_pred" in name, 
                    "enc_pos_embedding" in name,
                    "dec_pos_embedding" in name,
                    "norm" in name,
                    # "decoder.decode_layers.2.MLP1" in name,
                ])
                param.requires_grad = trainable
                if trainable:
                    trainable_params += param.numel()
        
        print(f"Total parameters: {total_params/(1024*1024):.2f}M")
        print(f"Trainable parameters: {trainable_params/(1024*1024):.2f}M ({trainable_params/total_params*100:.2f}%)")
        print("===================================\n")

    def _apply_standard_augmentation(self, raw_context, raw_future, context_len, horizon_len, test_data=None):
        """应用标准数据增强方法生成多样化样本，保持多特征间的相关性
        先将数据逆归一化到原始尺度，进行增强后再归一化回来

        Args:
            raw_context: 原始上下文数据 [context_len, num_features]
            raw_future: 原始未来数据 [label_len+pred_len, num_features]
            context_len: 上下文长度
            horizon_len: 预测窗口总长度(label_len + pred_len)
            test_data: 测试数据集实例，用于获取scale信息和执行逆归一化

        Returns:
            list: 增强后的样本列表[(x_context, x_future, x_mark, y_mark), ...]
        """
        import random
        augmented_samples = []
        
        # 计算label_len和pred_len
        label_len = context_len // 2
        pred_len = horizon_len - label_len  # 实际需要预测的长度

        num_features = raw_context.shape[1]
        
        # 分离raw_future中的label部分和pred部分
        raw_future_label = raw_future[:label_len] 
        raw_future_pred = raw_future[label_len:]  

        # 逆归一化数据到原始尺度进行增强
        context_original = raw_context.copy()
        future_pred_original = raw_future_pred.copy()  # 只对预测部分进行增强
        
        if test_data.scale:
            # 调整形状以适配inverse_transform方法
            context_shape = raw_context.shape
            future_pred_shape = raw_future_pred.shape
            
            # 重新整形数据
            context_flat = raw_context.reshape(context_shape[0] * context_shape[1], -1)
            future_pred_flat = raw_future_pred.reshape(future_pred_shape[0] * future_pred_shape[1], -1)
            
            # 执行逆归一化
            context_original_flat = test_data.inverse_transform(context_flat)
            future_pred_original_flat = test_data.inverse_transform(future_pred_flat)
            
            # 恢复原始形状
            context_original = context_original_flat.reshape(context_shape)
            future_pred_original = future_pred_original_flat.reshape(future_pred_shape)
        
        # 生成3个增强样本
        for _ in range(3):
            # 使用主特征计算增强参数
            main_context = context_original[:, -1]  # 使用最后一个特征作为主特征
            main_future_pred = future_pred_original[:, -1]  # 只对预测部分进行增强
            
            # 计算主特征趋势和统计量
            trend = np.polyfit(np.arange(len(main_context)), main_context, 1)[0]
            main_std = np.std(main_context) + 1e-6
            trend_strength = np.abs(trend) * len(main_context) / main_std
            
            # 根据趋势强度确定增强参数
            if trend_strength > 0.5:
                if random.random() < 0.30:
                    trend_noise = -trend * np.random.uniform(0.8, 1.2)
                else:
                    trend_noise = trend * np.random.uniform(0.8, 1.2)
            else:
                trend_noise = np.random.normal(0, main_std * 0.05)
            
            # 生成噪声
            noise_scale = main_std * 0.05
            context_noise = np.random.normal(0, noise_scale, len(main_context))
            future_pred_noise = np.random.normal(0, noise_scale, len(main_future_pred))
            
            # 生成时序变化噪声
            changes = np.diff(main_context)
            change_noise = np.random.normal(0, noise_scale, len(changes))
            perturbed_changes = changes * (1 + change_noise)
            
            # 创建增强后的数据(仅上下文和预测部分)
            aug_context = np.zeros_like(context_original)
            aug_future_pred = np.zeros_like(future_pred_original)
            
            # 对每个特征应用增强，保持增强超参数一致但调整幅度
            for dim in range(num_features):
                # 获取当前特征
                dim_context = context_original[:, dim]
                dim_future_pred = future_pred_original[:, dim]
                
                # 计算特征统计量
                dim_mean = np.mean(dim_context)
                dim_std = np.std(dim_context) + 1e-6
                
                # 根据主特征和当前特征的标准差比例调整增强幅度
                scale_ratio = dim_std / main_std if main_std > 0 else 1.0
                
                # 应用增强
                # 1. 趋势调整
                trended_context = dim_context + trend_noise * np.arange(len(dim_context)) * scale_ratio
                trended_future_pred = dim_future_pred + trend_noise * np.arange(len(dim_future_pred)) * scale_ratio
                
                # 2. 噪声添加
                noisy_context = trended_context + context_noise * scale_ratio
                noisy_future_pred = trended_future_pred + future_pred_noise * scale_ratio
                
                # 3. 缩放
                scale_factor = np.random.uniform(0.95, 1.05)
                scaled_context = (noisy_context - dim_mean) * scale_factor + dim_mean
                scaled_future_pred = (noisy_future_pred - dim_mean) * scale_factor + dim_mean
                
                # 4. 时序相关性变化
                if len(dim_context) > 1:
                    aug_context_ts = np.concatenate([[dim_context[0]], 
                                                  dim_context[0] + np.cumsum(perturbed_changes) * scale_ratio])
                    # 融合时序变化和其他增强效果
                    aug_context[:, dim] = (aug_context_ts + scaled_context) / 2
                else:
                    aug_context[:, dim] = scaled_context
                
                if len(dim_future_pred) > 1:
                    # 应用连贯的变化模式
                    if len(perturbed_changes) > 0:
                        # 使用最后几个变化的平均值作为未来趋势
                        last_changes = perturbed_changes[-min(3, len(perturbed_changes)):]
                        mean_change = np.mean(last_changes)
                        aug_future_pred_ts = dim_future_pred[0] + np.arange(len(dim_future_pred)) * mean_change * scale_ratio
                        aug_future_pred[:, dim] = (aug_future_pred_ts + scaled_future_pred) / 2
                    else:
                        aug_future_pred[:, dim] = scaled_future_pred
                else:
                    aug_future_pred[:, dim] = scaled_future_pred
            
            # 将增强后的数据归一化回来
            if test_data.scale:
                # 调整形状
                aug_context_shape = aug_context.shape
                aug_future_pred_shape = aug_future_pred.shape
                
                # 重新整形数据
                aug_context_flat = aug_context.reshape(aug_context_shape[0] * aug_context_shape[1], -1)
                aug_future_pred_flat = aug_future_pred.reshape(aug_future_pred_shape[0] * aug_future_pred_shape[1], -1)
                
                # 获取训练集的归一化器
                scalers = test_data._get_train_scaler()

                # 检查是否为单变量数据
                feature_dim = aug_context_flat.shape[1]
                if feature_dim == 1:
                    # 单变量情况，直接使用目标特征的scaler
                    target_feature = 'Solar8000/ART_MBP'
                    aug_context_norm_flat = scalers[target_feature].transform(aug_context_flat)
                    aug_future_pred_norm_flat = scalers[target_feature].transform(aug_future_pred_flat)
                else:
                    # 多特征处理情况，按特征循环处理
                    # 按照VitalDBLoader.time_series_features的顺序
                    time_series_features = [
                        'Solar8000/BT',       # 0
                        'Solar8000/HR',       # 1
                        'Solar8000/ART_DBP',  # 2
                        'Solar8000/ART_MBP',  # 3
                    ]
                    
                    # 创建结果数组
                    aug_context_norm_flat = np.zeros_like(aug_context_flat)
                    aug_future_pred_norm_flat = np.zeros_like(aug_future_pred_flat)
                    
                    # 确保不超出特征维度
                    feature_count = min(feature_dim, len(time_series_features))
                    
                    # 仅处理有效的特征维度
                    for i in range(feature_count):
                        feature = time_series_features[i]
                        if feature in scalers:
                            # 取出该特征的所有值并重塑为(n, 1)进行归一化
                            current_feature_context = aug_context_flat[:, i].reshape(-1, 1)
                            current_feature_future_pred = aug_future_pred_flat[:, i].reshape(-1, 1)
                            
                            # 使用对应特征的归一化器
                            feature_context_norm = scalers[feature].transform(current_feature_context)
                            feature_future_pred_norm = scalers[feature].transform(current_feature_future_pred)
                            
                            # 将结果放回
                            aug_context_norm_flat[:, i] = feature_context_norm.flatten()
                            aug_future_pred_norm_flat[:, i] = feature_future_pred_norm.flatten()
                
                # 恢复形状
                aug_context = aug_context_norm_flat.reshape(aug_context_shape)
                aug_future_pred = aug_future_pred_norm_flat.reshape(aug_future_pred_shape)
            

            # 组合标签部分和增强的预测部分形成完整的未来数据
            aug_future = np.concatenate([aug_context[-label_len:], aug_future_pred], axis=0)
            
            # 转换为张量
            context_tensor = torch.tensor(aug_context, dtype=torch.float32)
            future_tensor = torch.tensor(aug_future, dtype=torch.float32)
            
            # 生成时间特征（与原代码相同方式）
            x_mark = torch.zeros((context_len, 4), dtype=torch.float32)
            y_mark = torch.zeros((len(x_future_array), 4), dtype=torch.float32)
            for j in range(context_len):
                x_mark[j, 0] = 1
                x_mark[j, 1] = 1
                x_mark[j, 2] = 1
                x_mark[j, 3] = 1
                
            for j in range(len(x_future_array)):
                y_mark[j, 0] = 1
                y_mark[j, 1] = 1
                y_mark[j, 2] = 1
                y_mark[j, 3] = 1

            augmented_samples.append((context_tensor, future_tensor, x_mark, y_mark))
        
        return augmented_samples

    def _apply_reference_based_augmentation(self, ttt_train_data, original_sample_indices, context_len, horizon_len,test_data):
        """根据参考库进行基于样本平衡的增强
        
        Args:
            ttt_train_data: 当前训练数据
            original_sample_indices: 原始样本索引
            context_len: 上下文长度
            horizon_len: 预测长度
            
        Returns:
            list: 增强后的样本列表
        """
        if not hasattr(self, 'reference_library'):
            return ttt_train_data
            
        # 统计当前正负样本比例并保存每个样本的IOH状态，同时计算原始样本的均值
        pos_count = 0
        neg_count = 0
        sample_ioh_status = []  # 保存每个样本的IOH状态
        original_values = []  # 保存原始数据的值用于计算均值

        # 统计正负样本比例
        for i in range(len(ttt_train_data)):
            # 获取样本数据
            x_context = ttt_train_data[i][0]
            x_future = ttt_train_data[i][1]
            
            # 确定目标维度
            f_dim = -1 if self.args.features == 'MS' else 0
            
            # 获取样本数据
            if self.args.features == 'MS':
                context_data = x_context[:, f_dim].numpy()
                future_data = x_future[:, f_dim].numpy()
            else:
                context_data = x_context[:, 0].numpy()
                future_data = x_future[:, 0].numpy()
                
            # 反归一化数据用于IOH判断和计算均值
            if test_data and test_data.scale:
                # 提取并反归一化未来数据
                future_shape = future_data.shape
                future_flat = future_data.reshape(future_shape[0], -1)
                
                # 提取并反归一化上下文数据
                context_shape = context_data.shape
                context_flat = context_data.reshape(-1, 1)
                
                # 使用数据集的inverse_transform方法
                future_original = test_data.inverse_transform(future_flat)
                context_original = test_data.inverse_transform(context_flat).reshape(context_shape)
                
                future_original = future_original[-self.args.pred_len:]
                # 使用原始数据判断IOH
                is_ioh = run_metrics.user_definable_IOH(future_original.reshape(1, -1))[0]
                
                # 收集原始尺度的数据用于计算平均值
                if i in original_sample_indices:
                    original_values.extend(context_original)
                    original_values.extend(future_original)
            else:
                # 如果数据未归一化，直接使用
                future_data = future_data[-self.args.pred_len:]
                is_ioh = run_metrics.user_definable_IOH(future_data.reshape(1, -1))[0]
                
                # 收集数据用于计算平均值
                if i in original_sample_indices:
                    original_values.extend(context_data)
                    original_values.extend(future_data)
            
            # 保存样本的IOH状态
            sample_ioh_status.append(is_ioh)
            
            if is_ioh:
                pos_count += 1
            else:
                neg_count += 1
        
        print(f"Current positive samples: {pos_count}, negative samples: {neg_count}")

        # 计算原始样本的均值（使用已经收集的逆归一化后的值）
        avg_value = np.mean(original_values) if original_values else 0
        # print(f"Average value of original samples (original scale): {avg_value:.2f}")
        
        # 根据不同情况确定增强策略
        if pos_count == 0:  # 没有IOH正样本
            print("No IOH positive samples in current data")
            if avg_value < 95:  
                target_pos_ratio = 0.9  
                print(f"Low average value ({float(avg_value):.2f} < 90), using 1:6 ratio for augmentation")
            else:  
                target_pos_ratio = 0.7
                print(f"Normal average value ({float(avg_value):.2f} >= 90), not introducing positive samples")
        else: 
            target_pos_ratio = 0.9
            print(f"Found IOH positive samples, using 1:1 ratio for augmentation")
        
        # 记录原始数据大小，计算需要增加的样本数 ttt_train_data,2,128
        original_size = len(ttt_train_data)
        augment_total = min(original_size * 3, 128)
        
        # 计算目标正样本和负样本数量
        target_total = original_size + augment_total
        target_pos = int(target_total * target_pos_ratio)
        target_neg = target_total - target_pos
        
        # 计算需要额外增加的正样本和负样本数量
        pos_to_add = max(0, target_pos - pos_count)
        neg_to_add = max(0, target_neg - neg_count)
        
        print(f"Target: positive={target_pos}, negative={target_neg}")
        print(f"Need to add: positive={pos_to_add}, negative={neg_to_add}")
        
        # 优先从原始样本中检索
        augmented_data = ttt_train_data.copy()
        if pos_to_add > 0 or neg_to_add > 0:
            # 使用已有的IOH判断结果对原始样本进行分类
            pos_indices = []
            neg_indices = []
            for idx in original_sample_indices:
                # 直接使用之前保存的IOH状态
                if sample_ioh_status[idx]:
                    pos_indices.append(idx)
                else:
                    neg_indices.append(idx)
            
            # 逆序处理原始样本索引，优先使用较新的数据
            pos_indices = sorted(pos_indices, reverse=True)
            neg_indices = sorted(neg_indices, reverse=True)
            
            # 如果需要正样本但没有原始正样本，则使用所有原始样本检索
            pos_query_indices = pos_indices if pos_indices else sorted(original_sample_indices, reverse=True)
            neg_query_indices = neg_indices if neg_indices else sorted(original_sample_indices, reverse=True)
            
            # 对于每个原始样本，检索相似样本
            added_pos, added_neg = 0, 0
            
            # 增加正样本
            if pos_to_add > 0:
                n_samples_per_query = max(1, pos_to_add // max(1, len(pos_query_indices)))
                print(f"Retrieving {n_samples_per_query} positive samples per query")
                
                for idx in pos_query_indices:
                    if added_pos >= pos_to_add:
                        break
                        
                    x_context, x_future, x_mark, y_mark = ttt_train_data[idx]
                    
                    if self.args.features == 'MS':
                        # 多变量预测单变量，使用最后一个维度(目标变量)
                        raw_context = x_context[:, -1].numpy()
                        raw_future = x_future[:, -1].numpy()
                    else:
                        # 单变量模式，直接使用第一个维度
                        raw_context = x_context[:, 0].numpy()
                        raw_future = x_future[:, 0].numpy()
                    
                    # 在检索前反归一化数据
                    if test_data and test_data.scale:
                        # 先拼接数据
                        query_series = np.concatenate([raw_context, raw_future])
                        
                        # 整体进行反归一化
                        series_shape = query_series.shape
                        series_flat = query_series.reshape(-1, 1)
                        query_series = test_data.inverse_transform(series_flat).reshape(series_shape)
                    else:
                        # 如果数据未归一化，直接拼接使用
                        query_series = np.concatenate([raw_context, raw_future])
                    
                    # 检索正样本
                    similar_samples = self.retrieve_similar_samples(query_series, True, n_samples=n_samples_per_query)
                    for sample in similar_samples:
                            augmented_data.append((sample[0], sample[1], sample[2], sample[3]))
                            added_pos += 1
                            if added_pos >= pos_to_add:
                                break
            
            # 增加负样本
            if neg_to_add > 0:
                n_samples_per_query = max(1, neg_to_add // max(1, len(neg_query_indices)))
                print(f"Retrieving {n_samples_per_query} negative samples per query")
                
                for idx in neg_query_indices:
                    if added_neg >= neg_to_add:
                        break
                        
                    x_context, x_future, x_mark, y_mark = ttt_train_data[idx]
                    
                    if self.args.features == 'MS':
                        # 多变量预测单变量，使用最后一个维度(目标变量)
                        raw_context = x_context[:, -1].numpy()
                        raw_future = x_future[:, -1].numpy()
                    else:
                        # 单变量模式，直接使用第一个维度
                        raw_context = x_context[:, 0].numpy()
                        raw_future = x_future[:, 0].numpy()
                        
                    # 在检索前反归一化数据
                    if test_data and test_data.scale:
                        # 先拼接数据
                        query_series = np.concatenate([raw_context, raw_future])
                        
                        # 整体进行反归一化
                        series_shape = query_series.shape
                        series_flat = query_series.reshape(-1, 1)
                        query_series = test_data.inverse_transform(series_flat).reshape(series_shape)
                    else:
                        # 如果数据未归一化，直接拼接使用
                        query_series = np.concatenate([raw_context, raw_future])
                    
                    # 检索负样本
                    similar_samples = self.retrieve_similar_samples(query_series, False, n_samples=n_samples_per_query)
                    for sample in similar_samples:
                            augmented_data.append((sample[0], sample[1], sample[2], sample[3]))
                            added_neg += 1
                            if added_neg >= neg_to_add:
                                break
            
            # 更新最终的样本统计
            final_pos_count = pos_count + added_pos
            final_neg_count = neg_count + added_neg
            final_total = final_pos_count + final_neg_count
            final_ratio = final_pos_count / max(1, final_total)
            
            print(f"After balanced augmentation - positive: {final_pos_count}, negative: {final_neg_count}, ratio: {final_ratio:.4f}")
        
        return augmented_data
        
    def retrieve_similar_samples(self, query_series, is_ioh, n_samples=10):
        """从参考库中检索与查询序列最相似的样本，基于相似度排序
        
        Args:
            query_series: 查询时间序列
            is_ioh: 是否为IOH样本
            n_samples: 返回样本数量
            
        Returns:
            list: 相似样本列表，按相似度降序排列
        """
        if not hasattr(self, 'reference_library'):
            return []
            
        # 选择参考库
        lib_key = 'pos' if is_ioh else 'neg'
        reference = self.reference_library[lib_key]
        model = reference['model']
        clusters = reference['clusters']
        
        # 预测聚类
        query_scaled = query_series.reshape(1, -1, 1)  # (1, train_context+train_horizon, 1)
        query_scaled = reference['scaler'].transform(query_scaled)
        
        cluster_idx = model.predict(query_scaled)[0]
        
        # 从聚类中获取样本
        candidates = clusters[cluster_idx]
        
        # 如果聚类中样本数量小于n_samples，返回所有样本
        if len(candidates) <= n_samples:
            samples = candidates
        else:
            # 随机选择n_samples个样本
            samples = random.sample(candidates, n_samples)
        
        # 确保样本是PyTorch张量
        processed_samples = []
        for sample in samples:
            # 参考库中样本格式为(x, y, x_mark, y_mark)，其中x、y是NumPy数组
            x, y, x_mark, y_mark = sample
            
            # 转换为PyTorch张量
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            if not isinstance(y, torch.Tensor):
                y = torch.tensor(y, dtype=torch.float32)
            if not isinstance(x_mark, torch.Tensor):
                x_mark = torch.tensor(x_mark, dtype=torch.float32)
            if not isinstance(y_mark, torch.Tensor):
                y_mark = torch.tensor(y_mark, dtype=torch.float32)
                
            processed_samples.append((x, y, x_mark, y_mark))
            
        return processed_samples
            
    def build_reference_sample_bank(self):
        """构建时序参考库，对正负样本分别进行K-Shape聚类，并支持保存和加载
        """
        # 创建保存目录
        save_dir = os.path.join("./reference_sample_bank", "kshape_clusters_10000_30_60_5epoch")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"kshape_clusters_c{self.args.seq_len}_h{self.args.pred_len}.pkl")
        
        # 检查是否已存在保存的参考库
        if os.path.exists(save_path):
            try:
                with open(save_path, 'rb') as f:
                    self.reference_library = pickle.load(f)
                print(f"load reference sample bank from:{save_path} \n"
                      f"##### pos clusters {len(self.reference_library['pos']['clusters'])}, "
                      f"neg clusters {len(self.reference_library['neg']['clusters'])}")
                return
            except Exception as e:
                print(f"load reference sample bank failed: {e}, will rebuild")
        
        # 使用已有的训练数据集获取数据
        train_data, train_loader = self._get_data(flag='train')
        
        # 收集训练数据
        pos_samples = []  # 正样本(IOH)
        neg_samples = []  # 负样本(非IOH)
        pos_info = []     # 保存正样本的原始信息
        neg_info = []     # 保存负样本的原始信息
        
        print("构建时序参考库中...")
        
        # 收集所有训练样本
        for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
            batch_size = batch_x.shape[0]
            
            for i in range(batch_size):
                # 提取单样本
                x = batch_x[i].numpy()
                y = batch_y[i].numpy()
                x_mark = batch_x_mark[i].numpy()
                y_mark = batch_y_mark[i].numpy()
                
                # 提取时序样本的特征向量 - 主要是目标变量
                f_dim = -1 if self.args.features == 'MS' else 0
                if self.args.features == 'MS':
                    time_series = np.concatenate([x[:, f_dim], y[:, f_dim]])
                else:
                    time_series = np.concatenate([x.squeeze(), y.squeeze()])
                
                # 反归一化数据用于IOH判断
                if train_data.scale:
                    # 提取未来数据并反归一化
                    future_shape = y.shape
                    future_flat = y.reshape(future_shape[0], -1)
                    
                    # 使用数据集的inverse_transform方法
                    future_original = train_data.inverse_transform(future_flat)
                    future_original = future_original.reshape(future_shape)
                    
                    # 获取目标变量
                    future_data_original = future_original[-self.args.pred_len:, f_dim]
                    
                    # 使用原始数据判断IOH
                    is_ioh = run_metrics.user_definable_IOH(future_data_original.reshape(1, -1))[0]
                else:
                    # 如果数据未归一化，直接使用
                    future_data = y[-self.args.pred_len:, f_dim]
                    is_ioh = run_metrics.user_definable_IOH(future_data.reshape(1, -1))[0]
                
                # 根据标签分类
                if is_ioh:
                    pos_samples.append(time_series)
                    pos_info.append((x, y, x_mark, y_mark))  # 保存原始信息
                else:
                    neg_samples.append(time_series)
                    neg_info.append((x, y, x_mark, y_mark))  # 保存原始信息
        
        print(f"pos_samples: {len(pos_samples)}, neg_samples: {len(neg_samples)}, ratio: {len(pos_samples) / len(neg_samples)}")
        # 限制样本数量以加快聚类速度
        max_samples = 10000
        if len(pos_samples) > max_samples:
            indices = random.sample(range(len(pos_samples)), max_samples)
            pos_samples = [pos_samples[i] for i in indices]
            pos_info = [pos_info[i] for i in indices]
        
        if len(neg_samples) > max_samples:
            indices = random.sample(range(len(neg_samples)), 3*max_samples)
            neg_samples = [neg_samples[i] for i in indices]
            neg_info = [neg_info[i] for i in indices]
        
        print(f"Collected {len(pos_samples)} positive samples and {len(neg_samples)} negative samples")
        
        # 预处理 - 标准化
        pos_scaler = TimeSeriesScalerMeanVariance()
        neg_scaler = TimeSeriesScalerMeanVariance()
        
        if len(pos_samples) > 0:
            pos_samples = np.array(pos_samples).reshape(len(pos_samples), -1, 1)  # 转换为tslearn格式
            pos_samples = pos_scaler.fit_transform(pos_samples)
            
        if len(neg_samples) > 0:
            neg_samples = np.array(neg_samples).reshape(len(neg_samples), -1, 1)
            neg_samples = neg_scaler.fit_transform(neg_samples)
        
        # 设置聚类参数
        n_clusters_pos = min(max(len(pos_samples) // 10, 3), 30)
        n_clusters_neg = min(max(len(neg_samples) // 10, 3), 60)  # 负样本聚类数
        
        # 对正样本进行K-Shape聚类
        pos_model = None
        pos_clusters = []
        if len(pos_samples) >= n_clusters_pos:
            print(f"augment positive samples: {len(pos_samples)}, clusters: {n_clusters_pos}")
            pos_model = KShape(n_clusters=n_clusters_pos, random_state=42, n_init=3, verbose=True) 
            pos_labels = pos_model.fit_predict(pos_samples)
            pos_clusters = [[] for _ in range(n_clusters_pos)]
            
            # 按聚类整理样本
            for i, label in enumerate(pos_labels):
                pos_clusters[label].append(pos_info[i])
        else:
            pos_clusters = [pos_info]  # 样本太少时不聚类
        
        # 对负样本进行K-Shape聚类
        neg_model = None
        neg_clusters = []
        if len(neg_samples) >= n_clusters_neg:
            print(f"augment negative samples: {len(neg_samples)}, clusters: {n_clusters_neg}")
            neg_model = KShape(n_clusters=n_clusters_neg, random_state=42, n_init=3, verbose=True)
            neg_labels = neg_model.fit_predict(neg_samples)
            neg_clusters = [[] for _ in range(n_clusters_neg)]
            
            # 按聚类整理样本
            for i, label in enumerate(neg_labels):
                neg_clusters[label].append(neg_info[i])
        else:
            neg_clusters = [neg_info]  # 样本太少时不聚类
        
        # 保存聚类结果
        self.reference_library = {
            'pos': {
                'model': pos_model,
                'clusters': pos_clusters,
                'scaler': pos_scaler
            },
            'neg': {
                'model': neg_model,
                'clusters': neg_clusters,
                'scaler': neg_scaler
            }
        }
        
        # 将参考库保存到文件
        try:
            with open(save_path, 'wb') as f:
                pickle.dump(self.reference_library, f)
            print(f"save reference library to: {save_path}")
        except Exception as e:
            print(f"save reference library failed: {e}")
        
        print(f"build reference sample bank: positive samples {len(pos_clusters)}, negative samples {len(neg_clusters)}")

    def _create_ttt_train_data(self, case_data, start_idx, end_idx, context_len, horizon_len, sample_step, test_data=None):
        """创建TTT训练数据，确保不会发生数据泄露
        
        Args:
            case_data: 患者数据
            start_idx: 当前预测的起始索引
            end_idx: 当前预测的结束索引
            context_len: 输入序列长度
            horizon_len: 预测长度
            sample_step: 采样步长
            test_data: 测试数据集实例,用于获取scale信息和执行逆归一化
        """
        ttt_data = []
        original_sample_indices = []  # 保存原始样本的索引
        required_length = context_len + horizon_len
            
        # 处理 case_data 为字典类型的情况
        features = list(case_data.keys())
        if len(features) == 0:
            return []
        
        ordered_features = [
            'Solar8000/BT',
            'Solar8000/HR',
            'Solar8000/ART_DBP',
            'Solar8000/ART_MBP',
        ]
        
        # # reset start_id
        tmp_start = end_idx - context_len - horizon_len - sample_step * 31 - 1
        if tmp_start > 0:
            start_idx = tmp_start
        else:
            start_idx = 0
        print(f'ttt_train: {(start_idx, end_idx)}')

        # 生成TTT训练样本
        for i in range(start_idx, end_idx - required_length, sample_step):
            segment_data = case_data['Solar8000/ART_MBP'][i:i+required_length]
            
            # 处理归一化数据的情况
            if test_data.scale:
                segment_data_original = test_data.scalers['Solar8000/ART_MBP'].inverse_transform(
                    segment_data.reshape(-1, 1)).flatten()
                # 使用原始数据的阈值(60 mmHg)检测突变
                if (np.abs(np.diff(segment_data_original)) > 60).any():
                    continue  # abrupt change -> noise
            else:
                # 数据未归一化，直接使用原始阈值
                if (np.abs(np.diff(segment_data)) > 60).any():
                    continue

            # 提取上下文和未来数据 - 严格按照定义的特征顺序
            x_context_list = []
            x_future_list = []
            
            for feature in ordered_features:
                if feature in case_data:
                    feature_data = case_data[feature]
                    x_context_list.append(feature_data[i:i+context_len])
                    x_future_list.append(feature_data[i+context_len-(context_len//2):i+context_len+horizon_len])
                
            # 将列表转换为NumPy数组，且保持特征顺序
            x_context_array = np.array(x_context_list).transpose()  # [context_len, num_features]
            x_future_array = np.array(x_future_list).transpose()  # [horizon_len+label_len, num_features]
                
            # 保存原始样本
            x_context = torch.tensor(x_context_array, dtype=torch.float32)
            x_future = torch.tensor(x_future_array, dtype=torch.float32)
            
            # 生成时间特征
            x_mark = torch.zeros((context_len, 4), dtype=torch.float32)
            y_mark = torch.zeros((len(x_future_array), 4), dtype=torch.float32)
            for j in range(context_len):
                x_mark[j, 0] = 1
                x_mark[j, 1] = 1
                x_mark[j, 2] = 1
                x_mark[j, 3] = 1
                
            for j in range(len(x_future_array)):
                y_mark[j, 0] = 1
                y_mark[j, 1] = 1
                y_mark[j, 2] = 1
                y_mark[j, 3] = 1
            
            # 添加原始样本
            ttt_data.append((x_context, x_future, x_mark, y_mark))
            original_sample_indices.append(len(ttt_data) - 1)
            
            # 应用标准数据增强
            # augmented_samples = self._apply_standard_augmentation(
            #     x_context_array, x_future_array, context_len, horizon_len, test_data
            # )
            
            # # 添加增强样本
            # for sample in augmented_samples:
            #     ttt_data.append(sample)
        
        if len(ttt_data) == 0:
            return []
            
        # 应用参考库增强来平衡样本
        if hasattr(self, 'reference_library'):
            ttt_data = self._apply_reference_based_augmentation(
                ttt_data, original_sample_indices, context_len, horizon_len,test_data
            )

        print(f"Final TTT train data size: {len(ttt_data)}")
        return ttt_data
        
    def reset_ttt_environment(self):
        self.model = copy.deepcopy(self.base_model)

        self.choose_training_parts()
        self.model_optim = self._select_optimizer()
        self.model_optim.load_state_dict(self.optimizer_status['optimizer_state_dict'])
        
class TTTDataset(Dataset):
    """自定义TTT训练数据集"""
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x_context, x_future, x_mark, y_mark = self.data[idx]
        
        # 确保所有返回的数据都是PyTorch张量
        if not isinstance(x_context, torch.Tensor):
            x_context = torch.tensor(x_context, dtype=torch.float32)
        if not isinstance(x_future, torch.Tensor):
            x_future = torch.tensor(x_future, dtype=torch.float32)
        if not isinstance(x_mark, torch.Tensor):
            x_mark = torch.tensor(x_mark, dtype=torch.float32)
        if not isinstance(y_mark, torch.Tensor):
            y_mark = torch.tensor(y_mark, dtype=torch.float32)
            
        return x_context, x_future, x_mark, y_mark
