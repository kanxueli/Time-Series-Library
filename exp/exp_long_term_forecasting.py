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

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

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
                batch_y = batch_y.float()

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

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

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

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device) # (batch_size, seq_len, features)
                batch_y = batch_y.float().to(self.device) # (batch_size, pred_len, features)
                batch_x_mark = batch_x_mark.float().to(self.device) # (batch_size, seq_len, 4)
                batch_y_mark = batch_y_mark.float().to(self.device) # (batch_size, pred_len, 4)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # print(batch_x.shape, batch_x_mark.shape, batch_y.shape, batch_y_mark.shape)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
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
                # encoder - decoder
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

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        return

    def test_ttt(self, setting, test=0):
        """使用test-time training (TTT)技术对模型进行测试时微调"""
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
            self.base_model = self.model

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
            test_batches = test_data.create_case_dataloader(
                caseid=caseid,
                batch_size=self.args.ttt_test_batch_size,
                seq_len=self.args.seq_len,
                pred_len=self.args.pred_len,
                sample_step=self.args.sample_step
            )
            
            if not test_batches:
                continue
                
            case_preds = []
            case_trues = []
            
            # 处理每个batch
            for batch_idx, (batch_x, batch_x_mark, batch_y_mark, batch_y) in enumerate(test_batches):
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
                    
                    case_preds.append(outputs)
                    case_trues.append(batch_y)
                
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
                        self.args.sample_step
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
                            self.reset_ttt_environment(base_model=self.base_model, optimizer_status=self.optimizer_status)
                            
                            # TTT训练
                            self.model.train()
                            for ttt_epoch in range(self.args.ttt_train_epochs):
                                for tt_idx, batch in enumerate(ttt_loader):
                                    ttt_batch_x, ttt_batch_y, ttt_batch_x_mark, ttt_batch_y_mark = batch
                                    ttt_batch_x = ttt_batch_x.float().to(self.device)
                                    ttt_batch_y = ttt_batch_y.float().to(self.device)
                                    ttt_batch_x_mark = ttt_batch_x_mark.float().to(self.device)
                                    ttt_batch_y_mark = ttt_batch_y_mark.float().to(self.device)
                                    
                                    ttt_dec_inp = torch.zeros_like(ttt_batch_y).float()
                                    ttt_dec_inp = torch.cat([ttt_batch_y[:, :self.args.label_len, :], ttt_dec_inp], dim=1).float().to(self.device)
                                    
                                    if self.args.use_amp:
                                        with torch.cuda.amp.autocast():
                                            ttt_outputs = self.model(ttt_batch_x, ttt_batch_x_mark, ttt_dec_inp, ttt_batch_y_mark)
                                            ttt_loss = self._select_criterion()(ttt_outputs, ttt_batch_y)
                                    else:
                                        ttt_outputs = self.model(ttt_batch_x, ttt_batch_x_mark, ttt_dec_inp, ttt_batch_y_mark)
                                        ttt_loss = self._select_criterion()(ttt_outputs, ttt_batch_y)
                                    
                                    self.model_optim.zero_grad()
                                    if self.args.use_amp:
                                        scaler.scale(ttt_loss).backward()
                                        scaler.step(self.model_optim)
                                        scaler.update()
                                    else:
                                        ttt_loss.backward()
                                        self.model_optim.step()
                            
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
        
        # 计算总体指标
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('Overall metrics - mse:{:.4f}, mae:{:.4f}'.format(mse, mae))
        
        return

    def _create_ttt_train_data(self, case_data, start_idx, end_idx, context_len, horizon_len, sample_step):
        """创建TTT训练数据，确保不会发生数据泄露
        
        Args:
            case_data: 患者数据
            start_idx: 当前预测的起始索引
            end_idx: 当前预测的结束索引
            context_len: 输入序列长度
            horizon_len: 预测长度
            sample_step: 采样步长
        """
        ttt_data = []
        required_length = context_len + horizon_len
            
        # 处理 case_data 为字典类型的情况
        features = list(case_data.keys())
        if len(features) == 0:
            return []
        
        # # reset start_id
        tmp_start = end_idx - context_len - horizon_len - sample_step * 31
        if tmp_start > 0:
            start_idx = tmp_start
        else:
            start_idx = 0
        print(f'ttt_train: {(start_idx, end_idx)}')

        # 生成TTT训练样本
        for i in range(start_idx, end_idx - required_length + 1, sample_step):
            # 提取上下文和未来数据
            x_context_list = []
            x_future_list = []
            
            for feature in features:
                feature_data = case_data[feature]
                x_context_list.append(feature_data[i:i+context_len])
                x_future_list.append(feature_data[i+context_len:i+context_len+horizon_len])
                
            # 将列表转换为NumPy数组
            x_context_array = np.array(x_context_list).transpose()  # [context_len, num_features]
            x_future_array = np.array(x_future_list).transpose()  # [horizon_len, num_features]
            
            # # 数据增强
            # if self.args.use_data_augmentation:
            #     # 添加噪声
            #     if self.args.jitter:
            #         x_context_array = self._add_jitter(x_context_array)
            #         x_future_array = self._add_jitter(x_future_array)
                
            #     # 缩放
            #     if self.args.scaling:
            #         x_context_array = self._scale_data(x_context_array)
            #         x_future_array = self._scale_data(x_future_array)
            
            x_context = torch.tensor(x_context_array, dtype=torch.float32)
            x_future = torch.tensor(x_future_array, dtype=torch.float32)
            
            # 生成时间特征
            x_mark = torch.zeros((context_len, 4), dtype=torch.float32)
            y_mark = torch.zeros((horizon_len, 4), dtype=torch.float32)
            
            ttt_data.append((x_context, x_future, x_mark, y_mark))
        
        if len(ttt_data) == 0:
            return []

        return ttt_data
        
    def _add_jitter(self, data, scale=0.1):
        """添加随机噪声"""
        noise = np.random.normal(0, scale, data.shape)
        return data + noise
        
    def _scale_data(self, data, scale_range=(0.8, 1.2)):
        """随机缩放数据"""
        scale = np.random.uniform(scale_range[0], scale_range[1])
        return data * scale
        
    def reset_ttt_environment(self, base_model=None, optimizer_status=None):
        self.model = copy.deepcopy(base_model)

        # self.choose_training_parts()

        self.model_optim = self._select_optimizer()
        self.model_optim.load_state_dict(optimizer_status['optimizer_state_dict'])
        
class TTTDataset(Dataset):
    """自定义TTT训练数据集"""
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x_context, x_future, x_mark, y_mark = self.data[idx]
        return x_context, x_future, x_mark, y_mark
