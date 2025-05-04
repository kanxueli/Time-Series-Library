import numpy as np
import json
from sklearn.metrics import auc, classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score, f1_score, precision_recall_curve

# @njit
def Check_If_IOH(time_series, srate, IOH_value, duration):
    """
    Check if there is a period of intraoperative hypotension (IOH) in the time series.

    Parameters:
    - time_series (1D array-like): The blood pressure time series.
    - srate: Sampling rate of the time series (samples per second).
    - IOH_value: Threshold value for IOH (blood pressure below this is considered hypotensive).
    - duration: duration in seconds that defines IOH (must stay below IOH_value for this period).

    Returns:
    - bool: True if IOH is detected, otherwise False.
    """
    # 确保 time_series 是 numpy array
    if isinstance(time_series, list):
        time_series = np.array(time_series)

    # 将Duration转换为采样点数
    duration_samples = int(duration * srate)
    
    # 如果时间序列长度小于duration_samples，不可能满足IOH条件，直接返回False
    if len(time_series) < duration_samples:
        return False
    
    # 创建一个布尔掩码数组，标记低于IOH阈值的点
    below_threshold = time_series < IOH_value
    
    # 使用滑动窗口检查是否存在连续的duration_samples个值都低于IOH_value
    for i in range(len(below_threshold) - duration_samples + 1):
        # 检查当前滑动窗口内的所有值是否都为True（即都低于IOH_value）
        if np.all(below_threshold[i:i + duration_samples]):
            return True
    
    return False

def sliding_window_average(time_series, slide_len):
    if slide_len <= 0:
        raise ValueError("slide_len must be greater than 0")
    
    # 存储滑动窗口的平均值
    window_averages = []
    
    # 遍历序列，按滑动窗口大小取值
    for i in range(0, len(time_series), slide_len):
        # 获取当前窗口的值
        window = time_series[i:i + slide_len]
        # 计算窗口的平均值并存储
        window_avg = round(np.nanmean(window), 2)
        window_averages.append(window_avg)
    
    return window_averages


# 只适用s为单位的抽样,且 slide_len必须能被stime整除，以及Duration能被slide_len整除，不然求出来的序列有误差。
def Check_If_IOH_Combined_S(time_series, stime, IOH_value, Duration, slide_len):
    # Duration 和 滑动窗口长度转为采样点
    duration_samples = int(Duration / slide_len)
    slide_samples = slide_len
    
    # 计算滑动窗口的平均值
    if slide_samples == 1:
        smoothed_series = time_series
    else:
        # smoothed_series = Count_Windows_MovingAvg(time_series, slide_samples)
        smoothed_series = sliding_window_average(time_series, slide_samples)

    # Step 2: 对平滑后的序列进行低血压检测
    if duration_samples == 1:
        evt = np.nanmax(smoothed_series) < IOH_value
    else:
        # Step 3: 逐点判断
        evt = Check_If_IOH(smoothed_series, 1, IOH_value, duration_samples)
        # evt = np.nanmax(sliding_window_value) < IOH_value

    # print("evt:", evt, "max:", np.nanmax(sliding_window_count) )
    return evt

def user_definable_IOH(time_series):
    predcition_lables = []
    for i in time_series:
        predcition_lables.append(Check_If_IOH_Combined_S(i, 1, 65, 2, 1))
    return predcition_lables

def Classify_Metrics(pred, true):
    # for i in range(len(pred)):
    ground_true_labels = user_definable_IOH(true.reshape(true.shape[0], true.shape[1]))
    prediction_lables = user_definable_IOH(pred.reshape(pred.shape[0], pred.shape[1]))

    print("##### Prediction Results ####")
    precision, recall, thmbps = precision_recall_curve(ground_true_labels, prediction_lables)
    auprc = auc(recall, precision)

    fpr, tpr, thmbps = roc_curve(ground_true_labels, prediction_lables)
    auroc = auc(fpr, tpr)
    f1 = f1_score(ground_true_labels, prediction_lables)
    acc = accuracy_score(ground_true_labels, prediction_lables)
    tn, fp, fn, tp = confusion_matrix(ground_true_labels, prediction_lables).ravel()

    testres = 'auroc={:.3f}, auprc={:.3f} acc={:.3f}, F1={:.3f}, Recall={:.1f}, Prec={:.1f}, NPV={:.1f}, TN={}, fp={}, fn={}, TP={}'.format(auroc*100, auprc*100, acc*100, f1*100, tp/(tp+fn)*100, tp/(tp+fp)*100, tn/(tn+fn)*100, tn, fp, fn, tp)
    print(testres)

    return auroc*100, auprc*100, acc*100, f1*100, tp/(tp+fn)*100, tp/(tp+fp)*100, tn/(tn+fn)*100, tn, fp, fn, tp


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 *
                (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def run_metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    auroc, auprc, acc, f1, recall, precision, npv, tn, fp, fn, tp = Classify_Metrics(pred, true)
    print(f'Metrics: MAE={mae:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}, MAPE={mape:.4f}, MSPE={mspe:.4f}')

    return mae, mse, rmse, mape, mspe, auroc, auprc, acc, f1, recall, precision, npv, tn, fp, fn, tp

def load_predictions(file_path):
    """
    Load the predictions and true values from a file.

    Parameters:
    - file_path (str): The file path from which to load the data.

    Returns:
    - preds (ndarray): The loaded predicted values.
    - trues (ndarray): The loaded ground truth values.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    preds = np.array(data['predictions'])
    trues = np.array(data['ground_truths'])

    return preds, trues



# # 读取保存的预测和真实值
# preds, trues = load_predictions('predictions_and_trues.json')

# # 调用 metric 函数进行评测
# mae, mse, rmse, mape, mspe = run_metric(preds, trues)
# print(f'Metrics: MAE={mae}, MSE={mse}, RMSE={rmse}, MAPE={mape}, MSPE={mspe}')
