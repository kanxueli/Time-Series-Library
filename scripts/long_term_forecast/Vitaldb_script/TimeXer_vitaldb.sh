root_path=/home/likx/time_series_forecasting/cma_dataset_preprocess/vitaldb_dataset/sample_step30/
device_index=1
seq_len=30
batch_size=512
model_name=TimeXer

# 多任务学习参数
use_multi_task=1
mask_rate=0.20    # 掩码率
mr_loss_ratio=0.5

CUDA_VISIBLE_DEVICES=$device_index python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --model_id VitalDB_15m_pred_5m \
  --model $model_name \
  --data VitalDB \
  --features MS \
  --seq_len $seq_len \
  --label_len $((seq_len / 2)) \
  --pred_len 10 \
  --train_epochs 10 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 4 \
  --dec_in 4 \
  --c_out 1 \
  --d_model 256 \
  --batch_size $batch_size \
  --des 'Exp_15to5min' \
  --itr 1 \
  --inverse \
  --use_multi_task $use_multi_task \
  --mask_rate $mask_rate \
  --mr_loss_ratio $mr_loss_ratio

CUDA_VISIBLE_DEVICES=$device_index python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --model_id VitalDB_15m_pred_10m \
  --model $model_name \
  --data VitalDB \
  --features MS \
  --seq_len $seq_len \
  --label_len $((seq_len / 2)) \
  --pred_len 20 \
  --e_layers 2 \
  --factor 3 \
  --enc_in 4 \
  --dec_in 4 \
  --c_out 1 \
  --des 'Exp_15to10min' \
  --d_model 128 \
  --batch_size $batch_size \
  --itr 1 \
  --inverse \
  --use_multi_task $use_multi_task \
  --mask_rate $mask_rate \
  --mr_loss_ratio $mr_loss_ratio

CUDA_VISIBLE_DEVICES=$device_index python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --model_id VitalDB_15m_pred_15m \
  --model $model_name \
  --data VitalDB \
  --features MS \
  --seq_len $seq_len \
  --label_len $((seq_len / 2)) \
  --pred_len 30 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 4 \
  --dec_in 4 \
  --c_out 1 \
  --des 'Exp_15to15min' \
  --d_model 512 \
  --d_ff 1024 \
  --batch_size $batch_size \
  --itr 1 \
  --inverse \
  --use_multi_task $use_multi_task \
  --mask_rate $mask_rate \
  --mr_loss_ratio $mr_loss_ratio

