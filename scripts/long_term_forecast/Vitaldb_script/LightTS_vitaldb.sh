root_path=/home/likx/time_series_forecasting/cma_dataset_preprocess/vitaldb_dataset/sample_step30/
device_index=0
seq_len=30
batch_size=512
model_name=LightTS

# CUDA_VISIBLE_DEVICES=$device_index python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path $root_path \
#   --model_id VitalDB_15m_pred_5m \
#   --model $model_name \
#   --data VitalDB \
#   --features MS \
#   --seq_len $seq_len \
#   --label_len $((seq_len / 2)) \
#   --pred_len 10 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 4 \
#   --dec_in 4 \
#   --c_out 1 \
#   --batch_size $batch_size \
#   --des 'Exp_15to5min' \
#   --itr 1 \
#   --freq s \
#   --inverse

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
  --d_layers 1 \
  --factor 3 \
  --enc_in 4 \
  --dec_in 4 \
  --c_out 1 \
  --batch_size $batch_size \
  --des 'Exp_15to10min' \
  --itr 1 \
  --freq s \
  --inverse

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
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 4 \
  --dec_in 4 \
  --c_out 1 \
  --batch_size $batch_size \
  --des 'Exp_15to15min' \
  --itr 1 \
  --freq s \
  --inverse