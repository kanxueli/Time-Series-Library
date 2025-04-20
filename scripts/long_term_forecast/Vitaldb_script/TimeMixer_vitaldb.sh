model_name=TimeMixer

seq_len=32
e_layers=2
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=16
d_ff=32
train_epochs=10
patience=10
batch_size=16
device_index=0

root_path=/home/likx/time_series_forecasting/cma_dataset_preprocess/vitaldb_dataset/sample_step30/

# 15min->5min
CUDA_VISIBLE_DEVICES=$device_index python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path  $root_path\
  --model_id ETTh1_$seq_len'_'5min \
  --model $model_name \
  --data VitalDB \
  --features M\
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 10 \
  --e_layers $e_layers \
  --enc_in 4 \
  --dec_in 4 \
  --c_out 4 \
  --des 'Exp_15to5min' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size 32 \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window \
  --inverse

# # 15min->10min
# CUDA_VISIBLE_DEVICES=$device_index python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path $root_path \
#   --model_id ETTh1_$seq_len'_'10min \
#   --model $model_name \
#   --data VitalDB \
#   --features M \
#   --seq_len $seq_len \
#   --label_len 0 \
#   --pred_len 20 \
#   --e_layers $e_layers \
#   --enc_in 4 \
#   --dec_in 4 \
#   --c_out 1 \
#   --des 'Exp_15to10min' \
#   --itr 1 \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --learning_rate $learning_rate \
#   --train_epochs $train_epochs \
#   --patience $patience \
#   --batch_size 32 \
#   --down_sampling_layers $down_sampling_layers \
#   --down_sampling_method avg \
#   --down_sampling_window $down_sampling_window \
#   --inverse

# # 15min->15min
# CUDA_VISIBLE_DEVICES=$device_index python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path $root_path \
#   --model_id ETTh1_$seq_len'_'15min \
#   --model $model_name \
#   --data VitalDB \
#   --features M \
#   --seq_len $seq_len \
#   --label_len 0 \
#   --pred_len $seq_len \
#   --e_layers $e_layers \
#   --enc_in 4 \
#   --dec_in 4 \
#   --c_out 1 \
#   --des 'Exp_15to15min' \
#   --itr 1 \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --learning_rate $learning_rate \
#   --train_epochs $train_epochs \
#   --patience $patience \
#   --batch_size 32 \
#   --down_sampling_layers $down_sampling_layers \
#   --down_sampling_method avg \
#   --down_sampling_window $down_sampling_window \
#   --inverse
