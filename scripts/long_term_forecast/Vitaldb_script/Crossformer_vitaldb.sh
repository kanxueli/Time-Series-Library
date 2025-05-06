root_path=/home/likx/time_series_forecasting/cma_dataset_preprocess/vitaldb_dataset/sample_step30/
device_index=1
seq_len=30
batch_size=512
model_name=Crossformer

# 训练参数
is_training=0
use_multi_task=1
mask_rate=0.1
mr_loss_ratio=0.1

# ttt 参数
use_ttt=1
ttt_test_batch_size=8
ttt_train_batch_size=64
ttt_lr=1e-4
ttt_train_epochs=1

CUDA_VISIBLE_DEVICES=$device_index python -u run.py \
  --task_name long_term_forecast \
  --is_training $is_training \
  --root_path $root_path \
  --model_id VitalDB_15m_pred_5m \
  --model $model_name \
  --data VitalDB \
  --features MS \
  --seq_len $seq_len \
  --label_len $((seq_len / 2)) \
  --pred_len 10 \
  --batch_size $batch_size \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 4 \
  --dec_in 4 \
  --c_out 1 \
  --des 'Exp_15to5min' \
  --itr 1 \
  --inverse \
  --use_multi_task $use_multi_task \
  --mask_rate $mask_rate \
  --mr_loss_ratio $mr_loss_ratio \
  --use_ttt $use_ttt \
  --ttt_test_batch_size $ttt_test_batch_size \
  --ttt_train_batch_size $ttt_train_batch_size \
  --ttt_lr $ttt_lr \
  --ttt_train_epochs $ttt_train_epochs

CUDA_VISIBLE_DEVICES=$device_index python -u run.py \
  --task_name long_term_forecast \
  --is_training $is_training \
  --root_path $root_path \
  --model_id VitalDB_15m_pred_10m \
  --model $model_name \
  --data VitalDB \
  --features MS \
  --seq_len $seq_len \
  --label_len $((seq_len / 2)) \
  --pred_len 20 \
  --batch_size $batch_size \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 4 \
  --dec_in 4 \
  --c_out 1 \
  --des 'Exp_15to10min' \
  --itr 1 \
  --inverse \
  --use_multi_task $use_multi_task \
  --mask_rate $mask_rate \
  --mr_loss_ratio $mr_loss_ratio \
  --use_ttt $use_ttt \
  --ttt_test_batch_size $ttt_test_batch_size \
  --ttt_train_batch_size $ttt_train_batch_size \
  --ttt_lr $ttt_lr \
  --ttt_train_epochs $ttt_train_epochs

CUDA_VISIBLE_DEVICES=$device_index python -u run.py \
  --task_name long_term_forecast \
  --is_training $is_training \
  --root_path $root_path \
  --model_id VitalDB_15m_pred_15m \
  --model $model_name \
  --data VitalDB \
  --features MS \
  --seq_len $seq_len \
  --label_len $((seq_len / 2)) \
  --pred_len 30 \
  --batch_size $batch_size \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 4 \
  --dec_in 4 \
  --c_out 1 \
  --des 'Exp_15to15min' \
  --itr 1 \
  --inverse \
  --use_multi_task $use_multi_task \
  --mask_rate $mask_rate \
  --mr_loss_ratio $mr_loss_ratio \
  --use_ttt $use_ttt \
  --ttt_test_batch_size $ttt_test_batch_size \
  --ttt_train_batch_size $ttt_train_batch_size \
  --ttt_lr $ttt_lr \
  --ttt_train_epochs $ttt_train_epochs
