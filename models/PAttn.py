import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from einops import rearrange


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2406.16964
    """
    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_size = patch_len 
        self.stride = stride
        self.task_name = configs.task_name
        
        self.d_model = configs.d_model
       
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 2
        self.padding_patch_layer = nn.ReplicationPad1d((0,  self.stride)) 
        self.in_layer = nn.Linear(self.patch_size, self.d_model)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(1)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )
        self.out_layer = nn.Linear(self.d_model * self.patch_num, configs.pred_len)
        
        # 添加重建任务的头部，用于多任务学习中的掩码重建
        self.reconstruction_head = nn.Linear(self.d_model * self.patch_num, configs.seq_len)
            
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # 判断是否为多任务学习模式
        multi_task_mode = mask is not None
        
        # 对数据进行标准化处理
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        
        # 如果是多任务学习模式，则同时进行预测和重建
        if multi_task_mode:
            # 预测任务：使用原始数据（未掩码数据）
            B, _, C = x_enc.shape
            x_enc_orig = x_enc.permute(0, 2, 1)
            x_enc_orig = self.padding_patch_layer(x_enc_orig)
            x_enc_orig = x_enc_orig.unfold(dimension=-1, size=self.patch_size, step=self.stride)
            enc_out_orig = self.in_layer(x_enc_orig)
            enc_out_orig = rearrange(enc_out_orig, 'b c m l -> (b c) m l')
            dec_out_orig, _ = self.encoder(enc_out_orig)
            dec_out_orig = rearrange(dec_out_orig, '(b c) m l -> b c (m l)', b=B, c=C)
            
            # 预测输出
            forecast_out = self.out_layer(dec_out_orig)
            forecast_out = forecast_out.permute(0, 2, 1)
            forecast_out = forecast_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            forecast_out = forecast_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            
            # 重建任务：使用掩码数据
            # 应用掩码
            inp = x_enc.clone().detach()
            inp = inp.masked_fill(mask == 0, 0)
            
            # 重建处理
            inp = inp.permute(0, 2, 1)
            inp = self.padding_patch_layer(inp)
            inp = inp.unfold(dimension=-1, size=self.patch_size, step=self.stride)
            enc_out_mask = self.in_layer(inp)
            enc_out_mask = rearrange(enc_out_mask, 'b c m l -> (b c) m l')
            dec_out_mask, _ = self.encoder(enc_out_mask)
            dec_out_mask = rearrange(dec_out_mask, '(b c) m l -> b c (m l)', b=B, c=C)
            
            # 重建输出
            imputation_out = self.reconstruction_head(dec_out_mask)
            imputation_out = imputation_out.permute(0, 2, 1)
            imputation_out = imputation_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
            imputation_out = imputation_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
            
            return forecast_out, imputation_out
            
        # 单任务预测模式
        else:
            B, _, C = x_enc.shape
            x_enc = x_enc.permute(0, 2, 1)
            x_enc = self.padding_patch_layer(x_enc)
            x_enc = x_enc.unfold(dimension=-1, size=self.patch_size, step=self.stride)
            enc_out = self.in_layer(x_enc)
            enc_out = rearrange(enc_out, 'b c m l -> (b c) m l')
            dec_out, _ = self.encoder(enc_out)
            dec_out = rearrange(dec_out, '(b c) m l -> b c (m l)', b=B, c=C)
            
            # 预测任务的输出
            forecast_out = self.out_layer(dec_out)
            forecast_out = forecast_out.permute(0, 2, 1)
            
            forecast_out = forecast_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            forecast_out = forecast_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            
            return forecast_out