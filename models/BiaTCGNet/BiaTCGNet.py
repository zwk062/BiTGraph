from models.BiaTCGNet.BiaTCGNet_layer import *
import torch.nn as nn
import torch
from torch.nn.utils import weight_norm

class Model(nn.Module):
    def __init__(self, gcn_true, buildA_true, gcn_depth, num_nodes, kernel_set, device, predefined_A=None, static_feat=None, dropout=0.3, subgraph_size=5, node_dim=40, dilation_exponential=1, conv_channels=32, residual_channels=32, skip_channels=64, end_channels=128, seq_length=12, in_dim=2, out_len=12, out_dim=1, layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True):
        super(Model, self).__init__()
        
        # GCN相关参数
        self.gcn_true = gcn_true # 是否使用GCN
        self.buildA_true = buildA_true # 是否构建邻接矩阵
        self.num_nodes = num_nodes # 节点数目
        self.kernel_set = kernel_set # 卷积核大小集合
        self.dropout = dropout # Dropout比率
        self.predefined_A = predefined_A # 预定义的邻接矩阵
        
        # 各种层的ModuleList
        self.filter_convs = nn.ModuleList() # 过滤卷积层
        self.gate_convs = nn.ModuleList() # 门控卷积层
        self.residual_convs = nn.ModuleList() # 残差卷积层
        self.skip_convs = nn.ModuleList() # 跳跃连接卷积层
        self.gconv1 = nn.ModuleList() # 第一组GCN
        self.gconv2 = nn.ModuleList() # 第二组GCN
        self.norm = nn.ModuleList() # 归一化层
        self.output_dim = out_dim # 输出维度
        
        # 输入特征的初始卷积
        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1))
        
        # 生成动态图
        self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha, static_feat=static_feat)
        
        # 计算感受野大小
        self.seq_length = seq_length
        kernel_size = self.kernel_set[-1]  # 选取最大卷积核
        if dilation_exponential > 1:
            self.receptive_field = int(1 + (kernel_size - 1) * (dilation_exponential ** layers - 1) / (dilation_exponential - 1))
        else:
            self.receptive_field = layers * (kernel_size - 1) + 1
        
        # 构建多个层的扩张卷积
        for i in range(1):
            new_dilation = 1
            dilationsize = []  # 记录每层的扩张大小
            for j in range(1, layers + 1):
                rf_size_j = i * layers * (kernel_size - 1) + j * (kernel_size - 1)
                dilationsize.append(seq_length - (kernel_size - 1) * j)
                
                # 添加扩张卷积层
                self.filter_convs.append(dilated_inception(residual_channels, conv_channels, kernel_set, dilation_factor=new_dilation))
                self.gate_convs.append(dilated_inception(residual_channels, conv_channels, kernel_set, dilation_factor=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=conv_channels, out_channels=residual_channels, kernel_size=(1, 1)))
                
                # 添加跳跃连接层
                self.skip_convs.append(nn.Conv2d(in_channels=conv_channels, out_channels=skip_channels, kernel_size=(1, self.seq_length - rf_size_j + 1)))
                
                # 添加GCN层
                if self.gcn_true:
                    self.gconv1.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha, dilationsize[j - 1], num_nodes, self.seq_length, out_len))
                    self.gconv2.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha, dilationsize[j - 1], num_nodes, self.seq_length, out_len))
                
                # 添加归一化层
                self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1), elementwise_affine=layer_norm_affline))
                new_dilation *= dilation_exponential
        
        self.layers = layers
        
        # 结束部分的卷积层
        self.end_conv_1 = weight_norm(nn.Conv2d(in_channels=skip_channels, out_channels=end_channels, kernel_size=(1, 1), bias=True))
        self.end_conv_2 = weight_norm(nn.Conv2d(in_channels=end_channels, out_channels=out_len * out_dim, kernel_size=(1, 1), bias=True))
        
        self.idx = torch.arange(self.num_nodes).cuda()

    def forward(self, input, mask, k, idx=None):
        # 转置input和mask，使时间步位于最后一维
        input = input.transpose(1, 3)
        mask = mask.transpose(1, 3).float()
        
        # 应用掩码
        input = input * mask
        seq_len = input.size(3)
        assert seq_len == self.seq_length, '输入序列长度与预设长度不匹配'
        
        # 填充以匹配感受野
        if self.seq_length < self.receptive_field:
            input = nn.functional.pad(input, (self.receptive_field - self.seq_length, 0, 0, 0))
        
        # 计算邻接矩阵
        if self.gcn_true:
            adp = self.gc(self.idx) if idx is None else self.gc(idx)
        else:
            adp = self.predefined_A
        
        # 通过初始卷积层
        x = self.start_conv(input)
        
        # 跳跃连接初始化
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
        
        # 逐层计算
        for i in range(self.layers):
            residual = x
            filter, mask_filter = self.filter_convs[i](x, mask)
            filter = torch.tanh(filter)
            gate, mask_gate = self.gate_convs[i](x, mask)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = self.skip_convs[i](x)
            skip = s + skip
            
            # 通过GCN层
            if self.gcn_true:
                state1, mask = self.gconv1[i](x, adp, mask_filter, k, flag=0)
                state2, mask2 = self.gconv2[i](x, adp.transpose(1, 0), mask_filter, k, flag=0)
                x = state1 + state2
            else:
                x = self.residual_convs[i](x)
            
            x = x + residual[:, :, :, -x.size(3):]
            x = self.norm[i](x, self.idx if idx is None else idx)
        
        # 结束部分的卷积
        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        
        # 调整输出形状
        B, T, N, D = x.shape
        x = x.reshape(B, -1, self.output_dim, N).permute(0, 1, 3, 2)
        return x
