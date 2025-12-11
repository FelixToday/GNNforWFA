import copy

import torch.nn as nn
import math
import torch
from einops.layers.torch import Rearrange

import time
from functools import wraps
def timer(func):
    """支持类方法的计时装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time

        # 获取函数名，如果是类方法，显示类名
        func_name = func.__name__
        if args and hasattr(args[0], '__class__'):
            class_name = args[0].__class__.__name__
            print(f"方法 {class_name}.{func_name} 运行时间: {elapsed_time:.6f} 秒")
        else:
            print(f"函数 {func_name} 运行时间: {elapsed_time:.6f} 秒")
        return result
    return wrapper
class CNN(nn.Module):
    def __init__(self, num_classes=100, input_features=1):
        super(CNN, self).__init__()

        self.model = CNN_model(num_classes=num_classes, input_features=input_features)

        # Adaptive average pooling layer for classification
        self.classifier = nn.AdaptiveAvgPool1d(1)

    def forward(self, A, x):
        x = Rearrange('b t n f -> b 1 f (t n)')(x)
        feat = self.model(x)

        x = self.classifier(feat)
        x = x.view(x.size(0), -1)
        return x


class CNN_model(nn.Module):
    def __init__(self, num_classes=100, input_features = 1):
        """
        Initialize the RF model.

        Parameters:
        num_classes (int): Number of output classes.
        num_tab (int): Number of tabs (not used in this model).
        """
        super(CNN_model, self).__init__()

        # Create feature extraction layers
        features = make_layers([128, 128, 'M', 256, 256, 'M', 512] + [num_classes])
        init_weights = True
        self.first_layer_in_channel = input_features
        self.first_layer_out_channel = 32

        # Create the initial convolutional layers
        self.first_layer = make_first_layers()
        self.features = features
        self.class_num = num_classes

        # Initialize weights
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        """
        Forward pass of the model.

        Parameters:
        x (Tensor): Input tensor.

        Returns:
        Tensor: Output tensor after passing through the network.
        """
        x = self.first_layer(x)
        x = x.view(x.size(0), self.first_layer_out_channel, -1)
        x = self.features(x)
        return x

    def _initialize_weights(self):
        """
        Initialize weights for the network layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, in_channels=32):
    """
    Create a sequence of convolutional and pooling layers.

    Parameters:
    cfg (list): Configuration list specifying the layers.
    in_channels (int): Number of input channels.

    Returns:
    nn.Sequential: Sequential container with the layers.
    """
    layers = []

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool1d(3), nn.Dropout(0.3)]
        else:
            conv1d = nn.Conv1d(in_channels, v, kernel_size=3, stride=1, padding=1)
            layers += [conv1d, nn.BatchNorm1d(v, eps=1e-05, momentum=0.1, affine=True), nn.ReLU()]
            in_channels = v

    return nn.Sequential(*layers)


def make_first_layers(in_channels=1, out_channel=32):
    """
    Create the initial convolutional layers.

    Parameters:
    in_channels (int): Number of input channels.
    out_channel (int): Number of output channels.

    Returns:
    nn.Sequential: Sequential container with the initial layers.
    """
    layers = []
    conv2d1 = nn.Conv2d(in_channels, out_channel, kernel_size=(3, 6), stride=1, padding=(1, 1))
    layers += [conv2d1, nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True), nn.ReLU()]

    conv2d2 = nn.Conv2d(out_channel, out_channel, kernel_size=(3, 6), stride=1, padding=(1, 1))
    layers += [conv2d2, nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True), nn.ReLU()]

    layers += [nn.MaxPool2d((1, 3)), nn.Dropout(0.1)]

    conv2d3 = nn.Conv2d(out_channel, 64, kernel_size=(3, 6), stride=1, padding=(1, 1))
    layers += [conv2d3, nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True), nn.ReLU()]

    conv2d4 = nn.Conv2d(64, 64, kernel_size=(3, 6), stride=1, padding=(1, 1))
    layers += [conv2d4, nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True), nn.ReLU()]

    layers += [nn.MaxPool2d((2, 2)), nn.Dropout(0.1)]

    return nn.Sequential(*layers)












class STGCN_Sequential(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, A, X):
        for layer in self.layers:
            if isinstance(layer, STGCN_Conv1D):
                X = layer(A, X)
            else:  # 对于 nn.ReLU 等单输入层
                X = layer(X)
        return X
class STGCN_Conv1D(nn.Module):
    """
    单模块 ST-GCN，带 batch 维度
    输入:
        A: (B, T, N, N)
        X: (B, T, N, Fin)
    输出:
        Y: (B, T_out, N, Fout)
    """
    def __init__(self, in_features, out_features, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.K = kernel_size
        self.stride = stride
        self.padding = padding
        self.W = nn.Parameter(torch.Tensor(in_features, out_features))
        nn.init.xavier_uniform_(self.W)
    def build_TS_adjacency(self, A_batch):
        B, K, N, _ = A_batch.shape
        total_size = K * N

        device = A_batch.device
        dtype = A_batch.dtype

        # 初始化全零矩阵
        A_local = torch.zeros((B, total_size, total_size), device=device, dtype=dtype)
        # 批量设置对角线块
        for i in range(K):
            start_row = i * N
            end_row = (i + 1) * N
            A_local[:, start_row:end_row, start_row:end_row] = A_batch[:, i, :, :]

        # 创建单位矩阵并扩展到批量
        eye_batch = torch.eye(N, device=device, dtype=dtype).unsqueeze(0).expand(B, N, N)

        # 设置上对角线和下对角线
        for i in range(K - 1):
            # 上对角线
            start_row = i * N
            end_row = (i + 1) * N
            start_col = (i + 1) * N
            end_col = (i + 2) * N
            A_local[:, start_row:end_row, start_col:end_col] = eye_batch

            # 下对角线
            A_local[:, start_col:end_col, start_row:end_row] = eye_batch

        return A_local

    def forward(self, A, X):
        """
        A: (B, T, N, N)
        X: (B, T, N, Fin)
        """
        B, T, N, Fin = X.shape

        # ---------- 1. 时间维 padding ----------
        if self.padding > 0:
            pad_A = A[:, 0:1].repeat(1, self.padding, 1, 1)  # (B, padding, N, N)
            pad_X = torch.zeros(B, self.padding, N, Fin, device=X.device)
            A = torch.cat([pad_A, A, pad_A], dim=1)  # 时间维拼接
            X = torch.cat([pad_X, X, pad_X], dim=1)
            T = A.shape[1]

        # ---------- 2. 输出时间长度 ----------
        T_out = (T - self.K) // self.stride + 1
        outputs = []

        # ---------- 3. 滑动窗口 ----------
        for i in range(T_out):
            t = i * self.stride
            A_win = A[:, t:t+self.K]  # (B, K, N, N)
            X_win = X[:, t:t+self.K]  # (B, K, N, Fin)
            # 将其组合为大矩阵

            A_local = self.build_TS_adjacency(A_win)
            X_local = X_win.reshape(B, -1, Fin)  # (B, N*K, Fin)
            Y_t = A_local @ X_local @ self.W  # (B, N*K, Fout)
            Y_t = Y_t.reshape(B, self.K, N,self.out_features)  # (B, N, K, Fout)
            Y_t = torch.mean(Y_t, dim=1).reshape(B, N, self.out_features)  # (B, N, Fout)
            # # ---------- 4. 时空卷积 ----------
            # Y_t = 0
            # for k in range(self.K):
            #     # batch 矩阵乘法
            #     AX = torch.matmul(A_win[:, k], X_win[:, k])  # (B, N, Fin)
            #     AXW = torch.matmul(AX, self.W[k])           # (B, N, Fout)
            #     Y_t += AXW
            #
            outputs.append(Y_t)
        outputs = torch.stack(outputs, dim=1)  # (B, T_out, N, Fout)
        return outputs


class STGCN_Pool1D(nn.Module):
    """
    ST-GCN 时间池化层，带 batch 维度
    输入:
        X: (B, T, N, F)
    输出:
        Y: (B, T_out, N, F)
    """
    def __init__(self, kernel_size=3, stride=None):
        super().__init__()
        self.K = kernel_size
        if stride is None:
            self.stride = kernel_size
        else:
            self.stride = stride

    def forward(self, X):
        B, T, N, F = X.shape
        K = self.K
        stride = self.stride
        T_out = (T - K) // stride + 1
        outputs = []

        for i in range(T_out):
            t = i * stride
            X_win = X[:, t:t+K]        # (B, K, N, F)
            Y_t = X_win.mean(dim=1)    # 时间维平均 -> (B, N, F)
            outputs.append(Y_t)

        return torch.stack(outputs, dim=1)  # (B, T_out, N, F)


class STGCN_AdaptivePool1D(nn.Module):
    """
    ST-GCN 自适应时间池化层，带 batch 维度
    输入:
        X: (B, T, N, F)
    输出:
        Y: (B, output_size, N, F)
    """
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, X):
        B, T, N, F = X.shape
        if self.output_size == T:
            return X

        out = []
        for i in range(self.output_size):
            start = int(round(i * T / self.output_size))
            end = int(round((i + 1) * T / self.output_size))
            if end <= start:
                end = start + 1
            X_win = X[:, start:end]      # (B, window_size, N, F)
            Y_i = X_win.mean(dim=1)      # 时间维平均 -> (B, N, F)
            out.append(Y_i)

        return torch.stack(out, dim=1)    # (B, output_size, N, F)

class STGCN_G1(nn.Module):
    def __init__(self, num_classes, input_dim, level_count, embed_dim=128, is_parallel=True, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.is_parallel = is_parallel

        self.embed = nn.Linear(input_dim, self.embed_dim)
        self.share_net = STGCN_Sequential(
            STGCN_Conv1D(in_features=self.embed_dim, out_features=32, kernel_size=3),
            STGCN_Conv1D(in_features=32, out_features=64, kernel_size=7),
            #STGCN_Pool1D(2),
            nn.ReLU(),
            STGCN_Conv1D(in_features=64, out_features=128, kernel_size=5),
            #STGCN_Pool1D(2),
            # nn.ReLU(),
            # STGCN_Conv1D(in_features=128, out_features=256, kernel_size=3),
            # nn.ReLU(),
        )

        output_dim = level_count * 128
        if is_parallel:
            self.ind_net = nn.Sequential(
                # 将输入在通道维度上复制num_classes份
                # 使用groups=num_classes实现独立卷积
                nn.Conv1d(output_dim * num_classes, 1024 * num_classes, 3,
                          groups=num_classes, padding=1),
                nn.AdaptiveAvgPool1d(1),
                nn.ReLU(),
                Rearrange('B (C H) 1 -> B C H', C=num_classes),  # (B, num_classes, 1024)
                nn.Linear(1024, 1)  # 每个类别独立线性层
            )
        else:
            self.ind_net = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(output_dim, 1024, 3),
                    nn.AdaptiveAvgPool1d(1),
                    nn.ReLU(),
                    Rearrange('B T D -> B (T D)'),
                    nn.Linear(1024, 1)
                )
                for _ in range(num_classes)
            ])
    #@timer
    def forward(self, A, x):
        x = self.embed(x)
        x = self.share_net(A, x)
        x = Rearrange('B T N D -> B (N D) T')(x)
        if self.is_parallel:
            x = x.repeat(1, self.num_classes, 1)  # (B, 7680*num_classes, T)
            x = self.ind_net(x)
            x = x.squeeze(-1)  # (B, num_classes)
        else:
            x = [ind_net_i(copy.deepcopy(x)) for ind_net_i in self.ind_net]
            x = torch.cat(x,dim=1)
            pass
        return x

class STGCN_G2(nn.Module):
    def __init__(self, num_classes, input_dim, level_count, embed_dim=128, is_parallel=True, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.is_parallel = is_parallel

        self.embed = nn.Linear(input_dim, self.embed_dim)
        self.share_cnn = nn.Sequential(
            nn.Conv1d(self.embed_dim, 128, kernel_size=7),
            nn.MaxPool1d(kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=7),
            nn.MaxPool1d(kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=5),
            nn.MaxPool1d(kernel_size=3),
            nn.ReLU(),
        )
        self.share_atten = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048, batch_first=True),
            num_layers=2,
        )


        self.ind_net = nn.ModuleList([
            nn.Sequential(
            nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048, batch_first=True),
            Rearrange('B T D -> B D T'),
            nn.AdaptiveAvgPool1d(1),
            Rearrange('B D T -> B (D T)'),
            nn.Linear(512, 1)
            )
            for _ in range(num_classes)
        ])

    #@timer
    def forward(self, A, x):
        x = self.embed(x)
        x = Rearrange('B T N D -> B D (T N)')(x)
        x = self.share_cnn(x)
        x = Rearrange('B D T -> B T D')(x)
        x = self.share_atten(x)
        x = [ind_net_i(x.detach().clone()) for ind_net_i in self.ind_net]
        x = torch.cat(x, dim=1)
        return x


if __name__ == "__main__":
    from torchinfo import summary
    # 测试参数
    B, T, N, n_dim = 32, 100, 30, 4
    embed_dim = 128
    num_classes = 95
    # ===========================
    # 随机输入数据
    # ===========================
    X = torch.randn(B, T, N, n_dim)           # 节点特征
    A = torch.randn(B, T, N, N)              # 邻接矩阵（随机测试用）
    y = torch.randint(0, num_classes, (B,))  # 随机标签

    # ===========================
    # 初始化模型
    # ===========================
    model = STGCN_G2(num_classes=num_classes, input_dim=n_dim, embed_dim=embed_dim, is_parallel=False,
                     level_count=N)
    summary(model, input_data=(A, X), depth=1, col_names=["input_size", "output_size", "num_params", "trainable"])

    # model = STGCN_G1(num_classes=num_classes, input_dim=n_dim, embed_dim=embed_dim, is_parallel=True,
    #                  level_count=N)
    # summary(model, input_data=(A, X), depth=1, col_names=["input_size", "output_size", "num_params", "trainable"])