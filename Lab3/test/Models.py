import torch.nn as nn
import torch
import torch.nn.functional as F


class SCCNet(nn.Module):
    """SCCNet model from Wei et al. 2019.

    Parameters
    ----------
    C : int
        Number of EEG input channels.
    N : int
        Number of EEG input time samples.
    nb_classes : int
        Number of classes to predict.
    Nu : int, optional
        Number of spatial kernel (default: C).
    Nt : int, optional
        Length of spatial kernel (default: 1).
    Nc : int, optional
        Number of spatial-temporal kernel (default: 20).
    fs : float, optional
        Sampling frequency of EEG input (default: 1000.0).
    dropoutRate : float, optional
        Dropout ratio (default: 0.5).
    """

    def __init__(self, C, N, nb_classes, Nu=None, Nt=1, Nc=20, fs=1000.0, dropoutRate=0.5):
        super(SCCNet, self).__init__()
        self.Nu = Nu if Nu is not None else C
        self.Nc = Nc

        # Spatial Convolution (across channels)
        self.conv1 = nn.Conv2d(  # (batch_size, in_channels, height, width)
            in_channels=1,  # 輸入圖像通道數
            out_channels=self.Nu,  # 捲基產生通道數量
            kernel_size=(C, Nt),  # 捲基 kernel
            padding=0,
        )
        self.per = Permute2d(shape=(0, 2, 1, 3))
        self.bn1 = nn.BatchNorm2d(1)

        # Temporal Convolution (depthwise)
        self.conv2 = nn.Conv2d(
            in_channels=1,
            out_channels=self.Nc,
            kernel_size=(self.Nu, 12),  # kernel length from paper
            padding=0,  # 'same' padding for length 12
        )
        self.bn2 = nn.BatchNorm2d(self.Nc)

        # Dropout
        self.dp = nn.Dropout(dropoutRate)

        # Pooling (Average)
        self.pool = nn.AvgPool2d(kernel_size=(1, 62), stride=(1, 12))

        # We determine the FC layer input size dynamically
        dummy_input = torch.zeros(1, 1, C, N)

        with torch.no_grad():
            dummy_out = self._forward_features(dummy_input)
            self.feature_dim = dummy_out.shape[1]

        self.fc = nn.Linear(self.feature_dim, nb_classes)

    def _forward_features(self, x):
        # print(f"self.conv1 {x.shape}")
        x = self.conv1(x)  # Spatial conv # input shape: ([16, 1, 22, 437]), output shape: ([16, 22, 1, 437])
        # print(x.shape)
        x = self.per(x)  # permute layer # output shape: ([16, 1, 22, 437])
        # print(x.shape)
        x = self.bn1(x)  # output shape: ([16, 1, 22, 437])
        # print(f"self.conv2 {x.shape}")
        x = self.conv2(x)  # Temporal conv # input shape: ([16, 22, 1, 437]), output shape: ([16, 20, 1, 426])
        # print(x.shape)
        x = self.bn2(x)  # output shape: ([16, 20, 1, 426])
        # print(x.shape)
        # x = F.relu(x)  # output shape: ([16, 20, 1, 426])
        x = x ** 2  # amplify important features and enhance nonlinearity
        x = self.dp(x)
        # print(f"pool {x.shape}")
        x = self.pool(x)  # output shape: ([16, 20, 1, 31])
        x = torch.log(x)
        # print(f"out {x.shape}")
        return x.view(x.size(0), -1)  # x.size(0)

    def forward(self, x):
        x = self._forward_features(x)
        x = self.fc(x)
        return x

class Permute2d(nn.Module):
    def __init__(self, shape):
        super(Permute2d, self).__init__()
        self.shape = shape

    def forward(self, x):
        return torch.permute(x, self.shape)
    
###SSVEPformer
class SSVEPformer(nn.Module):
    def __init__(self, C, N, nb_classes):
        super(SSVEPformer, self).__init__()

        # channel combination
        self.conv1 = nn.Conv1d(C, 2*C, 1)
        self.layernorm1 = nn.LayerNorm(N)
       
        #CNN module
        self.layernorm2 = nn.LayerNorm(N)
        self.conv2 = nn.Conv1d(2*C, 2*C, 31,padding=15)
        
        #channel MPL module
        self.layernorm3 = nn.LayerNorm(N)
        self.fc1=nn.Linear(N,N)
        
        #MLP head
        self.flatten=nn.Flatten()
        self.fc2=nn.Linear(2*C*N,6*nb_classes)
        self.layernorm4=nn.LayerNorm(6*nb_classes)
        self.fc3=nn.Linear(6*nb_classes,nb_classes)
        
    def forward(self, x):
        # channel combination
        x = self.conv1(x)
        x = self.layernorm1(x)
        x = F.gelu(x)
        x = F.dropout(x, 0.5)
        
        ### 1st encoder###
        #CNN module
        y = self.layernorm2(x)
        x = self.conv2(y)
        x = self.layernorm2(x)
        x = F.gelu(x)
        x = F.dropout(x, 0.5)
        x=x+y
        #channel MLP module
        y2 = self.layernorm3(x)
        x = self.fc1(y2)
        x = F.gelu(x)
        x = F.dropout(x, 0.5)
        x=x+y2
        
        ###2nd encoder###
        #CNN module
        y = self.layernorm2(x)
        x = self.conv2(y)
        x = self.layernorm2(x)
        x = F.gelu(x)
        x = F.dropout(x, 0.5)
        x=x+y
        #channel MPL module
        y2 = self.layernorm3(x)
        x = self.fc1(y2)
        x = F.gelu(x)
        x = F.dropout(x, 0.5)
        x=x+y2

        #MLP head
        x=self.flatten(x)
        x = F.dropout(x, 0.5)
        x = self.fc2(x)
        x = self.layernorm4(x)
        x = F.gelu(x)
        x = F.dropout(x, 0.5)
        x = self.fc3(x)
    
        return x
    
class EEGNet_SSVEP(nn.Module):
    def __init__(self, n_channels, n_samples, n_classes):
        super().__init__()
        
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, 64), padding=0),
            nn.BatchNorm2d(8)
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(4, 1), groups=8),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(0.5)
        )

        # 利用 dummy input 自動計算 linear 輸入維度
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, n_channels, n_samples)
            x = self.firstconv(dummy_input)
            x = self.depthwiseConv(x)
            flatten_dim = x.view(1, -1).size(1)

        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, n_classes)
        )

    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.classify(x)
        return x