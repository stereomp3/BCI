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


class LSTM(nn.Module):
    '''
        Employ the Bi-LSTM to learn the reliable dependency between spatio-temporal features
    '''

    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=True, num_layers=1)

    def forward(self, x):
        b, c, T = x.size()
        x = x.view(x.size(-1), -1, c)  # (b, c, T) -> (T, b, c)
        r_out, _ = self.rnn(x)  # r_out shape [time_step * 2, batch_size, output_size]
        out = r_out.view(b, 2 * T * c, -1)
        return out

# from https://github.com/YuDongPan/SSVEPNet/blob/master/Model/SSVEPNet.py
class ESNet(nn.Module):
    def calculateOutSize(self, model, nChan, nTime):
        '''
            Calculate the output based on input size
            model is from nn.Module and inputSize is a array
        '''
        data = torch.randn(1, 1, nChan, nTime)
        out = model(data).shape
        return out[1:]

    def spatial_block(self, nChan, dropout_level):
        '''
           Spatial filter block,assign different weight to different channels and fuse them
        '''
        block = []
        block.append(Conv2dWithConstraint(in_channels=1, out_channels=nChan * 2, kernel_size=(nChan, 1),
                                                     max_norm=1.0))
        block.append(nn.BatchNorm2d(num_features=nChan * 2))
        block.append(nn.PReLU())
        block.append(nn.Dropout(dropout_level))
        layer = nn.Sequential(*block)
        return layer

    def enhanced_block(self, in_channels, out_channels, dropout_level, kernel_size, stride):
        '''
           Enhanced structure block,build a CNN block to absorb data and output its stable feature
        '''
        block = []
        block.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, kernel_size),
                               stride=(1, stride)))
        block.append(nn.BatchNorm2d(num_features=out_channels))
        block.append(nn.PReLU())
        block.append(nn.Dropout(dropout_level))
        layer = nn.Sequential(*block)
        return layer

    def __init__(self, num_channels, T, num_classes):
        super(ESNet, self).__init__()
        self.dropout_level = 0.5
        self.F = [num_channels * 2] + [num_channels * 4]
        self.K = 10
        self.S = 2

        net = []
        net.append(self.spatial_block(num_channels, self.dropout_level))
        net.append(self.enhanced_block(self.F[0], self.F[1], self.dropout_level,
                                       self.K, self.S))

        self.conv_layers = nn.Sequential(*net)

        self.fcSize = self.calculateOutSize(self.conv_layers, num_channels, T)
        self.fcUnit = self.fcSize[0] * self.fcSize[1] * self.fcSize[2] * 2
        self.D1 = self.fcUnit // 10
        self.D2 = self.D1 // 5

        self.rnn = LSTM(input_size=self.F[1], hidden_size=self.F[1])

        self.dense_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.fcUnit, self.D1),
            nn.PReLU(),
            nn.Linear(self.D1, self.D2),
            nn.PReLU(),
            nn.Dropout(self.dropout_level),
            nn.Linear(self.D2, num_classes))

    def forward(self, x):
        out = self.conv_layers(x)
        out = out.squeeze(2)
        r_out = self.rnn(out)
        out = self.dense_layers(r_out)
        return out


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, X):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(X)


class EEGNet_SSVEP(nn.Module):
    def __init__(self, n_channels, n_samples, n_classes):
        super().__init__()

        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, 64), padding=0),
            nn.BatchNorm2d(8)
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(n_channels, 1), groups=8),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, n_channels)),
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