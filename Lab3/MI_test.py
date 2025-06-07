import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
import mne
from mne.viz import plot_topomap
from torch.utils.data import TensorDataset, DataLoader

from Models import SCCNet
from Utils import EEGDataReader, Visualiser
from Conformer import EEGConformer
from SSL2 import ContrastiveModel, EEGClassifier


class EEGEvaluator:
    def __init__(self, model_class, test_data, checkpoint_path, data_shape, batch_size=16,
                 device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.model_class = model_class
        self.test_data = test_data
        self.checkpoint_path = checkpoint_path
        self.data_shape = data_shape

        self.loss_fn = nn.CrossEntropyLoss()
        self.kwargs = {}  # 可加入模型自訂參數
        self.model = None

    def load_model_and_weights(self):
        # channel, samples, class
        self.model = self.model_class(self.data_shape[0][1], self.data_shape[0][2], self.data_shape[1][0]).to(
            self.device)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()  # for test or validation, effect Dropout/Batch Normalization (stop this layer)

    def load_ssl_model(self):
        # channel, samples, class
        m = SCCNet(22, 437, 4).to(self.device)
        checkpoint = torch.load("_checkpoint/sccnet_train_5e_9e-epoch498.pth", map_location=self.device)
        m.load_state_dict(checkpoint["state_dict"])
        m = ContrastiveModel(m).to(self.device)
        checkpoint = torch.load("checkpoints/pretrained_encoder-epoch199.pth", map_location=self.device)
        m.load_state_dict(checkpoint["state_dict"])
        self.model = self.model_class(m).to(self.device)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()  # for test or validation, effect Dropout/Batch Normalization (stop this layer)

    def load_eeg_conformer(self):
        self.model = self.model_class(n_outputs=4, n_chans=22, n_times=437, n_filters_time=40, filter_time_length=25,
                                      pool_time_length=75, pool_time_stride=15, drop_prob=0.5, att_depth=2, att_heads=5,
                                      att_drop_prob=0.5, final_fc_length='auto')
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # Move to device
        self.model.to(self.device)
        self.model.eval()

    def get_model(self):
        return self.model

    def set_test_data(self, test_data):
        self.test_data = test_data

    def get_evaluate_result(self):
        predicts, targets = [], []
        test_loader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)
        with torch.no_grad():  # for test
            for x_batch, y_batch in test_loader:
                # x_batch = x_batch.squeeze(1)  # for conformer
                # y_batch = y_batch.squeeze(1)  # for conformer
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = self.model(x_batch)
                # print(f"outputs.argmax(dim=1).cpu(): {outputs.argmax(dim=1).cpu()}")
                # print(f"y_batch.argmax(dim=1).cpu(): {y_batch.argmax(dim=1).cpu()}")
                predicts.append(outputs.argmax(dim=1).cpu())  # dim=1 -> select the max prob class.
                targets.append(y_batch.argmax(dim=1).cpu())
        y_pred = torch.cat(predicts).numpy()
        y_true = torch.cat(targets).numpy()
        print(f"accuracy: {np.sum(y_pred == y_true) / len(y_true)}")
        return predicts, targets

    def get_model_weights(self):  # get conv weight
        return self.model.conv1.weight.detach().cpu().numpy()  # shape: [22, 1, 22, 1]


if __name__ == '__main__':
    # 寫自動化測試
    DATASET_DIR = "nycu-bci-homework-3/BCI_hw3_dataset/train"
    DATASET_DIR_TEST = "nycu-bci-homework-3/BCI_hw3_dataset/labeled_test"
    data = EEGDataReader(DATASET_DIR, DATASET_DIR_TEST, scheme="sd", subject_id="01")  # si, sift, sd, ind
    train_data = data.load_train_data()
    test_subject_id = "01"  # 01~04
    test_data = data.load_single_test_data(test_subject_id)

    data_shape = [train_data.__getitem__(0)[0].shape, train_data.__getitem__(0)[1].shape]
    print(train_data.__getitem__(0)[0].shape)
    print(train_data.__getitem__(0)[1].shape)
    print(test_data.__getitem__(0)[0].shape)
    print(test_data.__getitem__(0)[1].shape)
    visualiser = Visualiser()
    evaluator = EEGEvaluator(
        model_class=EEGClassifier,  # model_class=EEGConformer, # SCCNet # EEGNet # ContrastiveModel # EEGClassifier
        data_shape=data_shape,
        test_data=test_data,
        # checkpoint_path="checkpoints/model-epoch98.pth",
        # checkpoint_path="checkpoints/ft-epoch80.pth",
        # checkpoint_path="_checkpoint/sccnet_train_5e_9e-epoch498.pth",
        # checkpoint_path="checkpoints/pretrained_encoder-epoch199.pth",
        # checkpoint_path="other_work/checkpoints/model-epoch96.pth",
        # checkpoint_path="checkpoints/si_ft/ft-epoch198.pth",
        checkpoint_path="checkpoints/si_ft/ssl_encoder-epoch195.pth",
        batch_size=16
    )
    test_subject = ["01", "02", "03", "04"]
    # test_subject = ["05", "06", "07", "08", "09"]
    for sid in test_subject:  # test_subject_id
        evaluator.set_test_data(data.load_single_test_data(sid))
        # evaluator.load_model_and_weights()
        evaluator.load_ssl_model()
        # evaluator.load_eeg_conformer()
        predicts, targets = evaluator.get_evaluate_result()
    # visualiser.plot_confusion_matrix(predicts, targets)
    # visualiser.plot_topomap_weights(evaluator.get_model_weights())
