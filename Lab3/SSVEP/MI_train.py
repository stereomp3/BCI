import os
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader


class EEGTrainerFineTune:  # train without K-fold
    def __init__(self, model_class, ft_data, checkpoint_path=None, savepath="checkpoints", device=None, ft_epochs=100,
                 batch_size=16, lr=1e-4, freeze_layers=False, vl_data=None):
        self.model_class = model_class
        self.ft_data = ft_data  # subject-specific fine-tuning dataset
        self.vl_data = vl_data  # subject-specific fine-tuning dataset
        self.savepath = savepath
        self.checkpoint_path = checkpoint_path
        os.makedirs(savepath, exist_ok=True)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.lr = lr
        self.ft_epochs = ft_epochs
        self.freeze_layers = freeze_layers
        self.model = None
        self.loss_fn = nn.CrossEntropyLoss()

    def load_model_and_weights(self):
        data_shape = [self.ft_data.__getitem__(0)[0].shape, self.ft_data.__getitem__(0)[1].shape]
        # channel, samples, class
        self.model = self.model_class(data_shape[0][1], data_shape[0][2], 2).to( # data_shape[0][2]
            self.device)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])

    def init_model(self):  # if don't use the checkpoint
        data_shape = [self.ft_data.__getitem__(0)[0].shape, self.ft_data.__getitem__(0)[1].shape]
        # channel, samples, class
        # self.model = self.model_class(data_shape[0][1], data_shape[0][2], data_shape[1][0]).to(
        #     self.device)
        self.model = self.model_class(data_shape[0][1], data_shape[0][2], 2).to(
            self.device)

    def eval_epoch(self, loader):
        self.model.eval()
        total_correct, total_samples = 0, 0
        epoch_loss = []

        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(self.device, dtype=torch.float), y_batch.to(self.device,
                                                                                          dtype=torch.long)  # float
                output = self.model(x_batch)
                loss = self.loss_fn(output, y_batch)
                # print(f"output: {output}")
                epoch_loss.append(loss.item())
                total_samples += y_batch.size(0)
                # total_correct += (output.argmax(dim=1) == y_batch.argmax(dim=1)).sum().item()
                total_correct += (output.argmax(dim=1) == y_batch).sum().item()

        return np.mean(epoch_loss), total_correct / total_samples

    def train(self):
        tag = "Train"
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)

        loader = DataLoader(self.ft_data, batch_size=self.batch_size, shuffle=True)
        vl_loader = None
        history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}
        if self.vl_data is None:
            history = {'loss': [], 'acc': []}
        else:
            vl_loader = DataLoader(self.vl_data, batch_size=self.batch_size, shuffle=True)
            print("#################load vl data#########################")

        if self.freeze_layers:
            print("[INFO] Freezing conv layers...")
            for name, param in self.model.named_parameters():
                if "conv1" in name or "conv2" in name:
                    param.requires_grad = False
        for ep in range(self.ft_epochs):
            self.model.train()
            correct, total, losses = 0, 0, []

            for x, y in loader:
                x = x.to(self.device, dtype=torch.float)
                y = y.to(self.device, dtype=torch.long)  # float

                # print(f"x.shape: {x.shape}")  # batch size, 1, 4, 1249 (如果資料 < batch size，會直接餵剩下資料數量)
                optimizer.zero_grad()
                out = self.model(x)
                loss = self.loss_fn(out, y)
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                # correct += (out.argmax(dim=1) == y.argmax(dim=1)).sum().item()
                correct += (out.argmax(dim=1) == y).sum().item()
                total += y.size(0)

            avg_loss = np.mean(losses)
            acc = correct / total

            history["loss"].append(avg_loss)
            history["acc"].append(acc)

            if self.vl_data is None:
                print(f"[{tag}] Epoch {ep}: loss={avg_loss:.4f}, acc={acc:.4f}")
            else:
                val_loss, val_acc = self.eval_epoch(vl_loader)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                print(
                    f"[{tag}] Epoch {ep}: loss={avg_loss:.4f}, acc={acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

            # save
            torch.save({'state_dict': self.model.state_dict()},
                       os.path.join(self.savepath, f"{tag.lower()}-epoch{ep}.pth"))

        return history
