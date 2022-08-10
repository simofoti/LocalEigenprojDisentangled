import os
import shutil
import tqdm
import torch
import torch.nn as nn

from torch_geometric.data import InMemoryDataset
from sklearn.model_selection import train_test_split

from data_generation_and_loading import MeshLoader
from model import SpiralEnblock


class VariationPredictability:
    def __init__(self, generated_data_list, train_split_ratio, main_model,
                 lr, epochs, batch_s, workers=0, device='cpu'):
        self._device = device
        self._epochs = epochs
        tr_set, ts_set = train_test_split(generated_data_list,
                                          train_size=train_split_ratio)
        self._train_loader = MeshLoader(VpListDataset(tr_set),
                                        batch_s, shuffle=True,
                                        drop_last=True, feature_swapper=None,
                                        num_workers=workers)
        self._test_loader = MeshLoader(VpListDataset(ts_set),
                                       batch_s, shuffle=True,
                                       drop_last=True, feature_swapper=None,
                                       num_workers=workers)
        self._model = VpModel(in_channels=main_model.in_channels,
                              out_channels=main_model.out_channels,
                              latent_size=main_model.latent_size,
                              spiral_indices=main_model.spiral_indices,
                              down_transform=main_model.down_transform,
                              up_transform=main_model.up_transform).to(device)
        self._opt = torch.optim.Adam(self._model.parameters(), lr=float(lr))

    def __call__(self):
        print("train and test VariationPredictability")
        best_acc = 0
        for e in tqdm.tqdm(range(self._epochs)):
            self.train()
            if (e + 1) % (self._epochs // 10):
                test_acc = self.test_acc()
                if test_acc > best_acc:
                    best_acc = test_acc
        shutil.rmtree(os.path.join('.', 'delete_me'))
        return best_acc

    def train(self):
        self._model.train()
        loss_func = nn.CrossEntropyLoss().to(self._device)
        for data in self._train_loader:
            data = data.to(self._device)
            out = self._model(data.x)
            loss = loss_func(out, data.y.squeeze())

            self._opt.zero_grad()
            loss.backward()
            self._opt.step()

    @torch.no_grad()
    def test_acc(self):
        self._model.eval()
        acc_sum, i = 0, 0
        for i, data in enumerate(self._test_loader):
            data = data.to(self._device)
            out = self._model(data.x)
            acc = self._compute_accuracies(out, data.y.squeeze())[0]
            acc_sum += acc.item()
        return acc_sum / i

    @staticmethod
    def _compute_accuracies(output, target, topk=(1, )):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class VpListDataset(InMemoryDataset):
    def __init__(self, data_list):
        super(VpListDataset, self).__init__(os.path.join('.', 'delete_me'))
        self.data, self.slices = self.collate(data_list)

    @property
    def processed_file_names(self):
        return []

    @property
    def raw_file_names(self):
        return []

    def process(self):
        pass

    def download(self):
        pass


class VpModel(nn.Module):
    def __init__(self, in_channels, out_channels, latent_size,
                 spiral_indices, down_transform, up_transform):
        super(VpModel, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_size = latent_size
        self.spiral_indices = spiral_indices
        self.down_transform = down_transform
        self.up_transform = up_transform
        self.num_vert = self.down_transform[-1].size(0)

        # encoder
        self.en_layers = nn.ModuleList()
        for idx in range(len(out_channels)):
            if idx == 0:
                self.en_layers.append(
                    SpiralEnblock(in_channels, out_channels[idx],
                                  self.spiral_indices[idx]))
            else:
                self.en_layers.append(
                    SpiralEnblock(out_channels[idx - 1], out_channels[idx],
                                  self.spiral_indices[idx]))
        self.en_layers.append(
            nn.Linear(self.num_vert * out_channels[-1], latent_size))
        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)

    def forward(self, x):
        for i, layer in enumerate(self.en_layers):
            if i < len(self.en_layers) - 1:
                x = layer(x, self.down_transform[i])

        x = x.view(-1, self.en_layers[-1].weight.size(1))
        return self.en_layers[-1](x)



