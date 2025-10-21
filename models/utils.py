import torch
from torch.nn import functional as F
import numpy as np
from torch.nn import Parameter
import torch.nn as nn


class LinearModule(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearModule, self).__init__(in_features, out_features, bias)

    def forward(self, input):
        x, cell_line = input[0], input[1]
        return [super().forward(x), cell_line]


class ReLUModule(nn.ReLU):
    def __init__(self):
        super(ReLUModule, self).__init__()

    def forward(self, input):
        x, cell_line = input[0], input[1]
        return [super().forward(x), cell_line]


class DropoutModule(nn.Dropout):
    def __init__(self, p):
        super(DropoutModule, self).__init__(p)

    def forward(self, input):
        x, cell_line = input[0], input[1]
        return [super().forward(x), cell_line]


class FilmModule(torch.nn.Module):
    def __init__(self, num_cell_lines, out_dim):
        super(FilmModule, self).__init__()
        film_init = 1 / 100 * torch.randn(num_cell_lines, 2 * out_dim)
        film_init = film_init + torch.Tensor([([1] * out_dim) + ([0] * out_dim)])

        self.film = Parameter(film_init)

    def forward(self, input):
        x, cell_line = input[0], input[1]
        return [
            self.film[cell_line][:, : x.shape[1]] * x
            + self.film[cell_line][:, x.shape[1]:],
            cell_line,
        ]

class CellLineSpecificLinearModule(nn.Module):
    def __init__(self, in_features, out_features, num_cell_lines):
        super(CellLineSpecificLinearModule, self).__init__()

        self.cell_line_matrices = Parameter(
            1 / 100 * torch.randn((num_cell_lines, out_features, in_features))
        )
        self.cell_line_offsets = Parameter(
            1 / 100 * torch.randn((num_cell_lines, out_features, 1))
        )

    def forward(self, input):
        x, cell_line = input[0], input[1]

        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1, 1)

        x = (
                self.cell_line_matrices[cell_line].matmul(x)
                + self.cell_line_offsets[cell_line]
        )
        return [x[:, :, 0], cell_line]


class ResidualModule(torch.nn.Module):
    def __init__(
            self,
            ConvLayer,
            drug_channels,
            prot_channels,
            pass_d2p_msg,
            pass_p2d_msg,
            pass_p2p_msg,
            drug_self_loop,
            prot_self_loop,
            data,
    ):
        super(ResidualModule, self).__init__()
        self.conv1 = ConvLayer(
            drug_channels,
            prot_channels,
            drug_channels,
            prot_channels,
            pass_d2p_msg,
            pass_p2d_msg,
            pass_p2p_msg,
            drug_self_loop,
            prot_self_loop,
            data,
        )

        self.conv2 = ConvLayer(
            drug_channels,
            prot_channels,
            drug_channels,
            prot_channels,
            pass_d2p_msg,
            pass_p2d_msg,
            pass_p2p_msg,
            drug_self_loop,
            prot_self_loop,
            data,
        )

    def forward(self, h_drug, h_prot, data):
        out_drug, out_prot = self.conv1(h_drug, h_prot, data)
        out_drug = F.relu(out_drug)
        out_prot = F.relu(out_prot)
        out_drug, out_prot = self.conv2(out_drug, out_prot, data)

        return F.relu(h_drug + out_drug), F.relu(h_prot + out_prot)

class Atten1(torch.nn.Module):

    def __init__(self, k, d, dropout):

        super().__init__()
        self.w = torch.nn.Sequential(torch.nn.Linear(d, 4 * k), torch.nn.ReLU())
        self.activation = torch.nn.ReLU()
        self.apply(weight_init)
        self.k = k
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, X):
        tmp = self.w(X)
        # temp:52*256
        # U：52*[0，64）  V：52*[64,128)  Z:52*[128,192)  V:52*[192,256)
        U = tmp[:, : self.k]
        V = tmp[:, self.k: 2 * self.k]
        Z = tmp[:, 2 * self.k: 3 * self.k]
        T = tmp[:, 3 * self.k:]
        V_T = torch.t(V)
        # normalization
        D = joint_normalize2(U, V_T)
        # res = m1(x)*(m2(x)_T*m3(x))
        res = torch.mm(torch.mm(U, V_T), Z)# 52*128——>52*64
        res = res * D
        res = F.softmax(res, dim=1)
        return self.dropout(res)

class Atten(torch.nn.Module):

    def __init__(self, k, d, dropout):

        super().__init__()
        self.w = torch.nn.Sequential(torch.nn.Linear(d, 4 * k), torch.nn.ReLU())
        self.activation = torch.nn.ReLU()
        self.apply(weight_init)
        self.k = k
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, X):
        tmp = self.w(X)
        U = tmp[:, : self.k]
        V = tmp[:, self.k: 2 * self.k]
        Z = tmp[:, 2 * self.k: 3 * self.k]
        T = tmp[:, 3 * self.k:]
        V_T = torch.t(V)

        D = joint_normalize2(U, V_T)
        # res = m1(x)*(m2(x)_T*m3(x))
        res = torch.mm(U, torch.mm(V_T, Z))# 52*128——>52*64
        res = res * D
        res = F.softmax(res, dim=1)
        return self.dropout(res)


class LowRankAttention(torch.nn.Module):


    def __init__(self, k, d, dropout):
        super().__init__()
        self.w = torch.nn.Sequential(torch.nn.Linear(d, 4 * k), torch.nn.ReLU())
        self.activation = torch.nn.ReLU()
        self.apply(weight_init)
        self.k = k
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, X):
        tmp = self.w(X)
        # temp:52*256
        # U：52*[0，64）  V：52*[64,128)  Z:52*[128,192)  V:52*[192,256)
        U = tmp[:, : self.k]
        V = tmp[:, self.k: 2 * self.k]
        Z = tmp[:, 2 * self.k: 3 * self.k]
        T = tmp[:, 3 * self.k:]
        V_T = torch.t(V)
        # normalization
        D = joint_normalize2(U, V_T)
        # res = m1(x)*(m2(x)_T*m3(x))
        res = torch.mm(U, torch.mm(V_T, Z))# 52*128——>52*64
        res = res * D
        #
        res = torch.cat((res * D, T), dim=1)
        return self.dropout(res)

class PLRGA(torch.nn.Module):

    def __init__(self, k, d, dropout):
        super().__init__()
        self.w = torch.nn.Sequential(torch.nn.Linear(d, 4 * k), torch.nn.ReLU())
        self.activation = torch.nn.ReLU()
        self.apply(weight_init)
        self.k = k
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, X):
        tmp = self.w(X)
        # temp:52*256
        # U：52*[0，64）  V：52*[64,128)  Z:52*[128,192)  V:52*[192,256)
        U = tmp[:, : self.k]
        V = tmp[:, self.k: 2 * self.k]
        Z = tmp[:, 2 * self.k: 3 * self.k]
        T = tmp[:, 3 * self.k:]
        V_T = torch.t(V)
        T_T = torch.t(T)
        # normalization
        D = joint_normalize2(U, V_T)
        # res = m1(x)*(m2(x)_T*m3(x))
        res = torch.mm(U, torch.mm(V_T, Z))# 52*128——>52*64
        res = torch.mm(res, T_T)# 52*52
        res = res * D
        res = torch.cat((res * D, T), dim=1)
        return self.dropout(res)

def joint_normalize2(U, V_T):

    if torch.cuda.is_available():
        tmp_ones = torch.ones((V_T.shape[1], 1)).to("cuda")
    else:
        tmp_ones = torch.ones((V_T.shape[1], 1))
    norm_factor = torch.mm(U, torch.mm(V_T, tmp_ones))
    norm_factor = (torch.sum(norm_factor) / U.shape[0]) + 1e-6
    return 1 / norm_factor


def weight_init(layer):
    if type(layer) == torch.nn.Linear:
        torch.nn.init.xavier_normal_(layer.weight.data)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias.data, 0)
    return

def get_batch(
        data,
        drug_drug_batch,
        h_drug,
        drug2target_dict,
        with_fp=False,
        with_expr=False,
        with_prot=False,
):

    batch_size = drug_drug_batch[0].shape[0]
    drug_1s = drug_drug_batch[0][:, 0]
    drug_2s = drug_drug_batch[0][:, 1]
    cell_lines = drug_drug_batch[1]

    if h_drug is not None:#
        batch_data_1 = h_drug[drug_1s]
        batch_data_2 = h_drug[drug_2s]
    else:
        batch_data_1 = data.x_drugs[drug_1s]
        batch_data_2 = data.x_drugs[drug_2s]

    if (
            with_fp and h_drug is not None#
    ):
        x_drug_1s = data.x_drugs[drug_1s, : data.fp_bits]# x_drug_1s：[1024,256]，x_drugs:[52,2212]
        x_drug_2s = data.x_drugs[drug_2s, : data.fp_bits]# fp_bits=256, batch_data_1:h_drug:[1024,288]
        batch_data_1 = torch.cat((batch_data_1, x_drug_1s), dim=1) # batch_data_1:(1024,544)
        batch_data_2 = torch.cat((batch_data_2, x_drug_2s), dim=1)

    if with_expr and h_drug is not None:
        expr_drug_1s = data.x_drugs[drug_1s, data.fp_bits:]
        expr_drug_2s = data.x_drugs[drug_2s, data.fp_bits:]
        batch_data_1 = torch.cat((batch_data_1, expr_drug_1s), dim=1)
        batch_data_2 = torch.cat((batch_data_2, expr_drug_2s), dim=1)

    if with_prot:
        prot_1 = torch.zeros((batch_size, data.x_prots.shape[0]))
        prot_2 = torch.zeros((batch_size, data.x_prots.shape[0]))

        if torch.cuda.is_available():
            prot_1 = prot_1.to("cuda")
            prot_2 = prot_2.to("cuda")

        for i in range(batch_size):
            prot_1[i, drug2target_dict[int(drug_1s[i])]] = 1
            prot_2[i, drug2target_dict[int(drug_2s[i])]] = 1

        batch_data_1 = torch.cat((batch_data_1, prot_1), dim=1)
        batch_data_2 = torch.cat((batch_data_2, prot_2), dim=1)

    return batch_data_1, batch_data_2, cell_lines


def get_layer_dims(
        predictor_layers,
        fp_dim,
        expr_dim,
        prot_numb,
        with_fp=False,
        with_expr=False,
        with_prot=False,
):

    if with_expr:
        predictor_layers[0] += expr_dim
    if with_fp:
        predictor_layers[0] += fp_dim
    if with_prot:
        predictor_layers[0] += prot_numb
    return predictor_layers

