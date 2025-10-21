import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Parameter
import pandas as pd
from models.utils import (
    get_batch,
    get_layer_dims,
    ReLUModule,
    DropoutModule,
    FilmModule,
    LinearModule,
    CellLineSpecificLinearModule,
)

class AbstractPredictor(torch.nn.Module):
    def __init__(self, data, config, predictor_layers):

        self.num_cell_lines = len(torch.unique(data.ddi_edge_classes))
        self.device = config["device"]
        self.with_fp = config["with_fp"]
        self.with_expr = config["with_expr"]
        self.with_prot = config["with_prot"]
        self.layer_dims = predictor_layers

        (
            self.layer_dims,
            self.output_dim_comb,
            self.output_dim_mono,
        ) = self.get_layer_dims(
            self.layer_dims,
            fp_dim=int(data.fp_bits),
            expr_dim=data.x_drugs.shape[1] - int(data.fp_bits),
            prot_numb=data.x_prots.shape[0],
        )

        self.drug2target_dict = {i: [] for i in range(data.x_drugs.shape[0])}
        for edge in data.dpi_edge_idx.T:
            self.drug2target_dict[int(edge[0])].append(
                int(edge[1]) - data.x_drugs.shape[0]
            )

        self.merge_n_layers_before_the_end = config["merge_n_layers_before_the_end"]
        self.merge_dim = self.layer_dims[-self.merge_n_layers_before_the_end - 1]
        assert 0 < self.merge_n_layers_before_the_end < len(predictor_layers)

        super(AbstractPredictor, self).__init__()

    def forward(self, data, drug_drug_batch, h_drug, n_forward_passes=1):

        h_drug_1s, h_drug_2s, cell_lines = self.get_batch(data, drug_drug_batch, h_drug)

        ground_truth_scores = drug_drug_batch[2][:, None, None]
        ground_truth_scores = ground_truth_scores[:, 0, 0].detach().cpu().numpy()
        ground_truth_scores = ground_truth_scores.tolist()
        drug1 = drug_drug_batch[0][:, 0].detach().cpu().numpy().tolist()
        drug2 = drug_drug_batch[0][:, 1].detach().cpu().numpy().tolist()

        comb = torch.empty([drug_drug_batch[0].shape[0], self.output_dim_comb, 0]).to(  # empty([1024,1,0])
            self.device
        )
        h_1 = torch.empty([drug_drug_batch[0].shape[0], self.output_dim_mono, 0]).to(  # empty([1024,1,0])
            self.device
        )
        h_2 = torch.empty([drug_drug_batch[0].shape[0], self.output_dim_mono, 0]).to(
            self.device
        )

        for i in range(n_forward_passes):  # comb_i:[1024, 1]
            comb_i, h_1_i, h_2_i = self.single_forward_pass(
                h_drug_1s, h_drug_2s, cell_lines
            )
            comb = torch.cat((comb, comb_i[:, :, None]), dim=2)
            if h_1_i is not None:
                h_1 = torch.cat((h_1, h_1_i[:, :, None]), dim=2)
                h_2 = torch.cat((h_2, h_2_i[:, :, None]), dim=2)
        return comb, h_1, h_2

    def get_layer_dims(self, predictor_layers, fp_dim, expr_dim, prot_numb):

        return (
            get_layer_dims(
                predictor_layers,
                fp_dim,
                expr_dim,
                prot_numb,
                with_fp=self.with_fp,
                with_expr=self.with_expr,
                with_prot=self.with_prot,
            ),
            1,
            1,
        )


    def get_batch(self, data, drug_drug_batch, h_drug):
        return get_batch(
            data,
            drug_drug_batch,
            h_drug,
            self.drug2target_dict,
            with_fp=self.with_fp,
            with_expr=self.with_expr,
            with_prot=self.with_prot,
        )

    def single_forward_pass(self, h_drug_1s, h_drug_2s, cell_lines):
        raise NotImplementedError

class BilinearMLPPredictor(torch.nn.Module):
    def __init__(self, data, config, predictor_layers):

        super(BilinearMLPPredictor, self).__init__()

        self.num_cell_lines = len(data.cell_line_to_idx_dict.keys())
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_type)
        self.layer_dims = predictor_layers

        self.merge_n_layers_before_the_end = config["merge_n_layers_before_the_end"]
        self.merge_dim = self.layer_dims[-self.merge_n_layers_before_the_end - 1]

        assert 0 < self.merge_n_layers_before_the_end < len(predictor_layers)

        layers_before_merge = []
        layers_after_merge = []

        for i in range(len(self.layer_dims) - 1 - self.merge_n_layers_before_the_end):
            layers_before_merge = self.add_layer(
                layers_before_merge,
                i,
                self.layer_dims[i],
                self.layer_dims[i + 1]
            )

        for i in range(
                len(self.layer_dims) - 1 - self.merge_n_layers_before_the_end,
                len(self.layer_dims) - 1,
        ):
            layers_after_merge = self.add_layer(
                layers_after_merge,
                i,
                self.layer_dims[i],
                self.layer_dims[i + 1]
            )

        self.before_merge_mlp = nn.Sequential(*layers_before_merge)
        self.after_merge_mlp = nn.Sequential(*layers_after_merge)

        self.bilinear_weights = Parameter(
            1 / 100 * torch.randn((self.merge_dim, self.merge_dim, self.merge_dim))
            + torch.cat([torch.eye(self.merge_dim)[None, :, :]] * self.merge_dim, dim=0)
        )
        self.bilinear_offsets = Parameter(1 / 100 * torch.randn((self.merge_dim)))

        self.allow_neg_eigval = True
        if self.allow_neg_eigval:
            self.bilinear_diag = Parameter(1 / 100 * torch.randn((self.merge_dim, self.merge_dim)) + 1)

    def forward(self, data, drug_drug_batch, n_forward_passes):
        h_drug_1, h_drug_2, cell_lines = self.get_batch(data, drug_drug_batch)

        h_1 = self.before_merge_mlp([h_drug_1, cell_lines])[0]
        h_2 = self.before_merge_mlp([h_drug_2, cell_lines])[0]

        h_1 = self.bilinear_weights.matmul(h_1.T).T
        h_2 = self.bilinear_weights.matmul(h_2.T).T

        if self.allow_neg_eigval:
            h_2 *= self.bilinear_diag

        h_1 = h_1.permute(0, 2, 1)
        h_1_scal_h_2 = (h_1 * h_2).sum(1)
        h_1_scal_h_2 += self.bilinear_offsets

        comb = self.after_merge_mlp([h_1_scal_h_2, cell_lines])[0]

        return comb, h_1, h_2

    def get_batch(self, data, drug_drug_batch):

        drug_1s = drug_drug_batch[0][:, 0]
        drug_2s = drug_drug_batch[0][:, 1]
        cell_lines = drug_drug_batch[1]

        h_drug_1 = data.x_drugs[drug_1s]
        h_drug_2 = data.x_drugs[drug_2s]

        return h_drug_1, h_drug_2, cell_lines

    def add_layer(self, layers, i, dim_i, dim_i_plus_1):
        layers.extend(self.linear_layer(i, dim_i, dim_i_plus_1))
        if i != len(self.layer_dims) - 2:
            layers.append(ReLUModule())

        return layers

    def linear_layer(self, i, dim_i, dim_i_plus_1):
        return [LinearModule(dim_i, dim_i_plus_1)]


class MLPAbstractPredictor(AbstractPredictor):

    def __init__(self, data, config, predictor_layers):
        super(MLPAbstractPredictor, self).__init__(
            data, config, predictor_layers
        )
        layers_before_merge = []
        layers_after_merge = []

        for i in range(len(self.layer_dims) - 1 - self.merge_n_layers_before_the_end):  # 0，1
            layers_before_merge = self.add_layer(
                layers_before_merge,
                i,
                self.layer_dims[i],
                self.layer_dims[i + 1],
                config,
            )

        self.layer_dims[
            -1
        ] = (
            self.output_dim_comb

        )
        for i in range(
                len(self.layer_dims) - 1 - self.merge_n_layers_before_the_end,
                len(self.layer_dims) - 1,
        ):
            layers_after_merge = self.add_layer(
                layers_after_merge,
                i,
                self.layer_dims[i],
                self.layer_dims[i + 1],
                config,
            )

        self.before_merge_mlp = nn.Sequential(*layers_before_merge)
        self.after_merge_mlp = nn.Sequential(*layers_after_merge)

        self.single_drug_mlp = None

    def add_layer(self, layers, i, dim_i, dim_i_plus_1, config):

        layers.extend(self.linear_layer(i, dim_i, dim_i_plus_1))
        if i != len(self.layer_dims) - 2:
            layers.append(ReLUModule())

            should_do_dropout = (
                    "num_dropout_lyrs" not in config
                    or len(self.layer_dims) - 2 - config["num_dropout_lyrs"] <= i
            )

            if should_do_dropout:
                layers.append(DropoutModule(p=config["dropout_proba"]))

        return layers

    def linear_layer(self, i, dim_i, dim_i_plus_1):

        raise NotImplementedError

    def single_forward_pass(self, h_drug_1, h_drug_2, cell_lines):
        comb = self.after_merge_mlp(
            [
                self.before_merge_mlp([h_drug_1, cell_lines])[0]
                + self.before_merge_mlp([h_drug_2, cell_lines])[0],
                cell_lines,
            ]
        )[0]
        return (
            comb,
            self.transform_single_drug(h_drug_1, cell_lines),
            self.transform_single_drug(h_drug_2, cell_lines),
        )

    def transform_single_drug(self, h, cell_lines):
        if self.single_drug_mlp is None:
            return None
        else:
            return self.single_drug_mlp([h, cell_lines])[0]


class BilinearMLPAbstractPredictor(MLPAbstractPredictor):

    def __init__(self, data, config, predictor_layers):
        super(BilinearMLPAbstractPredictor, self).__init__(
            data, config, predictor_layers
        )

        self.bilinear_weights = Parameter(
            1 / 100 * torch.randn((self.merge_dim, self.merge_dim, self.merge_dim))
            + torch.cat([torch.eye(self.merge_dim)[None, :, :]] * self.merge_dim, dim=0)
        )
        self.bilinear_offsets = Parameter(1 / 100 * torch.randn((self.merge_dim)))

    def single_forward_pass(self, h_drug_1, h_drug_2, cell_lines):
        h_1 = self.before_merge_mlp([h_drug_1, cell_lines])[0]
        h_2 = self.before_merge_mlp([h_drug_2, cell_lines])[0]

        h_1 = self.bilinear_weights.matmul(h_1.T).T
        h_2 = self.bilinear_weights.matmul(h_2.T).T

        h_1 = h_1.permute(0, 2, 1)
        h_1_scal_h_2 = (h_1 * h_2).sum(1)
        h_1_scal_h_2 += self.bilinear_offsets
        a = self.after_merge_mlp
        comb = self.after_merge_mlp([h_1_scal_h_2, cell_lines])[0]

        return (
            comb,
            self.transform_single_drug(h_drug_1, cell_lines),
            self.transform_single_drug(h_drug_2, cell_lines),
        )


class BasicMLPPredictor(MLPAbstractPredictor):
    def __init__(self, data, config, predictor_layers):
        super(BasicMLPPredictor, self).__init__(
            data, config, predictor_layers
        )

    def linear_layer(self, i, dim_i, dim_i_plus_1):
        return [LinearModule(dim_i, dim_i_plus_1)]


class FilmMLPPredictor(MLPAbstractPredictor):
    def __init__(self, data, config, predictor_layers):
        super(FilmMLPPredictor, self).__init__(
            data, config, predictor_layers
        )

    def linear_layer(self, i, dim_i, dim_i_plus_1):
        if i != len(self.layer_dims) - 2:
            return [
                LinearModule(dim_i, dim_i_plus_1),
                FilmModule(self.num_cell_lines, dim_i_plus_1),
            ]
        else:
            return [LinearModule(self.layer_dims[i], self.layer_dims[i + 1])]


class SharedLayersMLPPredictor(MLPAbstractPredictor):

    def __init__(self, data, config, predictor_layers):
        super(SharedLayersMLPPredictor, self).__init__(
            data, config, predictor_layers
        )

    def linear_layer(self, i, dim_i, dim_i_plus_1):
        if i < len(self.layer_dims) - 3:
            return [LinearModule(dim_i, dim_i_plus_1)]
        else:
            return [
                CellLineSpecificLinearModule(dim_i, dim_i_plus_1, self.num_cell_lines)
            ]

class BilinearBasicMLPPredictor(BilinearMLPAbstractPredictor):

    def __init__(self, data, config, predictor_layers):
        super(BilinearBasicMLPPredictor, self).__init__(
            data, config, predictor_layers
        )

    def linear_layer(self, i, dim_i, dim_i_plus_1):
        return [LinearModule(dim_i, dim_i_plus_1)]




class BilinearFilmMLPPredictor(BilinearMLPAbstractPredictor):

    def __init__(self, data, config, predictor_layers):
        super(BilinearFilmMLPPredictor, self).__init__(
            data, config, predictor_layers
        )

    def linear_layer(self, i, dim_i, dim_i_plus_1):
        if i != len(self.layer_dims) - 2:
            return [
                LinearModule(dim_i, dim_i_plus_1),
                FilmModule(self.num_cell_lines, dim_i_plus_1),
            ]
        else:
            return [LinearModule(dim_i, dim_i_plus_1)]


class BilinearSharedLayersMLPPredictor(BilinearMLPAbstractPredictor):
    def __init__(self, data, config, predictor_layers):
        super(BilinearSharedLayersMLPPredictor, self).__init__(
            data, config, predictor_layers
        )
    def linear_layer(self, i, dim_i, dim_i_plus_1):
        if i < len(self.layer_dims) - 3:

            return [LinearModule(dim_i, dim_i_plus_1)]
        else:
            return [
                CellLineSpecificLinearModule(dim_i, dim_i_plus_1, self.num_cell_lines)
            ]


class AbstractPredictor1(torch.nn.Module):
    def __init__(self, data, config, predictor_layers):
        super(AbstractPredictor1, self).__init__()

        self.num_cell_lines = len(torch.unique(data.ddi_edge_classes))
        self.device = config["device"]
        self.with_fp = config["with_fp"]
        self.with_expr = config["with_expr"]
        self.with_prot = config["with_prot"]
        self.layer_dims = predictor_layers

        (
            self.layer_dims,  #
            self.output_dim_comb,
            self.output_dim_mono,
        ) = self.get_layer_dims(
            self.layer_dims,
            fp_dim=int(data.fp_bits),
            expr_dim=data.x_drugs.shape[1] - int(data.fp_bits),
            prot_numb=data.x_prots.shape[0],
        )

        self.inter_net = nn.Sequential(
            nn.Linear(self.layer_dims[0], self.layer_dims[1]),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.layer_dims[1])
        )

        self.network = nn.Sequential(
            nn.Linear(self.layer_dims[1], self.layer_dims[1] // 2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.layer_dims[1] // 2),
            nn.Linear(self.layer_dims[1] // 2, 1)
        )

        self.drug2target_dict = {i: [] for i in range(data.x_drugs.shape[0])}
        for edge in data.dpi_edge_idx.T:
            self.drug2target_dict[int(edge[0])].append(
                int(edge[1]) - data.x_drugs.shape[0]
            )

        self.merge_n_layers_before_the_end = config["merge_n_layers_before_the_end"]
        self.merge_dim = self.layer_dims[-self.merge_n_layers_before_the_end - 1]
        assert 0 < self.merge_n_layers_before_the_end < len(predictor_layers)

        super(AbstractPredictor1, self).__init__()

    def forward(self, data, drug_drug_batch, h_drug, n_forward_passes=1):
        h_drug_1s, h_drug_2s, cell_lines = self.get_batch(data, drug_drug_batch, h_drug)
        xc = torch.cat((h_drug_1s, h_drug_2s), 1)
        xc = F.normalize(xc, 2, 1)
        xc = nn.Linear(xc.detach().numpy().shape[1], 2048)(xc)
        xc = nn.ReLU()(xc)
        xc = nn.Dropout(0.2)(xc)
        xc = nn.Linear(2048, 512)(xc)
        xc = nn.ReLU()(xc)
        xc = nn.Dropout(0.2)(xc)
        xc = nn.Linear(512, 128)(xc)
        xc = nn.ReLU()(xc)
        xc = nn.Dropout(0.2)(xc)
        out = nn.Linear(128, 2)(xc)

        return out

    def get_layer_dims(self, predictor_layers, fp_dim, expr_dim, prot_numb):
        return (
            get_layer_dims(
                predictor_layers,
                fp_dim,
                expr_dim,
                prot_numb,
                with_fp=self.with_fp,
                with_expr=self.with_expr,
                with_prot=self.with_prot,
            ),
            1,
            1,
        )


    def get_batch(self, data, drug_drug_batch, h_drug):
        return get_batch(
            data,
            drug_drug_batch,
            h_drug,
            self.drug2target_dict,
            with_fp=self.with_fp,
            with_expr=self.with_expr,
            with_prot=self.with_prot,
        )

    def single_forward_pass(self, h_drug_1s, h_drug_2s, cell_lines):
        raise NotImplementedError


class StackProjDNN(nn.Module):
    def __init__(self, drug_size: int, cell_size: int, stack_size: int, hidden_size: int):
        super(StackProjDNN, self).__init__()

        self.projectors = nn.Parameter(torch.zeros(size=(stack_size, cell_size, cell_size)))
        nn.init.xavier_uniform_(self.projectors.data, gain=1.414)

        self.network = nn.Sequential(
            nn.Linear(2 * drug_size + cell_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, drug1_feat: torch.Tensor, drug2_feat: torch.Tensor, cell_feat: torch.Tensor):
        cell_feat = cell_feat.unsqueeze(-1)
        cell_feats = torch.matmul(self.projectors, cell_feat).squeeze(-1)
        cell_feat = torch.sum(cell_feats, 1)
        feat = torch.cat([drug1_feat, drug2_feat, cell_feat], 1)
        out = self.network(feat)
        return out


class InteractionNet(nn.Module):
    def __init__(self, drug_size: int, cell_size: int, hidden_size: int):
        super(InteractionNet, self).__init__()

        self.inter_net = nn.Sequential(
            nn.Linear(drug_size + cell_size, hidden_size),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_size)
        )

        self.network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, h_drug_1: torch.Tensor, h_drug_2: torch.Tensor, cell_lines: torch.Tensor):
        dc1 = torch.cat([h_drug_1, cell_lines], 1)
        dc2 = torch.cat([h_drug_2, cell_lines], 1)
        inter1 = self.inter_net(dc1)
        inter2 = self.inter_net(dc2)
        inter3 = inter1 + inter2
        out = self.network(inter3)
        return out


# ----------------------------------------------------------------------------
class AbstractPredictor2(torch.nn.Module):
    def __init__(self, data, config, predictor_layers):
        #
        self.num_cell_lines = len(torch.unique(data.ddi_edge_classes))
        self.device = config["device"]
        self.with_fp = config["with_fp"]
        self.with_expr = config["with_expr"]
        self.with_prot = config["with_prot"]

        self.layer_dims = predictor_layers
        (
            self.layer_dims,  # [2500,1024,64,32,2]
            self.output_dim_comb,  # 1
            self.output_dim_mono,  # 1
        ) = self.get_layer_dims(
            self.layer_dims,
            fp_dim=int(data.fp_bits),
            expr_dim=data.x_drugs.shape[1] - int(data.fp_bits),
            prot_numb=data.x_prots.shape[0],
        )

        self.drug2target_dict = {i: [] for i in range(data.x_drugs.shape[0])}
        for edge in data.dpi_edge_idx.T:
            self.drug2target_dict[int(edge[0])].append(
                int(edge[1]) - data.x_drugs.shape[0]
            )

        self.merge_n_layers_before_the_end = config["merge_n_layers_before_the_end"]  # 在结束前合并n层，1
        self.merge_dim = self.layer_dims[-self.merge_n_layers_before_the_end - 1]  # 32
        assert 0 < self.merge_n_layers_before_the_end < len(predictor_layers)

        super(AbstractPredictor2, self).__init__()

    def forward(self, data, drug_drug_batch, h_drug, n_forward_passes=1):
        h_drug_1s, h_drug_2s, cell_lines = self.get_batch(data, drug_drug_batch, h_drug)
        comb = torch.empty([drug_drug_batch[0].shape[0], self.output_dim_comb * 2, 0]).to(  # empty([1024,1,0])
            self.device
        )

        for i in range(n_forward_passes):
            comb = self.single_forward_pass(
                h_drug_1s, h_drug_2s, cell_lines
            )
        return comb

    def get_layer_dims(self, predictor_layers, fp_dim, expr_dim, prot_numb):
        return (
            get_layer_dims(
                predictor_layers,
                fp_dim,
                expr_dim,
                prot_numb,
                with_fp=self.with_fp,
                with_expr=self.with_expr,
                with_prot=self.with_prot,
            ),
            1,
            1,
        )


    def get_batch(self, data, drug_drug_batch, h_drug):
        return get_batch(
            data,
            drug_drug_batch,
            h_drug,
            self.drug2target_dict,
            with_fp=self.with_fp,
            with_expr=self.with_expr,
            with_prot=self.with_prot,
        )

    def single_forward_pass(self, h_drug_1s, h_drug_2s, cell_lines):
        raise NotImplementedError


class MLPAbstractPredictor2(AbstractPredictor2):
    def __init__(self, data, config, predictor_layers):
        super(MLPAbstractPredictor2, self).__init__(
            data, config, predictor_layers
        )
        # predict layers:[2500,1024,64,32,2]
        layers_before_merge = []
        layers_after_merge = []

        for i in range(len(self.layer_dims) - 1 - self.merge_n_layers_before_the_end):  # 0，1
            layers_before_merge = self.add_layer(
                layers_before_merge,
                i,
                self.layer_dims[i],
                self.layer_dims[i + 1],
                config,
            )

        self.layer_dims[
            -1
        ] = (
            2
        )
        for i in range(
                len(self.layer_dims) - 1 - self.merge_n_layers_before_the_end,
                len(self.layer_dims) - 1,
        ):
            layers_after_merge = self.add_layer(
                layers_after_merge,
                i,
                self.layer_dims[i],
                self.layer_dims[i + 1],
                config,
            )

        self.before_merge_mlp = nn.Sequential(*layers_before_merge)
        self.after_merge_mlp = nn.Sequential(*layers_after_merge)
        self.before_merge_mlp1 = nn.Sequential(
            LinearModule(in_features=2500, out_features=1024, bias=True),
            ReLUModule(),
            nn.BatchNorm1d(1024),
            LinearModule(in_features=1024, out_features=64, bias=True),
            ReLUModule(),
            nn.BatchNorm1d(64),
            LinearModule(in_features=64, out_features=32, bias=True),
            ReLUModule(),
            nn.BatchNorm1d(32),
            DropoutModule(p=0.0))
        self.after_merge_mlp1 = nn.Sequential(LinearModule(in_features=32, out_features=2, bias=True), )
        self.single_drug_mlp = None

    def add_layer(self, layers, i, dim_i, dim_i_plus_1, config):
        layers.extend(self.linear_layer(i, dim_i, dim_i_plus_1))
        if i != len(self.layer_dims) - 2:
            layers.append(ReLUModule())
            should_do_dropout = (
                    "num_dropout_lyrs" not in config
                    or len(self.layer_dims) - 2 - config["num_dropout_lyrs"] <= i
            )
            if should_do_dropout:
                layers.append(DropoutModule(p=config["dropout_proba"]))

        return layers

    def linear_layer(self, i, dim_i, dim_i_plus_1):
        return [LinearModule(dim_i, dim_i_plus_1)]

    def single_forward_pass(self, h_drug_1, h_drug_2, cell_lines):
        comb = self.after_merge_mlp(
            [
                self.before_merge_mlp([h_drug_1, cell_lines])[0]
                + self.before_merge_mlp([h_drug_2, cell_lines])[0],
                cell_lines,
            ]
        )[0]
        return comb

    def transform_single_drug(self, h, cell_lines):

        if self.single_drug_mlp is None:
            return None
        else:
            return self.single_drug_mlp([h, cell_lines])[0]


class BilinearMLPAbstractPredictor2(MLPAbstractPredictor2):

    def __init__(self, data, config, predictor_layers):
        super(BilinearMLPAbstractPredictor2, self).__init__(
            data, config, predictor_layers
        )
        self.bilinear_weights = Parameter(
            1 / 100 * torch.randn((self.merge_dim, self.merge_dim, self.merge_dim))
            + torch.cat([torch.eye(self.merge_dim)[None, :, :]] * self.merge_dim, dim=0)
        )
        self.bilinear_offsets = Parameter(1 / 100 * torch.randn((self.merge_dim)))

    def single_forward_pass(self, h_drug_1, h_drug_2, cell_lines):
        h_1 = self.before_merge_mlp([h_drug_1, cell_lines])[0]
        h_2 = self.before_merge_mlp([h_drug_2, cell_lines])[0]
        h_1 = self.bilinear_weights.matmul(h_1.T).T
        h_2 = self.bilinear_weights.matmul(h_2.T).T

        h_1 = h_1.permute(0, 2, 1)
        h_1_scal_h_2 = (h_1 * h_2).sum(1)
        h_1_scal_h_2 += self.bilinear_offsets
        comb = self.after_merge_mlp1([h_1_scal_h_2, cell_lines])[0]

        return comb


class BilinearSharedLayersMLPPredictor2(BilinearMLPAbstractPredictor2):

    def __init__(self, data, config, predictor_layers):
        super(BilinearSharedLayersMLPPredictor2, self).__init__(
            data, config, predictor_layers
        )

    def linear_layer(self, i, dim_i, dim_i_plus_1):
        if i < len(self.layer_dims) - 3:
            return [LinearModule(dim_i, dim_i_plus_1)]
        else:
            return [
                CellLineSpecificLinearModule(dim_i, dim_i_plus_1, self.num_cell_lines)
            ]


# ----------------------------------------------------------------------------
class AbstractPredictor3(torch.nn.Module):
    def __init__(self, data, config, predictor_layers):
        #
        self.num_cell_lines = len(torch.unique(data.ddi_edge_classes))
        self.device = config["device"]
        self.with_fp = config["with_fp"]
        self.with_expr = config["with_expr"]
        self.with_prot = config["with_prot"]

        self.layer_dims = predictor_layers

        (
            self.layer_dims,  # [2500,1024,64,32,2]
            self.output_dim_comb,  # 1
            self.output_dim_mono,  # 1
        ) = self.get_layer_dims(
            self.layer_dims,
            fp_dim=int(data.fp_bits),
            expr_dim=data.x_drugs.shape[1] - int(data.fp_bits),
            prot_numb=data.x_prots.shape[0],
        )

        self.drug2target_dict = {i: [] for i in range(data.x_drugs.shape[0])}
        for edge in data.dpi_edge_idx.T:
            self.drug2target_dict[int(edge[0])].append(
                int(edge[1]) - data.x_drugs.shape[0]
            )

        self.merge_n_layers_before_the_end = config["merge_n_layers_before_the_end"]
        self.merge_dim = self.layer_dims[-self.merge_n_layers_before_the_end - 1]  # 32
        assert 0 < self.merge_n_layers_before_the_end < len(predictor_layers)

        super(AbstractPredictor3, self).__init__()

    def forward(self, data, drug_drug_batch, h_drug, n_forward_passes=1):
        h_drug_1s, h_drug_2s, cell_lines = self.get_batch(data, drug_drug_batch, h_drug)
        comb = torch.empty([drug_drug_batch[0].shape[0], self.output_dim_comb * 2, 0]).to(  # empty([1024,1,0])
            self.device
        )
        for i in range(n_forward_passes):  # comb_i:[1024, 1]
            comb = self.single_forward_pass(
                h_drug_1s, h_drug_2s, cell_lines
            )

        return comb

    def get_layer_dims(self, predictor_layers, fp_dim, expr_dim, prot_numb):

        return (
            get_layer_dims(
                predictor_layers,
                fp_dim,
                expr_dim,
                prot_numb,
                with_fp=self.with_fp,
                with_expr=self.with_expr,
                with_prot=self.with_prot,
            ),
            1,
            1,
        )

    def get_batch(self, data, drug_drug_batch, h_drug):
        return get_batch(
            data,
            drug_drug_batch,
            h_drug,
            self.drug2target_dict,
            with_fp=self.with_fp,
            with_expr=self.with_expr,
            with_prot=self.with_prot,
        )

    def single_forward_pass(self, h_drug_1s, h_drug_2s, cell_lines):
        raise NotImplementedError


class MLPAbstractPredictor3(AbstractPredictor3):
    def __init__(self, data, config, predictor_layers):
        super(MLPAbstractPredictor3, self).__init__(
            data, config, predictor_layers
        )
        layers_before_merge = []
        layers_after_merge = []

        for i in range(len(self.layer_dims) - 1 - self.merge_n_layers_before_the_end):  # 0，1
            layers_before_merge = self.add_layer(
                layers_before_merge,
                i,
                self.layer_dims[i],
                self.layer_dims[i + 1],
                config,
            )

        self.layer_dims[
            -1
        ] = (
            2
        )
        for i in range(
                len(self.layer_dims) - 1 - self.merge_n_layers_before_the_end,
                len(self.layer_dims) - 1,
        ):
            layers_after_merge = self.add_layer(
                layers_after_merge,
                i,
                self.layer_dims[i],
                self.layer_dims[i + 1],
                config,
            )

        self.before_merge_mlp = nn.Sequential(*layers_before_merge)
        self.after_merge_mlp = nn.Sequential(*layers_after_merge)

        self.single_drug_mlp = None

    def add_layer(self, layers, i, dim_i, dim_i_plus_1, config):
        layers.extend(self.linear_layer(i, dim_i, dim_i_plus_1))
        if i != len(self.layer_dims) - 2:
            layers.append(ReLUModule())
            should_do_dropout = (
                    "num_dropout_lyrs" not in config
                    or len(self.layer_dims) - 2 - config["num_dropout_lyrs"] <= i
            )
            if should_do_dropout:
                layers.append(DropoutModule(p=config["dropout_proba"]))

        return layers

    def linear_layer(self, i, dim_i, dim_i_plus_1):
        return [LinearModule(dim_i, dim_i_plus_1)]

    def single_forward_pass(self, h_drug_1, h_drug_2, cell_lines):
        inter1 = nn.Sequential(
            nn.Linear(h_drug_1.detach().numpy().shape[1], self.layer_dims[1]),
            nn.ReLU(),
            nn.BatchNorm1d(self.layer_dims[1]),
            nn.Linear(self.layer_dims[1], self.layer_dims[2]),
            nn.ReLU(),
            nn.BatchNorm1d(self.layer_dims[2]),
            nn.Linear(self.layer_dims[2], self.layer_dims[3]),
            nn.ReLU(),
            nn.BatchNorm1d(self.layer_dims[3]),
        )(h_drug_1)
        inter2 = nn.Sequential(
            nn.Linear(h_drug_1.detach().numpy().shape[1], self.layer_dims[1]),
            nn.ReLU(),
            nn.BatchNorm1d(self.layer_dims[1]),
            nn.Linear(self.layer_dims[1], self.layer_dims[2]),
            nn.ReLU(),
            nn.BatchNorm1d(self.layer_dims[2]),
            nn.Linear(self.layer_dims[2], self.layer_dims[3]),
            nn.ReLU(),
            nn.BatchNorm1d(self.layer_dims[3]),
        )(h_drug_2)
        comb = self.after_merge_mlp(
            [
                inter1 + inter2, cell_lines
            ]
        )[0]
        return comb

    def transform_single_drug(self, h, cell_lines):

        if self.single_drug_mlp is None:
            return None
        else:
            return self.single_drug_mlp([h, cell_lines])[0]


class BilinearMLPAbstractPredictor3(MLPAbstractPredictor3):
    def __init__(self, data, config, predictor_layers):
        super(BilinearMLPAbstractPredictor3, self).__init__(
            data, config, predictor_layers
        )

        self.bilinear_weights = Parameter(
            1 / 100 * torch.randn((self.merge_dim, self.merge_dim, self.merge_dim))
            + torch.cat([torch.eye(self.merge_dim)[None, :, :]] * self.merge_dim, dim=0)
        )
        self.bilinear_offsets = Parameter(1 / 100 * torch.randn((self.merge_dim)))

    def single_forward_pass(self, h_drug_1, h_drug_2, cell_lines):
        h_1 = self.before_merge_mlp([h_drug_1, cell_lines])[0]
        h_2 = self.before_merge_mlp([h_drug_2, cell_lines])[0]

        h_1 = self.bilinear_weights.matmul(h_1.T).T
        h_2 = self.bilinear_weights.matmul(h_2.T).T

        h_1 = h_1.permute(0, 2, 1)

        h_1_scal_h_2 = (h_1 * h_2).sum(1)

        h_1_scal_h_2 += self.bilinear_offsets

        comb = self.after_merge_mlp1([h_1_scal_h_2, cell_lines])[0]

        return comb


class Predictor(torch.nn.Module):
    def __init__(self, data, config, predictor_layers):
        #
        self.num_cell_lines = len(torch.unique(data.ddi_edge_classes))
        self.device = config["device"]
        self.with_fp = config["with_fp"]
        self.with_expr = config["with_expr"]
        self.with_prot = config["with_prot"]

        self.layer_dims = predictor_layers
        (
            self.layer_dims,  # [2500,1024,64,32,2]
            self.output_dim_comb,  # 1
            self.output_dim_mono,  # 1
        ) = self.get_layer_dims(
            self.layer_dims,
            fp_dim=int(data.fp_bits),
            expr_dim=data.x_drugs.shape[1] - int(data.fp_bits),
            prot_numb=data.x_prots.shape[0],
        )

        self.drug2target_dict = {i: [] for i in range(data.x_drugs.shape[0])}
        for edge in data.dpi_edge_idx.T:
            self.drug2target_dict[int(edge[0])].append(
                int(edge[1]) - data.x_drugs.shape[0]
            )

        self.merge_n_layers_before_the_end = config["merge_n_layers_before_the_end"]
        self.merge_dim = self.layer_dims[-self.merge_n_layers_before_the_end - 1]  # 32
        assert 0 < self.merge_n_layers_before_the_end < len(predictor_layers)

        super(Predictor, self).__init__()

    def forward(self, data, drug_drug_batch, h_drug, n_forward_passes=1):
        h_drug_1s, h_drug_2s, cell_lines = self.get_batch(data, drug_drug_batch, h_drug)
        self.fc1 = nn.Linear(5000, 512)
        self.fc2 = nn.Linear(512, 128)
        self.out = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        xc = torch.cat((h_drug_1s, h_drug_2s), 1)
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out

    def get_layer_dims(self, predictor_layers, fp_dim, expr_dim, prot_numb):

        return (
            get_layer_dims(
                predictor_layers,
                fp_dim,
                expr_dim,
                prot_numb,
                with_fp=self.with_fp,
                with_expr=self.with_expr,
                with_prot=self.with_prot,
            ),
            1,
            1,
        )

    def get_batch(self, data, drug_drug_batch, h_drug):
        return get_batch(
            data,
            drug_drug_batch,
            h_drug,
            self.drug2target_dict,
            with_fp=self.with_fp,
            with_expr=self.with_expr,
            with_prot=self.with_prot,
        )

    def single_forward_pass(self, h_drug_1s, h_drug_2s, cell_lines):
        raise NotImplementedError


import torch
import torch.nn as nn
from torch.nn import Parameter


########################################################################################################################
# Modules
########################################################################################################################


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
            cell_line]


class FilmWithFeatureModule(torch.nn.Module):
    def __init__(self, num_cell_line_features, out_dim):
        super(FilmWithFeatureModule, self).__init__()

        self.out_dim = out_dim

        self.condit_lin_1 = nn.Linear(num_cell_line_features, num_cell_line_features)
        self.condit_relu = nn.ReLU()
        self.condit_lin_2 = nn.Linear(num_cell_line_features, 2 * out_dim)

        # Change initialization of the bias so that the expectation of the output is 1 for the first columns
        self.condit_lin_2.bias.data[: out_dim] += 1

    def forward(self, input):
        x, cell_line_features = input[0], input[1]

        # Compute conditioning
        condit = self.condit_lin_2(self.condit_relu(self.condit_lin_1(cell_line_features)))

        return [
            condit[:, :self.out_dim] * x
            + condit[:, self.out_dim:],
            cell_line_features
        ]


class LinearFilmWithFeatureModule(torch.nn.Module):
    def __init__(self, num_cell_line_features, out_dim):
        super(LinearFilmWithFeatureModule, self).__init__()

        self.out_dim = out_dim

        self.condit_lin_1 = nn.Linear(num_cell_line_features, 2 * out_dim)

        # Change initialization of the bias so that the expectation of the output is 1 for the first columns
        self.condit_lin_1.bias.data[: out_dim] += 1

    def forward(self, input):
        x, cell_line_features = input[0], input[1]

        # Compute conditioning
        condit = self.condit_lin_1(cell_line_features)

        return [
            condit[:, :self.out_dim] * x
            + condit[:, self.out_dim:],
            cell_line_features
        ]


########################################################################################################################
# Bilinear MLP
########################################################################################################################


class BilinearMLPPredictor(torch.nn.Module):
    def __init__(self, data, config, predictor_layers):

        super(BilinearMLPPredictor, self).__init__()

        self.num_cell_lines = len(data.cell_line_to_idx_dict.keys())
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_type)

        self.layer_dims = predictor_layers

        self.merge_n_layers_before_the_end = config["merge_n_layers_before_the_end"]
        self.merge_dim = self.layer_dims[-self.merge_n_layers_before_the_end - 1]

        assert 0 < self.merge_n_layers_before_the_end < len(predictor_layers)

        layers_before_merge = []
        layers_after_merge = []

        # Build early layers (before addition of the two embeddings)
        for i in range(len(self.layer_dims) - 1 - self.merge_n_layers_before_the_end):
            layers_before_merge = self.add_layer(
                layers_before_merge,
                i,
                self.layer_dims[i],
                self.layer_dims[i + 1]
            )

        # Build last layers (after addition of the two embeddings)
        for i in range(
                len(self.layer_dims) - 1 - self.merge_n_layers_before_the_end,
                len(self.layer_dims) - 1,
        ):
            layers_after_merge = self.add_layer(
                layers_after_merge,
                i,
                self.layer_dims[i],
                self.layer_dims[i + 1]
            )

        self.before_merge_mlp = nn.Sequential(*layers_before_merge)
        self.after_merge_mlp = nn.Sequential(*layers_after_merge)

        # Number of bilinear transformations == the dimension of the layer at which the merge is performed
        # Initialize weights close to identity
        self.bilinear_weights = Parameter(
            1 / 100 * torch.randn((self.merge_dim, self.merge_dim, self.merge_dim))
            + torch.cat([torch.eye(self.merge_dim)[None, :, :]] * self.merge_dim, dim=0)
        )
        self.bilinear_offsets = Parameter(1 / 100 * torch.randn((self.merge_dim)))

        # self.allow_neg_eigval = config["allow_neg_eigval"]
        # if self.allow_neg_eigval:
        #     self.bilinear_diag = Parameter(1 / 100 * torch.randn((self.merge_dim, self.merge_dim)) + 1)

    def forward(self, data, drug_drug_batch):
        h_drug_1, h_drug_2, cell_lines = self.get_batch(data, drug_drug_batch)

        # Apply before merge MLP
        h_1 = self.before_merge_mlp([h_drug_1, cell_lines])[0]
        h_2 = self.before_merge_mlp([h_drug_2, cell_lines])[0]

        # compute <W.h_1, W.h_2> = h_1.T . W.T.W . h_2
        h_1 = self.bilinear_weights.matmul(h_1.T).T
        h_2 = self.bilinear_weights.matmul(h_2.T).T

        if self.allow_neg_eigval:
            # Multiply by diagonal matrix to allow for negative eigenvalues
            h_2 *= self.bilinear_diag

        # "Transpose" h_1
        h_1 = h_1.permute(0, 2, 1)

        # Multiplication
        h_1_scal_h_2 = (h_1 * h_2).sum(1)

        # Add offset
        h_1_scal_h_2 += self.bilinear_offsets

        comb = self.after_merge_mlp([h_1_scal_h_2, cell_lines])[0]

        return comb

    def get_batch(self, data, drug_drug_batch):

        drug_1s = drug_drug_batch[0][:, 0]  # Edge-tail drugs in the batch
        drug_2s = drug_drug_batch[0][:, 1]  # Edge-head drugs in the batch
        cell_lines = drug_drug_batch[1]  # Cell line of all examples in the batch

        h_drug_1 = data.x_drugs[drug_1s]
        h_drug_2 = data.x_drugs[drug_2s]

        return h_drug_1, h_drug_2, cell_lines

    def add_layer(self, layers, i, dim_i, dim_i_plus_1):
        layers.extend(self.linear_layer(i, dim_i, dim_i_plus_1))
        if i != len(self.layer_dims) - 2:
            layers.append(ReLUModule())

        return layers

    def linear_layer(self, i, dim_i, dim_i_plus_1):
        return [LinearModule(dim_i, dim_i_plus_1)]


########################################################################################################################
# Bilinear MLP with Film conditioning
########################################################################################################################


class BilinearFilmMLPPredictor(BilinearMLPPredictor):
    def __init__(self, data, config, predictor_layers):
        super(BilinearFilmMLPPredictor, self).__init__(data, config, predictor_layers)

    def linear_layer(self, i, dim_i, dim_i_plus_1):
        return [LinearModule(dim_i, dim_i_plus_1), FilmModule(self.num_cell_lines, self.layer_dims[i + 1])]


class BilinearFilmWithFeatMLPPredictor(BilinearMLPPredictor):
    def __init__(self, data, config, predictor_layers):
        self.cl_features_dim = data.cell_line_features.shape[1]
        super(BilinearFilmWithFeatMLPPredictor, self).__init__(data, config, predictor_layers)

    def linear_layer(self, i, dim_i, dim_i_plus_1):
        return [LinearModule(dim_i, dim_i_plus_1), FilmWithFeatureModule(self.cl_features_dim, self.layer_dims[i + 1])]

    def get_batch(self, data, drug_drug_batch):
        drug_1s = drug_drug_batch[0][:, 0]  # Edge-tail drugs in the batch
        drug_2s = drug_drug_batch[0][:, 1]  # Edge-head drugs in the batch
        cell_lines = drug_drug_batch[1]  # Cell line of all examples in the batch
        batch_cl_features = data.cell_line_features[cell_lines]

        h_drug_1 = data.x_drugs[drug_1s]
        h_drug_2 = data.x_drugs[drug_2s]

        return h_drug_1, h_drug_2, batch_cl_features


class BilinearLinFilmWithFeatMLPPredictor(BilinearFilmWithFeatMLPPredictor):
    def __init__(self, data, config, predictor_layers):
        super(BilinearLinFilmWithFeatMLPPredictor, self).__init__(data, config, predictor_layers)

    def linear_layer(self, i, dim_i, dim_i_plus_1):
        return [LinearModule(dim_i, dim_i_plus_1), LinearFilmWithFeatureModule(self.cl_features_dim,
                                                                               self.layer_dims[i + 1])]


########################################################################################################################
# Bilinear MLP with Cell line features as input
########################################################################################################################


class BilinearCellLineInputMLPPredictor(BilinearMLPPredictor):
    def __init__(self, data, config, predictor_layers):
        self.cl_features_dim = data.cell_line_features.shape[1]
        predictor_layers[0] += self.cl_features_dim
        super(BilinearCellLineInputMLPPredictor, self).__init__(data, config, predictor_layers)

    def get_batch(self, data, drug_drug_batch):
        drug_1s = drug_drug_batch[0][:, 0]  # Edge-tail drugs in the batch
        drug_2s = drug_drug_batch[0][:, 1]  # Edge-head drugs in the batch
        cell_lines = drug_drug_batch[1]  # Cell line of all examples in the batch
        batch_cl_features = data.cell_line_features[cell_lines]

        h_drug_1 = data.x_drugs[drug_1s]
        h_drug_2 = data.x_drugs[drug_2s]

        # Add cell line features to drug representations
        h_drug_1 = torch.cat((h_drug_1, batch_cl_features), dim=1)
        h_drug_2 = torch.cat((h_drug_2, batch_cl_features), dim=1)

        return h_drug_1, h_drug_2, cell_lines
