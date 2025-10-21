import torch
import numpy as np
from torch.nn import functional as F
from torch.nn import Parameter
from models.utils import ResidualModule, Atten, Atten1, PLRGA, LowRankAttention
from utils1 import get_dropout_modules_recursive


#######################################################################################################################
# Abstract Model
#######################################################################################################################

class Abstractloss(torch.nn.Module):
    def __init__(self, data, config):
        self.device = config["device"]
        super(Abstractloss, self).__init__()
        self.criterion = torch.nn.MSELoss()
        self.mu_predictor = Baseline(data, config)
        self.std_predictor = Baseline(data, config)
        self.data = data

    def forward(self, data, drug_drug_batch):
        raise NotImplementedError

    def enable_dropout(self):
        for m in get_dropout_modules_recursive(self):
            m.train()

    def loss(self, output, drug_drug_batch):
        comb = output[0]
        ground_truth_scores = drug_drug_batch[2][:, None, None]
        ground_truth_scores = torch.cat([ground_truth_scores] * comb.shape[2], dim=2)

        self.criterion = torch.nn.SmoothL1Loss(size_average=None, reduce=None, reduction='mean', beta=1.0)
        loss = self.criterion(comb, ground_truth_scores)

        return loss

    def enable_periodic_backprop(self):
        pass

    def disable_periodic_backprop(self):
        pass

class Baseloss(Abstractloss):
    def __init__(self, data, config):

        super(Baseloss, self).__init__(data, config)

        self.do_periodic_backprop = (
            config["do_periodic_backprop"]
            if "do_periodic_backprop" in config
            else False
        )
        self.backprop_period = (
            config["backprop_period"] if "backprop_period" in config else None
        )
        self.periodic_backprop_enabled = (
            False
        )
        self.curr_backprop_status = (
            False
        )
        self.backprop_iter = 0

    def forward(self, data, drug_drug_batch):
        if self.periodic_backprop_enabled:
            should_enable = self.backprop_iter % self.backprop_period == 0
            self.set_backprop_enabled_status(should_enable)
            self.backprop_iter += 1

        return self._forward(data, drug_drug_batch)

    def _forward(self, data, drug_drug_batch):
        raise NotImplementedError

    def set_backprop_enabled_status(self, status):

        if status != self.curr_backprop_status:
            for var in self.get_periodic_backprop_vars():
                var.requires_grad = status

            self.curr_backprop_status = status

    def get_periodic_backprop_vars(self):

        raise NotImplementedError

    def enable_periodic_backprop(self):

        assert self.backprop_period is not None
        self.periodic_backprop_enabled = self.do_periodic_backprop

    def disable_periodic_backprop(self):

        assert self.backprop_period is not None
        self.periodic_backprop_enabled = False

# class AbstractModel(torch.nn.Module):
#     def __init__(self, data, config):
#         """
#         Abstract class for Models. Models take as input the features of all drugs and all proteins, and
#         compute embeddings for the drugs.
#         The embeddings of drugs are then fed two at a time to a predictor model
#         :param data: `torch_geometric.data.Data` object
#         :param config: configuration dictionary
#         """
#         """
#         模型的抽象类。 模型将所有药物和所有蛋白质的特征作为输入，并且计算药物的嵌入。
#         然后将药物的嵌入一次两次输入预测模型
#          :param 数据: `torch_geometric.data.Data` 对象
#          :param config: 配置字典
#         """
#         self.device = config["device"]
#         super(AbstractModel, self).__init__()
#         # 原始是这个
#         self.criterion = torch.nn.MSELoss()
#         # 自己设计的loss
#         # self.criterion = weight_loss()
#         # self.criterion = logcosh_loss()
#
#     def forward(self, data, drug_drug_batch, n_forward_passes=1):
#         raise NotImplementedError
#
#     def enable_dropout(self):
#         """
#         手动启用模型的dropout模块。 这很有用，因为我们需要在eval时激活 dropout
#         """
#         for m in get_dropout_modules_recursive(self):
#             m.train()
#
#     def loss(self, output, drug_drug_batch):
#         """
#          协同预测管道的损失函数
#          :param output: 预测器的输出
#          :param drug_drug_batch：药物-药物组合示例的批次
#          ：返回：
#         """
#         comb = output[0]  # 我们不考虑单一药物的代表性
#         # comb.shape[2]=1
#         # 一顿操作
#         ground_truth_scores = drug_drug_batch[2][:, None, None]
#         # 按照2维去拼接
#         ground_truth_scores = torch.cat([ground_truth_scores] * comb.shape[2], dim=2)
#         loss = self.criterion(comb, ground_truth_scores)
#         # 使用新的loss
#         # 是matchmaker的loss
#         # loss = weight_loss(ground_truth_scores, comb)
#         # log_cosh loss
#         # loss = logcosh_loss(ground_truth_scores, comb)
#
#         # mse(pred, target)
#         # comb.shape torch.Size([64, 1, 1])
#         # ground_truth_scores.shape torch.Size([64, 1, 1])
#         # 解释方差
#         var = drug_drug_batch[2].var().item()
#         r_squared = (var - loss.item()) / var
#
#         return loss, r_squared, 0
#
#     def enable_periodic_backprop(self):
#         """
#         如果需要，将被此类的实现覆盖的虚拟方法
#         """
#         pass
#
#     def disable_periodic_backprop(self):
#         pass
#
#
# class Baseline(AbstractModel):
#     def __init__(self, data, config):
#         """
#         用于协同预测管道的基线模型（无 GCN）。
#         预测器是单独定义的（c.f. predictors.py），用于将药物嵌入映射到协同作用
#         """
#         super(Baseline, self).__init__(data, config)
#         config[
#             "with_fp"
#         ] = False  # 强制设置为 False 因为在这种情况下默认使用 fps
#         config[
#             "with_expr"
#         ] = False  # 强制设置为 False，因为如果可用，默认使用基因 expr
#
#         # 计算预测器输入的维度
#         predictor_layers = [data.x_drugs.shape[1]] + config["predictor_layers"]
#
#         self.predictor = self.get_predictor(data, config, predictor_layers)
#
#     def forward(self, data, drug_drug_batch, n_forward_passes=1):
#         return self.predictor(
#             data, drug_drug_batch, h_drug=None
#         )
#
#     def get_predictor(self, data, config, predictor_layers):
#         return config["predictor"](
#             data, config, predictor_layers
#         )

class Giantloss(Baseloss):
    def __init__(self, data, config):

        super(Giantloss, self).__init__(data, config)

        self.use_prot_emb = (
            config["use_prot_emb"] if "use_prot_emb" in config else False
        )
        self.use_prot_feats = (
            config["use_prot_feats"] if "use_prot_feats" in config else False
        )

        if (not self.use_prot_emb) and (not self.use_prot_feats):
            print(
                f"NOTE: 'use_prot_emb' and 'use_prot_feats' are missing, using both embeddings and features"
            )
            self.use_prot_emb = True
            self.use_prot_feats = True
        # config["prot_emb_dim"] = 16   data.x_prots.shape[1] = 1024
        self.prot_emb_dim = (
                self.use_prot_emb * config["prot_emb_dim"]
                + self.use_prot_feats * data.x_prots.shape[1]
        )
        # self.prot_emb_dim = 16+1024=1040
        if self.use_prot_emb:
            self.prot_emb = Parameter(
                1 / 100 * torch.randn((data.x_prots.shape[0], config["prot_emb_dim"]))
            )

        self.conv1 = config["conv_layer"](
            data.x_drugs.shape[1],  # 2212
            self.prot_emb_dim,  # 1040
            config["residual_layers_dim"],  # 32
            config["residual_layers_dim"],
            config["pass_d2p_msg"],
            config["pass_p2d_msg"],
            config["pass_p2p_msg"],  # True
            config["drug_self_loop"],
            config["prot_self_loop"],
            data,
        )

        if "drug_attention" in config:
            self.has_drug_attention = config["drug_attention"]["attention"]
            self.drug_attention_conf = config["drug_attention"]
            # "attention": True, "attention_rank": 64, "dropout_proba": 0.4,
        else:
            self.has_drug_attention = False

        if self.has_drug_attention:
            self.low_rank_drug_attention = []
            # data.x_drugs.shape 52*2212
            self.low_rank_drug_attention.append(
                LowRankAttention(
                    k=self.drug_attention_conf["attention_rank"],  # 64
                    d=data.x_drugs.shape[1],  # 2212
                    dropout=self.drug_attention_conf["dropout_proba"],  # 0.4

                )
            )

        if "prot_attention" in config:
            self.has_prot_attention = config["prot_attention"]["attention"]
            self.prot_attention_conf = config["prot_attention"]
        else:
            self.has_prot_attention = False

        if self.has_prot_attention:
            self.low_rank_prot_attention = []
            self.low_rank_prot_attention.append(
                LowRankAttention(
                    k=self.prot_attention_conf["attention_rank"],
                    d=self.prot_emb_dim,  # 1040
                    dropout=self.prot_attention_conf["dropout_proba"],
                )
            )

        self.residual_layers = []
        # residual_layers_dim：32
        drug_channels, prot_channels = (
            config["residual_layers_dim"],
            config["residual_layers_dim"],
        )

        for i in range(config["num_res_layers"]):
            if self.has_drug_attention:
                drug_channels += 2 * self.drug_attention_conf["attention_rank"]
                self.low_rank_drug_attention.append(
                    LowRankAttention(
                        k=self.drug_attention_conf["attention_rank"],
                        d=drug_channels,
                        dropout=self.drug_attention_conf["dropout_proba"],
                    )
                )

            if self.has_prot_attention:
                prot_channels += 2 * self.prot_attention_conf["attention_rank"]
                self.low_rank_prot_attention.append(
                    LowRankAttention(
                        k=self.prot_attention_conf["attention_rank"],
                        d=prot_channels,
                        dropout=self.prot_attention_conf["dropout_proba"],
                    )
                )

            self.residual_layers.append(
                ResidualModule(
                    ConvLayer=config["conv_layer"],
                    drug_channels=drug_channels,
                    prot_channels=prot_channels,
                    pass_d2p_msg=config["pass_d2p_msg"],
                    pass_p2d_msg=config["pass_p2d_msg"],
                    pass_p2p_msg=config["pass_p2p_msg"],
                    drug_self_loop=config["drug_self_loop"],
                    prot_self_loop=config["prot_self_loop"],
                    data=data,
                )
            )

        self.residual_layers = torch.nn.ModuleList(self.residual_layers)
        if self.has_drug_attention:
            self.low_rank_drug_attention = torch.nn.ModuleList(
                self.low_rank_drug_attention
            )

        if self.has_prot_attention:
            self.low_rank_prot_attention = torch.nn.ModuleList(
                self.low_rank_prot_attention
            )

        if self.has_drug_attention:
            predictor_layers = [  # 160+64*2 = 288
                drug_channels + 2 * self.drug_attention_conf["attention_rank"]
            ]
        else:
            predictor_layers = [drug_channels]  # 160

        # predictor_layers：[1024,64,32,1]---->[288,1024,64,32,1]
        predictor_layers += config["predictor_layers"]

        self.predictor = self.get_predictor(data, config, predictor_layers)

    def get_predictor(self, data, config, predictor_layers):
        return config["predictor"](
            data, config, predictor_layers
        )

    def get_periodic_backprop_vars(self):

        if self.use_prot_emb:
            yield self.prot_emb

        yield from self.conv1.parameters()
        yield from self.residual_layers.parameters()
        if self.has_drug_attention:
            yield from self.low_rank_drug_attention.parameters()
        if self.has_prot_attention:
            yield from self.low_rank_prot_attention.parameters()

    def _forward(self, data, drug_drug_batch):
        h_drug = data.x_drugs
        # h_drug:52*2212
        if self.use_prot_emb and self.use_prot_feats:
            # (19182,1040) = (19182, 16) , (19182, 1024)
            h_prot = torch.cat((self.prot_emb, data.x_prots), dim=1)
        elif self.use_prot_emb:
            h_prot = self.prot_emb
        elif self.use_prot_feats:
            h_prot = data.x_prots

        # h_drug_next, h_prot_next == (52,32) (19182,32)        # conv1 = ThreeMessageConvLayer
        h_drug_next, h_prot_next = self.conv1(h_drug, h_prot, data)
        if self.has_drug_attention:
            att = self.low_rank_drug_attention[0](h_drug)  # att:(52,128)
            h_drug_next = torch.cat((h_drug_next, att), dim=1)  # h_drug_next:(52,160)

        if self.has_prot_attention:
            att = self.low_rank_prot_attention[0](h_prot)  # att:(19182,128)
            h_prot_next = torch.cat((h_prot_next, att), dim=1)  # h_prot_next:(19182,160)
        # h_drug_next, h_prot_next
        h_drug, h_prot = F.relu(h_drug_next), F.relu(h_prot_next)

        # len(self.residual_layers)=1
        for i in range(len(self.residual_layers)):  # # h_drug_next, h_prot_next == (52,160) (19182,160)
            h_drug_next, h_prot_next = self.residual_layers[i](h_drug, h_prot, data)
            if self.has_drug_attention:
                att = self.low_rank_drug_attention[i + 1](h_drug)  # att:(52,128)
                h_drug_next = torch.cat((h_drug_next, att), dim=1)  # h_drug_next :(52, 288)

            if self.has_prot_attention:
                att = self.low_rank_prot_attention[i + 1](h_prot)  # att:(19182,128)
                h_prot_next = torch.cat((h_prot_next, att), dim=1)  # h_prot_next :(19182,288)
            h_drug, h_prot = F.relu(h_drug_next), F.relu(h_prot_next)

        return self.predictor(
            data, drug_drug_batch, h_drug
        )

class Baseline1(torch.nn.Module):
    def __init__(self, data, config):
        super(Baseline1, self).__init__()

        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_type)

        self.criterion = torch.nn.MSELoss()

        predictor_layers = [data.x_drugs.shape[1]] + config["predictor_layers"]

        assert predictor_layers[-1] == 1

        self.predictor = self.get_predictor(data, config, predictor_layers)

    def forward(self, data, drug_drug_batch):
        return self.predictor(data, drug_drug_batch)

    def get_predictor(self, data, config, predictor_layers):
        return config["predictor"](data, config, predictor_layers)

    def enable_periodic_backprop(self):
        pass

    def loss(self, output, drug_drug_batch):
        comb = output
        ground_truth_scores = drug_drug_batch[2][:, None]
        loss = self.criterion(comb, ground_truth_scores)

        return loss, 0, 0


class AbstractModel(torch.nn.Module):
    def __init__(self, data, config):
        self.device = config["device"]
        super(AbstractModel, self).__init__()
        self.criterion = torch.nn.MSELoss()


    def forward(self, data, drug_drug_batch):
        raise NotImplementedError

    def enable_dropout(self):

        for m in get_dropout_modules_recursive(self):
            m.train()

    def loss(self, output, drug_drug_batch):
        comb = output[0]
        ground_truth_scores = drug_drug_batch[2][:, None, None]
        ground_truth_scores = torch.cat([ground_truth_scores] * comb.shape[2], dim=2)
        loss = self.criterion(comb, ground_truth_scores)

        var = drug_drug_batch[2].var().item()
        r_squared = (var - loss.item()) / var

        return loss, r_squared, torch.tensor(0.0,requires_grad=True)

    def enable_periodic_backprop(self):

        pass

    def disable_periodic_backprop(self):
        pass


#without GCN
class Baseline(AbstractModel):
    def __init__(self, data, config):
        super(Baseline, self).__init__(data, config)
        config[
            "with_fp"
        ] = False
        config[
            "with_expr"
        ] = False

        predictor_layers = [data.x_drugs.shape[1]] + config["predictor_layers"]
        self.predictor = self.get_predictor(data, config, predictor_layers)

    def forward(self, data, drug_drug_batch):
        return self.predictor(
            data, drug_drug_batch, h_drug=None
        )
    def get_predictor(self, data, config, predictor_layers):
        return config["predictor"](
            data, config, predictor_layers
        )

class BasePeriodicBackpropModel(AbstractModel):
    def __init__(self, data, config):
        super(BasePeriodicBackpropModel, self).__init__(data, config)

        self.do_periodic_backprop = (
            config["do_periodic_backprop"]
            if "do_periodic_backprop" in config
            else False
        )
        self.backprop_period = (
            config["backprop_period"] if "backprop_period" in config else None
        )
        self.periodic_backprop_enabled = (
            False
        )
        self.curr_backprop_status = (
            False
        )
        self.backprop_iter = 0

    def forward(self, data, drug_drug_batch):
        if self.periodic_backprop_enabled:
            should_enable = self.backprop_iter % self.backprop_period == 0
            self.set_backprop_enabled_status(should_enable)
            self.backprop_iter += 1

        return self._forward(data, drug_drug_batch)

    def _forward(self, data, drug_drug_batch):
        raise NotImplementedError

    def set_backprop_enabled_status(self, status):
        if status != self.curr_backprop_status:
            for var in self.get_periodic_backprop_vars():
                var.requires_grad = status

            self.curr_backprop_status = status

    def get_periodic_backprop_vars(self):

        raise NotImplementedError

    def enable_periodic_backprop(self):
        assert self.backprop_period is not None
        self.periodic_backprop_enabled = self.do_periodic_backprop

    def disable_periodic_backprop(self):
        assert self.backprop_period is not None
        self.periodic_backprop_enabled = False


class GiantGAT(Baseloss):
    def __init__(self, data, config):

        super(GiantGAT, self).__init__(data, config)

        self.use_prot_emb = (
            config["use_prot_emb"] if "use_prot_emb" in config else False
        )
        self.use_prot_feats = (
            config["use_prot_feats"] if "use_prot_feats" in config else False
        )

        if (not self.use_prot_emb) and (not self.use_prot_feats):
            print(
                f"NOTE: 'use_prot_emb' and 'use_prot_feats' are missing, using both embeddings and features"
            )
            self.use_prot_emb = True
            self.use_prot_feats = True
        # config["prot_emb_dim"] = 16   data.x_prots.shape[1] = 1024
        self.prot_emb_dim = (
                self.use_prot_emb * config["prot_emb_dim"]
                + self.use_prot_feats * data.x_prots.shape[1]
        )
        # self.prot_emb_dim = 16+1024=1040

        if self.use_prot_emb:
            self.prot_emb = Parameter(
                # 19182*16
                1 / 100 * torch.randn((data.x_prots.shape[0], config["prot_emb_dim"]))
            )

        self.conv1 = config["conv_layer"](
            data.x_drugs.shape[1],  # 2212
            self.prot_emb_dim,  # 1040
            config["residual_layers_dim"],  # 32
            config["residual_layers_dim"],
            config["pass_d2p_msg"],
            config["pass_p2d_msg"],
            config["pass_p2p_msg"],  # True
            config["drug_self_loop"],
            config["prot_self_loop"],
            data,
        )

        if "drug_attention" in config:
            self.has_drug_attention = config["drug_attention"]["attention"]
            self.drug_attention_conf = config["drug_attention"]
            # "attention": True, "attention_rank": 64, "dropout_proba": 0.4,
        else:
            self.has_drug_attention = False

        if self.has_drug_attention:
            self.low_rank_drug_attention = []
            # data.x_drugs.shape 52*2212
            self.low_rank_drug_attention.append(
                Atten1(
                    k=self.drug_attention_conf["attention_rank"],  # 64
                    d=data.x_drugs.shape[1],  # 2212
                    dropout=self.drug_attention_conf["dropout_proba"],  # 0.4

                )
            )

        if "prot_attention" in config:
            self.has_prot_attention = config["prot_attention"]["attention"]
            self.prot_attention_conf = config["prot_attention"]
        else:
            self.has_prot_attention = False

        if self.has_prot_attention:
            self.low_rank_prot_attention = []
            self.low_rank_prot_attention.append(
                Atten1(
                    k=self.prot_attention_conf["attention_rank"],
                    d=self.prot_emb_dim,  # 1040
                    dropout=self.prot_attention_conf["dropout_proba"],
                )
            )

        self.residual_layers = []
        # residual_layers_dim：32
        drug_channels, prot_channels = (
            config["residual_layers_dim"],
            config["residual_layers_dim"],
        )

        for i in range(config["num_res_layers"]):
            if self.has_drug_attention:
                drug_channels += self.drug_attention_conf["attention_rank"]
                self.low_rank_drug_attention.append(
                    Atten1(
                        k=self.drug_attention_conf["attention_rank"],
                        d=drug_channels,
                        dropout=self.drug_attention_conf["dropout_proba"],
                    )
                )

            if self.has_prot_attention:
                prot_channels += self.prot_attention_conf["attention_rank"]
                self.low_rank_prot_attention.append(
                    Atten1(
                        k=self.prot_attention_conf["attention_rank"],
                        d=prot_channels,
                        dropout=self.prot_attention_conf["dropout_proba"],
                    )
                )

            self.residual_layers.append(
                ResidualModule(
                    ConvLayer=config["conv_layer"],
                    drug_channels=drug_channels,
                    prot_channels=prot_channels,
                    pass_d2p_msg=config["pass_d2p_msg"],
                    pass_p2d_msg=config["pass_p2d_msg"],
                    pass_p2p_msg=config["pass_p2p_msg"],
                    drug_self_loop=config["drug_self_loop"],
                    prot_self_loop=config["prot_self_loop"],
                    data=data,
                )
            )

        self.residual_layers = torch.nn.ModuleList(self.residual_layers)
        if self.has_drug_attention:
            self.low_rank_drug_attention = torch.nn.ModuleList(
                self.low_rank_drug_attention
            )

        if self.has_prot_attention:
            self.low_rank_prot_attention = torch.nn.ModuleList(
                self.low_rank_prot_attention
            )

        if self.has_drug_attention:
            predictor_layers = [  # 160+64*2 = 288
                drug_channels + self.drug_attention_conf["attention_rank"]
            ]
        else:
            predictor_layers = [drug_channels]  # 160

        # predictor_layers：[1024,64,32,1]---->[288,1024,64,32,1]
        predictor_layers += config["predictor_layers"]
        self.predictor = self.get_predictor(data, config, predictor_layers)

    def get_predictor(self, data, config, predictor_layers):
        return config["predictor"](
            data, config, predictor_layers
        )

    def get_periodic_backprop_vars(self):

        if self.use_prot_emb:
            yield self.prot_emb

        yield from self.conv1.parameters()
        yield from self.residual_layers.parameters()
        if self.has_drug_attention:
            yield from self.low_rank_drug_attention.parameters()
        if self.has_prot_attention:
            yield from self.low_rank_prot_attention.parameters()

    def _forward(self, data, drug_drug_batch):
        h_drug = data.x_drugs
        # h_drug:52*2212
        if self.use_prot_emb and self.use_prot_feats:
            # (19182,1040) = (19182, 16) , (19182, 1024)
            h_prot = torch.cat((self.prot_emb, data.x_prots), dim=1)
        elif self.use_prot_emb:
            h_prot = self.prot_emb
        elif self.use_prot_feats:
            h_prot = data.x_prots

        # h_drug_next, h_prot_next == (52,32) (19182,32)        # conv1 = ThreeMessageConvLayer
        h_drug_next, h_prot_next = self.conv1(h_drug, h_prot, data)
        if self.has_drug_attention:
            att = self.low_rank_drug_attention[0](h_drug)  # att:(52,128)
            h_drug_next = torch.cat((h_drug_next, att), dim=1)  # h_drug_next:(52,160)

        if self.has_prot_attention:
            att = self.low_rank_prot_attention[0](h_prot)  # att:(19182,128)
            h_prot_next = torch.cat((h_prot_next, att), dim=1)  # h_prot_next:(19182,160)
        # h_drug_next, h_prot_next
        h_drug, h_prot = F.relu(h_drug_next), F.relu(h_prot_next)

        for i in range(len(self.residual_layers)):  # # h_drug_next, h_prot_next == (52,160) (19182,160)
            h_drug_next, h_prot_next = self.residual_layers[i](h_drug, h_prot, data)
            if self.has_drug_attention:
                att = self.low_rank_drug_attention[i + 1](h_drug)  # att:(52,128)
                h_drug_next = torch.cat((h_drug_next, att), dim=1)  # h_drug_next :(52, 288)

            if self.has_prot_attention:
                att = self.low_rank_prot_attention[i + 1](h_prot)  # att:(19182,128)
                h_prot_next = torch.cat((h_prot_next, att), dim=1)  # h_prot_next :(19182,288)
            h_drug, h_prot = F.relu(h_drug_next), F.relu(h_prot_next)

        return self.predictor(
            data, drug_drug_batch, h_drug
        )


class GiantPLRGA(Baseloss):
    def __init__(self, data, config):
        super(GiantPLRGA, self).__init__(data, config)

        self.use_prot_emb = (
            config["use_prot_emb"] if "use_prot_emb" in config else False
        )
        self.use_prot_feats = (
            config["use_prot_feats"] if "use_prot_feats" in config else False
        )

        if (not self.use_prot_emb) and (not self.use_prot_feats):
            print(
                f"NOTE: 'use_prot_emb' and 'use_prot_feats' are missing, using both embeddings and features"
            )
            self.use_prot_emb = True
            self.use_prot_feats = True
        # config["prot_emb_dim"] = 16   data.x_prots.shape[1] = 1024
        self.prot_emb_dim = (
                self.use_prot_emb * config["prot_emb_dim"]
                + self.use_prot_feats * data.x_prots.shape[1]
        )
        # self.prot_emb_dim = 16+1024=1040

        if self.use_prot_emb:
            self.prot_emb = Parameter(
                1 / 100 * torch.randn((data.x_prots.shape[0], config["prot_emb_dim"]))
            )

        self.conv1 = config["conv_layer"](
            data.x_drugs.shape[1],  # 2212
            self.prot_emb_dim,  # 1040
            config["residual_layers_dim"],  # 32
            config["residual_layers_dim"],
            config["pass_d2p_msg"],
            config["pass_p2d_msg"],
            config["pass_p2p_msg"],  # True
            config["drug_self_loop"],
            config["prot_self_loop"],
            data,
        )

        if "drug_attention" in config:
            self.has_drug_attention = config["drug_attention"]["attention"]
            self.drug_attention_conf = config["drug_attention"]
            # "attention": True, "attention_rank": 64, "dropout_proba": 0.4,
        else:
            self.has_drug_attention = False

        if self.has_drug_attention:
            self.low_rank_drug_attention = []
            # data.x_drugs.shape 52*2212
            self.low_rank_drug_attention.append(
                PLRGA(
                    k=self.drug_attention_conf["attention_rank"],  # 64
                    d=data.x_drugs.shape[1],  # 2212
                    dropout=self.drug_attention_conf["dropout_proba"],  # 0.4

                )
            )

        if "prot_attention" in config:
            self.has_prot_attention = config["prot_attention"]["attention"]
            self.prot_attention_conf = config["prot_attention"]
        else:
            self.has_prot_attention = False

        if self.has_prot_attention:
            self.low_rank_prot_attention = []
            self.low_rank_prot_attention.append(
                PLRGA(
                    k=self.prot_attention_conf["attention_rank"],
                    d=self.prot_emb_dim,  # 1040
                    dropout=self.prot_attention_conf["dropout_proba"],
                )
            )

        self.residual_layers = []
        drug_channels, prot_channels = (
            config["residual_layers_dim"],
            config["residual_layers_dim"],
        )

        for i in range(config["num_res_layers"]):
            if self.has_drug_attention:
                drug_channels = drug_channels + 64 + 52
                self.low_rank_drug_attention.append(
                    PLRGA(
                        k=self.drug_attention_conf["attention_rank"],
                        d=drug_channels,
                        dropout=self.drug_attention_conf["dropout_proba"],
                    )
                )

            if self.has_prot_attention:
                prot_channels = prot_channels + 19246
                self.low_rank_prot_attention.append(
                    PLRGA(
                        k=self.prot_attention_conf["attention_rank"],
                        d=prot_channels,
                        dropout=self.prot_attention_conf["dropout_proba"],
                    )
                )

            self.residual_layers.append(
                ResidualModule(
                    ConvLayer=config["conv_layer"],
                    drug_channels=drug_channels,
                    prot_channels=prot_channels,
                    pass_d2p_msg=config["pass_d2p_msg"],
                    pass_p2d_msg=config["pass_p2d_msg"],
                    pass_p2p_msg=config["pass_p2p_msg"],
                    drug_self_loop=config["drug_self_loop"],
                    prot_self_loop=config["prot_self_loop"],
                    data=data,
                )
            )

        self.residual_layers = torch.nn.ModuleList(self.residual_layers)
        if self.has_drug_attention:
            self.low_rank_drug_attention = torch.nn.ModuleList(
                self.low_rank_drug_attention
            )

        if self.has_prot_attention:
            self.low_rank_prot_attention = torch.nn.ModuleList(
                self.low_rank_prot_attention
            )

        if self.has_drug_attention:
            predictor_layers = [  # 160+64*2 = 288
                drug_channels + 64 + 52
            ]
        else:
            predictor_layers = [drug_channels]  # 160

        # predictor_layers：[1024,64,32,1]---->[288,1024,64,32,1]
        predictor_layers += config["predictor_layers"]

        self.predictor = self.get_predictor(data, config, predictor_layers)

    def get_predictor(self, data, config, predictor_layers):
        return config["predictor"](
            data, config, predictor_layers
        )

    def get_periodic_backprop_vars(self):

        if self.use_prot_emb:
            yield self.prot_emb

        yield from self.conv1.parameters()
        yield from self.residual_layers.parameters()
        if self.has_drug_attention:
            yield from self.low_rank_drug_attention.parameters()
        if self.has_prot_attention:
            yield from self.low_rank_prot_attention.parameters()

    def _forward(self, data, drug_drug_batch):
        h_drug = data.x_drugs
        # h_drug:52*2212
        if self.use_prot_emb and self.use_prot_feats:
            # (19182,1040) = (19182, 16) , (19182, 1024)
            h_prot = torch.cat((self.prot_emb, data.x_prots), dim=1)
        elif self.use_prot_emb:
            h_prot = self.prot_emb
        elif self.use_prot_feats:
            h_prot = data.x_prots

        # h_drug_next, h_prot_next == (52,32) (19182,32)        # conv1 = ThreeMessageConvLayer
        h_drug_next, h_prot_next = self.conv1(h_drug, h_prot, data)
        if self.has_drug_attention:
            att = self.low_rank_drug_attention[0](h_drug)  # att:(52,128)
            h_drug_next = torch.cat((h_drug_next, att), dim=1)  # h_drug_next:(52,160)

        if self.has_prot_attention:
            att = self.low_rank_prot_attention[0](h_prot)  # att:(19182,128)
            h_prot_next = torch.cat((h_prot_next, att), dim=1)  # h_prot_next:(19182,160)
        # h_drug_next, h_prot_next
        h_drug, h_prot = F.relu(h_drug_next), F.relu(h_prot_next)

        # len(self.residual_layers)=1
        for i in range(len(self.residual_layers)):  # # h_drug_next, h_prot_next == (52,160) (19182,160)
            h_drug_next, h_prot_next = self.residual_layers[i](h_drug, h_prot, data)
            if self.has_drug_attention:
                att = self.low_rank_drug_attention[i + 1](h_drug)  # att:(52,128)
                h_drug_next = torch.cat((h_drug_next, att), dim=1)  # h_drug_next :(52, 288)

            if self.has_prot_attention:
                att = self.low_rank_prot_attention[i + 1](h_prot)  # att:(19182,128)
                h_prot_next = torch.cat((h_prot_next, att), dim=1)  # h_prot_next :(19182,288)
            h_drug, h_prot = F.relu(h_drug_next), F.relu(h_prot_next)

        return self.predictor(
            data, drug_drug_batch, h_drug
        )


class GiantGraphGCN(Baseloss):
    def __init__(self, data, config):

        super(GiantGraphGCN, self).__init__(data, config)

        self.use_prot_emb = (
            config["use_prot_emb"] if "use_prot_emb" in config else False
        )
        self.use_prot_feats = (
            config["use_prot_feats"] if "use_prot_feats" in config else False
        )

        if (not self.use_prot_emb) and (not self.use_prot_feats):
            print(
                f"NOTE: 'use_prot_emb' and 'use_prot_feats' are missing, using both embeddings and features"
            )
            self.use_prot_emb = True
            self.use_prot_feats = True
        # config["prot_emb_dim"] = 16   data.x_prots.shape[1] = 1024
        self.prot_emb_dim = (
                self.use_prot_emb * config["prot_emb_dim"]
                + self.use_prot_feats * data.x_prots.shape[1]
        )
        # self.prot_emb_dim = 16+1024=1040

        if self.use_prot_emb:
            self.prot_emb = Parameter(
                1 / 100 * torch.randn((data.x_prots.shape[0], config["prot_emb_dim"]))
            )

        self.conv1 = config["conv_layer"](
            data.x_drugs.shape[1],  # 2212
            self.prot_emb_dim,  # 1040
            config["residual_layers_dim"],  # 32
            config["residual_layers_dim"],
            config["pass_d2p_msg"],
            config["pass_p2d_msg"],
            config["pass_p2p_msg"],  # True
            config["drug_self_loop"],
            config["prot_self_loop"],
            data,
        )

        if "drug_attention" in config:
            self.has_drug_attention = config["drug_attention"]["attention"]
            self.drug_attention_conf = config["drug_attention"]
            # "attention": True, "attention_rank": 64, "dropout_proba": 0.4,
        else:
            self.has_drug_attention = False

        if self.has_drug_attention:
            self.low_rank_drug_attention = []
            # data.x_drugs.shape 52*2212
            self.low_rank_drug_attention.append(
                LowRankAttention(
                    k=self.drug_attention_conf["attention_rank"],  # 64
                    d=data.x_drugs.shape[1],
                    dropout=self.drug_attention_conf["dropout_proba"],  # 0.4

                )
            )

        if "prot_attention" in config:
            self.has_prot_attention = config["prot_attention"]["attention"]
            self.prot_attention_conf = config["prot_attention"]
        else:
            self.has_prot_attention = False

        if self.has_prot_attention:
            self.low_rank_prot_attention = []
            self.low_rank_prot_attention.append(
                LowRankAttention(
                    k=self.prot_attention_conf["attention_rank"],
                    d=self.prot_emb_dim,  # 1040
                    dropout=self.prot_attention_conf["dropout_proba"],
                )
            )

        self.residual_layers = []
        # residual_layers_dim：32
        drug_channels, prot_channels = (
            config["residual_layers_dim"],
            config["residual_layers_dim"],
        )

        for i in range(config["num_res_layers"]):
            if self.has_drug_attention:
                drug_channels += 2 * self.drug_attention_conf["attention_rank"]
                self.low_rank_drug_attention.append(
                    LowRankAttention(
                        k=self.drug_attention_conf["attention_rank"],
                        d=drug_channels,
                        dropout=self.drug_attention_conf["dropout_proba"],
                    )
                )

            if self.has_prot_attention:
                prot_channels += 2 * self.prot_attention_conf["attention_rank"]
                self.low_rank_prot_attention.append(
                    LowRankAttention(
                        k=self.prot_attention_conf["attention_rank"],
                        d=prot_channels,
                        dropout=self.prot_attention_conf["dropout_proba"],
                    )
                )

            self.residual_layers.append(
                ResidualModule(
                    ConvLayer=config["conv_layer"],
                    drug_channels=drug_channels,
                    prot_channels=prot_channels,
                    pass_d2p_msg=config["pass_d2p_msg"],
                    pass_p2d_msg=config["pass_p2d_msg"],
                    pass_p2p_msg=config["pass_p2p_msg"],
                    drug_self_loop=config["drug_self_loop"],
                    prot_self_loop=config["prot_self_loop"],
                    data=data,
                )
            )

        # 转换为模块列表
        self.residual_layers = torch.nn.ModuleList(self.residual_layers)
        if self.has_drug_attention:
            self.low_rank_drug_attention = torch.nn.ModuleList(
                self.low_rank_drug_attention
            )

        if self.has_prot_attention:
            self.low_rank_prot_attention = torch.nn.ModuleList(
                self.low_rank_prot_attention
            )

        if self.has_drug_attention:
            predictor_layers = [  # 160+64*2 = 288
                drug_channels + 2 * self.drug_attention_conf["attention_rank"]
            ]
        else:
            predictor_layers = [drug_channels]  # 160

        # predictor_layers：[1024,64,32,1]---->[288,1024,64,32,1]
        predictor_layers += config["predictor_layers"]

        self.predictor = self.get_predictor(data, config, predictor_layers)

    def get_predictor(self, data, config, predictor_layers):
        return config["predictor"](
            data, config, predictor_layers
        )

    def get_periodic_backprop_vars(self):

        if self.use_prot_emb:
            yield self.prot_emb

        yield from self.conv1.parameters()
        yield from self.residual_layers.parameters()
        if self.has_drug_attention:
            yield from self.low_rank_drug_attention.parameters()
        if self.has_prot_attention:
            yield from self.low_rank_prot_attention.parameters()

    def _forward(self, data, drug_drug_batch):
        h_drug = data.x_drugs
        # h_drug:52*2212
        if self.use_prot_emb and self.use_prot_feats:
            # (19182,1040) = (19182, 16) , (19182, 1024)
            h_prot = torch.cat((self.prot_emb, data.x_prots), dim=1)
        elif self.use_prot_emb:
            h_prot = self.prot_emb
        elif self.use_prot_feats:
            h_prot = data.x_prots

        # h_drug_next, h_prot_next == (52,32) (19182,32)        # conv1 = ThreeMessageConvLayer
        h_drug_next, h_prot_next = self.conv1(h_drug, h_prot, data)
        if self.has_drug_attention:
            att = self.low_rank_drug_attention[0](h_drug)  # att:(52,128)
            h_drug_next = torch.cat((h_drug_next, att), dim=1)  # h_drug_next:(52,160)

        if self.has_prot_attention:
            att = self.low_rank_prot_attention[0](h_prot)  # att:(19182,128)
            h_prot_next = torch.cat((h_prot_next, att), dim=1)  # h_prot_next:(19182,160)
        # h_drug_next, h_prot_next
        h_drug, h_prot = F.relu(h_drug_next), F.relu(h_prot_next)

        # len(self.residual_layers)=1
        for i in range(len(self.residual_layers)):  # # h_drug_next, h_prot_next == (52,160) (19182,160)
            h_drug_next, h_prot_next = self.residual_layers[i](h_drug, h_prot, data)
            if self.has_drug_attention:
                att = self.low_rank_drug_attention[i + 1](h_drug)  # att:(52,128)
                h_drug_next = torch.cat((h_drug_next, att), dim=1)  # h_drug_next :(52, 288)

            if self.has_prot_attention:
                att = self.low_rank_prot_attention[i + 1](h_prot)  # att:(19182,128)
                h_prot_next = torch.cat((h_prot_next, att), dim=1)  # h_prot_next :(19182,288)
            h_drug, h_prot = F.relu(h_drug_next), F.relu(h_prot_next)

        return self.predictor(
            data, drug_drug_batch, h_drug
        )

class EnsembleModel(AbstractModel):
    def __init__(self, data, config):
        super(EnsembleModel, self).__init__(data, config)

        models = []
        for _ in range(config["ensemble_size"]):
            models.append(config["model"](data, config))

        self.models = torch.nn.ModuleList(models)

    def forward(self, data, drug_drug_batch):
        out = []
        for m in self.models:
            out.append(m(data, drug_drug_batch))
        return torch.cat(out, dim=1)

class AbstractModel1(torch.nn.Module):
    def __init__(self, data, config):

        self.device = config["device"]
        super(AbstractModel1, self).__init__()
        self.criterion = torch.nn.MSELoss()
        self.criterion1 = torch.nn.CrossEntropyLoss()

    def forward(self, data, drug_drug_batch):
        raise NotImplementedError

    def enable_dropout(self):
        for m in get_dropout_modules_recursive(self):
            m.train()

    def loss(self, output, drug_drug_batch):

        comb = output

        ground_truth_scores = drug_drug_batch[2].long().clone().detach()
        loss = self.criterion(comb, ground_truth_scores)
        return loss

    def enable_periodic_backprop(self):

        pass

    def disable_periodic_backprop(self):
        pass


class BasePeriodicBackpropModel1(AbstractModel1):
    def __init__(self, data, config):
        super(BasePeriodicBackpropModel1, self).__init__(data, config)

        self.do_periodic_backprop = (
            config["do_periodic_backprop"]
            if "do_periodic_backprop" in config
            else False
        )
        self.backprop_period = (
            config["backprop_period"] if "backprop_period" in config else None
        )
        self.periodic_backprop_enabled = (
            False
        )
        self.curr_backprop_status = (
            False
        )
        self.backprop_iter = 0

    def forward(self, data, drug_drug_batch):
        if self.periodic_backprop_enabled:
            should_enable = self.backprop_iter % self.backprop_period == 0
            self.set_backprop_enabled_status(should_enable)
            self.backprop_iter += 1

        return self._forward(data, drug_drug_batch)

    def _forward(self, data, drug_drug_batch):
        raise NotImplementedError

    def set_backprop_enabled_status(self, status):

        if status != self.curr_backprop_status:
            for var in self.get_periodic_backprop_vars():
                var.requires_grad = status

            self.curr_backprop_status = status

    def get_periodic_backprop_vars(self):

        raise NotImplementedError

    def enable_periodic_backprop(self):

        assert self.backprop_period is not None
        self.periodic_backprop_enabled = self.do_periodic_backprop

    def disable_periodic_backprop(self):

        assert self.backprop_period is not None
        self.periodic_backprop_enabled = False



class GiantGraphGCN1(BasePeriodicBackpropModel1):
    def __init__(self, data, config):

        super(GiantGraphGCN1, self).__init__(data, config)

        self.use_prot_emb = (
            config["use_prot_emb"] if "use_prot_emb" in config else False
        )
        self.use_prot_feats = (
            config["use_prot_feats"] if "use_prot_feats" in config else False
        )

        if (not self.use_prot_emb) and (not self.use_prot_feats):
            print(
                f"NOTE: 'use_prot_emb' and 'use_prot_feats' are missing, using both embeddings and features"
            )
            self.use_prot_emb = True
            self.use_prot_feats = True
        # config["prot_emb_dim"] = 16   data.x_prots.shape[1] = 1024
        self.prot_emb_dim = (
                self.use_prot_emb * config["prot_emb_dim"]
                + self.use_prot_feats * data.x_prots.shape[1]
        )
        # self.prot_emb_dim = 16+1024=1040

        if self.use_prot_emb:
            self.prot_emb = Parameter(
                1 / 100 * torch.randn((data.x_prots.shape[0], config["prot_emb_dim"]))
            )

        self.conv1 = config["conv_layer"](
            data.x_drugs.shape[1],  # 2212
            self.prot_emb_dim,  # 1040
            config["residual_layers_dim"],  # 32
            config["residual_layers_dim"],
            config["pass_d2p_msg"],
            config["pass_p2d_msg"],
            config["pass_p2p_msg"],  # True
            config["drug_self_loop"],
            config["prot_self_loop"],
            data,
        )

        if "drug_attention" in config:
            self.has_drug_attention = config["drug_attention"]["attention"]
            self.drug_attention_conf = config["drug_attention"]
            # "attention": True, "attention_rank": 64, "dropout_proba": 0.4,
        else:
            self.has_drug_attention = False

        if self.has_drug_attention:
            self.low_rank_drug_attention = []
            # data.x_drugs.shape 52*2212
            self.low_rank_drug_attention.append(
                LowRankAttention(
                    k=self.drug_attention_conf["attention_rank"],  # 64
                    d=data.x_drugs.shape[1],  # 2212
                    dropout=self.drug_attention_conf["dropout_proba"],  # 0.4

                )
            )

        if "prot_attention" in config:
            self.has_prot_attention = config["prot_attention"]["attention"]
            self.prot_attention_conf = config["prot_attention"]
        else:
            self.has_prot_attention = False

        if self.has_prot_attention:
            self.low_rank_prot_attention = []
            self.low_rank_prot_attention.append(
                LowRankAttention(
                    k=self.prot_attention_conf["attention_rank"],
                    d=self.prot_emb_dim,  # 1040
                    dropout=self.prot_attention_conf["dropout_proba"],
                )
            )

        self.residual_layers = []
        # residual_layers_dim：32
        drug_channels, prot_channels = (
            config["residual_layers_dim"],
            config["residual_layers_dim"],
        )

        for i in range(config["num_res_layers"]):
            if self.has_drug_attention:
                drug_channels += 2 * self.drug_attention_conf["attention_rank"]
                self.low_rank_drug_attention.append(
                    LowRankAttention(
                        k=self.drug_attention_conf["attention_rank"],
                        d=drug_channels,
                        dropout=self.drug_attention_conf["dropout_proba"],
                    )
                )

            if self.has_prot_attention:
                prot_channels += 2 * self.prot_attention_conf["attention_rank"]
                self.low_rank_prot_attention.append(
                    LowRankAttention(
                        k=self.prot_attention_conf["attention_rank"],
                        d=prot_channels,
                        dropout=self.prot_attention_conf["dropout_proba"],
                    )
                )

            self.residual_layers.append(
                ResidualModule(
                    ConvLayer=config["conv_layer"],
                    drug_channels=drug_channels,
                    prot_channels=prot_channels,
                    pass_d2p_msg=config["pass_d2p_msg"],
                    pass_p2d_msg=config["pass_p2d_msg"],
                    pass_p2p_msg=config["pass_p2p_msg"],
                    drug_self_loop=config["drug_self_loop"],
                    prot_self_loop=config["prot_self_loop"],
                    data=data,
                )
            )

        self.residual_layers = torch.nn.ModuleList(self.residual_layers)
        if self.has_drug_attention:
            self.low_rank_drug_attention = torch.nn.ModuleList(
                self.low_rank_drug_attention
            )

        if self.has_prot_attention:
            self.low_rank_prot_attention = torch.nn.ModuleList(
                self.low_rank_prot_attention
            )

        if self.has_drug_attention:
            predictor_layers = [  # 160+64*2 = 288
                drug_channels + self.drug_attention_conf["attention_rank"] * 2
            ]
        else:
            predictor_layers = [drug_channels]  # 160

        # predictor_layers：[1024,64,32,1]---->[288,1024,64,32,1]
        predictor_layers += config["predictor_layers"]

        self.predictor = self.get_predictor(data, config, predictor_layers)

    def get_predictor(self, data, config, predictor_layers):
        return config["predictor"](
            data, config, predictor_layers
        )

    def get_periodic_backprop_vars(self):

        if self.use_prot_emb:
            yield self.prot_emb

        yield from self.conv1.parameters()
        yield from self.residual_layers.parameters()
        if self.has_drug_attention:
            yield from self.low_rank_drug_attention.parameters()
        if self.has_prot_attention:
            yield from self.low_rank_prot_attention.parameters()

    def _forward(self, data, drug_drug_batch):
        h_drug = data.x_drugs
        # h_drug:52*2212
        if self.use_prot_emb and self.use_prot_feats:
            # (19182,1040) = (19182, 16) , (19182, 1024)
            h_prot = torch.cat((self.prot_emb, data.x_prots), dim=1)
        elif self.use_prot_emb:
            h_prot = self.prot_emb
        elif self.use_prot_feats:
            h_prot = data.x_prots


        # 第一层
        # h_drug_next, h_prot_next == (52,32) (19182,32)        # conv1 = ThreeMessageConvLayer
        h_drug_next, h_prot_next = self.conv1(h_drug, h_prot, data)
        if self.has_drug_attention:
            att = self.low_rank_drug_attention[0](h_drug)  # att:(52,128)
            h_drug_next = torch.cat((h_drug_next, att), dim=1)  # h_drug_next:(52,160)

        if self.has_prot_attention:
            att = self.low_rank_prot_attention[0](h_prot)  # att:(19182,128)
            h_prot_next = torch.cat((h_prot_next, att), dim=1)  # h_prot_next:(19182,160)
        # h_drug_next, h_prot_next
        h_drug, h_prot = F.relu(h_drug_next), F.relu(h_prot_next)

        # len(self.residual_layers)=1
        for i in range(len(self.residual_layers)):  # # h_drug_next, h_prot_next == (52,160) (19182,160)
            h_drug_next, h_prot_next = self.residual_layers[i](h_drug, h_prot, data)
            if self.has_drug_attention:
                att = self.low_rank_drug_attention[i + 1](h_drug)  # att:(52,128)
                h_drug_next = torch.cat((h_drug_next, att), dim=1)  # h_drug_next :(52, 288)

            if self.has_prot_attention:
                att = self.low_rank_prot_attention[i + 1](h_prot)  # att:(19182,128)
                h_prot_next = torch.cat((h_prot_next, att), dim=1)  # h_prot_next :(19182,288)
            h_drug, h_prot = F.relu(h_drug_next), F.relu(h_prot_next)

        return self.predictor(
            data, drug_drug_batch, h_drug
        )


class Baselineloss(torch.nn.Module):
    def __init__(self, data, config):

        super(Baselineloss, self).__init__()

        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_type)

        self.criterion = torch.nn.MSELoss()

        predictor_layers = [data.x_drugs.shape[1]] + config["predictor_layers"]

        assert predictor_layers[-1] == 1

        self.predictor = self.get_predictor(data, config, predictor_layers)

    def forward(self, data, drug_drug_batch):
        return self.predictor(data, drug_drug_batch)

    def get_predictor(self, data, config, predictor_layers):
        return config["predictor"](data, config, predictor_layers)

    def loss(self, output, drug_drug_batch):
        comb = output
        ground_truth_scores = drug_drug_batch[2][:, None]
        loss = self.criterion(comb, ground_truth_scores)

        return loss
