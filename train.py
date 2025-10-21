import torch
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
# from config.test_mse import configuration, pipeline_config
from models.model import GiantGraphGCN, EnsembleModel, Baseline
import os
from torch.utils.data import DataLoader
from DGSSynADR.datasets.utils import get_tensor_dataset
from sklearn.metrics import mean_squared_error as mse
from ray import tune
import ray
import time
import argparse
import importlib
from scipy.stats import pearsonr
import numpy as np
import torch.nn.functional as F
# from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, precision_score, recall_score, balanced_accuracy_score
from sklearn.metrics import mean_squared_error, cohen_kappa_score, r2_score

from DGSSynADR.models.acquisition import (
    RandomAcquisition,
    ExpectedImprovement,
    GreedyAcquisition,
    GPUCB,
    Thompson,
)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
########################################################################################################################
# Epoch loops
########################################################################################################################

# data: torch_geometric.data.Data, loader: dataloader进行迭代、optim: 使用的优化器、max_examp_per_epoch: epoch的最大大小
# n_forward_passes: 每批执行的前向传递次数（对于return: 字典的loss）
def train_epoch(
        data, loader, model, optim, max_examp_per_epoch=None, n_forward_passes=1
):
    """
    训练一个 epoch 的模型
     data: `torch_geometric.data.Data` 对象
     loader: dataloader 进行迭代
     参数模型：
     optim: 使用的优化器
     max_examp_per_epoch: epoch 的最大大小
     n_forward_passes: 每批执行的前向传递次数（对于 MC dropout）
     return: 字典的loss
    """

    global ground_truth_scores, comb1
    model.train()
    # 启用定期反向传播
    model.enable_periodic_backprop()

    epoch_loss = 0
    num_batches = len(loader)
    # "num:", num_batches=884
    examples_seen = 0
    b = c = d = e = 0
    comb_all = []
    ground_truth_scores_all = []
    pred_labels_all = []
    true_labels_all = []
    for _, drug_drug_batch in enumerate(loader):
        optim.zero_grad()

        out = model.forward(data, drug_drug_batch, n_forward_passes=n_forward_passes)
        # len(out)=3, len(drug_drug_batch)=11
        # loss函数返回的是loss, r_squared, 0三个
        loss, comb_r_squared, mono_r_squared = model.loss(out, drug_drug_batch)
        # auc_roc计算
        loss.backward()
        optim.step()
        # comb是预测分数
        comb = out[0]
        comb1 = comb[:, 0, 0].detach().cpu().numpy()
        comb1 = comb1.tolist()

        # 判断Drugbatch的大小，每个批次
        # ground_truth_scores是真实分数
        # comb是预测分数
        # true_labels是阈值下的真实标签
        # pred_labels是阈值下的预测标签

        ground_truth_scores = drug_drug_batch[2][:, None, None]
        ground_truth_scores = torch.cat([ground_truth_scores] * comb.shape[2], dim=2)
        ground_truth_scores = ground_truth_scores[:, 0, 0].detach().numpy()
        ground_truth_scores1 = ground_truth_scores.tolist()
        ground_truth_scores_all = ground_truth_scores_all + ground_truth_scores1
        comb_all = comb_all + comb1

        epoch_loss += loss.item()
        # If we have seen enough examples in this epoch, break
        # 如果我们在这个epoch看到了足够多的例子，就break
        examples_seen += drug_drug_batch[0].shape[0]
        if max_examp_per_epoch is not None:
            if examples_seen >= max_examp_per_epoch:
                break

    mse = mean_squared_error(ground_truth_scores_all, comb_all)
    r2 = r2_score(ground_truth_scores_all, comb_all)
    pearson, _ = pearsonr(comb_all, ground_truth_scores_all)

    print("---------训练---------")
    print(" train mse:", mse)
    print(" train r2:", r2)
    print(" train pearson:", pearson)
    print("---------训练结束---------")

    return {"loss_sum": epoch_loss, "loss_mean": epoch_loss / num_batches}


# 返回summary_dict指标和分数字典（如果提供了获取功能）
def eval_epoch(
        data, loader, model, acquisition=None, n_forward_passes=1, message="Mean valid loss"):
    """
     评估模型
     data: `torch_geometric.data.Data` 对象
     loader: dataloader 进行迭代
     参数模型：
     acquisition：用于计算分数的获取函数。 如果没有，将不计算分数
     n_forward_passes: 每批执行的前向传递次数（对于 MC dropout）
     message: 在损失前绘制的消息
     return: 指标和分数字典（如果提供了获取功能）
    """
    model.eval()
    # eval()模式下dropout层会让所有的激活单元都通过
    model.disable_periodic_backprop()
    # 如果在评估时使用多个(>=2)前向传递，我们需要启用 dropout
    if n_forward_passes > 1:
        model.enable_dropout()

    epoch_loss = 0
    epoch_comb_r_squared = 0
    epoch_mono_r_squared = 0
    num_batches = len(loader)
    active_scores = torch.empty(0)
    comb_all = []
    ground_truth_scores_all = []
    pred_labels_all = []
    mse_all = r2_all = pearson_all = []
    cell_lines = []
    drug1s = []
    drug2s = []
    with torch.no_grad():
        for _, drug_drug_batch in enumerate(loader):
            out = model.forward(
                data, drug_drug_batch, n_forward_passes=n_forward_passes
            )

            if acquisition is not None:
                active_scores = torch.cat((active_scores, acquisition.get_scores(out)))

            loss, comb_r_squared, mono_r_squared = model.loss(out, drug_drug_batch)
            epoch_loss += loss.item()
            epoch_comb_r_squared += comb_r_squared
            epoch_mono_r_squared += mono_r_squared

            comb = out[0]
            # comb = F.softmax(comb)
            comb1 = comb[:, 0, 0].detach().cpu().numpy()
            # print("comb1:", comb1)
            comb1 = comb1.tolist()

            # data.
            # 判断Drugbatch的大小，每个批次
            ground_truth_scores = drug_drug_batch[2][:, None, None]
            ground_truth_scores = torch.cat([ground_truth_scores] * comb.shape[2], dim=2)
            # ground_truth_scores = F.softmax(ground_truth_scores, 1)
            ground_truth_scores = ground_truth_scores[:, 0, 0].detach().cpu().numpy()
            ground_truth_scores1 = ground_truth_scores.tolist()

            cell = drug_drug_batch[1].cpu().numpy().tolist()
            drug1 = drug_drug_batch[0][:, 0].detach().cpu().numpy().tolist()
            drug2 = drug_drug_batch[0][:, 1].detach().cpu().numpy().tolist()


            ground_truth_scores_all = ground_truth_scores_all + ground_truth_scores1
            comb_all = comb_all + comb1
            cell_lines = cell_lines + cell
            drug1s = drug1s + drug1
            drug2s = drug2s + drug2

        mse = mean_squared_error(ground_truth_scores_all, comb_all)
        r2 = r2_score(ground_truth_scores_all, comb_all)
        pearson, _ = pearsonr(comb_all, ground_truth_scores_all)


    file = pd.DataFrame({'y_pred_score': comb_all,
                         'y_true_score': ground_truth_scores_all,
                         "cell_line": cell_lines,
                         'drug1s:': drug1s,
                         'drug2s:': drug2s})


    # | r |≥0.8时，可认为两变量间极高度相关；0.6≤ | r | < 0.8，
    # 可认为两变量高度相关；0.4≤ | r | < 0.6，可认为两变量中度相关；
    # 0.2≤ | r | < 0.4，可认为两变量低度相关； | r | < 0.2，可认为两变量基本不相关。

    # R2的范围在0-1之间，越接近1，表示越好
    print("---------测试---------")
    print(" vaild mse:", mse)
    print(" vaild r2:", r2)
    print(" vaild pearson:", pearson)
    print("---------测试结束---------")

    file = pd.DataFrame({'vaild mse': mse,
                         'vaild r2': r2,
                         "vaild pearson": pearson}, index=[0])

    file.to_csv("/home/Ganyanglan/HXY/DGSSynADR/result2/result1.csv", mode="a", header=False)

    print(message + ": {:.4f}".format(epoch_loss / num_batches))


    summary_dict = {
        "loss_sum": epoch_loss,
        "loss_mean": epoch_loss / num_batches,
        "comb_r_squared": epoch_comb_r_squared / num_batches,
        "mono_r_squared": epoch_mono_r_squared / num_batches,
    }

    if acquisition is not None:
        return summary_dict, active_scores

    return summary_dict


########################################################################################################################
# Abstract trainer
########################################################################################################################
class AbstractTrainer(tune.Trainable):
    # 初始化
    def setup(self, config):
        self.batch_size = config["batch_size"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_train_forward_passes = config["n_train_forward_passes"]
        self.training_it = 0
        config["device"] = self.device

        # 我修改的
        self.target = config["target"]

        # 初始化数据集
        dataset = config["dataset"](
            transform=config["transform"],
            pre_transform=config["pre_transform"],
            fp_bits=config["fp_bits"],
            fp_radius=config["fp_radius"],
            ppi_graph=config["ppi_graph"],
            dti_graph=config["dti_graph"],
            cell_line=config["cell_line"],
            use_l1000=config["use_l1000"],
            restrict_to_l1000_covered_drugs=config["restrict_to_l1000_covered_drugs"],
        )

        self.data = dataset[0].to(self.device)  # len(dataset[0])=27

        # If a score is the target, we store it in the ddi_edge_response attribute of the data object
        # 如果分数是目标，我们将其存储在数据对象的 ddi_edge_response 属性中
        if "target" in config.keys():
            possible_target_dicts = {
                "css": self.data.ddi_edge_css,
                "bliss": self.data.ddi_edge_bliss,
                "zip": self.data.ddi_edge_zip,
                "hsa": self.data.ddi_edge_hsa,
                "loewe": self.data.ddi_edge_loewe,
            }
            self.data.ddi_edge_response = possible_target_dicts[config["target"]]

        torch.manual_seed(config["seed"])

        # Perform train/valid/test split. Test split is fixed regardless of the user defined seed
        # 执行训练/有效/测试拆分。 无论用户定义的种子如何，测试拆分都是固定的
        self.train_idxs, self.val_idxs, self.test_idxs = dataset.random_split(
            config["test_set_prop"], config["val_set_prop"], config["split_level"]
        )

        # Valid loader 有效的装载机
        valid_ddi_dataset = get_tensor_dataset(self.data, self.val_idxs)

        self.valid_loader = DataLoader(
            valid_ddi_dataset,
            batch_size=config["batch_size"],
            pin_memory=(self.device == "cuda"),
        )

        # Initialize model初始化模型
        if config["ensemble"] is True:
            self.model = EnsembleModel(self.data, config).to(self.device)
        else:
            self.model = config["model"](self.data, config).to(self.device)

        # Initialize optimizer初始化优化器
        self.optim = torch.optim.Adam(
            self.model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )

        self.train_epoch = config["train_epoch"]
        self.eval_epoch = config["eval_epoch"]

        self.loggers = self._build_loggers(config)

    def log_result(self, result):
        super().log_result(result)

        for logger in self.loggers:
            logger.log_result(result)

    def _build_loggers(self, config):
        logger_classes = config.get("logger_classes", [])
        return [
            logger_class(config, self.logdir, self.data)
            for logger_class in logger_classes
        ]

    def cleanup(self):
        for logger in self.loggers:
            logger.flush()
            logger.close()

    def reset_config(self, new_config):
        for logger in self.loggers:
            logger.flush()
            logger.close()

        self.loggers = self._build_loggers(new_config)

    def step(self):
        raise NotImplementedError

    def save_checkpoint(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))


########################################################################################################################
# Basic trainer
########################################################################################################################


class BasicTrainer(AbstractTrainer):
    """
    Trainer to use if one does not wish to use active learning
    如果不想使用主动学习，可以使用训练器，初始化常规训练管道
    """
    # 返回 train_loss_sum、...mean、eval_loss_sum、eval_loss_mean、eval_comb_r_squared、eval_mono_r_squared

    def setup(self, config):
        print("Initializing regular training pipeline")
        super(BasicTrainer, self).setup(config)
        # Train loader
        train_ddi_dataset = get_tensor_dataset(self.data, self.train_idxs)
        d = self.train_idxs
        b = self.test_idxs
        c = self.val_idxs

        self.train_loader = DataLoader(
            train_ddi_dataset,
            batch_size=config["batch_size"],
            pin_memory=(self.device == "cuda"),
        )

    def step(self):
        train_metrics = self.train_epoch(
            self.data,
            self.train_loader,
            self.model,
            self.optim,
            n_forward_passes=self.n_train_forward_passes,
        )
        # 没有主动查询，eval_metrics只返回sum_dict
        eval_metrics = self.eval_epoch(self.data, self.valid_loader, self.model)

        train_metrics = [("train_" + k, v) for k, v in train_metrics.items()]
        eval_metrics = [("eval_" + k, v) for k, v in eval_metrics.items()]
        metrics = dict(train_metrics + eval_metrics)

        metrics["training_iteration"] = self.training_it
        metrics["all_space_explored"] = 0  # For compatibility with active trainer
        self.training_it += 1
        # print(self.model)
        return metrics


########################################################################################################################
# Active Trainer
########################################################################################################################


class ActiveTrainer(AbstractTrainer):
    """
    Trainer class to perform active learning
    执行主动学习的Trainer class
    """
    def setup(self, config):
        print("Initializing active training pipeline")
        super(ActiveTrainer, self).setup(config)

        self.acquire_n_at_a_time = config["acquire_n_at_a_time"]
        self.max_examp_per_epoch = config["max_examp_per_epoch"]
        self.acquisition = config["acquisition"](config)
        self.n_scoring_forward_passes = config["n_scoring_forward_passes"]
        self.n_epoch_between_queries = config["n_epoch_between_queries"]

        # randomly acquire data at the beginning开始时随机获取数据
        # 这是一个切片操作
        self.seen_idxs = self.train_idxs[: config["n_initial"]]
        self.unseen_idxs = self.train_idxs[config["n_initial"]:]

        # Initialize variable that saves the last query初始化保存上次查询的变量
        self.last_query_idxs = self.seen_idxs

        # Initialize dataloaders初始化数据加载器
        (   self.seen_loader,
            self.unseen_loader,
            self.last_query_loader,
        ) = self.update_loaders(self.seen_idxs, self.unseen_idxs, self.last_query_idxs)
        # Get the set of top 1% most synergistic combinations
        # 获取一组前 1% 最具协同性的组合
        # self.seen_idxs 128，unseen_idxs 40354左右
        one_perc = int(0.01 * len(self.unseen_idxs))
        scores = self.data.ddi_edge_response[self.unseen_idxs]
        # scores的长度和unseen_idxs一样
        # 创建一个无序不重复元素集
        # argsort()函数是对数组中的元素进行从小到大排序，只返回相应序列元素的数组下标，数组元素位置不变
        # [:one_perc] 表示从端点开始到index为“one_perc”,即取前one_percd的数组下标，【即得到了分数最高的那些scores的索引】
        self.top_one_perc = set(
            self.unseen_idxs[torch.argsort(scores, descending=True)[:one_perc]].numpy()
        )
        self.count = 0

        # 获取的是排序后的ddi_edge_response所对应的索引值

    def step(self):
        # Check whether we have explored everything
        # 检查我们是否已经探索了一切
        global seen_metrics
        if len(self.unseen_loader) == 0:
            print("All space has been explored已探索所有空间")
            return {"all_space_explored": 1, "training_iteration": self.training_it}

        # Evaluate on last query before training在训练前评估最后一个查询
        last_query_before_metric = self.eval_epoch(
            self.data,
            self.last_query_loader,
            self.model,
            message="Last query before training loss",
        )

        # Train on seen examples在见过的例子上训练
        for _ in range(self.n_epoch_between_queries):
            # Perform several training epochs. Save only metrics from the last epoch
            # 执行几个训练时期。 仅保存上一个时期的指标
            seen_metrics = self.train_epoch(
                self.data,
                self.seen_loader,
                self.model,
                self.optim,
                self.max_examp_per_epoch,
                n_forward_passes=self.n_train_forward_passes,
            )

        # Evaluate on last query after training在训练后评估最后一个查询
        last_query_after_metric = self.eval_epoch(
            self.data,
            self.last_query_loader,
            self.model,
            message="Last query after training loss",
        )
        print("----评估验证集----")
        # Evaluate on valid set评估验证集
        eval_metrics = self.eval_epoch(
            self.data, self.valid_loader, self.model, message="Eval loss"
        )
        print("------end------")
        # Score unseen examples给看不见的例子打分
        # 返回summary_dict指标和分数字典（提供了获取功能）
        unseen_metrics, active_scores = self.eval_epoch(
            self.data,
            self.unseen_loader,
            self.model,
            self.acquisition,
            self.n_scoring_forward_passes,
            message="Unseen loss",
        )

        # Build summary构建摘要
        seen_metrics = [("seen_" + k, v) for k, v in seen_metrics.items()]
        last_query_before_metric = [
            ("last_query_before_tr_" + k, v) for k, v in last_query_before_metric.items()
        ]
        last_query_after_metric = [
            ("last_query_after_tr_" + k, v) for k, v in last_query_after_metric.items()
        ]
        unseen_metrics = [("unseen_" + k, v) for k, v in unseen_metrics.items()]
        eval_metrics = [("eval_" + k, v) for k, v in eval_metrics.items()]
        # dict()创建一个新的字典
        metrics = dict(
            seen_metrics
            + unseen_metrics
            + eval_metrics
            + last_query_before_metric
            + last_query_after_metric
        )

        # Acquire new data获取新数据
        print("query data, Acquire new data, 获取新数据")
        query = self.unseen_idxs[
            torch.argsort(active_scores, descending=True)[: self.acquire_n_at_a_time]
        ]
        # remove the query from the unseen examples从看不见的示例中删除查询
        # 获取acquire_n_at_a_time到最后的index作为unseen的index
        self.unseen_idxs = self.unseen_idxs[
            torch.argsort(active_scores, descending=True)[self.acquire_n_at_a_time:]
        ]

        # Add the query to the seen examples将查询添加到看到的示例中
        self.seen_idxs = torch.cat((self.seen_idxs, query))
        metrics["seen_idxs"] = self.data.ddi_edge_idx[:, self.seen_idxs]

        # Compute proportion of top 1% synergistic drugs which have been discovered
        # 计算已发现的前1%协同药物的比例
        query_set = set(query.detach().numpy())
        self.count += len(query_set & self.top_one_perc)
        # 输出比例
        metrics["top"] = self.count / len(self.top_one_perc)

        self.last_query_idxs = query

        # Update the dataloaders更新数据加载器
        (
            self.seen_loader,
            self.unseen_loader,
            self.last_query_loader,
        ) = self.update_loaders(self.seen_idxs, self.unseen_idxs, self.last_query_idxs)

        metrics["training_iteration"] = self.training_it
        metrics["all_space_explored"] = 0
        self.training_it += 1
        # 返回的metrics有22项内容
        return metrics

    def update_loaders(self, seen_idxs, unseen_idxs, last_query_idxs):
        # Seen loader看到装载机
        seen_ddi_dataset = get_tensor_dataset(self.data, seen_idxs)

        seen_loader = DataLoader(
            seen_ddi_dataset,
            batch_size=self.batch_size,
            pin_memory=(self.device == "cuda"),
        )

        # Unseen loader看不见的装载机
        unseen_ddi_dataset = get_tensor_dataset(self.data, unseen_idxs)

        unseen_loader = DataLoader(
            unseen_ddi_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=(self.device == " "),
        )

        # Last query loader最后一个查询加载器
        last_query_ddi_dataset = get_tensor_dataset(self.data, last_query_idxs)
        # DataLoader本质上就是一个iterable（跟python的内置类型list等一样），并利用多进程来加速batch data的处理，使用yield来使用有限的内存
        last_query_loader = DataLoader(
            last_query_ddi_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=(self.device == "cuda"),
        )

        # Update acquisition function更新获取功能
        self.acquisition.update_with_seen(self.data.ddi_edge_response[seen_idxs])

        return seen_loader, unseen_loader, last_query_loader


def train(configuration):
    """
    Train with or without tune depending on configuration根据配置进行调整或不调整训练
    """
    if configuration["trainer_config"]["use_tune"]:
        ###########################################
        # Use tune使用调
        ###########################################
        ray.init(num_cpus=20)  # 此处定义了系统有20个CPU核
        time_to_sleep = 5
        print("Sleeping for %d seconds" % time_to_sleep)
        time.sleep(time_to_sleep)
        # 推迟调用线程的运行,可通过参数time_to_sleep指秒数,表示进程挂起的时间
        print("Woke up.. Scheduling")

        tune.run(
            configuration["trainer"],
            name=configuration["name"],
            config=configuration["trainer_config"],
            stop=configuration["stop"],
            resources_per_trial=configuration["resources_per_trial"],
            num_samples=1,
            checkpoint_at_end=configuration["checkpoint_at_end"],
            local_dir=configuration["summaries_dir"],
            checkpoint_freq=configuration["checkpoint_freq"],
            scheduler=configuration["scheduler"],
            search_alg=configuration["search_alg"],
        )

    else:
        ###########################################
        # Do not use tune
        ###########################################

        trainer = configuration["trainer"](configuration["trainer_config"])
        for i in range(configuration["trainer_config"]["num_epoch_without_tune"]):
            trainer.train()




if __name__ == "__main__":
    # Parser
    for i in range(1):
        a = pd.DataFrame([i], index=[0])
        a.to_csv("/home/Ganyanglan/HXY/Recover/recover/result2/result_1.csv", mode="a", header=False)

        from time import time as get_time
        time_start = get_time()
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--config",
            type=str,
            help='Name of the configuration file without ".py" at the end',
        )
        args = parser.parse_args()
        # Retrieve configuration检索配置
        args.config = "test_mse_loewe"
        my_config = importlib.import_module("recover.config." + args.config)
        print("Running with configuration from", "recover.config." + args.config)
        my_config.configuration["name"] = args.config
        train(my_config.configuration)
        print("Running time:", get_time()-time_start)
