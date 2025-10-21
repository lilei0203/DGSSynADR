import torch
import pandas as pd
from models.model import GiantGraphGCN, EnsembleModel, Baseline
import os
from torch.utils.data import DataLoader
from datasets.utils import get_tensor_dataset
from ray import tune
import ray
import time
import argparse
import importlib
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, cohen_kappa_score, r2_score
from time import time as get_time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train_epoch(
        data, loader, model, optim, n_forward_passes=1
):
    global ground_truth_scores, comb1
    model.train()
    model.enable_periodic_backprop()

    epoch_loss = 0
    num_batches = len(loader)
    comb_all = []
    ground_truth_scores_all = []
    for _, drug_drug_batch in enumerate(loader):
        optim.zero_grad()

        out = model.forward(data, drug_drug_batch, n_forward_passes=n_forward_passes)
        loss = model.loss(out, drug_drug_batch)
        loss.backward()
        optim.step()
        comb = out[0]
        comb1 = comb[:, 0, 0].detach().cpu().numpy()
        comb1 = comb1.tolist()

        ground_truth_scores = drug_drug_batch[2][:, None, None]
        ground_truth_scores = torch.cat([ground_truth_scores] * comb.shape[2], dim=2)
        ground_truth_scores = ground_truth_scores[:, 0, 0].detach().numpy()
        ground_truth_scores1 = ground_truth_scores.tolist()
        ground_truth_scores_all = ground_truth_scores_all + ground_truth_scores1
        comb_all = comb_all + comb1

        epoch_loss += loss.item()



    mse = mean_squared_error(ground_truth_scores_all, comb_all)
    r2 = r2_score(ground_truth_scores_all, comb_all)
    pearson, _ = pearsonr(comb_all, ground_truth_scores_all)

    print("---------Train---------")
    print(" train mse:", mse)
    print(" train r2:", r2)
    print(" train pearson:", pearson)
    print("---------Train Over---------")

    return {"loss_sum": epoch_loss, "loss_mean": epoch_loss / num_batches}

def eval_epoch(
        data, loader, model, n_forward_passes=1, message="Mean valid loss"):
    model.eval()
    model.disable_periodic_backprop()
    if n_forward_passes > 1:
        model.enable_dropout()

    epoch_loss = 0
    epoch_comb_r_squared = 0
    epoch_mono_r_squared = 0
    num_batches = len(loader)
    comb_all = []
    ground_truth_scores_all = []
    cell_lines = []
    drug1s = []
    drug2s = []
    with torch.no_grad():
        for _, drug_drug_batch in enumerate(loader):
            out = model.forward(
                data, drug_drug_batch, n_forward_passes=n_forward_passes
            )

            loss = model.loss(out, drug_drug_batch)
            epoch_loss += loss.item()

            comb = out[0]
            comb1 = comb[:, 0, 0].detach().cpu().numpy()
            comb1 = comb1.tolist()

            ground_truth_scores = drug_drug_batch[2][:, None, None]
            ground_truth_scores = torch.cat([ground_truth_scores] * comb.shape[2], dim=2)
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
    file.to_csv("/home/Ganyanglan/HXY/DGSSynADR/result/drug_pair_score.csv", header=False)

    print("---------Test---------")
    print(" vaild mse:", mse)
    print(" vaild r2:", r2)
    print(" vaild pearson:", pearson)
    print("---------Test Over---------")

    file = pd.DataFrame({'vaild mse': mse,
                         'vaild r2': r2,
                         "vaild pearson": pearson}, index=[0])

    file.to_csv("/home/Ganyanglan/HXY/DGSSynADR/result/result_loss.csv", mode="a", header=False)

    print(message + ": {:.4f}".format(epoch_loss / num_batches))


    summary_dict = {
        "loss_sum": epoch_loss,
        "loss_mean": epoch_loss / num_batches,
        "comb_r_squared": epoch_comb_r_squared / num_batches,
        "mono_r_squared": epoch_mono_r_squared / num_batches,
    }


    return summary_dict


class AbstractTrainer(tune.Trainable):
    def setup(self, config):
        self.batch_size = config["batch_size"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_train_forward_passes = config["n_train_forward_passes"]
        self.training_it = 0
        config["device"] = self.device
        self.target = config["target"]

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

        self.data = dataset[0].to(self.device)
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

        self.train_idxs, self.val_idxs, self.test_idxs = dataset.random_split(
            config["test_set_prop"], config["val_set_prop"], config["split_level"]
        )

        valid_ddi_dataset = get_tensor_dataset(self.data, self.val_idxs)

        self.valid_loader = DataLoader(
            valid_ddi_dataset,
            batch_size=config["batch_size"],
            pin_memory=(self.device == "cuda"),
        )

        if config["ensemble"] is True:
            self.model = EnsembleModel(self.data, config).to(self.device)
        else:
            self.model = config["model"](self.data, config).to(self.device)

        self.optim = torch.optim.Adam(
            self.model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )

        self.train_epoch = config["train_epoch"]
        self.eval_epoch = config["eval_epoch"]



    def step(self):
        raise NotImplementedError

    def save_checkpoint(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))


class BasicTrainer(AbstractTrainer):
    def setup(self, config):
        print("Initializing regular training pipeline")
        super(BasicTrainer, self).setup(config)
        train_ddi_dataset = get_tensor_dataset(self.data, self.train_idxs)
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

        eval_metrics = self.eval_epoch(self.data, self.valid_loader, self.model)

        train_metrics = [("train_" + k, v) for k, v in train_metrics.items()]
        eval_metrics = [("eval_" + k, v) for k, v in eval_metrics.items()]
        metrics = dict(train_metrics + eval_metrics)

        metrics["training_iteration"] = self.training_it
        metrics["all_space_explored"] = 0
        self.training_it += 1
        return metrics


def train(configuration):

    if configuration["trainer_config"]["use_tune"]:
        ray.init(num_cpus=20)
        time_to_sleep = 5
        print("Sleeping for %d seconds" % time_to_sleep)
        time.sleep(time_to_sleep)
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
        trainer = configuration["trainer"](configuration["trainer_config"])
        for i in range(configuration["trainer_config"]["num_epoch_without_tune"]):
            trainer.train()




if __name__ == "__main__":

    for i in range(1):
        data = pd.DataFrame([i], index=[0])
        data.to_csv("/home/Ganyanglan/HXY/DGSSynADR/result/result_loss.csv", mode="a", header=False)
        time_start = get_time()
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--config",
            type=str,
            help='Name of the configuration file without ".py" at the end',
        )
        args = parser.parse_args()
        args.config = "test_mse_loewe_loss"
        my_config = importlib.import_module("config." + args.config)
        print("Running with configuration from", args.config)
        my_config.configuration["name"] = args.config
        train(my_config.configuration)
        print("Running time:", get_time()-time_start)
