from datasets.drugcomb_matrix_data import DrugCombMatrix_onehot
from models.model import (
    GiantPLRGA
)
from models.message_conv_layers import ThreeMessageConvLayer
from hyperopt import hp
from models.predictors import (
    BilinearSharedLayersMLPPredictor,
)
from utils1 import get_project_root
from train_mse_loewe_PLRGA import train_epoch, eval_epoch, BasicTrainer
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler
import os

cell_line = None
pipeline_config = {
    "use_tune": False,
    "num_epoch_without_tune": 270,  # Used only if "use_tune" == False
    "seed": 1,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "batch_size": 1024,
    "train_epoch": train_epoch,
    "eval_epoch": eval_epoch,
}

trainer = BasicTrainer

model_config = {
    "model": GiantPLRGA,
    "inhibition_model_hid_lay_dim": 8,
    "inhibition_model_weight_decay": 1,
    "mono_loss_scale_factor": 1,
    "combo_loss_scale_factor": 1,
    "model_synergy": True,

    "conv_layer": ThreeMessageConvLayer,
    "pass_d2p_msg": True,
    "pass_p2d_msg": True,  # always set to True
    "pass_p2p_msg": True,
    "drug_self_loop": True,
    "prot_self_loop": True,

    "residual_layers_dim": 32,
    "num_res_layers": 1,
    "backprop_period": 4,
    "do_periodic_backprop": True,
    "use_prot_emb": True,
    "use_prot_feats": True,
    "prot_emb_dim": 16,
    "drug_attention": {
        "attention": True,
        "attention_rank": 64,
        "dropout_proba": 0.4,
    },
    "prot_attention": {
        "attention": True,
        "attention_rank": 64,
        "dropout_proba": 0.4,
    },
}

predictor_config = {
    "predictor": BilinearSharedLayersMLPPredictor,
    "predictor_layers": [
        1024,
        64,
        32,
        1,
    ],
    "merge_n_layers_before_the_end": 1,
    "with_fp": True,
    "with_expr": True,
    "with_prot": False,
}

dataset_config = {
    "dataset": DrugCombMatrix_onehot,
    "transform": None,
    "pre_transform": None,
    "split_level": "pair",
    "val_set_prop": 0.3,
    "test_set_prop": 0,
    "cell_line": cell_line,
    "target": "loewe",
    "fp_bits": 1024,
    "fp_radius": 3,
    "ppi_graph": "huri.csv",
    "dti_graph": "chembl_dtis.csv",
    "use_l1000": True,
    "restrict_to_l1000_covered_drugs": True,
}

bayesian_learning_config = {
    "ensemble": False,
    "ensemble_size": 5,
    "dropout_proba": 0.0,
    "num_dropout_lyrs": 1,
    "n_train_forward_passes": 1,
    "n_scoring_forward_passes": 1,
}

asha_scheduler = ASHAScheduler(
    time_attr="training_iteration",
    metric="valid_mse",
    mode="min",
    max_t=1000,
    grace_period=10,
    reduction_factor=3,
    brackets=1,
)

search_space = {
    "lr": hp.loguniform("lr", -16.118095651, -5.52146091786),
    "batch_size": hp.choice("batch_size", [128, 256, 512, 1024]),
}

current_best_params = [
    {
        "lr": 1e-4,
        "batch_size": 1024,
    }
]

search_alg = HyperOptSearch(
    search_space, metric="valid_mse", mode="min", points_to_evaluate=current_best_params
)

configuration = {
    "trainer": trainer,
    "trainer_config": {
        **pipeline_config,
        **predictor_config,
        **model_config,
        **dataset_config,
        **bayesian_learning_config,
    },
    "summaries_dir": os.path.join(get_project_root(), "RayLogs"),
    "memory": 1800,
    "checkpoint_freq": 0,
    "stop": {"training_iteration": 500, "all_space_explored": 1},
    "checkpoint_at_end": False,
    "resources_per_trial": {"cpu": 4, "gpu": 1},
    "scheduler": None,
    "search_alg": None,
}