import argparse
import json
import os
import re
import shutil
import warnings
from datetime import datetime
from time import time

import deepspeed
import numpy as np
import pandas as pd
import swanlab
import torch
import torch.distributed as dist
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, DistributedSampler, Sampler

from src.config.configs import get_cfg_defaults
from src.core.utils import graph_collate_func, set_seed
from src.data.dataloader import DTIDataset
from src.models.models import BINDTI, binary_cross_entropy

DEFAULT_DEEPSPEED_CONFIG = os.path.join("configs", "ds_zero2.json")
DEFAULT_SWANLAB_LOG_ROOT = "swanlog"


def is_dist_ready():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    if is_dist_ready():
        return dist.get_rank()
    return int(os.environ.get("RANK", "0"))


def is_main_process():
    return get_rank() == 0


def setup_device():
    if not torch.cuda.is_available():
        return torch.device("cpu")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return torch.device(f"cuda:{local_rank}")


def load_json_config(config_path):
    with open(config_path, "r", encoding="utf-8") as rf:
        return json.load(rf)


def normalize_metrics(metrics):
    if metrics is None:
        return None
    normalized = {key: float(value) for key, value in metrics.items()}
    if "auc" in normalized and "auroc" not in normalized:
        normalized["auroc"] = normalized["auc"]
    if "aupr" in normalized and "auprc" not in normalized:
        normalized["auprc"] = normalized["aupr"]
    if "acc" in normalized and "accuracy" not in normalized:
        normalized["accuracy"] = normalized["acc"]
    return normalized


def metric_to_score(metrics, metric_name):
    if metric_name not in metrics:
        raise KeyError(f"Unknown best metric '{metric_name}'. Available metrics: {sorted(metrics.keys())}")
    value = float(metrics[metric_name])
    return -value if metric_name == "loss" else value


def broadcast_object(obj, src=0):
    if not is_dist_ready():
        return obj

    object_list = [obj if get_rank() == src else None]
    dist.broadcast_object_list(object_list, src=src)
    return object_list[0]


def build_client_state(
    epoch,
    best_score,
    best_metric,
    best_epoch,
    best_val_metrics,
    best_test_metrics,
    training_state=None,
):
    client_state = {
        "epoch": int(epoch),
        "best_score": float(best_score),
        "best_metric": str(best_metric),
        "best_epoch": int(best_epoch),
        "best_val_metrics": normalize_metrics(best_val_metrics),
        "best_test_metrics": normalize_metrics(best_test_metrics),
    }
    if best_val_metrics is not None and "auc" in best_val_metrics:
        client_state["best_val_auc"] = float(best_val_metrics["auc"])
    if training_state is not None:
        client_state["training_state"] = training_state
    return client_state


def ensure_checkpoint_dirs(save_dir):
    os.makedirs(save_dir, exist_ok=True)


def resolve_resume_tag(resume_tag):
    if resume_tag in (None, "", "latest"):
        return None
    return resume_tag


def get_epoch_checkpoint_tag(epoch):
    return f"epoch_{epoch:04d}"


def prune_epoch_checkpoints(save_dir, keep_tags):
    keep_tags = {str(tag) for tag in keep_tags if tag and re.fullmatch(r"epoch_\d{4}", str(tag))}
    removed = []
    if not os.path.isdir(save_dir):
        return removed

    for name in os.listdir(save_dir):
        if name in keep_tags or not re.fullmatch(r"epoch_\d{4}", name):
            continue
        checkpoint_dir = os.path.join(save_dir, name)
        if os.path.isdir(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
            removed.append(name)

    return removed


def normalize_experiment_name(name):
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(name).strip())
    normalized = re.sub(r"-+", "-", normalized).strip(".-")
    if not normalized:
        raise ValueError("swanlab experiment name is empty after normalization")
    return normalized


def build_default_experiment_name(args, cfg):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_tag = normalize_experiment_name(args.data.replace("/", "_"))
    fusion_tag = normalize_experiment_name(cfg.CROSSINTENTION.FUSION_MODE)
    metric_tag = normalize_experiment_name(cfg.SOLVER.BEST_METRIC)
    return f"formal-{dataset_tag}-{args.split}-{fusion_tag}-best_{metric_tag}-{timestamp}"


def move_batch_to_device(batch, device):
    feature_vectors, feature, coor, conf_mask, energy, bg_d, v_p, protein_mask, y = batch
    feature_vectors = feature_vectors.to(device, non_blocking=True)
    feature = feature.to(device, non_blocking=True)
    coor = coor.to(device, non_blocking=True)
    conf_mask = conf_mask.to(device, non_blocking=True)
    energy = energy.to(device, non_blocking=True)
    bg_d = bg_d.to(device)
    v_p = v_p.to(device, non_blocking=True)
    protein_mask = protein_mask.to(device, non_blocking=True)
    y = y.float().to(device, non_blocking=True)
    return feature_vectors, feature, coor, conf_mask, energy, bg_d, v_p, protein_mask, y


def build_dataloader(
    dataset,
    batch_size,
    num_workers,
    shuffle,
    drop_last,
    sampler=None,
    persistent_workers=False,
):
    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = 2

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle if sampler is None else False),
        sampler=sampler,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=graph_collate_func,
        pin_memory=torch.cuda.is_available(),
        **loader_kwargs,
    )


class DistributedEvalSampler(Sampler):
    def __init__(self, dataset, num_replicas, rank):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self):
        if self.rank >= len(self.dataset):
            return 0
        return (len(self.dataset) - 1 - self.rank) // self.num_replicas + 1


def reduce_average(loss_sum, sample_count, device):
    stats = torch.tensor([loss_sum, sample_count], dtype=torch.float64, device=device)
    if is_dist_ready():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    global_loss_sum, global_sample_count = stats.tolist()
    return global_loss_sum / max(1.0, global_sample_count)


def find_best_threshold(labels, preds, metric_name="acc"):
    if len(labels) == 0:
        return 0.5, 0.0

    labels = labels.astype(int)
    thresholds = np.linspace(0.01, 0.99, 99)
    pred_labels = (preds[:, None] >= thresholds[None, :]).astype(int)
    metric_name = "acc" if metric_name in {"accuracy", "acc_05"} else metric_name
    metric_name = "f1" if metric_name == "f1_05" else metric_name

    if metric_name == "f1":
        scores = np.array([f1_score(labels, pred_labels[:, idx], zero_division=0) for idx in range(len(thresholds))])
    else:
        scores = (pred_labels == labels[:, None]).mean(axis=0)

    best_index = int(np.argmax(scores))
    return float(thresholds[best_index]), float(scores[best_index])


def get_optimizer_param_groups(optimizer):
    current = optimizer
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        param_groups = getattr(current, "param_groups", None)
        if param_groups is not None:
            return param_groups
        current = getattr(current, "optimizer", None)
    return None


def get_current_lr(optimizer, default_lr):
    param_groups = get_optimizer_param_groups(optimizer)
    if not param_groups:
        return float(default_lr)
    return float(param_groups[0]["lr"])


def set_optimizer_lr(optimizer, lr):
    param_groups = get_optimizer_param_groups(optimizer)
    if not param_groups:
        return
    for group in param_groups:
        group["lr"] = float(lr)


def compute_pos_weight(df_train):
    labels = df_train["Y"].astype(int)
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    if positives == 0:
        return 1.0
    return negatives / positives


@torch.no_grad()
def evaluate(model_engine, dataloader, device, threshold=0.5, tune_threshold=False, threshold_metric="acc"):
    model_engine.eval()
    all_labels = []
    all_preds = []
    total_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        feature_vectors, feature, coor, conf_mask, energy, bg_d, v_p, protein_mask, y = move_batch_to_device(batch, device)
        _, _, score, _ = model_engine(
            feature_vectors, feature, coor, conf_mask, energy, bg_d, v_p, protein_mask, mode="eval"
        )
        pred, loss = binary_cross_entropy(score, y)

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_labels.extend(y.detach().cpu().numpy().tolist())
        all_preds.extend(pred.detach().cpu().numpy().tolist())

    loss_stats = torch.tensor([total_loss, total_samples], dtype=torch.float64, device=device)
    if is_dist_ready():
        dist.all_reduce(loss_stats, op=dist.ReduceOp.SUM)

    metrics = {
        "loss": loss_stats[0].item() / max(1.0, loss_stats[1].item()),
        "auc": 0.0,
        "auroc": 0.0,
        "aupr": 0.0,
        "auprc": 0.0,
        "acc": 0.0,
        "accuracy": 0.0,
        "acc_05": 0.0,
        "f1": 0.0,
        "f1_05": 0.0,
        "threshold": float(threshold),
        "pred_mean": 0.0,
        "pred_pos_rate": 0.0,
        "pred_pos_rate_05": 0.0,
    }

    if is_dist_ready():
        gathered_labels = [None for _ in range(dist.get_world_size())]
        gathered_preds = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered_labels, all_labels)
        dist.all_gather_object(gathered_preds, all_preds)
        if is_main_process():
            all_labels = [item for part in gathered_labels for item in part]
            all_preds = [item for part in gathered_preds for item in part]

    if is_main_process():
        labels = np.array(all_labels).astype(int)
        preds = np.array(all_preds)
        selected_threshold = float(threshold)
        if tune_threshold:
            selected_threshold, _ = find_best_threshold(labels, preds, threshold_metric)

        pred_labels = (preds >= selected_threshold).astype(int)
        pred_labels_05 = (preds >= 0.5).astype(int)
        metrics["threshold"] = selected_threshold

        if len(labels) > 0 and len(np.unique(labels)) > 1:
            metrics["auc"] = roc_auc_score(labels, preds)
            metrics["auroc"] = metrics["auc"]
            metrics["aupr"] = average_precision_score(labels, preds)
            metrics["auprc"] = metrics["aupr"]

        if len(labels) > 0:
            metrics["acc"] = accuracy_score(labels, pred_labels)
            metrics["accuracy"] = metrics["acc"]
            metrics["acc_05"] = accuracy_score(labels, pred_labels_05)
            metrics["f1"] = f1_score(labels, pred_labels, zero_division=0)
            metrics["f1_05"] = f1_score(labels, pred_labels_05, zero_division=0)
            metrics["pred_mean"] = float(preds.mean())
            metrics["pred_pos_rate"] = float(pred_labels.mean())
            metrics["pred_pos_rate_05"] = float(pred_labels_05.mean())
    return broadcast_object(metrics)


def save_text_outputs(save_dir, model_engine, cfg, best_metric, best_score, best_epoch, best_val_metrics, best_test_metrics):
    model_to_save = model_engine.module if hasattr(model_engine, "module") else model_engine

    with open(os.path.join(save_dir, "model_architecture.txt"), "w") as wf:
        wf.write(str(model_to_save))

    with open(os.path.join(save_dir, "config.txt"), "w") as wf:
        wf.write(str(dict(cfg)))

    with open(os.path.join(save_dir, "best_metrics.txt"), "w") as wf:
        wf.write(f"best_metric={best_metric}\n")
        wf.write(f"best_score={best_score}\n")
        wf.write(f"best_epoch={best_epoch}\n")
        wf.write(f"best_val={best_val_metrics}\n")
        wf.write(f"best_test={best_test_metrics}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="LDMDTI training with DeepSpeed ZeRO-2")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--data", type=str, required=True, help="dataset path under data/")
    parser.add_argument(
        "--split",
        type=str,
        default="random",
        choices=["random", "random1", "random2", "random3", "random4"],
        help="dataset split name",
    )
    parser.add_argument("--config", type=str, default=None, help="experiment config yaml under configs/")
    parser.add_argument("--resume", action="store_true", help="resume training from a DeepSpeed checkpoint")
    parser.add_argument("--resume_dir", type=str, default=None, help="checkpoint directory to resume from")
    parser.add_argument("--resume_tag", type=str, default="latest", help="checkpoint tag to resume from")
    parser.add_argument("--save_dir", type=str, default=None, help="override checkpoint/output directory")
    parser.add_argument(
        "--warm_start",
        action="store_true",
        help="load matching model weights from a checkpoint but start a new training run",
    )
    parser.add_argument("--warm_start_dir", type=str, default=None, help="checkpoint directory for warm start")
    parser.add_argument("--warm_start_tag", type=str, default="latest", help="checkpoint tag for warm start")
    parser.add_argument(
        "--drug_fusion_gate_bias",
        type=float,
        default=None,
        help="override DRUG.FUSION_GATE_BIAS for 1D vs 2D/3D fusion initialization",
    )
    parser.add_argument(
        "--fusion_mode",
        type=str,
        default=None,
        choices=[
            "biintention",
            "interaction_field",
            "field_enhanced_bi",
            "mamba",
            "mamba_enhanced_bi",
            "protein_mamba_biintention",
            "protein_mamba_direct_gate_biintention",
        ],
        help="override CROSSINTENTION.FUSION_MODE",
    )
    parser.add_argument(
        "--field_enhance_gate_bias",
        type=float,
        default=None,
        help="override CROSSINTENTION.FIELD_ENHANCE_GATE_BIAS for field residual initialization",
    )
    parser.add_argument(
        "--cf_start_epoch",
        type=int,
        default=None,
        help="override CROSSINTENTION.CF_START_EPOCH",
    )
    parser.add_argument(
        "--cf_key_weight",
        type=float,
        default=None,
        help="override CROSSINTENTION.CF_KEY_WEIGHT",
    )
    parser.add_argument(
        "--cf_stable_weight",
        type=float,
        default=None,
        help="override CROSSINTENTION.CF_STABLE_WEIGHT",
    )
    parser.add_argument(
        "--field_entropy_weight",
        type=float,
        default=None,
        help="override CROSSINTENTION.FIELD_ENTROPY_WEIGHT",
    )
    parser.add_argument(
        "--best_metric",
        type=str,
        default=None,
        choices=["auc", "auroc", "aupr", "auprc", "acc", "accuracy", "acc_05", "f1", "f1_05", "loss"],
        help="override SOLVER.BEST_METRIC for checkpoint selection and early stopping",
    )
    parser.add_argument("--cache_root", type=str, default="cache/features", help="feature cache root directory")
    parser.add_argument("--swanlab_project", type=str, default="LDM-DTI", help="SwanLab project name")
    parser.add_argument(
        "--swanlab_experiment",
        type=str,
        default=None,
        help="SwanLab experiment name; also used as the local swanlog subdirectory name",
    )
    parser.add_argument(
        "--swanlab_log_root",
        type=str,
        default=DEFAULT_SWANLAB_LOG_ROOT,
        help="root directory for SwanLab logs; each run uses <root>/<swanlab_experiment>",
    )
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    if not getattr(args, "deepspeed_config", None):
        args.deepspeed_config = DEFAULT_DEEPSPEED_CONFIG
    return args


def main():
    args = parse_args()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    cfg = get_cfg_defaults()
    if args.config is not None:
        cfg.merge_from_file(args.config)
    if args.drug_fusion_gate_bias is not None:
        cfg.DRUG.FUSION_GATE_BIAS = float(args.drug_fusion_gate_bias)
    if args.fusion_mode is not None:
        cfg.CROSSINTENTION.FUSION_MODE = args.fusion_mode
    if args.field_enhance_gate_bias is not None:
        cfg.CROSSINTENTION.FIELD_ENHANCE_GATE_BIAS = float(args.field_enhance_gate_bias)
    if args.cf_start_epoch is not None:
        cfg.CROSSINTENTION.CF_START_EPOCH = int(args.cf_start_epoch)
    if args.cf_key_weight is not None:
        cfg.CROSSINTENTION.CF_KEY_WEIGHT = float(args.cf_key_weight)
    if args.cf_stable_weight is not None:
        cfg.CROSSINTENTION.CF_STABLE_WEIGHT = float(args.cf_stable_weight)
    if args.field_entropy_weight is not None:
        cfg.CROSSINTENTION.FIELD_ENTROPY_WEIGHT = float(args.field_entropy_weight)
    if args.best_metric is not None:
        cfg.SOLVER.BEST_METRIC = args.best_metric
    set_seed(cfg.SOLVER.SEED)
    device = setup_device()
    ds_config = load_json_config(args.deepspeed_config)

    data_dir = os.path.join("data", args.data, args.split)
    cache_dir = os.path.join(args.cache_root, args.data, args.split)
    save_dir = args.save_dir or os.path.join(cfg.RESULT.OUTPUT_DIR, args.data, args.split)
    ensure_checkpoint_dirs(save_dir)

    swanlab_run = None

    if is_main_process():
        print(f"start time: {datetime.now()}")
        print("start...")
        print(f"dataset: {args.data}")
        print(f"split: {args.split}")
        print(f"device: {device}")
        print(f"data_dir: {data_dir}")
        print(f"cache_dir: {cache_dir}")
        print(f"save_dir: {save_dir}")
        print(f"config: {args.config}")
        print(f"deepspeed_config: {args.deepspeed_config}")
        print(f"best_metric: val/{cfg.SOLVER.BEST_METRIC}")
        print(
            "v1 tuned params: "
            f"fusion_gate_bias={cfg.DRUG.FUSION_GATE_BIAS}, "
            f"fusion_mode={cfg.CROSSINTENTION.FUSION_MODE}, "
            f"field_enhance_gate_bias={cfg.CROSSINTENTION.FIELD_ENHANCE_GATE_BIAS}, "
            f"mamba_enhance_gate_bias={cfg.CROSSINTENTION.MAMBA_ENHANCE_GATE_BIAS}, "
            f"mamba_gate_hidden_dim={cfg.CROSSINTENTION.MAMBA_GATE_HIDDEN_DIM}, "
            f"mamba_gate_delta_scale={cfg.CROSSINTENTION.MAMBA_GATE_DELTA_SCALE}, "
            f"mamba_gate_detach_context={cfg.CROSSINTENTION.MAMBA_GATE_DETACH_CONTEXT}, "
            f"mamba_gate_bounded={cfg.CROSSINTENTION.MAMBA_GATE_BOUNDED}, "
            f"cf_start_epoch={cfg.CROSSINTENTION.CF_START_EPOCH}, "
            f"cf_key_weight={cfg.CROSSINTENTION.CF_KEY_WEIGHT}, "
            f"cf_stable_weight={cfg.CROSSINTENTION.CF_STABLE_WEIGHT}, "
            f"field_entropy_weight={cfg.CROSSINTENTION.FIELD_ENTROPY_WEIGHT}"
        )

        raw_experiment_name = args.swanlab_experiment or build_default_experiment_name(args, cfg)
        experiment_name = normalize_experiment_name(raw_experiment_name)
        if args.swanlab_experiment and experiment_name != args.swanlab_experiment:
            print(f"swanlab experiment name normalized: {args.swanlab_experiment} -> {experiment_name}")
        swanlab_log_dir = os.path.join(args.swanlab_log_root, experiment_name)
        print(f"swanlab_experiment: {experiment_name}")
        print(f"swanlab_log_dir: {swanlab_log_dir}")
        swanlab_run = swanlab.init(
            project=args.swanlab_project,
            experiment_name=experiment_name,
            logdir=swanlab_log_dir,
            config={
                "dataset": args.data,
                "split": args.split,
                "config": args.config,
                "cache_dir": cache_dir,
                "swanlab_experiment": experiment_name,
                "swanlab_log_dir": swanlab_log_dir,
                "learning_rate": cfg.SOLVER.LR,
                "batch_size_per_gpu": cfg.SOLVER.BATCH_SIZE,
                "max_epoch": cfg.SOLVER.MAX_EPOCH,
                "num_workers": cfg.SOLVER.NUM_WORKERS,
                "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
                "lr_scheduler": cfg.SOLVER.LR_SCHEDULER,
                "lr_decay": cfg.SOLVER.LR_DECAY,
                "lr_patience": cfg.SOLVER.LR_PATIENCE,
                "min_lr": cfg.SOLVER.MIN_LR,
                "early_stop_patience": cfg.SOLVER.EARLY_STOP_PATIENCE,
                "best_metric": cfg.SOLVER.BEST_METRIC,
                "min_delta": cfg.SOLVER.MIN_DELTA,
                "use_pos_weight": cfg.SOLVER.USE_POS_WEIGHT,
                "pos_weight": cfg.SOLVER.POS_WEIGHT,
                "drug_fusion_gate_bias": cfg.DRUG.FUSION_GATE_BIAS,
                "fusion_mode": cfg.CROSSINTENTION.FUSION_MODE,
                "field_enhance_gate_bias": cfg.CROSSINTENTION.FIELD_ENHANCE_GATE_BIAS,
                "mamba_enhance_gate_bias": cfg.CROSSINTENTION.MAMBA_ENHANCE_GATE_BIAS,
                "mamba_gate_hidden_dim": cfg.CROSSINTENTION.MAMBA_GATE_HIDDEN_DIM,
                "mamba_gate_delta_scale": cfg.CROSSINTENTION.MAMBA_GATE_DELTA_SCALE,
                "mamba_gate_detach_context": cfg.CROSSINTENTION.MAMBA_GATE_DETACH_CONTEXT,
                "mamba_gate_bounded": cfg.CROSSINTENTION.MAMBA_GATE_BOUNDED,
                "cf_start_epoch": cfg.CROSSINTENTION.CF_START_EPOCH,
                "cf_key_weight": cfg.CROSSINTENTION.CF_KEY_WEIGHT,
                "cf_stable_weight": cfg.CROSSINTENTION.CF_STABLE_WEIGHT,
                "field_entropy_weight": cfg.CROSSINTENTION.FIELD_ENTROPY_WEIGHT,
                "world_size": int(os.environ.get("WORLD_SIZE", "1")),
                "gradient_accumulation_steps": ds_config.get("gradient_accumulation_steps", 1),
                "train_micro_batch_size_per_gpu": ds_config.get("train_micro_batch_size_per_gpu", cfg.SOLVER.BATCH_SIZE),
                "deepspeed_config": args.deepspeed_config,
            },
        )

    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    test_path = os.path.join(data_dir, "test.csv")

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    pos_weight_value = None
    if cfg.SOLVER.USE_POS_WEIGHT:
        pos_weight_value = float(cfg.SOLVER.POS_WEIGHT)
        if pos_weight_value <= 0:
            pos_weight_value = compute_pos_weight(df_train)

    if is_main_process():
        print(f"train_pos_weight: {pos_weight_value if pos_weight_value is not None else 'disabled'}")

    train_dataset = DTIDataset(df_train.index.values, df_train, cache_dir=cache_dir)
    val_dataset = DTIDataset(df_val.index.values, df_val, cache_dir=cache_dir)
    test_dataset = DTIDataset(df_test.index.values, df_test, cache_dir=cache_dir)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    train_sampler = None
    val_sampler = None
    test_sampler = None
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        val_sampler = DistributedEvalSampler(val_dataset, num_replicas=world_size, rank=rank)
        test_sampler = DistributedEvalSampler(test_dataset, num_replicas=world_size, rank=rank)

    train_loader = build_dataloader(
        train_dataset,
        batch_size=cfg.SOLVER.BATCH_SIZE,
        num_workers=cfg.SOLVER.NUM_WORKERS,
        shuffle=True,
        drop_last=True,
        sampler=train_sampler,
        persistent_workers=True,
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=cfg.SOLVER.BATCH_SIZE,
        num_workers=cfg.SOLVER.NUM_WORKERS,
        shuffle=False,
        drop_last=False,
        sampler=val_sampler,
    )
    test_loader = build_dataloader(
        test_dataset,
        batch_size=cfg.SOLVER.BATCH_SIZE,
        num_workers=cfg.SOLVER.NUM_WORKERS,
        shuffle=False,
        drop_last=False,
        sampler=test_sampler,
    )

    model = BINDTI(device=device, **cfg)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.SOLVER.LR,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )

    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer,
        model_parameters=model.parameters(),
    )
    train_pos_weight = None
    if pos_weight_value is not None:
        train_pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=model_engine.device)

    best_metric = cfg.SOLVER.BEST_METRIC
    best_score = -float("inf")
    best_epoch = -1
    best_val_metrics = None
    best_test_metrics = None
    start_epoch = 0
    current_lr = get_current_lr(model_engine.optimizer, cfg.SOLVER.LR)
    early_stop_count = 0
    lr_bad_epochs = 0

    if args.resume and args.warm_start:
        raise ValueError("--resume and --warm_start are mutually exclusive")

    if args.warm_start:
        warm_start_dir = args.warm_start_dir or save_dir
        warm_start_tag = resolve_resume_tag(args.warm_start_tag)
        load_path, _ = model_engine.load_checkpoint(
            warm_start_dir,
            tag=warm_start_tag,
            load_module_strict=False,
            load_optimizer_states=False,
            load_lr_scheduler_states=False,
        )
        if load_path is None:
            raise FileNotFoundError(
                f"Failed to warm start from {warm_start_dir} with tag '{args.warm_start_tag}'."
            )
        if is_main_process():
            print(f"warm_start checkpoint: {load_path}")
            print("warm_start mode: loaded matching model weights only; optimizer and epoch state reset")

    if args.resume:
        resume_dir = args.resume_dir or save_dir
        resume_tag = resolve_resume_tag(args.resume_tag)
        try:
            load_path, client_state = model_engine.load_checkpoint(resume_dir, tag=resume_tag)
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to resume checkpoint strictly. This usually means the checkpoint was "
                "created by a different model architecture. Start a new run without --resume, "
                "or use --warm_start to load only matching model weights."
            ) from exc
        if load_path is None:
            raise FileNotFoundError(
                f"Failed to load checkpoint from {resume_dir} with tag '{args.resume_tag}'."
            )

        client_state = client_state or {}
        start_epoch = int(client_state.get("epoch", 0))
        best_metric = str(client_state.get("best_metric", best_metric))
        best_score = float(client_state.get("best_score", client_state.get("best_val_auc", -float("inf"))))
        best_epoch = int(client_state.get("best_epoch", -1))
        best_val_metrics = normalize_metrics(client_state.get("best_val_metrics"))
        best_test_metrics = normalize_metrics(client_state.get("best_test_metrics"))
        training_state = client_state.get("training_state", {})
        current_lr = get_current_lr(
            model_engine.optimizer,
            float(training_state.get("current_lr", current_lr)),
        )
        early_stop_count = int(training_state.get("early_stop_count", 0))
        lr_bad_epochs = int(training_state.get("lr_bad_epochs", 0))
        set_optimizer_lr(model_engine.optimizer, current_lr)

        if is_main_process():
            print(f"resume checkpoint: {load_path}")
            print(f"resume from epoch: {start_epoch}")
            print(f"resume best_metric: val/{best_metric}")
            print(f"resume best_score: {best_score:.4f}")
            print(f"resume lr: {current_lr:.6g}")

    start_time = time()

    for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        epoch_lr = current_lr
        epoch_start_time = time()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model_engine.train()
        epoch_loss_sum = 0.0
        epoch_bce_objective_sum = 0.0
        epoch_objective_sum = 0.0
        epoch_cf_loss_sum = 0.0
        epoch_cf_key_raw_sum = 0.0
        epoch_cf_key_weighted_sum = 0.0
        epoch_cf_stable_raw_sum = 0.0
        epoch_cf_stable_weighted_sum = 0.0
        epoch_cf_entropy_raw_sum = 0.0
        epoch_cf_entropy_weighted_sum = 0.0
        epoch_sample_count = 0

        for batch in train_loader:
            feature_vectors, feature, coor, conf_mask, energy, bg_d, v_p, protein_mask, y = move_batch_to_device(
                batch, model_engine.device
            )

            score, aux = model_engine(
                feature_vectors,
                feature,
                coor,
                conf_mask,
                energy,
                bg_d,
                v_p,
                protein_mask,
                mode="train",
                return_aux=True,
            )
            _, bce_objective = binary_cross_entropy(score, y, pos_weight=train_pos_weight)
            objective = bce_objective
            cf_loss = score.new_tensor(0.0)
            cf_parts = {
                "cf_key_raw": score.new_tensor(0.0),
                "cf_key_weighted": score.new_tensor(0.0),
                "cf_stable_raw": score.new_tensor(0.0),
                "cf_stable_weighted": score.new_tensor(0.0),
                "cf_entropy_raw": score.new_tensor(0.0),
                "cf_entropy_weighted": score.new_tensor(0.0),
            }
            if epoch + 1 >= cfg.CROSSINTENTION.CF_START_EPOCH:
                model_for_loss = model_engine.module if hasattr(model_engine, "module") else model_engine
                cf_loss, cf_parts = model_for_loss.counterfactual_loss(
                    score,
                    aux,
                    y,
                    return_parts=True,
                )
                objective = objective + cf_loss
                # objective =
                #     BCEWithLogitsLoss(pos_weight)
                # + CF_KEY_WEIGHT * cf_key_raw
                # + CF_STABLE_WEIGHT * cf_stable_raw
                # + FIELD_ENTROPY_WEIGHT * cf_entropy_raw
            with torch.no_grad():
                _, loss = binary_cross_entropy(score, y)

            model_engine.backward(objective)
            model_engine.step()

            batch_size = y.size(0)
            epoch_loss_sum += loss.item() * batch_size
            epoch_bce_objective_sum += bce_objective.item() * batch_size
            epoch_objective_sum += objective.item() * batch_size
            epoch_cf_loss_sum += cf_loss.detach().item() * batch_size
            epoch_cf_key_raw_sum += cf_parts["cf_key_raw"].item() * batch_size
            epoch_cf_key_weighted_sum += cf_parts["cf_key_weighted"].item() * batch_size
            epoch_cf_stable_raw_sum += cf_parts["cf_stable_raw"].item() * batch_size
            epoch_cf_stable_weighted_sum += cf_parts["cf_stable_weighted"].item() * batch_size
            epoch_cf_entropy_raw_sum += cf_parts["cf_entropy_raw"].item() * batch_size
            epoch_cf_entropy_weighted_sum += cf_parts["cf_entropy_weighted"].item() * batch_size
            epoch_sample_count += batch_size

        train_elapsed = time() - epoch_start_time
        avg_train_loss = reduce_average(epoch_loss_sum, epoch_sample_count, model_engine.device)
        avg_train_bce_objective = reduce_average(
            epoch_bce_objective_sum,
            epoch_sample_count,
            model_engine.device,
        )
        avg_train_objective = reduce_average(epoch_objective_sum, epoch_sample_count, model_engine.device)
        avg_train_cf_loss = reduce_average(epoch_cf_loss_sum, epoch_sample_count, model_engine.device)
        avg_train_cf_key_raw = reduce_average(epoch_cf_key_raw_sum, epoch_sample_count, model_engine.device)
        avg_train_cf_key_weighted = reduce_average(
            epoch_cf_key_weighted_sum,
            epoch_sample_count,
            model_engine.device,
        )
        avg_train_cf_stable_raw = reduce_average(
            epoch_cf_stable_raw_sum,
            epoch_sample_count,
            model_engine.device,
        )
        avg_train_cf_stable_weighted = reduce_average(
            epoch_cf_stable_weighted_sum,
            epoch_sample_count,
            model_engine.device,
        )
        avg_train_cf_entropy_raw = reduce_average(
            epoch_cf_entropy_raw_sum,
            epoch_sample_count,
            model_engine.device,
        )
        avg_train_cf_entropy_weighted = reduce_average(
            epoch_cf_entropy_weighted_sum,
            epoch_sample_count,
            model_engine.device,
        )
        val_start_time = time()
        val_metrics = evaluate(
            model_engine,
            val_loader,
            model_engine.device,
            tune_threshold=True,
            threshold_metric=best_metric,
        )
        val_elapsed = time() - val_start_time
        test_start_time = time()
        test_metrics = evaluate(
            model_engine,
            test_loader,
            model_engine.device,
            threshold=val_metrics["threshold"],
        )
        test_elapsed = time() - test_start_time

        should_save = torch.tensor([0], dtype=torch.int32, device=model_engine.device)
        control_state = None

        if is_main_process():
            val_score = metric_to_score(val_metrics, best_metric)
            improved = val_score > best_score + cfg.SOLVER.MIN_DELTA
            if improved:
                best_score = val_score
                best_epoch = epoch + 1
                best_val_metrics = normalize_metrics(val_metrics)
                best_test_metrics = normalize_metrics(test_metrics)
                early_stop_count = 0
                lr_bad_epochs = 0
                should_save[0] = 1
            else:
                early_stop_count += 1
                lr_bad_epochs += 1

            next_lr = current_lr
            lr_reduced = False
            if (
                cfg.SOLVER.LR_SCHEDULER == "plateau"
                and cfg.SOLVER.LR_PATIENCE > 0
                and lr_bad_epochs >= cfg.SOLVER.LR_PATIENCE
            ):
                next_lr = max(current_lr * cfg.SOLVER.LR_DECAY, cfg.SOLVER.MIN_LR)
                if next_lr < current_lr:
                    lr_bad_epochs = 0
                    lr_reduced = True

            stop_training = (
                cfg.SOLVER.EARLY_STOP_PATIENCE > 0
                and early_stop_count >= cfg.SOLVER.EARLY_STOP_PATIENCE
            )

            print(
                f"Epoch [{epoch + 1}/{cfg.SOLVER.MAX_EPOCH}] "
                f"lr={epoch_lr:.6g} "
                f"train_loss={avg_train_loss:.4f} "
                f"train_bce_obj={avg_train_bce_objective:.4f} "
                f"train_cf={avg_train_cf_loss:.4f} "
                f"train_obj={avg_train_objective:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_auc={val_metrics['auc']:.4f} "
                f"val_aupr={val_metrics['aupr']:.4f} "
                f"val_acc={val_metrics['acc']:.4f} "
                f"val_f1={val_metrics['f1']:.4f} "
                f"val_acc@0.5={val_metrics['acc_05']:.4f} "
                f"val_f1@0.5={val_metrics['f1_05']:.4f} "
                f"val_thr={val_metrics['threshold']:.2f} "
                f"test_loss={test_metrics['loss']:.4f} "
                f"test_auc={test_metrics['auc']:.4f} "
                f"test_aupr={test_metrics['aupr']:.4f} "
                f"test_acc={test_metrics['acc']:.4f} "
                f"test_f1={test_metrics['f1']:.4f} "
                f"test_acc@0.5={test_metrics['acc_05']:.4f} "
                f"test_f1@0.5={test_metrics['f1_05']:.4f} "
                f"early_stop={early_stop_count}/{cfg.SOLVER.EARLY_STOP_PATIENCE}"
            )

            if lr_reduced:
                print(f"reduce lr: {current_lr:.6g} -> {next_lr:.6g}")
            if stop_training:
                print(f"early stopping at epoch {epoch + 1}")

            swanlab.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": avg_train_loss,
                    "train/bce_objective": avg_train_bce_objective,
                    "train/cf_loss": avg_train_cf_loss,
                    "train/cf_key_raw": avg_train_cf_key_raw,
                    "train/cf_key_weighted": avg_train_cf_key_weighted,
                    "train/cf_stable_raw": avg_train_cf_stable_raw,
                    "train/cf_stable_weighted": avg_train_cf_stable_weighted,
                    "train/cf_entropy_raw": avg_train_cf_entropy_raw,
                    "train/cf_entropy_weighted": avg_train_cf_entropy_weighted,
                    "train/objective": avg_train_objective,
                    "train/lr": epoch_lr,
                    "train/next_lr": next_lr,
                    "train/pos_weight": pos_weight_value or 1.0,
                    "val/loss": val_metrics["loss"],
                    "val/auc": val_metrics["auc"],
                    "val/auroc": val_metrics["auroc"],
                    "val/aupr": val_metrics["aupr"],
                    "val/auprc": val_metrics["auprc"],
                    "val/acc": val_metrics["acc"],
                    "val/accuracy": val_metrics["accuracy"],
                    "val/acc_05": val_metrics["acc_05"],
                    "val/f1": val_metrics["f1"],
                    "val/f1_05": val_metrics["f1_05"],
                    "val/threshold": val_metrics["threshold"],
                    "val/pred_mean": val_metrics["pred_mean"],
                    "val/pred_pos_rate": val_metrics["pred_pos_rate"],
                    "val/pred_pos_rate_05": val_metrics["pred_pos_rate_05"],
                    "test/loss": test_metrics["loss"],
                    "test/auc": test_metrics["auc"],
                    "test/auroc": test_metrics["auroc"],
                    "test/aupr": test_metrics["aupr"],
                    "test/auprc": test_metrics["auprc"],
                    "test/acc": test_metrics["acc"],
                    "test/accuracy": test_metrics["accuracy"],
                    "test/acc_05": test_metrics["acc_05"],
                    "test/f1": test_metrics["f1"],
                    "test/f1_05": test_metrics["f1_05"],
                    "test/threshold": test_metrics["threshold"],
                    "test/pred_mean": test_metrics["pred_mean"],
                    "test/pred_pos_rate": test_metrics["pred_pos_rate"],
                    "test/pred_pos_rate_05": test_metrics["pred_pos_rate_05"],
                    "best/score": best_score,
                    "best/val_auc": best_val_metrics.get("auc", -1.0) if best_val_metrics is not None else -1.0,
                    "best/val_auroc": best_val_metrics.get("auroc", -1.0) if best_val_metrics is not None else -1.0,
                    "best/val_auprc": best_val_metrics.get("auprc", -1.0) if best_val_metrics is not None else -1.0,
                    "best/val_accuracy": best_val_metrics.get("accuracy", -1.0) if best_val_metrics is not None else -1.0,
                    "best/val_f1": best_val_metrics.get("f1", -1.0) if best_val_metrics is not None else -1.0,
                    "best/epoch": best_epoch,
                    "control/early_stop_count": early_stop_count,
                    "control/lr_bad_epochs": lr_bad_epochs,
                }
            )

            control_state = {
                "best_score": best_score,
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "best_val_metrics": best_val_metrics,
                "best_test_metrics": best_test_metrics,
                "current_lr": next_lr,
                "early_stop_count": early_stop_count,
                "lr_bad_epochs": lr_bad_epochs,
                "stop_training": stop_training,
            }

        control_state = broadcast_object(control_state)
        best_score = control_state["best_score"]
        best_metric = control_state["best_metric"]
        best_epoch = control_state["best_epoch"]
        best_val_metrics = control_state["best_val_metrics"]
        best_test_metrics = control_state["best_test_metrics"]
        current_lr = control_state["current_lr"]
        early_stop_count = control_state["early_stop_count"]
        lr_bad_epochs = control_state["lr_bad_epochs"]
        stop_training = control_state["stop_training"]
        set_optimizer_lr(model_engine.optimizer, current_lr)
        training_state = {
            "current_lr": current_lr,
            "early_stop_count": early_stop_count,
            "lr_bad_epochs": lr_bad_epochs,
            "last_val_threshold": val_metrics["threshold"],
            "pos_weight": pos_weight_value,
        }
        client_state = build_client_state(
            epoch + 1,
            best_score,
            best_metric,
            best_epoch,
            best_val_metrics,
            best_test_metrics,
            training_state=training_state,
        )

        if is_dist_ready():
            dist.broadcast(should_save, src=0)

        save_start_time = time()
        latest_tag = get_epoch_checkpoint_tag(epoch + 1)
        best_tag = get_epoch_checkpoint_tag(best_epoch) if best_epoch > 0 else None
        model_engine.save_checkpoint(
            save_dir,
            tag=latest_tag,
            client_state=client_state,
            save_latest=True,
        )

        if is_dist_ready():
            dist.barrier()

        removed_checkpoints = []
        if is_main_process():
            removed_checkpoints = prune_epoch_checkpoints(save_dir, {latest_tag, best_tag})

        if is_dist_ready():
            dist.barrier()

        save_elapsed = time() - save_start_time

        if is_main_process():
            checkpoint_msg = "saved_best_and_last" if should_save.item() == 1 else "saved_last"
            if removed_checkpoints:
                checkpoint_msg += f", pruned={len(removed_checkpoints)}"
            print(
                f"Epoch time: train={train_elapsed:.2f}s "
                f"val={val_elapsed:.2f}s "
                f"test={test_elapsed:.2f}s "
                f"checkpoint={checkpoint_msg}/{save_elapsed:.2f}s "
                f"total={time() - epoch_start_time:.2f}s"
            )

        if stop_training:
            break

    if is_main_process():
        save_text_outputs(
            save_dir,
            model_engine,
            cfg,
            best_metric,
            best_score,
            best_epoch,
            best_val_metrics,
            best_test_metrics,
        )
        elapsed = time() - start_time
        print(f"best_metric: val/{best_metric}")
        print(f"best_score: {best_score}")
        print(f"best_epoch: {best_epoch}")
        print(f"best_val: {best_val_metrics}")
        print(f"best_test: {best_test_metrics}")
        print(f"end time: {datetime.now()}")
        print(f"Total running time: {round(elapsed, 2)}s")
        if swanlab_run is not None:
            swanlab.finish()

    return best_test_metrics


if __name__ == "__main__":
    main()
