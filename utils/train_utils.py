import os
from omegaconf import OmegaConf
from collections import OrderedDict
import logging

mainlogger = logging.getLogger("mainlogger")

import torch
from collections import OrderedDict


def init_workspace(name, logdir, model_config, lightning_config, rank=0):
    workdir = os.path.join(logdir, name)
    ckptdir = os.path.join(workdir, "checkpoints")
    cfgdir = os.path.join(workdir, "configs")
    loginfo = os.path.join(workdir, "loginfo")

    # Create logdirs and save configs (all ranks will do to avoid missing directory error if rank:0 is slower)
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(ckptdir, exist_ok=True)
    os.makedirs(cfgdir, exist_ok=True)
    os.makedirs(loginfo, exist_ok=True)

    if rank == 0:
        if (
            "callbacks" in lightning_config
            and "metrics_over_trainsteps_checkpoint" in lightning_config.callbacks
        ):
            os.makedirs(os.path.join(ckptdir, "trainstep_checkpoints"), exist_ok=True)
        OmegaConf.save(model_config, os.path.join(cfgdir, "model.yaml"))
        OmegaConf.save(
            OmegaConf.create({"lightning": lightning_config}),
            os.path.join(cfgdir, "lightning.yaml"),
        )
    return workdir, ckptdir, cfgdir, loginfo


def check_config_attribute(config, name):
    if name in config:
        value = getattr(config, name)
        return value
    else:
        return None


def get_trainer_callbacks(lightning_config, config, logdir, ckptdir, logger):
    default_callbacks_cfg = {
        "model_checkpoint": {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch}",
                "verbose": True,
                "save_last": False,
            },
        },
        # "batch_logger": {
        #     "target": "utils.callbacks.ImageLogger",
        #     "params": {
        #         "save_dir": logdir,
        #         "batch_frequency": 1000,
        #         "max_images": 4,
        #         "clamp": True,
        #     },
        # },
        "learning_rate_logger": {
            "target": "pytorch_lightning.callbacks.LearningRateMonitor",
            "params": {"logging_interval": "step", "log_momentum": False},
        },
        "cuda_callback": {"target": "utils.callbacks.CUDACallback"},
        "keep_latest_checkpoints": {
            "target": "utils.callbacks.KeepLatestCheckpoints",
            "params": {"ckpt_dir": ckptdir, "keep_k": 1},
        },
    }

    ## optional setting for saving checkpoints
    monitor_metric = check_config_attribute(config.model.params, "monitor")
    if monitor_metric is not None:
        mainlogger.info(f"Monitoring {monitor_metric} as checkpoint metric.")
        default_callbacks_cfg["model_checkpoint"]["params"]["monitor"] = monitor_metric
        default_callbacks_cfg["model_checkpoint"]["params"]["save_top_k"] = 3
        default_callbacks_cfg["model_checkpoint"]["params"]["mode"] = "min"

    if "metrics_over_trainsteps_checkpoint" in lightning_config.callbacks:
        mainlogger.info(
            "Caution: Saving checkpoints every n train steps without deleting. This might require some free space."
        )
        default_metrics_over_trainsteps_ckpt_dict = {
            "metrics_over_trainsteps_checkpoint": {
                "target": "pytorch_lightning.callbacks.ModelCheckpoint",
                "params": {
                    "dirpath": os.path.join(ckptdir, "trainstep_checkpoints"),
                    "filename": "{epoch}-{step}",
                    "verbose": True,
                    "save_top_k": -1,
                    "every_n_train_steps": 10000,
                    "save_last": False,
                    "save_weights_only": False,  # Save full state for proper resumption
                },
            }
        }
        default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

    if "callbacks" in lightning_config:
        callbacks_cfg = lightning_config.callbacks
    else:
        callbacks_cfg = OmegaConf.create()
    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)

    return callbacks_cfg


def get_trainer_logger(lightning_config, logdir, on_debug):
    default_logger_cfgs = {
        "tensorboard": {
            "target": "pytorch_lightning.loggers.TensorBoardLogger",
            "params": {
                "save_dir": logdir,
                "name": "tensorboard",
            },
        },
        "testtube": {
            "target": "pytorch_lightning.loggers.CSVLogger",
            "params": {
                "name": "testtube",
                "save_dir": logdir,
            },
        },
    }
    os.makedirs(os.path.join(logdir, "tensorboard"), exist_ok=True)
    default_logger_cfg = default_logger_cfgs["tensorboard"]
    if "logger" in lightning_config:
        logger_cfg = lightning_config.logger
    else:
        logger_cfg = OmegaConf.create()
    logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
    return logger_cfg


def get_trainer_strategy(lightning_config):
    default_strategy_dict = {
        "target": "pytorch_lightning.strategies.DDPStrategy"
    }

    if "strategy" in lightning_config:
        strategy_cfg = lightning_config.strategy

        # If it's a string, return as-is
        if isinstance(strategy_cfg, str):
            return strategy_cfg

        # Convert OmegaConf to dict if it has params.config
        if hasattr(strategy_cfg, 'params') and hasattr(strategy_cfg.params, 'config'):
            strategy_cfg = OmegaConf.to_container(strategy_cfg, resolve=True)
            # Convert nested config to dict
            if 'params' in strategy_cfg and 'config' in strategy_cfg['params']:
                strategy_cfg['params']['config'] = dict(strategy_cfg['params']['config'])

        return strategy_cfg
    else:
        strategy_cfg = OmegaConf.create()

    # Add find_unused_parameters if specified in lightning_config
    if "find_unused_parameters" in lightning_config:
        if "params" not in default_strategy_dict:
            default_strategy_dict["params"] = {}
        default_strategy_dict["params"]["find_unused_parameters"] = lightning_config.find_unused_parameters

    strategy_cfg = OmegaConf.merge(default_strategy_dict, strategy_cfg)
    return strategy_cfg

def load_checkpoints(model, model_cfg):
    ## special load setting for adapter training
    if check_config_attribute(model_cfg, "adapter_only"):
        pretrained_ckpt = model_cfg.pretrained_checkpoint
        assert os.path.exists(pretrained_ckpt), (
            "Error: Pre-trained checkpoint NOT found at:%s" % pretrained_ckpt
        )
        mainlogger.info(
            ">>> Load weights from pretrained checkpoint (training adapter only)"
        )
        print(f"Loading model from {pretrained_ckpt}")
        ## only load weight for the backbone model (e.g. latent diffusion model)
        state_dict = torch.load(pretrained_ckpt, map_location=f"cpu")
        if "state_dict" in list(state_dict.keys()):
            state_dict = state_dict["state_dict"]
        else:
            # deepspeed
            dp_state_dict = OrderedDict()
            for key in state_dict["module"].keys():
                dp_state_dict[key[16:]] = state_dict["module"][key]
            state_dict = dp_state_dict
        model.load_state_dict(state_dict, strict=False)
        model.empty_paras = None
        return model
    empty_paras = None

    if check_config_attribute(model_cfg, "pretrained_checkpoint"):
        pretrained_ckpt = model_cfg.pretrained_checkpoint
        assert os.path.exists(pretrained_ckpt), (
            "Error: Pre-trained checkpoint NOT found at:%s" % pretrained_ckpt
        )
        mainlogger.info(">>> Load weights from pretrained checkpoint")
        # mainlogger.info(pretrained_ckpt)
        print("Loading model from {pretrained_ckpt}")
        pl_sd = torch.load(pretrained_ckpt, map_location="cpu")
        try:
            if "state_dict" in pl_sd.keys():
                model.load_state_dict(pl_sd["state_dict"])
            else:
                # deepspeed
                new_pl_sd = OrderedDict()
                for key in pl_sd["module"].keys():
                    new_pl_sd[key[16:]] = pl_sd["module"][key]
                model.load_state_dict(new_pl_sd)
        except:
            model.load_state_dict(pl_sd)
    else:
        empty_paras = None

    ## record empty params
    model.empty_paras = empty_paras
    return model


def get_autoresume_path(logdir):
    resume_checkpt_path = None
    trainstep_dir = os.path.join(logdir, "checkpoints", "trainstep_checkpoints")

    # Check if trainstep_checkpoints directory exists and is not empty
    if os.path.exists(trainstep_dir) and os.listdir(trainstep_dir):
        checkpoints = sorted(os.listdir(trainstep_dir))
        mainlogger.info(f"Available checkpoints: {checkpoints}")

        # Scenario 1: Check if last.ckpt exists in trainstep_checkpoints
        last_ckpt = os.path.join(trainstep_dir, "last.ckpt")
        if os.path.exists(last_ckpt):
            resume_checkpt_path = last_ckpt
            mainlogger.info("Using checkpoint: last.ckpt")

        # Scenario 2: If last.ckpt not found, use the last checkpoint by name
        if resume_checkpt_path is None:
            if checkpoints:
                last_checkpoint = checkpoints[-1]
                candidate_path = os.path.join(trainstep_dir, last_checkpoint)
                resume_checkpt_path = candidate_path
                mainlogger.info(f"Using checkpoint: {resume_checkpt_path}")
    else:
        mainlogger.info("trainstep_checkpoints directory not found or is empty")

    if resume_checkpt_path is None:
        mainlogger.info(
            f"[INFO] no checkpoint found in current workspace: {os.path.join(logdir, 'checkpoints')}"
        )

    return resume_checkpt_path


def set_logger(logfile, name="mainlogger"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logfile, mode="w")
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s-%(levelname)s: %(message)s"))
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
