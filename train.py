import argparse, os, sys, datetime
from omegaconf import OmegaConf
from transformers import logging as transf_logging
import signal

import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.strategies import FSDPStrategy
from pytorch_lightning.trainer import Trainer

sys.path.insert(0, os.getcwd())
from utils.common_utils import instantiate_from_config
from utils.train_utils import (
    get_trainer_callbacks,
    get_trainer_logger,
    get_trainer_strategy,
)
from utils.train_utils import (
    set_logger,
    init_workspace,
    load_checkpoints,
    get_autoresume_path,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# torch.backends.cuda.matmul.allow_tf32 = True

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "--seed", "-s", type=int, default=-1, help="seed for seed_everything"
    )
    parser.add_argument(
        "--name", "-n", type=str, default="", help="experiment name, as saving folder"
    )

    parser.add_argument(
        "--base",
        "-b",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )

    parser.add_argument(
        "--train", "-t", action="store_true", default=False, help="train"
    )
    parser.add_argument("--val", "-v", action="store_true", default=False, help="val")
    parser.add_argument("--test", action="store_true", default=False, help="test")

    parser.add_argument(
        "--logdir",
        "-l",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--auto_resume",
        action="store_true",
        default=False,
        help="resume from full-info checkpoint",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        default=False,
        help="enable post-mortem debugging",
    )

    return parser




if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    torch.autograd.set_detect_anomaly(True)
    torch.backends.cudnn.enabled = False
    try:
        local_rank = int(os.environ.get("LOCAL_RANK"))
        global_rank = int(os.environ.get("RANK"))
        num_rank = int(os.environ.get("WORLD_SIZE"))
    except:
        local_rank, global_rank, num_rank = 0, 0, 1
    print(f'local_rank: {local_rank} | global_rank:{global_rank} | num_rank:{num_rank}')

    parser = get_parser()
    args, unknown = parser.parse_known_args()
    ## disable transformer warning
    transf_logging.set_verbosity_error()
    if args.seed >= 0:
        seed_everything(args.seed)

    ## yaml configs: "model" | "data" | "lightning"
    configs = [OmegaConf.load(cfg) for cfg in args.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    lightning_config = config.pop("lightning", OmegaConf.create())
    trainer_config = lightning_config.get("trainer", OmegaConf.create())

    ## setup workspace directories
    workdir, ckptdir, cfgdir, loginfo = init_workspace(
        args.name, args.logdir, config, lightning_config, global_rank
    )
    logger = set_logger(
        logfile=os.path.join(loginfo, "log_%d:%s.txt" % (global_rank, now))
    )
    logger.info("@lightning version: %s [>=1.8 required]" % (pl.__version__))

    ## MODEL CONFIG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    logger.info("***** Configing Model *****")
    config.model.params.logdir = workdir
    # Ensure deterministic model initialization across all ranks
    model = instantiate_from_config(config.model)
    # model.compile_model()

    if args.auto_resume:
        resume_ckpt_path = get_autoresume_path(workdir)
        if resume_ckpt_path is not None:
            logger.info("Resuming from checkpoint: %s" % resume_ckpt_path)
        else:
            # model = load_checkpoints(model, config.model)
            logger.warning("Auto-resuming skipped as No checkpoint found!")
            resume_ckpt_path = None
    else:
        # model = load_checkpoints(model, config.model)
        resume_ckpt_path = None

    print(trainer_config)
    num_nodes = trainer_config.num_nodes
    ngpu_per_node = trainer_config.devices
    logger.info(f"Running on {num_rank}={num_nodes}x{ngpu_per_node} GPUs")

    ## setup learning rate
    base_lr = config.model.base_learning_rate
    bs = config.data.params.batch_size
    if getattr(config.model, "scale_lr", True):
        model.learning_rate = num_rank * bs * base_lr
    else:
        model.learning_rate = base_lr

    ## DATA CONFIG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    logger.info("***** Configing Data *****")
    data = instantiate_from_config(config.data)
    data.setup()
    for k in data.datasets:
        logger.info(
            f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}"
        )

    ## TRAINER CONFIG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    logger.info("***** Configing Trainer *****")
    if "accelerator" not in trainer_config:
        trainer_config["accelerator"] = "gpu"

    torch.set_float32_matmul_precision("medium")

    ## setup trainer args: pl-logger and callbacks
    trainer_kwargs = dict()
    trainer_kwargs["num_sanity_val_steps"] = 0
    logger_cfg = get_trainer_logger(lightning_config, workdir, args.debug)
    trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

    ## setup callbacks
    callbacks_cfg = get_trainer_callbacks(
        lightning_config, config, workdir, ckptdir, logger
    )
    trainer_kwargs["callbacks"] = [
        instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg
    ]


    # Using FSDP during stage 2 training to enable 17/16 frames training if memory is a bottleneck.

    # ----------------------------- Wan -----------------------------
    # from src.models.transformer_wan import WanTransformerBlock
    # from src.models.autoencoder_wanT import WanDecoder3d, WanEncoder3d

    # trainer_kwargs["strategy"] = FSDPStrategy(
    #     auto_wrap_policy={WanTransformerBlock, WanDecoder3d, WanEncoder3d},
    #     limit_all_gathers=True,
    # )

    # ------------------------- VideoVAEPlus -------------------------
    # from src.models.transformer_wan import WanTransformerBlock
    # from src.models.VideoVaePlus.videovaeplus_ref import Encoder2plus1D, Decoder2plus1D
    # from src.models.VideoVaePlus.autoencoder_temporal import EncoderTemporal1DCNN, DecoderTemporal1DCNN

    # trainer_kwargs["strategy"] = FSDPStrategy(
    #     auto_wrap_policy={Encoder2plus1D, Decoder2plus1D, EncoderTemporal1DCNN, DecoderTemporal1DCNN, WanTransformerBlock},
    #     limit_all_gathers=True,
    #  )


    # ------------------------- From Config -------------------------
    strategy_cfg = get_trainer_strategy(lightning_config)
    trainer_kwargs["strategy"] = (
        strategy_cfg
        if type(strategy_cfg) == str
        else instantiate_from_config(strategy_cfg)
    )

    trainer_kwargs["precision"] = "bf16-mixed"
    trainer_kwargs["sync_batchnorm"] = False

    ## trainer config: others
    if (
        "train" in config.data.params
        and config.data.params.train.target == "lvdm.data.hdvila.HDVila"
        or (
            "validation" in config.data.params
            and config.data.params.validation.target == "lvdm.data.hdvila.HDVila"
        )
    ):
        trainer_kwargs["use_distributed_sampler"] = False

    ## for debug
    # trainer_kwargs["fast_dev_run"] = 10
    # trainer_kwargs["limit_train_batches"] = 1./32
    # trainer_kwargs["limit_val_batches"] = 0.01
    # trainer_kwargs["check_val_every_n_epoch"] = 20  #float: epoch ratio | integer: batch num

    # completely disable validation
    trainer_kwargs["limit_val_batches"] = 0
    trainer = Trainer(**trainer_config, **trainer_kwargs)

    ## allow checkpointing via signals (USR1 or termination)
    def _save_checkpoint_from_signal():
        if trainer.global_rank != 0:
            if trainer.global_rank == 1:  # Just log once from rank 1
                print(f"[signal] Rank {trainer.global_rank} received signal, rank 0 will save")
            return
        ckpt_name = f"epoch={trainer.current_epoch:06}-step={trainer.global_step:09}.ckpt"
        ckpt_path = os.path.join(ckptdir, "trainstep_checkpoints", ckpt_name)
        print(f"[signal] Saving checkpoint to {ckpt_path}")
        trainer.save_checkpoint(ckpt_path)
        # Try to flush logger before exiting on termination signals
        logger.info(f"Checkpoint saved from signal: {ckpt_path}")
        if hasattr(logger, "handlers"):
            for h in logger.handlers:
                try:
                    h.flush()
                except Exception:
                    pass

    def handle_signal(*args, **kwargs):
        _save_checkpoint_from_signal()

    def handle_term(*args, **kwargs):
        _save_checkpoint_from_signal()
        # Exit after saving to let the job terminate cleanly
        raise SystemExit("Received termination signal; checkpoint saved, exiting.")

    def divein(*args, **kwargs):
        if trainer.global_rank == 0:
            import pudb

            pudb.set_trace()

    signal.signal(signal.SIGUSR1, handle_signal)
    # signal.signal(signal.SIGTERM, handle_term)
    # signal.signal(signal.SIGINT, handle_term)
    # signal.signal(signal.SIGUSR2, divein)

    ## Running LOOP >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    logger.info("***** Running the Loop *****")

    val_at_start = config.model.params.get("val_begining", False)


    if args.train:
        try:
            if val_at_start:
                logger.info("=" * 60)
                logger.info("***** Running FULL Validation at Start (Before Training) *****")
                logger.info("=" * 60)
                trainer.validate(model, data)
                logger.info("=" * 60)
                logger.info("***** Starting Training *****")
                logger.info("=" * 60)
                
            strategy_obj = getattr(trainer, "strategy", None)
            strategy_name = getattr(strategy_obj, "strategy_name", None) or (
                strategy_obj.__class__.__name__ if strategy_obj else "None"
            )
            logger.info(f"<Training with strategy: {strategy_name}>")

            # Explicitly set train/eval modes before training
            if hasattr(model, 'loss'):
                model.loss.eval()
                logger.info("Set loss module to eval mode")
            if hasattr(model, 'ae'):
                model.ae.train()
                logger.info("Set ae module to train mode")

            trainer.fit(model, data, ckpt_path=resume_ckpt_path)

        except Exception:
            # melk()
            raise
    if args.val:
        trainer.validate(model, data)
    if args.test or not trainer.interrupted:
        trainer.test(model, data)

