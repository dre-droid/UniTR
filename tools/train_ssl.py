"""
Self-supervised pretraining entry point for iBOT-style UniTR pretraining.

Usage:
    python tools/train_ssl.py --cfg_file tools/cfgs/nuscenes_models/unitr_ibot.yaml

For distributed:
    python -m torch.distributed.launch --nproc_per_node=N tools/train_ssl.py \
        --cfg_file tools/cfgs/nuscenes_models/unitr_ibot.yaml --launcher pytorch
"""

import _init_path
import argparse
import datetime
import glob
import os
from pathlib import Path

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models.ssl import iBOTUniTR
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model, save_checkpoint, checkpoint_state


def build_ssl_network(model_cfg, dataset):
    """Build the iBOT SSL network."""
    model = iBOTUniTR(
        model_cfg=model_cfg,
        num_class=0,  # no classes for SSL
        dataset=dataset,
    )
    return model


def model_fn_ssl(total_steps=None):
    """
    Model function for SSL training.
    Returns a callable compatible with the existing training loop.
    """
    from collections import namedtuple
    from pcdet.models import load_data_to_gpu

    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict'])

    def model_func(model, batch_dict):
        load_data_to_gpu(batch_dict)
        ret_dict, tb_dict, disp_dict = model(batch_dict)

        loss = ret_dict['loss'].mean()

        # EMA teacher update
        if hasattr(model, 'module'):
            model.module.update_teacher(total_steps or 1)
        else:
            model.update_teacher(total_steps or 1)

        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()

        return ModelReturn(loss, tb_dict, disp_dict)

    return model_func


def parse_config():
    parser = argparse.ArgumentParser(description='iBOT SSL Pretraining')
    parser.add_argument('--cfg_file', type=str, default=None, help='config file')
    parser.add_argument('--batch_size', type=int, default=None, help='batch size per GPU')
    parser.add_argument('--epochs', type=int, default=None, help='number of epochs')
    parser.add_argument('--workers', type=int, default=4, help='dataloader workers')
    parser.add_argument('--extra_tag', type=str, default='default', help='experiment tag')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to resume from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained model to load')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888)
    parser.add_argument('--sync_bn', action='store_true', default=False)
    parser.add_argument('--fix_random_seed', action='store_true', default=False)
    parser.add_argument('--ckpt_save_interval', type=int, default=1)
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument('--max_ckpt_save_num', type=int, default=10)
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--use_tqdm_to_record', action='store_true', default=False)
    parser.add_argument('--logger_iter_interval', type=int, default=50)
    parser.add_argument('--ckpt_save_time_interval', type=int, default=300)
    parser.add_argument('--wo_gpu_stat', action='store_true')
    parser.add_argument('--use_amp', action='store_true')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])

    args.use_amp = args.use_amp or cfg.OPTIMIZATION.get('USE_AMP', False)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()

    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0
        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs
    args.ckpt_save_interval = cfg.OPTIMIZATION.get('CKPT_SAVE_INTERVAL', 1) if args.ckpt_save_interval == 1 else args.ckpt_save_interval

    if args.fix_random_seed:
        common_utils.set_random_seed(666 + cfg.LOCAL_RANK)

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('train_ssl_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    logger.info('**********************Start iBOT SSL Pretraining**********************')
    gpu_list = os.environ.get('CUDA_VISIBLE_DEVICES', 'ALL')
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        logger.info('Distributed training: total_batch_size=%d' % (total_gpus * args.batch_size))
    else:
        logger.info('Single-process training')

    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    # ---- Build dataloader ----
    logger.info('----------- Building dataloader -----------')
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=False,
        total_epochs=args.epochs,
        seed=666 if args.fix_random_seed else None
    )

    # ---- Build network ----
    logger.info('----------- Building iBOT SSL network -----------')
    model = build_ssl_network(model_cfg=cfg.MODEL, dataset=train_set)
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Model created: {param_count:,} params total, {trainable_count:,} trainable')

    # ---- Optimizer ----
    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    # ---- Resume from checkpoint ----
    start_epoch = it = 0
    last_epoch = -1
    if args.ckpt is not None:
        it, start_epoch = model.load_params_with_optimizer(
            args.ckpt, to_cpu=dist_train, optimizer=optimizer, logger=logger
        )
        last_epoch = start_epoch + 1
    elif args.pretrained_model is not None:
        model.load_params_from_file(
            filename=args.pretrained_model, to_cpu=dist_train, logger=logger
        )
    else:
        ckpt_list = glob.glob(str(ckpt_dir / '*.pth'))
        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime)
            while len(ckpt_list) > 0:
                try:
                    it, start_epoch = model.load_params_with_optimizer(
                        ckpt_list[-1], to_cpu=dist_train, optimizer=optimizer, logger=logger
                    )
                    last_epoch = start_epoch + 1
                    break
                except:
                    ckpt_list = ckpt_list[:-1]

    model.train()
    if dist_train:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()]
        )

    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )

    # ---- Compute total steps for EMA schedule ----
    total_steps = len(train_loader) * args.epochs

    # ---- Train ----
    logger.info('**********************Start SSL training %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    train_model(
        model,
        optimizer,
        train_loader,
        model_func=model_fn_ssl(total_steps=total_steps),
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.OPTIMIZATION,
        start_epoch=start_epoch,
        total_epochs=args.epochs,
        start_iter=it,
        rank=cfg.LOCAL_RANK,
        tb_log=tb_log,
        ckpt_save_dir=ckpt_dir,
        train_sampler=train_sampler,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=False,
        logger=logger,
        logger_iter_interval=args.logger_iter_interval,
        ckpt_save_time_interval=args.ckpt_save_time_interval,
        use_logger_to_record=not args.use_tqdm_to_record,
        show_gpu_stat=not args.wo_gpu_stat,
        use_amp=args.use_amp,
        cfg=cfg,
    )

    if hasattr(train_set, 'use_shared_memory') and train_set.use_shared_memory:
        train_set.clean_shared_memory()

    logger.info('**********************End SSL training %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))


if __name__ == '__main__':
    main()
