from functools import partial

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched

from .fastai_optim import OptimWrapper
from .learning_schedules_fastai import CosineWarmupLR, OneCycle, CosineAnnealing

class OneCycleLRWrapper:
    """
    Compatibility wrapper for PyTorch's native OneCycleLR to match the 
    signature expected by the UniTR training loop: step(accumulated_iter, cur_epoch).
    """
    def __init__(self, scheduler):
        self.scheduler = scheduler
        
    def step(self, step, epoch=None):
        # Native OneCycleLR tracks its own steps internally
        self.scheduler.step()
        
    def __getattr__(self, name):
        return getattr(self.scheduler, name)


def build_optimizer(model, optim_cfg):
    if optim_cfg.OPTIMIZER == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY)
    elif optim_cfg.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY,
            momentum=optim_cfg.MOMENTUM
        )
    elif optim_cfg.OPTIMIZER in ['adam_onecycle','adam_cosineanneal']:
        def children(m: nn.Module):
            return list(m.children())

        def num_children(m: nn.Module) -> int:
            return len(children(m))

        flatten_model = lambda m: sum(map(flatten_model, m.children()), []) if num_children(m) else [m]
        betas = optim_cfg.get('BETAS', (0.9, 0.99))
        betas = tuple(betas)
        optimizer_func = partial(optim.Adam, betas=betas)

        # Check for backbone LR scaling (used for fine-tuning SSL models)
        backbone_lr_scale = optim_cfg.get('BACKBONE_LR_SCALE', None)
        
        if backbone_lr_scale is not None:
            # Differential LR mode: split params into groups
            # Scale down LR for pretrained modules: vfe, mm_backbone, map_to_bev
            backbone_params = []
            for module_name in ['vfe', 'mm_backbone', 'map_to_bev_module']:
                module = getattr(model, module_name, None)
                if module is not None:
                    backbone_params.extend(list(module.parameters()))
            
            backbone_param_ids = set(id(p) for p in backbone_params)
            other_params = [p for p in model.parameters() if id(p) not in backbone_param_ids]
            
            optimizer = optim.AdamW(
                [
                    {'params': backbone_params, 'lr': optim_cfg.LR * backbone_lr_scale},
                    {'params': other_params,    'lr': optim_cfg.LR},
                ],
                weight_decay=optim_cfg.WEIGHT_DECAY,
                betas=betas,
            )
            # Store metadata for build_scheduler
            optimizer._use_differential_lr = True
            optimizer._backbone_lr_scale = backbone_lr_scale
        else:
            get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]
            optimizer = OptimWrapper.create(
                optimizer_func, 3e-3, get_layer_groups(model), wd=optim_cfg.WEIGHT_DECAY, true_wd=True, bn_wd=True
            )
            optimizer._use_differential_lr = False
    else:
        raise NotImplementedError

    return optimizer


def build_scheduler(optimizer, total_iters_each_epoch, total_epochs, last_epoch, optim_cfg):
    decay_steps = [x * total_iters_each_epoch for x in optim_cfg.DECAY_STEP_LIST]
    def lr_lbmd(cur_epoch):
        cur_decay = 1
        for decay_step in decay_steps:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * optim_cfg.LR_DECAY
        return max(cur_decay, optim_cfg.LR_CLIP / optim_cfg.LR)

    lr_warmup_scheduler = None
    total_steps = total_iters_each_epoch * total_epochs

    # Check if differential mode was enabled in build_optimizer
    use_diff_lr = getattr(optimizer, '_use_differential_lr', False)

    if optim_cfg.OPTIMIZER == 'adam_onecycle':
        if use_diff_lr:
            # Native PyTorch OneCycleLR handles list of max_lrs perfectly
            backbone_lr_scale = optimizer._backbone_lr_scale
            peak_lrs = [optim_cfg.LR * backbone_lr_scale, optim_cfg.LR]
            
            native_scheduler = lr_sched.OneCycleLR(
                optimizer,
                max_lr=peak_lrs,
                total_steps=total_steps,
                pct_start=optim_cfg.PCT_START,
                div_factor=optim_cfg.DIV_FACTOR,
                final_div_factor=1e4, # standard in pcdet/fastai
                anneal_strategy='cos',
            )
            lr_scheduler = OneCycleLRWrapper(native_scheduler)
        else:
            lr_scheduler = OneCycle(
                optimizer, total_steps, optim_cfg.LR, list(optim_cfg.MOMS), optim_cfg.DIV_FACTOR, optim_cfg.PCT_START
            )
    elif optim_cfg.OPTIMIZER == 'adam_cosineanneal':
        warmup_iter = optim_cfg.get('WARMUP_ITER',
                                    optim_cfg.get('WARMUP_EPOCH', 1) * total_iters_each_epoch)
        lr_scheduler = CosineAnnealing(
            optimizer, total_steps, total_epochs, optim_cfg.LR, list(optim_cfg.MOMS), optim_cfg.PCT_START, warmup_iter
        )
    else:
        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd, last_epoch=last_epoch)

        if optim_cfg.LR_WARMUP:
            lr_warmup_scheduler = CosineWarmupLR(
                optimizer, T_max=optim_cfg.WARMUP_EPOCH * total_iters_each_epoch,
                eta_min=optim_cfg.LR / optim_cfg.DIV_FACTOR
            )

    return lr_scheduler, lr_warmup_scheduler
