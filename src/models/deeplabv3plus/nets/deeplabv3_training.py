import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def CE_Loss(inputs, target, cls_weights, num_classes=21):
    if isinstance(inputs, dict):
        inputs = inputs["logits"]
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    CE_loss  = nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes)(temp_inputs, temp_target)
    return CE_loss


def Softmax_CE_Loss(inputs, soft_target, cls_weights, num_classes=21):
    if isinstance(inputs, dict):
        inputs = inputs["logits"]
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = soft_target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    log_probs = F.log_softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), dim=-1)
    soft_target = soft_target.view(n, -1, ct)
    class_target = soft_target[..., :num_classes]
    valid_mask = (1.0 - soft_target[..., num_classes]).float()

    if cls_weights is not None:
        class_target = class_target * cls_weights.view(1, 1, -1)

    loss = -(class_target * log_probs).sum(dim=-1)
    loss = loss * valid_mask
    normalizer = valid_mask.sum().clamp(min=1.0)
    return loss.sum() / normalizer

def Focal_Loss(inputs, target, cls_weights, num_classes=21, alpha=0.5, gamma=2):
    if isinstance(inputs, dict):
        inputs = inputs["logits"]
    # Extended from the focal-loss reference under:
    # Integrated as plugins.FocalLoss / build_loss("focal").
    # This version keeps ignore-label behavior and also supports soft labels
    # produced by MixUp / CutMix.
    n, c, h, w = inputs.size()
    if target.dim() == 4:
        nt, ht, wt, ct = target.size()
        if h != ht and w != wt:
            inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

        probs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), dim=-1)
        log_probs = torch.log(probs.clamp(min=1e-8))
        target = target.view(n, -1, ct)
        class_target = target[..., :num_classes]
        valid_mask = (1.0 - target[..., num_classes]).float()

        if cls_weights is not None:
            weighted_target = class_target * cls_weights.view(1, 1, -1)
        else:
            weighted_target = class_target

        pt = (probs * class_target).sum(dim=-1).clamp(min=1e-8)
        ce = -(weighted_target * log_probs).sum(dim=-1)
        focal_weight = (1 - pt) ** gamma
        if alpha is not None:
            focal_weight = focal_weight * alpha
        loss = ce * focal_weight * valid_mask
        normalizer = valid_mask.sum().clamp(min=1.0)
        return loss.sum() / normalizer

    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    logpt  = -nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes, reduction='none')(temp_inputs, temp_target)
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt
    loss = loss.mean()
    return loss

def Dice_loss(inputs, target, beta=1, smooth = 1e-5):
    if isinstance(inputs, dict):
        inputs = inputs["logits"]
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
        
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_target = target.view(n, -1, ct)

    #--------------------------------------------#
    #   计算dice loss
    #--------------------------------------------#
    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
    fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss


def Focal_Tversky_Loss(inputs, target, alpha=0.3, beta=0.7, gamma=1.33, smooth=1e-5):
    if isinstance(inputs, dict):
        inputs = inputs["logits"]
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = target.view(n, -1, ct)
    class_target = temp_target[..., :c]
    if ct > c:
        valid_mask = (1.0 - temp_target[..., c]).unsqueeze(-1)
    else:
        valid_mask = torch.ones_like(class_target[..., :1])

    temp_inputs = temp_inputs * valid_mask
    class_target = class_target * valid_mask

    tp = torch.sum(temp_inputs * class_target, dim=[0, 1])
    fp = torch.sum(temp_inputs * (1.0 - class_target), dim=[0, 1])
    fn = torch.sum((1.0 - temp_inputs) * class_target, dim=[0, 1])

    score = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return torch.mean(torch.pow(1.0 - score, gamma))


def _binary_pos_weight(target, valid_mask):
    positives = (target[valid_mask] > 0.5).sum().float()
    negatives = (target[valid_mask] <= 0.5).sum().float()
    return (negatives / positives.clamp(min=1.0)).clamp(min=1.0, max=20.0)


def _build_component_targets(mask, num_classes):
    valid = mask != num_classes
    lesion_ids = torch.arange(2, num_classes, device=mask.device, dtype=mask.dtype)
    lesion = torch.isin(mask, lesion_ids) & valid
    lesion_float = lesion.float().unsqueeze(1)

    eroded = 1.0 - F.max_pool2d(1.0 - lesion_float, kernel_size=3, stride=1, padding=1)
    boundary = (lesion_float - eroded).clamp_min(0.0)
    center = F.avg_pool2d(lesion_float, kernel_size=9, stride=1, padding=4) * lesion_float
    return lesion_float, boundary, center, valid.unsqueeze(1)


def Component_Aux_Loss(outputs, target, num_classes, lesion_weight=0.4, boundary_weight=0.2, center_weight=0.2):
    if not isinstance(outputs, dict):
        return outputs.sum() * 0.0

    required = {"lesion_logits", "boundary_logits", "center_logits"}
    if not required.issubset(outputs):
        missing = ", ".join(sorted(required - set(outputs)))
        raise KeyError(f"Missing component auxiliary outputs: {missing}")

    lesion_target, boundary_target, center_target, valid = _build_component_targets(target, num_classes)
    losses = []
    for key, aux_target, weight in [
        ("lesion_logits", lesion_target, lesion_weight),
        ("boundary_logits", boundary_target, boundary_weight),
        ("center_logits", center_target, center_weight),
    ]:
        if weight <= 0:
            continue
        logits = outputs[key]
        if logits.shape[-2:] != aux_target.shape[-2:]:
            logits = F.interpolate(logits, size=aux_target.shape[-2:], mode="bilinear", align_corners=True)
        valid_logits = logits[valid]
        valid_target = aux_target[valid]
        if valid_target.numel() == 0:
            continue
        pos_weight = _binary_pos_weight(aux_target, valid)
        losses.append(
            weight
            * F.binary_cross_entropy_with_logits(
                valid_logits,
                valid_target,
                pos_weight=pos_weight,
            )
        )
    if not losses:
        return outputs["logits"].sum() * 0.0
    return sum(losses)


def Severity_Consistency_Loss(inputs, target, num_classes, loss_type="l1", smooth=1e-6):
    if isinstance(inputs, dict):
        inputs = inputs["logits"]

    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht or w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    valid = target != num_classes
    gt_leaf = ((target >= 1) & (target < num_classes) & valid).float()
    gt_lesion = ((target >= 2) & (target < num_classes) & valid).float()
    gt_leaf_area = gt_leaf.flatten(1).sum(dim=1)
    gt_lesion_area = gt_lesion.flatten(1).sum(dim=1)
    valid_images = gt_leaf_area > 0
    if not valid_images.any():
        return inputs.sum() * 0.0

    probs = torch.softmax(inputs, dim=1)
    pred_leaf = probs[:, 1:num_classes].sum(dim=1) * valid.float()
    pred_lesion = probs[:, 2:num_classes].sum(dim=1) * valid.float()
    pred_leaf_area = pred_leaf.flatten(1).sum(dim=1)
    pred_lesion_area = pred_lesion.flatten(1).sum(dim=1)

    gt_severity = gt_lesion_area[valid_images] / gt_leaf_area[valid_images].clamp(min=smooth)
    pred_severity = pred_lesion_area[valid_images] / pred_leaf_area[valid_images].clamp(min=smooth)

    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(pred_severity, gt_severity)
    if loss_type == "mse":
        return F.mse_loss(pred_severity, gt_severity)
    if loss_type != "l1":
        raise ValueError(f"Unsupported severity loss type: {loss_type}")
    return F.l1_loss(pred_severity, gt_severity)

def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.1, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.3, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
