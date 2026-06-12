import os

import torch
from nets.deeplabv3_training import (CE_Loss, Component_Aux_Loss, Dice_loss, Focal_Loss,
                                     Focal_Tversky_Loss, Lesion_Prior_Loss,
                                     Severity_Consistency_Loss, Softmax_CE_Loss, weights_init)
from tqdm import tqdm

from utils.batch_mix import apply_batch_mix
from utils.utils import get_lr
from utils.utils_metrics import f_score


def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes, \
    fp16, scaler, save_period, save_dir, local_rank=0, focal_alpha=0.5, focal_gamma=2.0, mix_mode="none", mix_prob=0.0, mixup_alpha=0.4, cutmix_alpha=1.0, \
    lbft_loss=False, lbft_lambda=1.0, lbft_alpha=0.3, lbft_beta=0.7, lbft_gamma=1.33, component_aux=False, component_lesion_weight=0.4, component_boundary_weight=0.2, component_center_weight=0.2, \
    severity_consistency_loss=False, severity_consistency_weight=0.1, severity_loss_type="l1", label_smoothing=0.0, dice_class_start=0, dice_loss_weight=1.0, lesion_prior_loss=False, lesion_prior_weight=0.05, lesion_prior_pos_weight_cap=5.0):
    total_loss      = 0
    total_f_score   = 0

    val_loss        = 0
    val_f_score     = 0

    def compute_train_loss(outputs, pngs, labels, weights):
        if lbft_loss:
            if mix_mode == "mixup":
                ce_loss = Softmax_CE_Loss(outputs, labels, weights, num_classes=num_classes, label_smoothing=label_smoothing)
            else:
                ce_loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes, label_smoothing=label_smoothing)
            tversky_loss = Focal_Tversky_Loss(
                outputs,
                labels,
                alpha=lbft_alpha,
                beta=lbft_beta,
                gamma=lbft_gamma,
            )
            return ce_loss + lbft_lambda * tversky_loss

        if focal_loss:
            focal_target = labels if mix_mode == "mixup" else pngs
            return Focal_Loss(outputs, focal_target, weights, num_classes=num_classes, alpha=focal_alpha, gamma=focal_gamma)
        if mix_mode == "mixup":
            return Softmax_CE_Loss(outputs, labels, weights, num_classes=num_classes, label_smoothing=label_smoothing)
        return CE_Loss(outputs, pngs, weights, num_classes=num_classes, label_smoothing=label_smoothing)

    def compute_val_loss(outputs, pngs, labels, weights):
        if lbft_loss:
            ce_loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes, label_smoothing=label_smoothing)
            tversky_loss = Focal_Tversky_Loss(
                outputs,
                labels,
                alpha=lbft_alpha,
                beta=lbft_beta,
                gamma=lbft_gamma,
            )
            return ce_loss + lbft_lambda * tversky_loss

        if focal_loss:
            return Focal_Loss(outputs, pngs, weights, num_classes=num_classes, alpha=focal_alpha, gamma=focal_gamma)
        return CE_Loss(outputs, pngs, weights, num_classes=num_classes, label_smoothing=label_smoothing)

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step: 
            break
        imgs, pngs, labels = batch

        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs    = imgs.cuda(local_rank)
                pngs    = pngs.cuda(local_rank)
                labels  = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)
            imgs, pngs, labels = apply_batch_mix(
                imgs, pngs, labels,
                mix_mode=mix_mode,
                mix_prob=mix_prob,
                mixup_alpha=mixup_alpha,
                cutmix_alpha=cutmix_alpha
            )
        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()
        if not fp16:
            #----------------------#
            #   前向传播
            #----------------------#
            outputs = model_train(imgs)
            #----------------------#
            #   计算损失
            #----------------------#
            loss = compute_train_loss(outputs, pngs, labels, weights)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels, class_start=dice_class_start)
                loss      = loss + dice_loss_weight * main_dice

            if component_aux:
                loss = loss + Component_Aux_Loss(
                    outputs,
                    pngs,
                    num_classes,
                    lesion_weight=component_lesion_weight,
                    boundary_weight=component_boundary_weight,
                    center_weight=component_center_weight,
                )

            if lesion_prior_loss:
                loss = loss + lesion_prior_weight * Lesion_Prior_Loss(
                    outputs,
                    pngs,
                    num_classes,
                    pos_weight_cap=lesion_prior_pos_weight_cap,
                )

            if severity_consistency_loss:
                loss = loss + severity_consistency_weight * Severity_Consistency_Loss(
                    outputs,
                    pngs,
                    num_classes,
                    loss_type=severity_loss_type,
                )

            with torch.no_grad():
                #-------------------------------#
                #   计算f_score
                #-------------------------------#
                _f_score = f_score(outputs, labels)

            #----------------------#
            #   反向传播
            #----------------------#
            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                #----------------------#
                #   前向传播
                #----------------------#
                outputs = model_train(imgs)
                #----------------------#
                #   计算损失
                #----------------------#
                loss = compute_train_loss(outputs, pngs, labels, weights)

                if dice_loss:
                    main_dice = Dice_loss(outputs, labels, class_start=dice_class_start)
                    loss      = loss + dice_loss_weight * main_dice

                if component_aux:
                    loss = loss + Component_Aux_Loss(
                        outputs,
                        pngs,
                        num_classes,
                        lesion_weight=component_lesion_weight,
                        boundary_weight=component_boundary_weight,
                        center_weight=component_center_weight,
                    )

                if lesion_prior_loss:
                    loss = loss + lesion_prior_weight * Lesion_Prior_Loss(
                        outputs,
                        pngs,
                        num_classes,
                        pos_weight_cap=lesion_prior_pos_weight_cap,
                    )

                if severity_consistency_loss:
                    loss = loss + severity_consistency_weight * Severity_Consistency_Loss(
                        outputs,
                        pngs,
                        num_classes,
                        loss_type=severity_loss_type,
                    )

                with torch.no_grad():
                    #-------------------------------#
                    #   计算f_score
                    #-------------------------------#
                    _f_score = f_score(outputs, labels)
                    
            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss      += loss.item()
        total_f_score   += _f_score.item()
            
        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'f_score'   : total_f_score / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs    = imgs.cuda(local_rank)
                pngs    = pngs.cuda(local_rank)
                labels  = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

            #----------------------#
            #   前向传播
            #----------------------#
            outputs     = model_train(imgs)
            #----------------------#
            #   计算损失
            #----------------------#
            loss = compute_val_loss(outputs, pngs, labels, weights)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels, class_start=dice_class_start)
                loss  = loss + dice_loss_weight * main_dice
            if lesion_prior_loss:
                loss = loss + lesion_prior_weight * Lesion_Prior_Loss(
                    outputs,
                    pngs,
                    num_classes,
                    pos_weight_cap=lesion_prior_pos_weight_cap,
                )
            if severity_consistency_loss:
                loss = loss + severity_consistency_weight * Severity_Consistency_Loss(
                    outputs,
                    pngs,
                    num_classes,
                    loss_type=severity_loss_type,
                )
            #-------------------------------#
            #   计算f_score
            #-------------------------------#
            _f_score    = f_score(outputs, labels)

            val_loss    += loss.item()
            val_f_score += _f_score.item()
            
            if local_rank == 0:
                pbar.set_postfix(**{'val_loss'  : val_loss / (iteration + 1),
                                    'f_score'   : val_f_score / (iteration + 1),
                                    'lr'        : get_lr(optimizer)})
                pbar.update(1)
            
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        current_miou = eval_callback.on_epoch_end(epoch + 1, model_train)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best val-loss model to best_val_loss_weights.pth and best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_val_loss_weights.pth"))
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

        best_miou_path = os.path.join(save_dir, "best_miou.txt")
        previous_best_miou = -1.0
        if os.path.exists(best_miou_path):
            try:
                with open(best_miou_path, "r", encoding="utf-8") as f:
                    previous_best_miou = float(f.read().strip())
            except ValueError:
                previous_best_miou = -1.0
        if current_miou is not None and current_miou >= previous_best_miou:
            print('Save best mIoU model to best_miou_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_miou_weights.pth"))
            with open(best_miou_path, "w", encoding="utf-8") as f:
                f.write(str(current_miou))
            
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
