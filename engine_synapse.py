from tqdm import tqdm
import numpy as np

from torch.cuda.amp import autocast as autocast
import torch

from sklearn.metrics import confusion_matrix

from scipy.ndimage.morphology import binary_fill_holes, binary_opening

from utils import test_single_volume

import time

def add_log_perturbation(predictions, labels, epsilon=0.05):
    """
    使用 PyTorch 根据样本比例计算 Log 值，并将其作为扰动加到分割预测结果上。

    参数:
    predictions: torch.Tensor, 形状为 [24, 9, 224, 224] 的分割预测结果
    labels: torch.Tensor, 形状为 [24, 224, 224] 的标签
    epsilon: float, 扰动强度，默认为 0.1

    返回:
    torch.Tensor: 加入扰动后的预测结果，形状为 [24, 9, 224, 224]
    """
    
    # 确保输入是 PyTorch 张量
    predictions = predictions.to(torch.float32)
    labels = labels.to(torch.long)

    # 获取类别数量
    num_classes = predictions.shape[1]

    # 计算每个类别的样本比例
    class_proportions = []
    for i in range(num_classes):
        class_mask = (labels == i)
        proportion = class_mask.float().mean()
        class_proportions.append(proportion)

    # 将比例转换为张量并添加小值以避免 log(0)
    class_proportions = torch.tensor(class_proportions, device=predictions.device) + 1e-10

    # 计算 log 值
    log_values = torch.log(class_proportions)

    # 将 log 值归一化到 [-1, 1] 范围
    log_values = 2 * (log_values - log_values.min()) / (log_values.max() - log_values.min()) - 1

    # 创建扰动张量
    perturbation = log_values.view(1, num_classes, 1, 1).expand_as(predictions)

    # 将扰动加到预测结果上
    perturbed_predictions = predictions + epsilon * perturbation

    return perturbed_predictions 
def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    logger, 
                    config, 
                    scaler=None):
    '''
    train model for one epoch
    '''
    stime = time.time()
    model.train() 
 
    loss_list = []

    for iter, data in enumerate(train_loader):
        optimizer.zero_grad()

        images, targets = data['image'], data['label']
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()   

        if config.amp:
            with autocast():
                out = model(images)
                out = add_log_perturbation(out,targets)
                loss = criterion(out, targets)      
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(images)
            out = add_log_perturbation(out,targets)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()

        loss_list.append(loss.item())
        now_lr = optimizer.state_dict()['param_groups'][0]['lr']
        mean_loss = np.mean(loss_list)
        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {loss.item():.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    scheduler.step()
    etime = time.time()
    log_info = f'Finish one epoch train: epoch {epoch}, loss: {mean_loss:.4f}, time(s): {etime-stime:.2f}'
    print(log_info)
    logger.info(log_info)
    return mean_loss





def val_one_epoch(test_datasets,
                    test_loader,
                    model,
                    epoch, 
                    logger,
                    config,
                    test_save_path,
                    val_or_test=False):
    # switch to evaluate mode
    stime = time.time()
    model.eval()
    with torch.no_grad():
        metric_list = 0.0
        i_batch = 0
        for data in tqdm(test_loader):
            img, msk, case_name = data['image'], data['label'], data['case_name'][0]
            metric_i = test_single_volume(img, msk, model, classes=config.num_classes, patch_size=[config.input_size_h, config.input_size_w],
                                    test_save_path=test_save_path, case=case_name, z_spacing=config.z_spacing, val_or_test=val_or_test)
            metric_list += np.array(metric_i)

            logger.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name,
                        np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
            i_batch += 1
        metric_list = metric_list / len(test_datasets)
        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        for i in range(1, config.num_classes):
            logger.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        etime = time.time()
        log_info = f'val epoch: {epoch}, mean_dice: {performance}, mean_hd95: {mean_hd95}, time(s): {etime-stime:.2f}'
        print(log_info)
        logger.info(log_info)
    
    return performance, mean_hd95