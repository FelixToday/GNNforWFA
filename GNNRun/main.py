# -*- coding: utf-8 -*-

# @Author: Xianjun Li
# @E-mail: xjli@mail.hnust.edu.cn
# @Date: 2025/12/8 下午9:07
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score
)
from typing import Dict, Optional, Union

from GNNModel.model import *
from GNNModel.dataset import GNNDataset
from GNNRun.utils import load_data
from lxj_utils_sys import str_to_bool, parse_args


# ===========================
# 指标计算函数
# ===========================
def compute_metrics_single_label(y_true, y_pred, verbose=False):
    """计算单标签分类指标"""
    metrics = {
        'accuracy': 100*accuracy_score(y_true, y_pred),
        'precision': 100*precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': 100*recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1': 100*f1_score(y_true, y_pred, average='macro', zero_division=0),
    }
    if verbose:
        print(f"accuracy: {metrics['accuracy']:.2f}%")
        print(f"precision: {metrics['precision']:.2f}%")
        print(f"recall: {metrics['recall']:.2f}%")
        print(f"f1: {metrics['f1']:.2f}%")
    return metrics
def compute_metrics_multi_label(
        y_true: Union[np.ndarray, torch.Tensor],
        y_pred_score: Union[np.ndarray, torch.Tensor],
        max_k: int = 5,
        return_k: Optional[int] = None,
        verbose: bool = True
) -> Dict[str, float]:
    """
    计算多标签分类的 Precision@k 和 Mean Precision@k 指标

    该实现计算的是 Micro Precision@k，即所有样本的 top-k 命中总数除以总预测次数。
    ap@k 是 p@1 到 p@k 的平均值（Mean Precision@k）。

    Args:
        y_true: 真实标签，形状为 (n_samples, n_classes)，二值矩阵（0或1）
        y_pred_score: 预测分数，形状为 (n_samples, n_classes)，数值越高表示预测为正类的概率越大
        max_k: 最大的 k 值，计算从 1 到 max_k 的所有指标，默认为 5
        return_k: 指定返回哪个 k 值的指标，如果为 None 则返回 max_k 对应的指标
        verbose: 是否打印每个 k 值的指标

    Returns:
        包含指标的字典，例如 {'p@5': 85.32, 'ap@5': 78.45}
        - p@k: Micro Precision at k (百分比)
        - ap@k: Mean Precision at k (p@1 到 p@k 的平均值，百分比)

    Example:
        >>> y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
        >>> y_pred = np.array([[0.8, 0.1, 0.7], [0.2, 0.9, 0.3], [0.7, 0.6, 0.2]])
        >>> metrics = compute_metrics_multi_label(y_true, y_pred, max_k=3, verbose=False)
        >>> print(metrics)
        {'p@3': 44.44, 'ap@3': 55.56}
    """
    # 转换为 numpy 数组
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred_score, torch.Tensor):
        y_pred_score = y_pred_score.cpu().numpy()

    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred_score = np.asarray(y_pred_score)

    # 验证输入形状
    if y_true.shape != y_pred_score.shape:
        raise ValueError(f"y_true 和 y_pred_score 形状不匹配: {y_true.shape} vs {y_pred_score.shape}")

    n_samples, n_classes = y_true.shape

    if n_samples == 0:
        raise ValueError("输入数据为空，无法计算指标")

    # 如果 max_k 大于类别数，自动调整
    if max_k > n_classes:
        max_k = n_classes
        if verbose:
            print(f"Warning: max_k 大于类别数，已自动调整为 {n_classes}")

    # 确定返回哪个 k 值的指标
    if return_k is None:
        return_k = max_k
    elif return_k > max_k:
        raise ValueError(f"return_k={return_k} 不能大于 max_k={max_k}")

    # 初始化真正例计数器 {k: count}
    # tp[k] 表示所有样本在 top-k 预测中的总命中次数
    tp = {k: 0 for k in range(1, max_k + 1)}

    # 获取每个样本的预测类别排序 (降序)
    # argsort 返回升序索引，所以用 [:, ::-1] 转为降序
    sorted_indices = np.argsort(y_pred_score, axis=1)[:, ::-1]

    # 遍历每个 k 值，向量化计算命中数
    for k in range(1, max_k + 1):
        # 获取所有样本的 top-k 预测索引
        top_k_preds = sorted_indices[:, :k]

        # 获取这些预测对应的真实标签
        # 使用高级索引：对每个样本，取其 top-k 预测位置的真实值
        rows = np.arange(n_samples)[:, np.newaxis]
        top_k_true = y_true[rows, top_k_preds]

        # 统计命中数（真实标签 > 0）
        tp[k] = np.sum(top_k_true > 0)

    # 计算指标
    results = {}
    cumulative_precision = 0.0

    for k in range(1, max_k + 1):
        # 计算 Precision@k
        # 分母是 总样本数 * k（所有预测次数）
        # 表示平均每个样本的 top-k 预测中，有多少比例是正确的
        p_k = tp[k] / (n_samples * k)
        cumulative_precision += p_k

        # 计算 Mean Precision@k (p@1 到 p@k 的平均值)
        mean_p_k = cumulative_precision / k

        # 存储结果（转换为百分比并保留4位小数）
        results[f'p@{k}'] = round(p_k * 100, 4)
        results[f'ap@{k}'] = round(mean_p_k * 100, 4)

        # 打印结果
        if verbose:
            print(f"p@{k}: {results[f'p@{k}']:.2f}%")
            print(f"ap@{k}: {results[f'ap@{k}']:.2f}%")

    # 返回指定 k 值的指标
    return {
        f'p@{return_k}': results[f'p@{return_k}'],
        f'ap@{return_k}': results[f'ap@{return_k}']
    }

dataset_lib = {
    "CW":{'num_tabs':1,'maximum_load_time': 80,'name':'CW'},
    "trafficsilver_bwr_CW":{'num_tabs':1,'maximum_load_time': 80,'name':'BWR'},
    "trafficsilver_rb_CW":{'num_tabs':1,'maximum_load_time': 80,'name':'RB'},
    "trafficsilver_bd_CW":{'num_tabs':1,'maximum_load_time': 80,'name':'BD'},
    "wtfpad_CW":{'num_tabs':1,'maximum_load_time': 80,'name':'Pad'},
    "front_CW":{'num_tabs':1,'maximum_load_time': 80,'name':'Front'},
    "regulator_CW":{'num_tabs':1,'maximum_load_time': 80,'name':'Regula'},
    "tamaraw_CW":{'num_tabs':1,'maximum_load_time': 80,'name':'Tamaraw'},

    "Closed_2tab":{'num_tabs':2,'maximum_load_time': 120,'name':'CW_2tab'},
    "Closed_3tab":{'num_tabs':3,'maximum_load_time': 120,'name':'CW_3tab'},
    "Closed_4tab":{'num_tabs':4,'maximum_load_time': 120,'name':'CW_4tab'},
    "Closed_5tab":{'num_tabs':5,'maximum_load_time': 120,'name':'CW_5tab'},
    "wtfpad_Closed_2tab":{'num_tabs':2,'maximum_load_time': 120,'name':'CW_Pad'},
    "front_Closed_2tab":{'num_tabs':2,'maximum_load_time': 120,'name':'CW_Fro'},
    "regulator_Closed_2tab":{'num_tabs':2,'maximum_load_time': 120,'name':'CW_Reg'},
    "Open_2tab":{'num_tabs':2,'maximum_load_time': 120,'name':'OW_2tab'},
    "Open_3tab":{'num_tabs':3,'maximum_load_time': 120,'name':'OW_3tab'},
    "Open_4tab":{'num_tabs':4,'maximum_load_time': 120,'name':'OW_4tab'},
    "Open_5tab":{'num_tabs':5,'maximum_load_time': 120,'name':'OW_5tab'},
    "wtfpad_Open_2tab":{'num_tabs':2,'maximum_load_time': 120,'name':'OW_Pad_2tab'},
    "front_Open_2tab":{'num_tabs':2,'maximum_load_time': 120,'name':'OW_Fro_2tab'},
    "regulator_Open_2tab":{'num_tabs':2,'maximum_load_time': 120,'name':'OW_Reg_2tab'},
    # "":{'num_tabs':,'maximum_load_time': ,'name':''},
}



# ===========================
# 配置参数
# ===========================
parser = argparse.ArgumentParser(description='模型训练参数配置')

# 训练参数
parser.add_argument('--batch_size', type=int, default=32, help='训练时每个 batch 的样本数量')
parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
parser.add_argument('--lr', type=float, default=0.001, help='学习率')
# 数据参数
parser.add_argument('--database_dir', type=str, default=r"/root/autodl-tmp/dataset/wfa/npz_dataset", help='数据存储的位置')
parser.add_argument('--dataset', type=str, default=r"Closed_2tab", help='选择的数据集')
parser.add_argument('--note', type=str, default=r"MG2_DG2", help='运行结果保存的文件夹')
parser.add_argument('--loaded_ratio', type=int, default=100, help='加载的数据比例（%）')
parser.add_argument('--TAM_type', type=str, default='G1', help='提取特征的方法')
parser.add_argument('--seq_len', type=int, default=5000, help='流量最大长度')
parser.add_argument('--level_count', type=int, default=18, help='图中节点数量')
parser.add_argument('--max_matrix_len', type=int, default=100, help='提取特征矩阵的最大长度')
parser.add_argument('--log_transform', type=str_to_bool, default=True, help='是否对特征做对数变换')
parser.add_argument('--maximum_load_time', type=int, default=80, help='最大加载时间（秒）')
parser.add_argument('--is_idx', type=str_to_bool, default=False, help='是否使用索引')
parser.add_argument('--is_test', type=str_to_bool, default=False, help='是否为测试模式')
parser.add_argument('--verbose_metrics', type=str_to_bool, default=False, help='是否输出metrics')

# 模型参数
parser.add_argument('--model', type=str, default='STGCN_G1', help='使用的模型')

# 系统参数
parser.add_argument('--checkpoint_path', type=str, default='../checkpoints', help='保存最优模型的路径')
parser.add_argument('--num_workers', type=int, default=16, help='数据集加载的进程数')
parser.add_argument('--early_stopping_patience', type=int, default=10, help='早停耐心值')

CONFIG = parse_args(parser, is_print_help=True)

CONFIG['note'] = f'M_{CONFIG["model"]}_D_{CONFIG["TAM_type"]}'
CONFIG['problem_type'] = 'single_label' if dataset_lib[CONFIG['dataset']]['num_tabs'] == 1 else'multi_label'
CONFIG["checkpoint_path"] = os.path.join(CONFIG["checkpoint_path"], CONFIG["dataset"], CONFIG["note"])
verbose = CONFIG['verbose_metrics']



if CONFIG['is_test']:
    CONFIG['epochs']=3
    CONFIG["n_samples"]=CONFIG['batch_size'] * 10
else:
    CONFIG["n_samples"] = -1

CONFIG['model_path'] = os.path.join(CONFIG["checkpoint_path"], "best_model.pth")
CONFIG['result_path'] = os.path.join(CONFIG["checkpoint_path"], 'training_results.json')
CONFIG['test_result_path'] = os.path.join(CONFIG["checkpoint_path"], 'test_results.json')

# ===========================
# 数据加载
# ===========================
database_dir = str(os.path.join(CONFIG['database_dir'], CONFIG['dataset']))
X_train, y_train = load_data(os.path.join(database_dir, "train.npz"))
X_val, y_val = load_data(os.path.join(database_dir, "valid.npz"))
X_test, y_test = load_data(os.path.join(database_dir, "test.npz"))

if dataset_lib[CONFIG['dataset']]['num_tabs'] == 1:
    num_classes = len(np.unique(y_train))
else:
    num_classes = y_train.shape[1]

max_k=5
return_k=dataset_lib[CONFIG['dataset']]['num_tabs']
# 限制样本数量（调试用）
if CONFIG["n_samples"] != -1:
    X_train, y_train = X_train[:CONFIG["n_samples"]], y_train[:CONFIG["n_samples"]]
    X_val, y_val = X_val[:CONFIG["n_samples"]], y_val[:CONFIG["n_samples"]]
    X_test, y_test = X_test[:CONFIG["n_samples"]], y_test[:CONFIG["n_samples"]]

# 创建数据集
train_dataset = GNNDataset(X_train, y_train,
                           loaded_ratio=CONFIG["loaded_ratio"],
                           TAM_type=CONFIG["TAM_type"],
                           seq_len=CONFIG["seq_len"],
                           max_matrix_len=CONFIG["max_matrix_len"],
                           log_transform=CONFIG["log_transform"],
                           maximum_load_time=CONFIG["maximum_load_time"],
                           is_idx=CONFIG["is_idx"],
                           level_count=CONFIG["level_count"])

val_dataset = GNNDataset(X_val, y_val,
                         loaded_ratio=CONFIG["loaded_ratio"],
                         TAM_type=CONFIG["TAM_type"],
                         seq_len=CONFIG["seq_len"],
                         max_matrix_len=CONFIG["max_matrix_len"],
                         log_transform=CONFIG["log_transform"],
                         maximum_load_time=CONFIG["maximum_load_time"],
                         is_idx=CONFIG["is_idx"],
                         level_count=CONFIG["level_count"])

test_dataset = GNNDataset(X_test, y_test,
                          loaded_ratio=CONFIG["loaded_ratio"],
                          TAM_type=CONFIG["TAM_type"],
                          seq_len=CONFIG["seq_len"],
                          max_matrix_len=CONFIG["max_matrix_len"],
                          log_transform=CONFIG["log_transform"],
                          maximum_load_time=CONFIG["maximum_load_time"],
                          is_idx=CONFIG["is_idx"],
                          level_count=CONFIG["level_count"])

# 创建数据加载器
num_workers = CONFIG["num_workers"]
train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"],
                          shuffle=True, num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"],
                        shuffle=False, num_workers=num_workers, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"],
                         shuffle=False, num_workers=num_workers, pin_memory=True)

# 获取特征维度
CONFIG["n_dim"] = train_dataset[0][0][0].shape[-1]

# ===========================
# 初始化模型
# ===========================
model = eval(CONFIG["model"])(input_dim=CONFIG["n_dim"],
                              num_classes=num_classes,
                              level_count=CONFIG["level_count"])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 定义损失函数
if CONFIG["problem_type"] == "single_label":
    criterion = nn.CrossEntropyLoss()
else:
    #criterion = nn.BCEWithLogitsLoss()
    criterion = nn.MultiLabelSoftMarginLoss()

optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])

# 学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                 factor=0.5, patience=5, verbose=verbose)

# 训练历史记录
training_history = {
    'config': CONFIG,
    'epochs': [],
    'best_epoch': 0,
    'best_metric': 0.0,
    'final_train_metrics': {},
    'final_val_metrics': {}
}

# 早停计数器
early_stopping_counter = 0

# 创建检查点目录
os.makedirs(CONFIG["checkpoint_path"], exist_ok=True)

# 保存配置
with open(os.path.join(CONFIG["checkpoint_path"], 'config.json'), 'w') as f:
    json.dump(CONFIG, f, indent=2, default=str)


# ===========================
# 训练函数
# ===========================
def train_epoch(model, train_loader, criterion, optimizer, device, config):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (data, y) in enumerate(pbar):
        # 数据移动到设备
        X, A, y = data[0].to(device), data[1].to(device), y.to(device)

        # 根据问题类型处理标签
        if config['problem_type'] == 'multi_label':
            y = y.float()
        elif config['problem_type'] == 'single_label':
            y = y.long()
            if y.dim() > 1 and y.size(1) > 1:
                y = torch.argmax(y, dim=1)

        # 前向传播
        optimizer.zero_grad()
        logits = model(A, X)
        # 计算损失
        loss = criterion(logits, y)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 累计损失
        running_loss += loss.item() * y.size(0)

        # 收集预测和标签用于计算指标
        with torch.no_grad():
            if config['problem_type'] == 'multi_label':
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).int()
                all_probs.append(probs.cpu())
            else:
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                all_probs.append(probs.cpu())

            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

        # 更新进度条
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # 计算平均损失
    train_loss = running_loss / len(train_loader.dataset)

    # 计算训练指标
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_probs = torch.cat(all_probs, dim=0).numpy()

    if config['problem_type'] == 'single_label':
        metrics = compute_metrics_single_label(all_labels, all_preds, verbose=verbose)
    else:
        metrics = compute_metrics_multi_label(
            y_true=all_labels,
            y_pred_score=all_probs,
            max_k=max_k,
            return_k=return_k,
            verbose=verbose
        )

    metrics['loss'] = train_loss

    return metrics


# ===========================
# 评估函数
# ===========================
def evaluate(model, data_loader, criterion, device, config, split_name="Validation"):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0.0

    with torch.no_grad():
        pbar = tqdm(data_loader, desc=f"{split_name}")
        for data, y in pbar:
            X, A, y = data[0].to(device), data[1].to(device), y.to(device)

            # 处理标签
            if config['problem_type'] == 'multi_label':
                y = y.float()
            elif config['problem_type'] == 'single_label':
                y = y.long()
                if y.dim() > 1 and y.size(1) > 1:
                    y = torch.argmax(y, dim=1)

            # 前向传播
            logits = model(A, X)

            # 计算损失
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)

            # 收集预测
            if config['problem_type'] == 'multi_label':
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).int()
                all_probs.append(probs.cpu())
            else:
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                all_probs.append(probs.cpu())

            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # 计算指标
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_probs = torch.cat(all_probs, dim=0).numpy()

    avg_loss = total_loss / len(data_loader.dataset)

    if config['problem_type'] == 'single_label':
        metrics = compute_metrics_single_label(all_labels, all_preds, verbose=verbose)
    else:
        metrics = compute_metrics_multi_label(
            y_true=all_labels,
            y_pred_score=all_probs,
            max_k=max_k,
            return_k=return_k,
            verbose=verbose
        )

    metrics['loss'] = avg_loss

    # 打印结果
    print(f"\n{split_name} Results:")
    print("=" * 50)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        elif isinstance(value, list):
            # 如果值是列表，只打印前5个和后5个避免输出过长
            if len(value) > 10:
                print(f"{key}: [{', '.join([f'{v:.4f}' for v in value[:5]])} ... {', '.join([f'{v:.4f}' for v in value[-5:]])}]")
            else:
                print(f"{key}: {[round(v, 4) for v in value]}")
    print("=" * 50)

    return metrics


# ===========================
# 模型保存函数
# ===========================

# ===========================
# 测试函数
# ===========================
def test(model, test_loader, criterion, device, config):
    """测试函数"""
    print("\n" + "=" * 60)
    print("开始测试...")
    print("=" * 60)

    # 加载最优模型
    if os.path.exists(config['model_path']):
        print(f"加载最优模型: {config['model_path']}")
        checkpoint = torch.load(config['model_path'], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_epoch = checkpoint['epoch']
    else:
        print("警告: 未找到最优模型，使用当前模型进行测试")
        best_epoch = config.get('best_epoch', 0)

    # 在测试集上评估
    test_metrics = evaluate(model, test_loader, criterion, device, config, split_name="Test")

    # 保存测试结果
    test_results = {
        'best_epoch': best_epoch,
        'test_metrics': test_metrics,
        'model_path': config['model_path']
    }

    with open(config['test_result_path'], 'w') as f:
        json.dump(test_results, f, indent=2, default=str)

    print(f"\n测试结果已保存到: {config['test_result_path']}")

    return test_metrics


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)
    print(f"问题类型: {CONFIG['problem_type']}")
    print(f"模型: {CONFIG['model']}")
    print(f"设备: {device}")
    print(f"检查点路径: {CONFIG['checkpoint_path']}")
    print("=" * 60)

    best_val_metric = 0.0
    best_epoch = 0
    early_stopping_counter = 0

    for epoch in range(CONFIG['epochs']):
        print(f"\nEpoch {epoch + 1}/{CONFIG['epochs']}")
        print("-" * 60)

        # 训练
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, CONFIG)

        # 验证
        val_metrics = evaluate(model, val_loader, criterion, device, CONFIG, split_name="Validation")

        # 记录历史
        epoch_info = {
            'epoch': epoch + 1,
            'train': train_metrics,
            'val': val_metrics
        }
        training_history['epochs'].append(epoch_info)

        # 确定用于选择最优模型的指标
        if CONFIG['problem_type'] == 'single_label':
            # 单标签：使用F1 macro作为优化指标
            current_metric = val_metrics['f1']
        else:
            # 多标签：使用MAP作为优化指标
            current_metric = val_metrics[f'ap@{return_k}']

        # 更新学习率
        scheduler.step(current_metric)

        # 检查是否是最优模型
        is_best = current_metric > best_val_metric
        if is_best:
            best_val_metric = current_metric
            best_epoch = epoch + 1
            training_history['best_epoch'] = best_epoch
            training_history['best_metric'] = best_val_metric
            early_stopping_counter = 0
            print(f"✓ 新最优模型! {('F1' if CONFIG['problem_type'] == 'single_label' else 'MAP')}: {best_val_metric:.2f}")
        else:
            early_stopping_counter += 1
            print(f"未改善，耐心值: {early_stopping_counter}/{CONFIG['early_stopping_patience']}")

        # 保存检查点
        if is_best:
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "epoch": best_epoch,
            }
            torch.save(checkpoint, CONFIG['model_path'])

        # 保存训练历史
        with open(CONFIG['result_path'], 'w') as f:
            json.dump(training_history, f, indent=2, default=str)

        # 早停检查
        if early_stopping_counter >= CONFIG['early_stopping_patience']:
            print(f"\n早停触发！连续 {CONFIG['early_stopping_patience']} 个epoch未改善")
            break

    # 保存最终训练指标
    training_history['final_train_metrics'] = train_metrics
    training_history['final_val_metrics'] = val_metrics

    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"最优验证指标: {best_val_metric:.4f} (Epoch {best_epoch})")
    print(f"训练历史已保存到: {CONFIG['result_path']}")
    print("=" * 60)

    # 测试
    test_metrics = test(model, test_loader, criterion, device, CONFIG)

