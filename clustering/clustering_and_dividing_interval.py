import os
import json
import time
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(path):
    """
    加载数据函数
    假设数据文件是.npz格式，包含'X'键
    """
    try:
        data = np.load(path)
        X = data['X']
        y = data.get('y', None)
        logger.info(f"成功加载数据: {path}, X形状: {X.shape}")
        return X, y
    except FileNotFoundError:
        logger.error(f"数据文件未找到: {path}")
        raise
    except Exception as e:
        logger.error(f"加载数据失败: {path}, 错误: {e}")
        raise


def clustring_traffic(database_dir, clustering_num, dataset_type="valid.npz"):
    """
    对网络流量数据按数据包大小进行聚类

    参数:
        database_dir: str, 数据集目录路径
        clustering_num: int, 聚类簇数量
        dataset_type: str, 数据集文件名，默认为"valid.npz"

    返回:
        tuple: (list[float], list[int], list[float], np.ndarray) -
               簇中心列表(保留两位小数)、每个簇的样本数量列表、每个簇的样本比率列表(小数形式，保留4位)、
               原始数据包大小数组（非零值，用于后续区间统计）
    """
    try:
        # 参数验证
        if not isinstance(clustering_num, int) or clustering_num <= 0:
            raise ValueError(f"clustering_num必须是正整数，当前值: {clustering_num}")

        # 加载数据
        data_path = os.path.join(database_dir, dataset_type)
        logger.info(f"开始加载数据: {data_path}")

        X_train, _ = load_data(data_path)

        # 验证数据维度
        if X_train.ndim != 3 or X_train.shape[2] != 2:
            raise ValueError(f"数据维度错误，期望 (n_samples, 10000, 2)，实际: {X_train.shape}")

        # 提取数据包大小并取绝对值（不分上下行）
        packet_sizes = np.abs(X_train[:, :, 1])

        # 展平为一维数组
        packet_sizes_flat = packet_sizes.flatten()
        logger.info(f"提取包大小，展平后形状: {packet_sizes_flat.shape}")

        # 过滤掉0值（无数据包的位置）
        packet_sizes_nonzero = packet_sizes_flat[packet_sizes_flat > 0]
        logger.info(f"过滤0值后，有效包数量: {packet_sizes_nonzero.shape[0]}")

        if len(packet_sizes_nonzero) == 0:
            logger.warning("没有有效的数据包大小值，返回空列表")
            return [], [], [], np.array([])

        # 如果数据量过大，进行随机采样以提高效率（仅用于聚类）
        max_samples = 1000000
        if len(packet_sizes_nonzero) > max_samples:
            logger.info(f"数据量过大 ({len(packet_sizes_nonzero):,}), 采样到 {max_samples:,}用于聚类")
            np.random.seed(42)
            indices = np.random.choice(len(packet_sizes_nonzero), max_samples, replace=False)
            packet_samples = packet_sizes_nonzero[indices]
        else:
            packet_samples = packet_sizes_nonzero

        # 重塑为二维数组
        packet_samples_reshaped = packet_samples.reshape(-1, 1)

        # 使用MiniBatchKMeans进行聚类
        logger.info(f"开始MiniBatchKMeans聚类，簇数量: {clustering_num}")
        kmeans = MiniBatchKMeans(
            n_clusters=clustering_num,
            random_state=42,
            batch_size=10000,
            n_init=10,
            max_iter=100,
            verbose=0
        )
        kmeans.fit(packet_samples_reshaped)

        # 获取簇标签并统计每个簇的样本数量
        labels = kmeans.labels_
        # 统计原始顺序的簇样本数量
        original_counts = np.bincount(labels, minlength=clustering_num)

        # 获取排序后的簇中心索引
        sorted_indices = np.argsort(kmeans.cluster_centers_.flatten())

        # 根据排序后的索引重新排列样本数量
        cluster_counts = original_counts[sorted_indices].tolist()

        # 计算总样本数和比率
        total_samples = sum(cluster_counts)
        cluster_ratios = [round(count / total_samples, 4) for count in cluster_counts]

        # 四舍五入到两位小数并转换为Python float列表
        cluster_centers = [round(float(c), 2) for c in np.sort(kmeans.cluster_centers_.flatten())]

        # 日志输出
        logger.info(f"聚类完成，簇中心 (已排序): {cluster_centers}")
        logger.info(f"每个簇的样本数量: {cluster_counts}")
        logger.info(f"每个簇的样本比率: {[f'{ratio:.2%}' for ratio in cluster_ratios]}")

        # 返回原始数据包大小数组（非零值），用于后续区间统计
        return cluster_centers, cluster_counts, cluster_ratios, packet_sizes_nonzero

    except Exception as e:
        logger.error(f"聚类过程出错: {e}")
        raise


def dividing_interval(lb, ub, cluster, type="direct", packet_sizes=None):
    """
    根据聚类中心生成数据包大小的划分区间，并可选择性地计算每个区间的样本数量

    参数:
        lb: float, 区间下界（最小边界）
        ub: float, 区间上界（最大边界）
        cluster: list[float], 已排序的聚类中心列表
        type: str, 划分类型
              "direct" - 直接使用聚类中心作为分割点
              "diff" - 使用相邻聚类中心的中点作为分割点
        packet_sizes: np.ndarray, 可选，原始数据包大小数组（非零值），用于统计每个区间的样本数量

    返回:
        tuple: (list[float], list[int], list[float]) - 区间边界列表、每个区间的样本数量列表、每个区间的样本比率列表
    """
    # 参数验证
    if not isinstance(lb, (int, float)) or not isinstance(ub, (int, float)):
        raise TypeError("lb 和 ub 必须是数值类型")

    if lb >= ub:
        raise ValueError(f"下界 lb ({lb}) 必须小于上界 ub ({ub})")

    if not isinstance(cluster, list):
        raise TypeError("cluster 必须是列表类型")

    # 处理空集群情况
    if not cluster:
        logger.warning("聚类中心列表为空，返回仅包含边界的区间")
        if packet_sizes is not None:
            total_samples = len(packet_sizes)
            interval_counts = [total_samples]
            interval_ratios = [1.0]
        else:
            interval_counts = []
            interval_ratios = []

        divide_interval = [round(float(lb), 2), round(float(ub), 2)]
        return divide_interval, interval_counts, interval_ratios

    # 确保所有聚类中心在 [lb, ub] 范围内（警告但不阻止）
    if min(cluster) < lb or max(cluster) > ub:
        logger.warning(f"部分聚类中心超出边界范围 [{lb}, {ub}]，可能导致不合理区间")

    # 根据类型生成分割点
    if type == "direct":
        # 直接使用聚类中心作为分割点
        divide_interval = [lb] + cluster + [ub]

    elif type == "diff":
        # 使用相邻聚类中心的中点作为分割点
        if len(cluster) == 1:
            # 只有一个聚类中心时，退化为 direct 模式
            logger.warning("只有一个聚类中心，diff 模式退化为 direct 模式")
            divide_interval = [lb] + cluster + [ub]
        else:
            # 计算相邻聚类中心的中点
            mid_points = [(cluster[i] + cluster[i + 1]) / 2.0
                          for i in range(len(cluster) - 1)]
            divide_interval = [lb] + mid_points + [ub]

    else:
        raise ValueError(f"不支持的 type 参数: '{type}'，仅支持 'direct' 或 'diff'")

    # 四舍五入到两位小数
    divide_interval = [round(float(x), 2) for x in divide_interval]

    # 验证结果是否单调递增
    if not all(divide_interval[i] < divide_interval[i + 1] for i in range(len(divide_interval) - 1)):
        logger.error(f"生成的区间不是单调递增: {divide_interval}")
        raise RuntimeError("生成的区间无效：不是单调递增")

    # 如果提供了原始数据包大小，统计每个区间的样本数量
    if packet_sizes is not None:
        logger.info(f"开始统计区间样本数量，有效包总数: {len(packet_sizes):,}")

        # 使用 np.histogram 统计每个区间的样本数量
        # bins 是区间边界，统计结果会落在 [bin[i], bin[i+1]) 区间内
        interval_counts, _ = np.histogram(packet_sizes, bins=divide_interval)

        # 转换为列表
        interval_counts = interval_counts.tolist()

        # 计算总样本数和比率
        total_samples = sum(interval_counts)
        if total_samples > 0:
            interval_ratios = [round(count / total_samples, 4) for count in interval_counts]
        else:
            interval_ratios = [0.0] * len(interval_counts)

        logger.info(f"区间统计完成，总样本数: {total_samples:,}")
        logger.info(f"区间边界: {divide_interval}")
        logger.info(f"区间样本数量: {interval_counts}")
        logger.info(f"区间样本比率: {[f'{ratio:.2%}' for ratio in interval_ratios]}")

        return divide_interval, interval_counts, interval_ratios
    else:
        logger.info(f"未提供原始数据包大小，仅返回区间边界: {divide_interval}")
        return divide_interval, [], []


def generate_packet_size_intervals(database_dir, interval_num, type="direct",
                                   dataset_type="valid.npz", lb=0.0, ub=1500.0,
                                   save_json=True, json_path="interval_info.json"):
    """
    根据指定的区间数量生成数据包大小划分区间，并保存结果到JSON文件

    参数:
        database_dir: str, 数据集目录路径
        interval_num: int, 期望的区间数量
        type: str, 划分类型 - "direct" 或 "diff"
        dataset_type: str, 数据集文件名，默认为"valid.npz"
        lb: float, 区间下界，默认为0.0
        ub: float, 区间上界，默认为1500.0
        save_json: bool, 是否保存结果到JSON文件，默认为True
        json_path: str, JSON文件保存路径，默认为"interval_info.json"

    返回:
        tuple: (list[float], list[int], list[float]) - 区间边界、区间样本数、区间样本比率

    示例:
        >>> intervals, counts, ratios = generate_packet_size_intervals(
        ...     "/path/to/data", interval_num=5, type="direct"
        ... )
        >>> print(intervals)  # [0.0, 167.39, 553.57, 1033.37, 1289.28, 1500.0]
        >>> print(counts)     # [14850, 22950, 17950, 11950, 4950, 250]
        >>> print(ratios)     # [0.2034, 0.3144, 0.2459, 0.1637, 0.0068, 0.0034]
    """
    # 参数验证
    if type not in ["direct", "diff"]:
        raise ValueError(f"不支持的 type: '{type}'，仅支持 'direct' 或 'diff'")

    if not isinstance(interval_num, int) or interval_num <= 0:
        raise ValueError(f"interval_num 必须是正整数，当前值: {interval_num}")

    # 根据 type 计算 cluster_num
    if type == "direct":
        cluster_num = interval_num - 1
        if cluster_num <= 0:
            raise ValueError(f"direct 模式下 interval_num 必须大于 1，当前值: {interval_num}")
        logger.info(f"Direct 模式: 区间数={interval_num} -> 聚类数={cluster_num}")
    else:  # diff
        cluster_num = interval_num
        logger.info(f"Diff 模式: 区间数={interval_num} -> 聚类数={cluster_num}")

    # 执行聚类，获取聚类中心和原始数据包大小
    logger.info(f"开始执行聚类，数据集: {dataset_type}")
    centers, cluster_counts, cluster_ratios, packet_sizes = clustring_traffic(
        database_dir, cluster_num, dataset_type
    )

    if not centers:
        logger.warning("聚类失败，返回空结果")
        return [], [], []

    # 生成区间并统计样本分布
    logger.info(f"开始生成区间，类型: {type}, 边界: [{lb}, {ub}]")
    intervals, interval_counts, interval_ratios = dividing_interval(
        lb, ub, centers, type=type, packet_sizes=packet_sizes
    )

    # 验证区间数量
    actual_interval_num = len(intervals) - 1
    if actual_interval_num != interval_num:
        logger.warning(f"实际生成的区间数量 ({actual_interval_num}) 与期望的 ({interval_num}) 不符")

    # 保存结果到JSON文件
    if save_json:
        try:
            # 构建JSON数据结构
            json_data = {
                "metadata": {
                    "database_dir": os.path.abspath(database_dir),
                    "dataset_type": dataset_type,
                    "interval_num": interval_num,
                    "type": type,
                    "lb": lb,
                    "ub": ub,
                    "cluster_num": cluster_num,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    "total_samples": sum(interval_counts),
                    "actual_interval_num": actual_interval_num
                },
                "intervals": intervals,
                "counts": interval_counts,
                "ratios": interval_ratios,
                "details": []
            }

            # 添加每个区间的详细信息
            for i in range(len(intervals) - 1):
                detail = {
                    "interval_index": i,
                    "range": f"[{intervals[i]}, {intervals[i + 1]})",
                    "lower_bound": intervals[i],
                    "upper_bound": intervals[i + 1],
                    "count": interval_counts[i],
                    "ratio": interval_ratios[i]
                }
                json_data["details"].append(detail)

            # 保存到JSON文件
            json_path = os.path.abspath(json_path)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)

            logger.info(f"结果已保存到 JSON 文件: {json_path}")
            print(f"\n✓ 结果已保存到: {json_path}")

        except Exception as e:
            logger.error(f"保存JSON文件失败: {e}")
            print(f"\n⚠ 警告: 保存JSON文件失败: {e}")

    # 打印最终摘要
    logger.info("=" * 60)
    logger.info("最终生成结果摘要:")
    logger.info(f"区间边界: {intervals}")
    logger.info(f"区间样本数: {interval_counts}")
    logger.info(f"区间样本比率: {[f'{r:.2%}' for r in interval_ratios]}")

    return intervals, interval_counts, interval_ratios


# 使用示例
if __name__ == "__main__":
    # 配置参数
    database_dir = "/root/autodl-tmp/dataset/wfa/npz_dataset/CW"
    dataset_type = "valid.npz"
    lb = 0.0
    ub = 1500.0
    json_output_dir = "./clustering_results"  # JSON文件保存目录

    # 确保输出目录存在
    os.makedirs(json_output_dir, exist_ok=True)

    # 测试不同的配置
    test_configs = [
        {"interval_num": 15, "type": "direct", "name": "direct"},
        {"interval_num": 15, "type": "diff", "name": "diff"},
    ]

    print("=" * 80)
    print("数据包大小区间划分工具 - JSON 保存模式")
    print("=" * 80)

    results = {}

    for config in test_configs:
        interval_num = config["interval_num"]
        type = config["type"]
        name = config["name"]
        json_path = os.path.join(json_output_dir, f"interval_info_{name}.json")

        print(f"\n{'=' * 80}")
        print(f"配置: interval_num={interval_num}, type='{type}'")
        print(f"JSON 文件: {json_path}")
        print(f"{'=' * 80}")

        try:
            # 调用主函数
            intervals, counts, ratios = generate_packet_size_intervals(
                database_dir=database_dir,
                interval_num=interval_num,
                type=type,
                dataset_type=dataset_type,
                lb=lb,
                ub=ub,
                save_json=True,
                json_path=json_path
            )

            # 存储结果
            results[name] = {
                "intervals": intervals,
                "counts": counts,
                "ratios": ratios,
                "json_path": json_path
            }

            # 打印详细结果
            if intervals and counts:
                print(f"\n✓ 成功生成 {len(intervals) - 1} 个区间")
                print(f"\n区间边界:")
                print(f"  {intervals}")

                print(f"\n区间样本分布:")
                total_samples = sum(counts)
                for i in range(len(intervals) - 1):
                    range_str = f"[{intervals[i]}, {intervals[i + 1]})"
                    print(f"  区间 {i + 1:2d}: {range_str:<25} 样本数: {counts[i]:>8,} ({ratios[i]:.2%})")

                print(f"\n总计: {total_samples:,} 个数据包")

        except Exception as e:
            print(f"\n✗ 执行失败: {e}")
            logger.error(f"配置失败: {config}, 错误: {e}")

    # 打印所有配置的对比总结
    print(f"\n{'=' * 80}")
    print("所有配置对比总结")
    print(f"{'=' * 80}")
    print(f"{'配置名称':<15} {'区间数':<8} {'总样本数':<12} {'JSON文件':<50}")
    print("-" * 80)

    for name, result in results.items():
        intervals = result["intervals"]
        counts = result["counts"]
        json_path = result["json_path"]
        total_samples = sum(counts) if counts else 0
        print(f"{name:<15} {len(intervals) - 1:<8} {total_samples:<12,} {json_path}")

    # 读取并展示一个JSON文件的内容示例
    if results:
        first_key = list(results.keys())[0]
        first_json_path = results[first_key]["json_path"]

        print(f"\n{'=' * 80}")
        print(f"JSON 文件内容示例: {os.path.basename(first_json_path)}")
        print(f"{'=' * 80}")

        try:
            with open(first_json_path, 'r', encoding='utf-8') as f:
                json_content = json.load(f)

            # 打印metadata部分
            print("\nMetadata 部分:")
            for key, value in json_content["metadata"].items():
                print(f"  {key}: {value}")

            # 打印前3个区间的详细信息
            print(f"\nDetails 部分 (前3个区间):")
            for i, detail in enumerate(json_content["details"][:3]):
                print(f"  区间 {i}: {detail['range']} - 样本数: {detail['count']:,} ({detail['ratio']:.2%})")

            if len(json_content["details"]) > 3:
                print(f"  ... 还有 {len(json_content['details']) - 3} 个区间")

        except Exception as e:
            print(f"读取JSON文件失败: {e}")

