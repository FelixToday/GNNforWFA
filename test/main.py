import torch


def construct_A_local_batch(A_batch):
    """
    构造批量的块三对角矩阵 A_local

    参数:
    A_batch: 形状为 (B, K, N, N) 的张量
        B: batch size
        K: 矩阵个数
        N: 每个矩阵的维度

    返回:
    A_local: 形状为 (B, K*N, K*N) 的块三对角矩阵
    """
    B, K, N, _ = A_batch.shape
    total_size = K * N

    # 获取设备信息
    device = A_batch.device
    dtype = A_batch.dtype

    # 初始化全零矩阵
    A_local = torch.zeros((B, total_size, total_size), device=device, dtype=dtype)

    # 填充矩阵
    for i in range(K):
        # 主对角线块：A_batch[:, i, :, :]
        start_row = i * N
        end_row = (i + 1) * N
        A_local[:, start_row:end_row, start_row:end_row] = A_batch[:, i, :, :]

        # 上对角线的单位矩阵块 (除了最后一个块)
        if i < K - 1:
            eye_matrix = torch.eye(N, device=device, dtype=dtype).unsqueeze(0).expand(B, N, N)
            A_local[:, start_row:end_row, end_row:end_row + N] = eye_matrix

        # 下对角线的单位矩阵块 (除了第一个块)
        if i > 0:
            eye_matrix = torch.eye(N, device=device, dtype=dtype).unsqueeze(0).expand(B, N, N)
            A_local[:, start_row:end_row, start_row - N:start_row] = eye_matrix

    return A_local


# 更高效的向量化版本
def construct_A_local_batch_vectorized(A_batch):
    """
    向量化版本的块三对角矩阵构造 (更高效)
    """
    B, K, N, _ = A_batch.shape
    total_size = K * N

    device = A_batch.device
    dtype = A_batch.dtype

    # 初始化全零矩阵
    A_local = torch.zeros((B, total_size, total_size), device=device, dtype=dtype)

    # 创建对角线索引
    diag_indices = torch.arange(K, device=device)
    row_indices = diag_indices * N
    col_indices = diag_indices * N

    # 批量设置对角线块
    for i in range(K):
        start_row = i * N
        end_row = (i + 1) * N
        A_local[:, start_row:end_row, start_row:end_row] = A_batch[:, i, :, :]

    # 创建单位矩阵并扩展到批量
    eye_batch = torch.eye(N, device=device, dtype=dtype).unsqueeze(0).expand(B, N, N)

    # 设置上对角线和下对角线
    for i in range(K - 1):
        # 上对角线
        start_row = i * N
        end_row = (i + 1) * N
        start_col = (i + 1) * N
        end_col = (i + 2) * N
        A_local[:, start_row:end_row, start_col:end_col] = eye_batch

        # 下对角线
        A_local[:, start_col:end_col, start_row:end_row] = eye_batch

    return A_local


# 示例用法
if __name__ == "__main__":
    # 创建批量数据
    B = 2  # batch size
    K = 3  # 矩阵个数
    N = 2  # 每个矩阵的维度

    # 创建批量数据 (B, K, N, N)
    A_batch = torch.tensor([
        # 第一个batch
        [
            [[1., 2.], [3., 4.]],  # A[0]
            [[5., 6.], [7., 8.]],  # A[1]
            [[9., 10.], [11., 12.]]  # A[2]
        ],
        # 第二个batch
        [
            [[13., 14.], [15., 16.]],
            [[17., 18.], [19., 20.]],
            [[21., 22.], [23., 24.]]
        ]
    ])

    print(f"输入形状: {A_batch.shape}")  # 应该是 (2, 3, 2, 2)

    # 使用函数
    A_local = construct_A_local_batch_vectorized(A_batch)

    print(f"输出形状: {A_local.shape}")  # 应该是 (2, 6, 6)

    print("\n第一个batch的A_local:")
    print(A_local[0])

    print("\n第二个batch的A_local:")
    print(A_local[1])

    # 验证结构
    print("\n验证结构:")
    for b in range(B):
        print(f"\nBatch {b}:")
        for i in range(K):
            start = i * N
            end = (i + 1) * N
            diagonal_block = A_local[b, start:end, start:end]
            print(f"  位置 {i}:")
            print(f"  期望: {A_batch[b, i]}")
            print(f"  实际: {diagonal_block}")
            print(f"  是否相等: {torch.allclose(A_batch[b, i], diagonal_block)}")
