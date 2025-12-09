import numpy as np
def load_data(data_path, drop_extra_time=False, load_time=None):
    data = np.load(data_path)
    X = data["X"]
    y = data["y"]
    # 时间负数调整
    X[:, :, 0] = np.abs(X[:, :, 0])
    # 去除大小信息
    #X[:, :, 1] = np.sign(X[:, :, 1])
    if drop_extra_time and load_time is not None:
        print(f"丢弃额外时间，时间上限：{load_time}")
        invalid_ind = X[:, :, 0]>load_time
        X[invalid_ind, :] = 0
    return X, y