from torch.utils.data import Dataset
import math
import numpy as np

def compute_idx(x, a, b, count):
    # idx = np.floor((x-a)*count/(b-a))
    # idx = np.clip(idx, 0, count-1)
    idx = int((x-a)/(b-a)*(count-1))
    idx = np.clip(idx, 0, count-1)
    return idx.astype(np.int32)

def pad_sequence(sequence, length):
    if len(sequence) >= length:
        sequence = sequence[:length]
    else:
        sequence = np.pad(sequence, (0, length - len(sequence)), "constant", constant_values=0.0)
    return sequence
class GNNDataset(Dataset):
    def __init__(self, X, labels,
                 loaded_ratio=100,
                 TAM_type='GNN',
                 seq_len=5000,
                 max_matrix_len=1800,
                 log_transform=False,
                 maximum_load_time=80,
                 is_idx=False,
                 level_count = 30,
                 **kwargs
                 ):
        self.X = X
        self.labels = labels
        self.loaded_ratio = loaded_ratio
        self.TAM = TAM_type
        self.is_idx = is_idx

        self.args = {
            "seq_len" : seq_len,
            "max_matrix_len" : max_matrix_len,
            "maximum_load_time" : maximum_load_time,
            "level_count" : level_count,
            "log_transform" : log_transform
        }

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        data = self.X[index]
        label = self.labels[index]
        return self.process_data(data), label

    def process_data(self, data):
        timestamp_seq = data[:, 0]
        packet_seq = data[:, 1]

        packet_seq = pad_sequence(packet_seq, self.args["seq_len"])
        timestamp_seq = pad_sequence(timestamp_seq, self.args["seq_len"])
        FM, TM, valid_idx = eval(f"get_TAM_{self.TAM}")(packet_seq, timestamp_seq, args=self.args)

        if self.args["log_transform"]:
            FM, TM = np.log1p(FM), np.log1p(TM)
        if self.is_idx:
            return FM.astype(np.float32), TM.astype(np.float32), valid_idx
        else:
            return FM.astype(np.float32), TM.astype(np.float32)

def get_TAM_G1(packet_length, time, args):
    # -1500,1500
    max_matrix_len = args["max_matrix_len"]
    maximum_load_time = args["maximum_load_time"]
    level_count = args["level_count"]

    # 统计窗口长度 返回datalen
    feature = np.zeros((max_matrix_len, level_count, 2))
    transfor_matrix = np.zeros((max_matrix_len, level_count, level_count))

    current_index = 0
    current_packets = []
    a = -1500
    b = 1500
    for l_k, t_k in zip(packet_length, time):
        if t_k == 0 and l_k == 0:
            break  # End of sequence

        i = compute_idx(t_k, 0, maximum_load_time, max_matrix_len)
        j = compute_idx(l_k, a, b, level_count)

        feature[i, j, 0] += 1# 数量
        feature[i, j, 1] += (l_k - (a+j * (b-a)/level_count))/512# 减去下界，size大小

        if j != current_index:
            if len(current_packets) < 2:
                pass
            else:
                current_idx = [compute_idx(p, a, b, level_count) for p in current_packets]
                row_ind = current_idx[:-1]
                col_ind = current_idx[1:]
                transfor_matrix[current_index, row_ind, col_ind] += 1

            current_index = j
            current_packets = [t_k]
        else:
            current_packets.append(t_k)

    if len(current_packets) < 2:
        pass
    else:
        current_idx = [compute_idx(p, a, b, level_count) for p in current_packets]
        row_ind = current_idx[:-1]
        col_ind = current_idx[1:]
        transfor_matrix[j, row_ind, col_ind] += 1
    valid_idx = j
    return feature, transfor_matrix, valid_idx

def get_TAM_G2(packet_length, time, args):
    # 0-1500
    max_matrix_len = args["max_matrix_len"]
    maximum_load_time = args["maximum_load_time"]
    level_count = args["level_count"]

    # 统计窗口长度 返回datalen
    feature = np.zeros((max_matrix_len, level_count, 4))
    transfor_matrix = np.zeros((max_matrix_len, level_count, level_count))

    current_index = 0
    current_packets = []
    a = 0
    b = 1500

    for l_k, t_k in zip(packet_length, time):
        if t_k == 0 and l_k == 0:
            break  # End of sequence

        dk = 0 if l_k >=0 else 1
        l_k = np.abs(l_k)
        i = compute_idx(t_k, 0, maximum_load_time, max_matrix_len)
        j = compute_idx(l_k, a, b, level_count)
        k = dk

        feature[i, j, k] += 1# 数量
        feature[i, j, 2+k] += (l_k - (a+j * (b-a)/level_count))/512# 减去下界，size大小

        if j != current_index:
            if len(current_packets) < 2:
                pass
            else:
                current_idx = [compute_idx(p, a, b, level_count) for p in current_packets]
                row_ind = current_idx[:-1]
                col_ind = current_idx[1:]
                transfor_matrix[current_index, row_ind, col_ind] += 1

            current_index = j
            current_packets = [t_k]
        else:
            current_packets.append(t_k)

    if len(current_packets) < 2:
        pass
    else:
        current_idx = [compute_idx(p, a, b, level_count) for p in current_packets]
        row_ind = current_idx[:-1]
        col_ind = current_idx[1:]
        transfor_matrix[j, row_ind, col_ind] += 1
    valid_idx = j
    return feature, transfor_matrix, valid_idx


def get_TAM_RF(packet_length, time, args):
    max_matrix_len = args["max_matrix_len"]
    sequence = np.sign(packet_length) * time
    maximum_load_time = args["maximum_load_time"]

    feature = np.zeros((2, max_matrix_len))  # Initialize feature matrix

    count = 0
    for pack in sequence:
        count += 1
        if pack == 0:
            count -= 1
            break  # End of sequence
        elif pack > 0:
            if pack >= maximum_load_time:
                feature[0, -1] += 1  # Assign to the last bin if it exceeds maximum load time
            else:
                idx = int(pack * (max_matrix_len - 1) / maximum_load_time)
                feature[0, idx] += 1
        else:
            pack = np.abs(pack)
            if pack >= maximum_load_time:
                feature[1, -1] += 1  # Assign to the last bin if it exceeds maximum load time
            else:
                idx = int(pack * (max_matrix_len - 1) / maximum_load_time)
                feature[1, idx] += 1
    feature = np.reshape(feature, (1, 2, max_matrix_len))
    feature = np.transpose(feature, (2, 0, 1))
    return feature, feature, None

if __name__ == "__main__":
    from GNNRun.utils import load_data
    import time
    from tqdm import tqdm
    X, y = load_data(r"/root/autodl-tmp/dataset/wfa/npz_dataset/Closed_2tab/valid.npz")
    dataset1 = GNNDataset(X, y,
                         loaded_ratio=100,
                         TAM_type='G1',
                         seq_len=5000,
                         max_matrix_len=100,
                         log_transform=False,
                         maximum_load_time=80,
                         is_idx=False,
                         level_count=30)
    dataset2 = GNNDataset(X, y,
                          loaded_ratio=100,
                          TAM_type='G2',
                          seq_len=5000,
                          max_matrix_len=100,
                          log_transform=False,
                          maximum_load_time=80,
                          is_idx=False,
                          level_count=30)
    def test_fun(dataset):
        tic = time.time()
        for i in tqdm(range(len(dataset))):
            dataset[i]
        toc = time.time()
        print(f"运行时长：{toc - tic:.2f}")
    test_fun(dataset1)
    test_fun(dataset2)
    # dataset_RF = GNNDataset(X, y,
    #                      loaded_ratio=100,
    #                      TAM_type='RF',
    #                      seq_len=5000,
    #                      max_matrix_len=1800,
    #                      log_transform=False,
    #                      maximum_load_time=80,
    #                      is_idx=False,
    #                      level_count=30)

    # x1 = dataset_RF[0][0][0]
    # x2 = dataset1[0][0][0]
    # x3 = dataset2[0][0][0]
    # from tqdm import tqdm
    # import time
    # start_time = time.time()
    # for i in tqdm(range(len(dataset))):
    #     data, label = dataset[i]
    # end_time = time.time()
    # print(f"valid Time cost: {end_time - start_time:.2f}s")
