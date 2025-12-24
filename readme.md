# 数据集位置
从 https://zenodo.org/records/14195051 下载数据
只需要下载Closed_5tab和CW两个数据集，解压后放在 /root/autodl-tmp/dataset/wfa/npz_dataset 目录下


# 安装依赖环境

```bash
# 安装环境
conda create -n myenv python=3.10
conda activate myenv
# 工具包
pip install lxj_utils_sys-1.0.0-py3-none-any.whl
# pytorch
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
# other packages
pip install tqdm
pip install einops
pip install matplotlib
pip install numpy
pip install scikit-learn
pip install tabulate
pip install torchinfo
pip install pytorch_metric_learning
pip install pandas
pip install natsort
```

# 进入工作目录
```bash
screen -S runner
conda activate myenv
cd /root/autodl-tmp/lixianjun/gnn/GNNRun
export PYTHONPATH=$PYTHONPATH:/root/autodl-tmp/lixianjun/gnn
```
# 单流运行程序
```bash
python main.py --batch_size 32 --epochs 100 --lr 0.001 --database_dir /root/autodl-tmp/dataset/wfa/npz_dataset --dataset CW --loaded_ratio 100 --TAM_type G1 --seq_len 5000 --level_count 18 --max_matrix_len 100 --log_transform True --maximum_load_time 80 --is_idx False --model STGCN_G1 --checkpoint_path ../checkpoints --num_workers 16 --early_stopping_patience 10 --is_test False --verbose_metrics False
```

# 多流运行程序
```bash
python main.py --batch_size 32 --epochs 100 --lr 0.001 --database_dir /root/autodl-tmp/dataset/wfa/npz_dataset --dataset Closed_5tab --loaded_ratio 100 --TAM_type G1 --seq_len 5000 --level_count 18 --max_matrix_len 100 --log_transform True --maximum_load_time 80 --is_idx False --model STGCN_G1 --checkpoint_path ../checkpoints --num_workers 16 --early_stopping_patience 10 --is_test False --verbose_metrics False
```