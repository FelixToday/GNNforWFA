# 安装依赖环境
```bash
pip install lxj_utils_sys-1.0.0-py3-none-any.whl
```

# 进入工作目录
```bash
screen -S runner
conda activate myenv
cd /root/autodl-tmp/lixianjun/gnn/GNNRun
export PYTHONPATH=$PYTHONPATH:/root/autodl-tmp/lixianjun/gnn
```
# 运行程序
```bash
python main.py --batch_size 32 --epochs 100 --lr 0.001 --num_classes 100 --database_dir /root/autodl-tmp/dataset/wfa/npz_dataset --dataset Closed_5tab --loaded_ratio 100 --TAM_type G1 --seq_len 5000 --level_count 18 --max_matrix_len 100 --log_transform True --maximum_load_time 80 --is_idx False --model STGCN_G1 --checkpoint_path ./checkpoints --num_workers 16 --early_stopping_patience 10 --is_test False --verbose_metrics False
```