# wtvae

Example command for training WTVAE 64
```{bash}
python3 train_wtvae_64.py --epochs=100 --root_dir=/disk_c/han/ --log_interval=10 --config=2wt_unflatten0 --unflatten=0 --num_wt=2
```
Change `root_dir`, `config`, `unflatten`, and `num_wt` as necessary.

Example command for training IWTVAE 64
```{bash}
python3 train_iwtvae_64.py --epochs=100 --root_dir=/disk_c/han/ --log_interval=10 --config=fc_2upsampling --unflatten=0 --num_wt=2 --upsampling=linear --num_upsampling=2 --wtvae_model=/disk_c/han/models/wtvae64_test/wtvae_epoch1.pth
```
Change `root_dir`, `config`, `unflatten`, `num_wt`, `upsampling`, `reuse`, `zero`, `num_upampling`, and `wtvae_model` as necessary.

Example command for training IWTVAE 512 pipeline (deterministic WT and IWT)
```{bash}
python3 train_iwtvae_512.py --root_dir=/disk_c/han/ --batch_size=16 --z_dim=500 --zero --config=dynmask_rest0_freezewtiwt_2wt --num_iwt=2
```

## Requirements

Under `requirements.yml` for conda environment setup

