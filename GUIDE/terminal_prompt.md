winget install git-xet
git clone https://huggingface.co/Qwen/Qwen3.5-0.8B
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/Qwen/Qwen3.5-0.8B

python src/data/download_subset.py --config configs/train_topk.yaml --output-dir ../data/fineweb_edu_subset --en-train-docs 100000 --zh-train-docs 100000 --en-val-docs 5000 --zh-val-docs 5000
python src/train.py --config ../data/fineweb_edu_subset/train_config_local.yaml --tag topk_l12_local --results-root ../drive/MyDrive/results

