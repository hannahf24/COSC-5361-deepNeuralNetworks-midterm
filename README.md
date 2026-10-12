## Requirements
- Python 3.12 (PyTorch has no 3.14 wheels yet)
- NVIDIA GPU with CUDA 12.6 (CPU works but is very slow)

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install pandas numpy matplotlib scikit-learn pillow

## Dataset
Download Food-101 from https://www.kaggle.com/datasets/dansbecker/food-101
Extract so the layout is:
    food-101/images/<class_name>/*.jpg
    food-101/meta/train.txt, test.txt
Not committed to this repo — 9 GB, far over GitHub's limits.

## Running
Set `dataPath` at the top of each script, then:
    python scratch_cnn.py        # ~3 h, writes per_class_scratch_cnn.csv
    python resnet_transfer.py    # several hours, reads that CSV for the comparison
Set `quickTest = True` first for a 2-config smoke test.
Order matters: the ResNet script needs per_class_scratch_cnn.csv to produce per_class_gain.csv.

Fixed: seed 42, test_size 0.10, class selection = first 21 alphabetically.
Not fixed: cuDNN kernel selection is nondeterministic, so accuracy varies ±1–2 pp between runs.
The reported numbers come from a single run of each script.
