# Project1 - Parser & Stamping
## File Structure
```
├── src (Source code)
│   ├── train.ipynb (Training notebook)
│   ├── simclr.py (SimCLR implementation)
│   ├── util.py (KNN and other utils)
│   ├── dataset
│   │   ├── dataset.py (Dataset Class)
│   │   └── pair_generator.py (Data Augmentation)
│   └── models
│       └── model.py (RESNET18)
├── environment.yml
└── README.md (Information of how to execute my codes)
```
## Usage
```
conda env create -f environment.yml --name hw2
cd src
# Then open train.ipynb to run the training process
```