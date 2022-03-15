# Music Memorability Prediction
## Regression of music memorability score
## File Structure
```
├── src (Source code)
│   ├── test.py   (testing script)
│   └── hw1.ipynb (training notebook)
├── model
│   └── success_model7.pt (trained model, use in test.py)
├── Pipfile
└── README.md
```
## Usage
```
cd src
python test.py --filename [list of audio file path to predict]
# Example: python test.py --filename normalize_5s_intro_0EVVKs6DQLo.wav normalize_5s_intro_d7to9URtLZ4.wav normalize_5s_intro_TzhhbYS9EO4.wav
```