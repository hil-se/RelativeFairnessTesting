# Relative Fairness Testing

#### Data (included in the [data/](https://github.com/hil-se/RelativeFairnessTesting/tree/main/synthetic/data) folder)

 - Adult, Bank, Default, German, Student, and Heart datasets
   + Raw data comes from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.php).
 - Compas dataset
   + Raw data comes from [propublica](https://github.com/propublica/compas-analysis/)

#### Usage
0. Install dependencies:
```
pip install -r requirements.txt
```
1. Navigate to the source code:
```
cd src
```
2. Generate results in [_inject\_results/_](https://github.com/hil-se/ContextualFairnessTesting/tree/main/inject_results)
```
python main.py
```

