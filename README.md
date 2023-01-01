# Contextual Fairness Testing

#### Data (included in the [data/](https://github.com/hil-se/ContextualFairnessTesting/tree/main/data) folder)

 - Adult, Bank, Default, German, Student, and Heart datasets
   + Raw data comes from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.php).
 - Compas dataset
   + Raw data comes from [propublica](https://github.com/propublica/compas-analysis/)

#### Function

  - inject_All() function in the [src/main.py](https://github.com/hil-se/ContextualFairnessTesting/blob/main/src/main.py#L76) file runs all experiments with injected bias.
  - run() function in the [src/cft.py](https://github.com/hil-se/ContextualFairnessTesting/blob/main/src/cft.py#L25) file has the main pipeline for each experiment.
  - inject_bias() function in the [src/cft.py](https://github.com/hil-se/ContextualFairnessTesting/blob/main/src/cft.py#L43) file injects bias to the training data.
  - Metrics class in the [src/metrics.py](https://github.com/hil-se/ContextualFairnessTesting/blob/main/src/metrics.py) file calculates the metrics of accuracy, CBT, and CBD.
  - [src/data_reader.py](https://github.com/hil-se/ContextualFairnessTesting/blob/main/src/data_reader.py) file loads and preprocesses every dataset.


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

