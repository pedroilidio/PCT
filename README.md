# Predictive Bi-Clustering Trees (PBCT)
This code implements PBCTs based on its original proposal by Pliakos, Geurts and Vens in 2018<sup>1</sup>. Functionality has been extended to n-dimensional interaction tensors, where n instances of n different classes would interact or not for each database instance.

<sup>1</sup>Pliakos, Konstantinos, Pierre Geurts, and Celine Vens. "Global multi-output decision trees for interaction prediction." *Machine Learning* 107.8 (2018): 1257-1281.

## Installation
The package is available at PyPI and can be installed by the usual `pip` command:
```
$ pip install pbct
```
Local installation can be done either by providing the `--user` flag to the above command or by cloning this repo and issuing `pip` afterwards, for example:
```
$ git clone https://github.com/pedroilidio/PCT
$ cd PCT
$ pip install -e .
```
Where the `-e` option installs it as symbolic links to the local cloned repository, so that changes in it will reflect on the installation directly.

## Usage
Usage and input/output examples are provided in the `examples` folder.
We provide a command-line utility to use PBCT models, that shows the following information when the `--help` option is used.

```
$ PBCT --help

usage: PBCT [-h] [--config CONFIG] [--XX XX [XX ...]]
            [--XX_names XX_NAMES [XX_NAMES ...]]
            [--XX_col_names XX_COL_NAMES [XX_COL_NAMES ...]] [--Y Y]
            [--path_model PATH_MODEL] [--max_depth MAX_DEPTH]
            [--min_samples_leaf MIN_SAMPLES_LEAF] [--verbose]
            [--outdir OUTDIR] [--k K [K ...]] [--diag]
            [--test_size TEST_SIZE [TEST_SIZE ...]]
            [--train_size TRAIN_SIZE [TRAIN_SIZE ...]] [--njobs NJOBS]
            [--random_state RANDOM_STATE]
            [{fit,predict,train_test,xval}]

Fit a PBCT model to data or use a trained model to predict new results. Input
files and options may be provided either with command-line options or by a
json config file (see --config option).

positional arguments:
  {fit,predict,train_test,xval}
                        fit: Use input data to train a PBCT. predict: Predict
                        interaction between input instances. train_test: Split
                        data between the 4 train/test sets, train and test a
                        PBCT. xval: run a 2D k-fold cross validation with the
                        given data. (default: None)

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Load options from json file. File example: {
                        "path_model": "/path/to/save/model.json", "fit":
                        "true", "XX": ["/path/to/X1.csv", "/path/to/X2.csv"],
                        "Y": "/path/to/Y.csv", }. Multiple dicts in a list
                        will cause this script to run multiple times.
                        (default: None)
  --XX XX [XX ...]      Paths to .csv files containing rows of numerical
                        attributes for each axis' instance. (default: None)
  --XX_names XX_NAMES [XX_NAMES ...]
                        Paths to files containing string identifiers for each
                        instance for each axis, being one file for each axis.
                        (default: None)
  --XX_col_names XX_COL_NAMES [XX_COL_NAMES ...]
                        Paths to files containing string identifiers for each
                        attributecolumn, being one file for each axis.
                        (default: None)
  --Y Y                 If fitting the model to data, it represents the path
                        to a .csv file containing the known interaction matrix
                        between rows and columns data.If predicting, Y is the
                        destination path for the predicted values, formatted
                        as an interaction matrix in the same way described.
                        (default: None)
  --path_model PATH_MODEL
                        When fitting: path to the location where the model
                        will be saved. When predicting: the saved model to be
                        used. (default: model.dict.pickle.gz)
  --max_depth MAX_DEPTH
                        Maximum PBCT depth allowed. -1 will disable this
                        stopping criterion. (default: -1)
  --min_samples_leaf MIN_SAMPLES_LEAF
                        Minimum number of sample pairs in the training set
                        required to arrive at each leaf. (default: 20)
  --verbose, -v         Show more detailed output (default: False)
  --outdir OUTDIR       Where to save results. (default: PBCT_results)
  --k K [K ...], -k K [K ...]
                        Number of folds for cross-validation. (default: [3])
  --diag                Use independent TrTc sets for cross-validation, i.e.
                        with no overllaping rows or columns. (default: False)
  --test_size TEST_SIZE [TEST_SIZE ...]
                        If between 0.0 and 1.0, represents the proportion of
                        the dataset to include in the TrTc split for each
                        axis, e.g.: .3 .5 means 30% of the rows and 50% of the
                        columns will be used as the TrTc set. If >= 1,
                        represents the absolute number of test samples in each
                        axis. If None, the values are set to the complements
                        of train_size. If a single value v is given, it will
                        be interpreted as (v, v). If train_size is also None,
                        it will be set to 0.25. (default: None)
  --train_size TRAIN_SIZE [TRAIN_SIZE ...]
                        Same as test_size, but refers to the LrLc set
                        dimensions. (default: None)
  --njobs NJOBS         How many processes to spawn when cross-validating.
                        (default: None)
  --random_state RANDOM_STATE
                        Random seed to use. (default: None)
```
