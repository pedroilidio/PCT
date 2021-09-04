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

usage: PBCT [-h] [--fit | --predict] [--config CONFIG] [--XX XX [XX ...]]
            [--XX_names XX_NAMES [XX_NAMES ...]]
            [--XX_col_names XX_COL_NAMES [XX_COL_NAMES ...]] [--Y Y]
            [--path_model PATH_MODEL] [--max_depth MAX_DEPTH]
            [--min_samples_leaf MIN_SAMPLES_LEAF] [--simple_mean] [--verbose]

Fit a PBCT model to data or use a trained model to predict new results. Input
files and options may be provided either with command-line options or by a
json config file (see --config option).

optional arguments:
  -h, --help            show this help message and exit
  --fit                 Use input data to train a PBCT. (default: False)
  --predict             Predict interaction between input instances. (default:
                        False)
  --config CONFIG       Load options from json file. File example:
                        {
                            "path_model": "/path/to/save/model.json",
                            "fit": "true",
                            "XX": ["/path/to/X1.csv", "/path/to/X2.csv"],
                            "Y": "/path/to/Y.csv"
                        }.
                        Multiple dicts in a list
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
                        used. (default: trained_model.json)
  --max_depth MAX_DEPTH
                        Maximum PBCT depth allowed. -1 will disable this
                        stopping criterion. (default: -1)
  --min_samples_leaf MIN_SAMPLES_LEAF
                        Minimum number of sample pairs in the training set
                        required to arrive at each leaf. (default: 20)
  --simple_mean         If provided, the prototype function will always return
                        the mean value over the entire sub interaction matrix
                        of the leaf, not considering possible known instances.
                        (default: False)
  --verbose, -v         Show more detailed output (default: False)
```
