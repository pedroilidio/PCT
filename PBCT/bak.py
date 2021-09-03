#!/usr/bin/env python
import argparse
from PBCT import PBCT, PBCTClassifier, DEFAULTS


def parse_args():
    arg_parser = argparse.ArgumentParser(
        description=(
            "Fit a PBCT model to data and/or use a trained model to predict new"
            " results.\n\nInput and options may be provided in three ways: Use --fit to train a PBCT and --predict to infer new values from data. If both options are given, the same saved model will be used for predictions."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument(
        '--fit', action='store_true',
        help='Use input data to train a PBCT.')
    arg_parser.add_argument(
        '--predict', action='store_true',
        help='Predict interaction between input instances.')

    arg_parser.add_argument(
        '--XX', nargs='+',
        help=('Paths to .csv files containing rows of numerical attributes'
              ' for each axis\' instance.'))
    arg_parser.add_argument(
        '--XX_names', nargs='+',
        help=('Paths to files containing string identifiers for each instance'
              ' for each axis, being one file for each axis.'))
    arg_parser.add_argument(
        '--XX_col_names', nargs='+',
        help=('Paths to files containing string identifiers for each attribute'
              'column, being one file for each axis.'))
    arg_parser.add_argument(
        '--Y', required=True,
        help=('If fitting the model to data (--fit), it represents the pat'
              'h to a .csv file containing the known interaction matrix be'
              'tween rows and columns data.'
              'If --predict is given, Y is the destination path for the pr'
              'edicted values, formatted as an interaction matrix in the s'
              'ame way described for --fit.'))
    arg_parser.add_argument(
        '--path_model', default=DEFAULTS['path_model'],
        help=('When fitting: path to the location where the model will be '
              'saved. When predicting: the saved model to be used.'))
    arg_parser.add_argument(
        '--max_depth', default=DEFAULTS['max_depth'],
        help='Maximum PBCT depth allowed.')
    arg_parser.add_argument(
        '--min_samples_leaf', default=DEFAULTS['min_samples_leaf'],
        help=('Minimum number of sample pairs in the training set required'
              ' to arrive at each leaf.'))
    arg_parser.add_argument(
        '--visualize', default=DEFAULTS['path_rendering'],
        help=('If provided, path to where a visualization of the trained t'
              'ree will be saved.'))
    arg_parser.add_argument(
        '--fromdir',
        help=('Read data from directory instead. In such case, filenames must be:'
              '\tX1, X2, Y, X1_names, X2_names, X1_col_names and X2_col_names,\n'
              'with any file extension. *_names files are optional.'))
    arg_parser.add_argument(
        '--fromjson',
        help=('Specify in a json file where to read data from. E.g.: {'
              '\n\t"path_model": "/path/to/save/model.json",'
              '\n\t"fit": "true",'
              '\n\t"predict": "false",'
              '\n\t"XX": ["/path/to/X1.csv", "/path/to/X2.csv"],'
              '\n\t"Y": "/path/to/Y.csv",'
              '\n\t"XX_names": ["/path/to/X1_names.txt",  "/path/to/X2_names.txt"],'
              '\n\t"XX_col_names": ["/path/to/X1_col_names.txt", "/path/to/X2_col_names.txt"],'
              '\n}'
              'being *_names files optional. Multiple dicts in a list will caus'
              'e this script to run multiple times. Set "fit" and "predict" to'
              '"false" in order to disable a specific run.'))

    return arg_parser.parse_args()


# FIXME: Not n-dimensional yet!
def main(path_Xrows, path_Xcols, path_Y, fit, predict,
         path_model=DEFAULTS['path_model'], max_depth=DEFAULTS['max_depth'],
         min_samples_leaf=DEFAULTS['min_samples_leaf'], path_render=DEFAULTS['path_rendering']):
    """
    Train a PBCT or predict values from the command-line. See `parse_args()`
    or use --help for parameters' descriptions.
    """

    if fit:
        print('Loading data...')
        Xrows, Xcols, Y = [pd.read_csv(p, index_col=0)
                           for p in (path_Xrows, path_Xcols, path_Y)]
        Tree = PBCT(min_samples_leaf=min_samples_leaf, max_depth=max_depth)
        print('Training PBCT...')
        Tree.fit(Xrows, Xcols, Y)
        print('Saving model...')
        joblib.dump(Tree, path_model)
        if path_render:
            print('Rendering tree...')
            render_tree(Tree, name=path_render)
        print('Done.')

    elif predict:
        print('Loading data...')
        Xrows, Xcols = [pd.read_csv(p, index_col=0)
                           for p in (path_Xrows, path_Xcols)]
        print('Loading model...')
        Tree = joblib.load(path_model)
        print('Predicting values...')
        predictions = Tree.predict(Xrows, Xcols)
        predictions_df = pd.DataFrame(predictions,
                                      index=Xrows.index,
                                      columns=Xcols.index)
        predictions_df.to_csv(path_Y)
        print('Done.')

    else:
        raise ValueError("Either 'fit' or 'predict' must be given.")


if __name__ == '__main__':
    args = parse_args()
    main(path_Xrows=args.Xrows, path_Xcols=args.Xcols, path_Y=args.Y,
         fit=args.fit, predict=args.predict, path_model=args.model,
         max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf,
         path_render=args.visualize)
