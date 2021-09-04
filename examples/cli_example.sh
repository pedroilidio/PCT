# Train a PBCT tree and save it to 'trained_model.json'.
PBCT --fit --XX input/X1.csv input/X2.csv --Y input/Y.csv \
     --path_model output/trained_model.dict.pickle.gz

# Identifier labels are optional arguments:
# --X_names input/X1_names.txt input/X2_names.txt
# --X_col_names input/X1_col_names.txt input/X2_col_names.txt

# Use the recently saved model to predict interactions from new data (not new
# in this example, we are just using the same files to demonstrate). Save
# predictions to 'predictions.csv'.
PBCT --predict --XX input/X1.csv input/X2.csv --Y output/predictions.csv \
     --path_model output/trained_model.dict.pickle.gz --verbose

# This same process could be done using parameters from a json config file
# (see 'config.json') providing the --config option:

# PBCT --config config.json
