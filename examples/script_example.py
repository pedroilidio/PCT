#!/usr/bin/env python
import numpy as np
from PBCT import PBCTClassifier 

# Load data.
X1 = np.loadtxt("input/X1.csv", delimiter=',')  # Row attributes matrix.
X2 = np.loadtxt("input/X2.csv", delimiter=',')  # Column attributes matrix.
Y = np.loadtxt("input/Y.csv", delimiter=',')  # Interaction matrix.

# Instantiate class.
cls = PBCTClassifier(min_samples_leaf=100, max_depth=30)
# Using PBCTClassifier with a binary interaction matrix will save some time.
# If IM is not binary, then go with PBCT class.

# Train model.
print('Training...')
cls.fit([X1, X2], Y)

# Save trained model.
cls.save("output/trained_model.dict.pickle.gz")

# Predict new interactions (using same X1 and X2 for the sake of demonstration).
print('Predicting...')
predictions = cls.predict([X1, X2], verbose=True)#, simple_mean=True) 
# simple_mean=True makes prediction values as mean values for the whole inter-
# action submatrix in the leaf, unconsidering the possibility of a known row or
# column instance. For now it is the default behaviour.

# Save predictions.
np.savetxt("output/predictions.csv", predictions, delimiter=',', fmt='%d')

# One can use PBCT.ax_predict to provide only row or column instances for new
# interactions inference, yielding a probability of interaction for each ins-
# tance of the training set on the other axis.

# Examples (again pretending X1 and X2 are unseen data):
print('Single axis multi-output prediction...')
cls.ax_predict(X1, axis=0, verbose=True)
cls.ax_predict(X2, axis=1, verbose=True)
