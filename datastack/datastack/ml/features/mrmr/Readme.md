
This directory has an information theoretic feature selector
due to Peng - MrMr, which uses the mutual information between
a variable and the target compared to the mutual information 
between the variable and previously selected variables to choose
the next variable to add.

The part of the algorithm about MI between the covariates could
use all our data, not just subjects with the target phenotype
measured. We have not done this yet.

For eye color it has not been helpful. If we start from 0
features, then we get better as we select more features, but
we do better to use our existing model.
Using this to add more features to our model does not improve
results.

But, we'll leave it here for future use.
ml/algorithms/mrselect.py is the top level estimator, which
does feature selection first and then trains a model on the
selected features.

The setup.py file works for me, but needs to be generalized to
make this work in general.

The underlying algorithm is in c++.
The main data structure is DataTable, in DataTable.cpp, and the
heart of the algorithm is in mrdata.cpp.
