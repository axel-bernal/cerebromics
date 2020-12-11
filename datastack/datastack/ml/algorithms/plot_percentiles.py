import GPy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datastack.ml.algorithms.linreg import Ridge


def get_quantiles_range(quantile_models, x_range, xlabel="x"):
    """
    evaluate the percentile_models over a range of x values

    Args:
        quantile_models:    the smoothed quantile predictors (dictionary)
        x_range:            numpy array of x values to evaluate the quantiles at
        xlabel:             label of the DataFrame column for x (default "x")

    Returns:
        pd.DataFrame containing all the predicted quantiles 
    """
    k = quantile_models.keys()

    result = {xlabel: x_range}

    for key in quantile_models.keys():
        result[key] = quantile_models[key].predict(x_range)
    return pd.DataFrame(result)


class SingleQuantileModel(object):
    def __init__(self, alpha_ridge, n_windows):
        self.alpha_ridge = alpha_ridge
        self.n_windows = n_windows

    def fit(self, x, y):
        X = x[:,np.newaxis]
        self.ridge = Ridge(alpha=self.alpha_ridge)
        self.ridge.fit(X=X, y=y)
        pred_mean_model = self.ridge.predict(X=X)
        self.kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=(X.max() - X.min())/self.n_windows)
        # import ipdb; ipdb.set_trace()
        self.gp = GPy.models.GPRegression(X, (y-pred_mean_model)[:,np.newaxis], self.kernel)

    def predict(self, x):
        X = x[:,np.newaxis]
        pred_mean_model = self.ridge.predict(X=X)
        return self.gp.predict(X)[0][:,0] + pred_mean_model    


class SingleQuantileModelLinear(object):
    def __init__(self, alpha_ridge, n_windows):
        self.ridge = Ridge(alpha=alpha_ridge)
        self.n_windows = n_windows

    def fit(self, x, y):
        X = x[:,np.newaxis]
        self.ridge.fit(X=X, y=y)
        # pred_ridge = self.ridge.predict(X=X)
        
    def predict(self, x):
        X = x[:,np.newaxis]
        pred_ridge = self.ridge.predict(X=X)
        return pred_ridge


def smooth_percentile_plot(x, y, percentiles=[1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0], plot_range=None, n_windows=3, n_windows_GP=None, ax=None, fbk={'lw': 0.0, 'edgecolor': None}, plot_kwargs={}, alpha_ridge=0.1, return_models=False, linear=False):
    """
    Visualize the percentiles of y as a function of x.
    estimates smooth percentiles of the variable x as a function of y, using a smoothed moving window average.

    Parameters
    ----------
    x           array-like (1 dimensional) the values of the variable x
    y           array-like (1 dimensional) the values of the variable y
    percentiles list of percentiles to plot   (default [1.0,5.0,25.0,50.0,75.0,95.0,99.0])
    plot_range  tuple of x-start and x-end value of x-values that is being plotted instead of the data range for None (default: None)
    n_windows   number of "independent" windows which defines the size of each sliding window
    n_windows_GP    number of "independent" windows which for defining the kernel length scale (default: None -> use n_windows)
    ax          axis handle fpr matplotlib (default None = current axis)
    fbk         plt.fill_between kwargs
    plot_kwargs plt.plot kwargs
    alpha_ridge regularization parameter of the linear fit from x -> to quantiles of y
    return_models   return the quantile prediction model_names
    linear      predict quantiles using linear model_names instead of the GP

    Returns
    -------
    figure handle
    """

    if n_windows_GP is None:
        n_windows_GP = n_windows

    # if axis handle is provided, set as current axis.
    if ax is not None:
        plt.sca(ax)

    if type(x) == pd.Series:
        x = x.values
    if type(y) == pd.Series:
        y = y.values

    idx_x = x.argsort()
    x_ = x[idx_x]
    y_ = y[idx_x]
    window_size = x_.shape[0] / n_windows
    start = np.arange(len(x_))
    stop = start + window_size
    stop[stop >= len(x_)] = len(x_)-1
    quantile_estimates = np.zeros((len(start), len(percentiles)))
    mean = np.zeros(len(start))
    median = np.zeros(len(start))
    x_plot = x_.copy()
    start = np.arange(x_plot.shape[0]) - window_size/2
    start[start < 0] = 0
    stop = np.arange(x_plot.shape[0]) + window_size/2
    stop[stop > len(x_)] = len(x_)

    if plot_range is not None:
        x_plot = np.arange(plot_range[0], plot_range[1], 0.01)

    # fist, estimate quantiles in moving windows
    for i in xrange(len(start)):
        y_window = y_[start[i]:stop[i]]
        mean[i] = y_window.mean()
        median[i] = np.median(y_window)
        quantile_estimates[i] = np.percentile(a=y_window, q=percentiles, interpolation='linear')

    # next smooth the quantile estimates using a GP regression
    models_quantiles = {}

    predicted_quantiles = {}
    for i, percentile in enumerate(percentiles):
        if linear:
            models_quantiles[percentile] = SingleQuantileModelLinear(alpha_ridge=alpha_ridge, n_windows=n_windows_GP)
        else:
            models_quantiles[percentile] = SingleQuantileModel(alpha_ridge=alpha_ridge, n_windows=n_windows_GP)
        models_quantiles[percentile].fit(x=x_, y=quantile_estimates[:,i])
        predicted_quantiles[percentile] = models_quantiles[percentile].predict(x=x_plot)
    
    if linear:
        models_quantiles["median"] = SingleQuantileModelLinear(alpha_ridge=alpha_ridge, n_windows=n_windows_GP)
    else:
        models_quantiles["median"] = SingleQuantileModel(alpha_ridge=alpha_ridge, n_windows=n_windows_GP)
    models_quantiles["median"].fit(x=x_, y=median)
    predicted_quantiles["median"] = models_quantiles["median"].predict(x=x_plot)

    for i in xrange(1, len(percentiles)):
        pi = percentiles[i]
        pi1 = percentiles[i-1]
        alpha = np.power(min(np.absolute((pi1+pi)/200.0), np.absolute(1.0-(pi1+pi)/200.0)), 1.0/3.0)
        plt.fill_between(x_plot, predicted_quantiles[pi1], predicted_quantiles[pi], alpha=alpha, **fbk)

    plt.plot(x_plot, predicted_quantiles["median"], **plot_kwargs)
    if return_models:
        return (plt.gca, models_quantiles)
    else:
        return plt.gca()


if __name__ == '__main__':
    import seaborn as sns

    # enable interactive plotting:
    plt.ion()

    from datastack.dbs import rdb as rosetta

    use_chi_ethnicity = True    #false would use all 29 ethnicities, which are mostly zero...
    rdb = rosetta.RosettaDBMongo(host="rosetta.hli.io")
    rdb.initialize(namespace="hg19")

    if use_chi_ethnicity:
        filters = {"lab.ProjectID": "FACE"}
    else:
        filters = None
    headers = ['qc.sample_key',"qc.sample_name"]
    if use_chi_ethnicity:
        eth_keys = [u'facepheno.ethnic.EUR',
            u'facepheno.ethnic.AFR',
            u'facepheno.ethnic.AMR',
            u'facepheno.ethnic.EAS',
            u'facepheno.ethnic.SAS']
    else:
        eth_keys = rdb.find_keys("dynamic.FACE.ethnicity.v1.")
    pheno_keys = ['facepheno.Age', 'facepheno.weight', 'facepheno.height']
    xy = rdb.query(filters=filters, keys=headers+eth_keys+pheno_keys, with_nans=False)

    print "plotting numpy arrays"
    x = xy["facepheno.Age"].values
    y = xy["facepheno.height"].values

    n_windows = 3
    percentiles = [1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0]

    if 0:
        print "plotting pandas.Series"
        fig = plt.figure()
        axis, model_names = smooth_percentile_plot(x=x, y=y, n_windows=n_windows, percentiles=percentiles, return_models=True, plot_range=[0,120])


        print "predicting the quantiles for a person of interest using return_models=True"
        age = 70.0
        for percentile in percentiles:
            prediction = model_names[percentile].predict(x=np.array([age]))
            print "%.1f percentile for %.1f years old person is %.1f" % (percentile, age, prediction)


    if 0:
        print "plotting pandas.Series"
        fig = plt.figure()
        axis, model_names = smooth_percentile_plot(x=x, y=y, n_windows=n_windows, percentiles=percentiles, return_models=True, plot_range=[0,120])


        print "predicting the quantiles for a person of interest using return_models=True"
        age = 70.0
        for percentile in percentiles:
            prediction = model_names[percentile].predict(x=np.array([age]))
            print "%.1f percentile for %.1f years old person is %.1f" % (percentile, age, prediction)


    if 1:
        print "plotting pandas.Series with linear quantiles model"
        fig = plt.figure()
        axis, model_names = smooth_percentile_plot(x=x, y=y, n_windows=n_windows, percentiles=percentiles, return_models=True, plot_range=[0,120], linear=True)


        print "predicting the linear quantiles for a person of interest using return_models=True"
        age = 70.0
        for percentile in percentiles:
            prediction = model_names[percentile].predict(x=np.array([age]))
            print "%.1f percentile for %.1f years old person is %.1f" % (percentile, age, prediction)

    if 1:
        print "plotting pandas.Series with quantiles model with smoother GP and more regularized linear regression part"
        fig = plt.figure()
        axis, model_names = smooth_percentile_plot(x=x, y=y, n_windows=n_windows, n_windows_GP=1, alpha_ridge=2, percentiles=percentiles, return_models=True, plot_range=[0,120], linear=False)


        print "predicting the quantiles for a person of interest using return_models=True with smoother GP and more regularized linear regression part"
        age = 70.0
        for percentile in percentiles:
            prediction = model_names[percentile].predict(x=np.array([age]))
            print "%.1f percentile for %.1f years old person is %.1f" % (percentile, age, prediction)

    df_range = get_quantiles_range(model_names, np.arange(20, 121), xlabel="age")

    x = xy["facepheno.Age"]
    y = xy["facepheno.height"]

    n_windows = 3
    percentiles = [1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0]
