import datastack.ml.cross_validation as cross_validation
import logging
import pandas as pd
import sklearn.linear_model

_logger = logging.getLogger(__name__)

IVECTOR_COEFFICIENTS = 100

# Map labeled phenotypes to ordinals. Some labeled phenotypes have
# ordinals that encode increasing importance or severity.

BASELINE_PHENOTYPES = ['facepheno.Age', 'facepheno.Sex']

SEX_MAP = {'Male': 0,
           'Female': 1}

DRINKS_FREQUENCY_MAP = {'Never': 0,
                        'Monthly or less': 1,
                        '2 - 4 times a month': 2,
                        '2 - 3 times a week': 3,
                        '4 or more times a week': 4}

DRINKS_PER_DAY_MAP = {'1 or 2': 0,
                      '3 or 4': 1,
                      '5 or 6': 2}

DRINKS_MORE_THAN_SIX_EPISODES_MAP = {'Never': 0,
                                     'Less than monthly': 1,
                                     'Monthly': 2,
                                     'Weekly': 3,
                                     'Daily or almost daily': 4}

HEALTH_STATUS_MAP = {'Excellent': 4,
                     'Very good': 3,
                     'Good': 2,
                     'Fair': 1,
                     'Poor': 0}

SMOKING_AFTER_AWAKE_MAP = {'31 - 60 minutes': 0,
                           '5 - 30 minutes': 1,
                           'Within 5 minutes': 2}

SMOKING_NUM_DAY_MAP = {'10 or less': 0,
                       '11 through 20': 1,
                       '21 through 30': 2,
                       '31 or more': 3}

ORDINAL_MAPS = {'facepheno.Sex': SEX_MAP,
                'facepheno.drink.frequency': DRINKS_FREQUENCY_MAP,
                'facepheno.drinks.day': DRINKS_PER_DAY_MAP,
                'facepheno.drinks.more.six': DRINKS_MORE_THAN_SIX_EPISODES_MAP,
                'facepheno.health.status': HEALTH_STATUS_MAP,
                'facepheno.smoking.after.awake': SMOKING_AFTER_AWAKE_MAP,
                'facepheno.smoking.num.day': SMOKING_NUM_DAY_MAP}

_VOICE_STUDY = 'FACE'


def set_debug_level(level):
    _logger.setLevel(level)


get_bmi = lambda mass_in_kg, height_in_meters: mass_in_kg / \
    height_in_meters ** 2


def get_xy_generator(
        target, covariates=None, *args, **kwargs):
    '''Create an XY-generator to use for cross-validation.

    Arguments:
        target: A target of a regression
        covariates: A list of covariates on which to regress; supports
        regex wildcards at the suffix

    Returns:
        An (X, y) generator
    '''
    if covariates is None:
        covariates = []
    _logger.debug(
        'Getting Xy generator to regress %s on %s covariates' %
        (target, covariates))

    def xy_generator(df):
        # Create a regex to extract covariate columns from the data frame
        regex = '|'.join(['^%s$' % (c) for c in covariates])
        _logger.debug('Filtering covariates with regex "%s"' % (regex))
        X = df.filter(regex=regex)
        return X, df[target]
    return xy_generator


def _evaluate_classification(target, kfolds, ordinals=None, nopreproc=False, covariates=None,
                             estimator=None, params=None, fit_params=None, pspec=None,
                             covariate_widget=None, kfold_widget=None, *args, **kwargs):
    if covariates is None:
        covariates = []
    _logger.debug(
        'Evaluating classification of %s with %s covariates%s' %
        (target, covariates, '' if estimator is None else ' using %s' % estimator.__class__.__name__))
    if covariate_widget is not None:
        covariate_widget.value = '\n'.join(covariates)
    xy_generator = get_xy_generator(target, covariates)
    return cross_validation.EvaluationClassification(
        xy_generator, kfolds, ordinals=ordinals, estimator=estimator, nopreproc=nopreproc, params=params, fit_params=fit_params,  pspec=pspec, widget=kfold_widget, *args, **kwargs)


def evaluate_classifications(target, kfolds, ordinals=None, predictor=None, key_covariate=None,
                             covariates=None, with_aggregate_covariates=True, covariate_widget=None, kfold_widget=None, *args, **kwargs):
    '''Evaluate the predictive power of a set of covariates for a given
    classification target.

    Arguments:
        target: A target of a classification problem
        kfolds: Pre-existing kfolds to use in the evaluation
        ordinals: A map that assigns ordinals to class labels; enables
        evaluation of classifiers based on linear regressions such as
        ridge regression
        predictor: The predictor covariate whose predictive power to
        measure
        covariates: A list of covariates on which to regress
        with_aggregate_covariates: If true, evaluate classifications
        using all covariates except the predictor as predictors,
        and then using all covariates including including the key
        covariate (if specified)
        covariate_widget: An IPython text-type widget to update with
        the covariates under evaluation; default None
        kfold_widget: An IPython integer widget (e.g., IntSlider or
        IntProgress) to update with the kfold under evaluation; default
        None

    Returns:
        The results of using baseline naive models, and a list of the
        evaluations performed using the covariates
    '''
    if not ((predictor is None) ^ (key_covariate is None)):
        raise Exception(
            'Expected only a predictor; got a predictor and a \'key covariate\'')
    if key_covariate is not None:
        predictor = key_covariate
    if covariates is None:
        covariates = []
    _logger.info(
        'Evaluating classifications of %s with %s%s covariates' %
        (target,
         '%s as a predictor and ' %
         (predictor) if predictor is not None else '',
            covariates))
    # Prepare a ridge classifier
    e_ridge = sklearn.linear_model.RidgeClassifier()
    e_params = cross_validation._REGRESSION_PARAMS_DEFAULT
    # Get the baseline using naive classifiers (e.g., returning the mode
    # class always).
    baseline = cross_validation.EvaluationClassificationBaseline(
        target, kfolds, ordinals=ordinals)
    baseline_df = baseline.metrics_df
    # Evaluate classification on the predictor only
    evaluators = []
    if predictor is not None:
        evaluators.append(
            _evaluate_classification(
                target,
                kfolds,
                covariates=[predictor],
                covariate_widget=covariate_widget,
                kfold_widget=kfold_widget,
                *args,
                **kwargs))
    if len(covariates) > 0:
        # Evaluate each non-predictor separately
        for c in covariates:
            evaluators.append(
                _evaluate_classification(
                    target,
                    kfolds,
                    covariates=[c],
                    covariate_widget=covariate_widget,
                    kfold_widget=kfold_widget,
                    *args,
                    **kwargs))
        # Evaluate all covariates together, without and with the key
        # covariate if specified
        if with_aggregate_covariates is True:
            if len(covariates) > 1:
                evaluators.append(
                    _evaluate_classification(
                        target,
                        kfolds,
                        covariates=covariates,
                        covariate_widget=covariate_widget,
                        kfold_widget=kfold_widget,
                        *args,
                        **kwargs))
            if predictor is not None:
                evaluators.append(
                    _evaluate_classification(
                        target,
                        kfolds,
                        covariates=[predictor] + covariates,
                        covariate_widget=covariate_widget,
                        kfold_widget=kfold_widget,
                        *args,
                        **kwargs))
            # Evaluate with ridge classification if we can treat the target
            # classes as ordinals
            if ordinals is not None:
                evaluators.append(
                    _evaluate_classification(
                        target,
                        kfolds,
                        ordinals=ordinals,
                        covariates=covariates,
                        estimator=e_ridge,
                        params=e_params,
                        covariate_widget=covariate_widget,
                        kfold_widget=kfold_widget,
                        *args,
                        **kwargs))
                if predictor is not None:
                    evaluators.append(
                        _evaluate_classification(
                            target,
                            kfolds,
                            ordinals=ordinals,
                            covariates=[predictor] + covariates,
                            estimator=e_ridge,
                            params=e_params,
                            covariate_widget=covariate_widget,
                            kfold_widget=kfold_widget,
                            *args,
                            **kwargs))
    return baseline_df, evaluators


def _evaluate_regression(target, kfolds, ordinals=None, covariates=None, nopreproc=False, estimator=None,
                         params=None, fit_params=None, idx_covariates=None, pspec=None,
                         covariate_widget=None, kfold_widget=None, *args, **kwargs):

    if covariates is None:
        covariates = []
    _logger.debug(
        'Evaluating regression of %s with %s covariates' %
        (target, covariates))
    if covariate_widget is not None:
        covariate_widget.value = '\n'.join(covariates)
    xy_generator = get_xy_generator(target, covariates)
    return cross_validation.EvaluationRegression(

        xy_generator, kfolds, ordinals=ordinals, nopreproc=nopreproc, estimator=estimator, params=params, fit_params=fit_params, 
        idx_covariates=idx_covariates, pspec=pspec, widget=kfold_widget, *args, **kwargs)


def evaluate_regressions(
        target,
        kfolds,
        ordinals=None,
        predictor=None,
        key_covariate=None,
        covariates=None,
        with_baseline=True,
        with_bootstrap=True,
        with_aggregate_covariates=True,
        covariate_widget=None,
        kfold_widget=None,
        *args, **kwargs):

    '''Evaluate the predictive power of a set of covariates for a given
    regression target.

    Arguments:
        target: A target of a regression problem
        kfolds: Pre-existing kfolds to use in the evaluation
        ordinals: A map that assigns ordinals to class labels; enables
        evaluation of classifiers based on linear regressions such as
        ridge regression
        predictor: The predictor whose predictive power to
        measure
        covariates: A list of covariates on which to regress
        with_aggregate_covariates: If true, evaluate regressions
        using all covariates except the predictor as predictors,
        and then using all covariates including including the key
        covariate (if specified)
        covariate_widget: An IPython text-type widget to update with
        the covariates under evaluation; default None
        kfold_widget: An IPython integer widget (e.g., IntSlider or
        IntProgress) to update with the kfold under evaluation; default
        None

    Returns:
        The results of using baseline naive models, and a list of the
        evaluations performed using the covariates
    '''
    if not ((predictor is None) ^ (key_covariate is None)):
        raise Exception(
            'Expected only a predictor; got a predictor and a \'key covariate\'')
    if key_covariate is not None:
        predictor = key_covariate
    if covariates is None:
        covariates = []
    _logger.info(
        'Evaluating classifications of %s with %s%s covariates' %
        (target,
         '%s as a predictor and ' %
         (predictor) if predictor is not None else '',
            covariates))
    baseline_df = None
    if with_baseline is True:
        baseline = cross_validation.EvaluationRegressionBaseline(
            target, kfolds, ordinals=ordinals)
        baseline_df = baseline.metrics_df
    # Evaluate classification on the predictor only
    evaluators = []
    if predictor is not None:
        evaluators.append(
            _evaluate_regression(
                target,
                kfolds,
                ordinals=ordinals,
                covariates=[predictor],
                covariate_widget=covariate_widget,
                kfold_widget=kfold_widget,
                *args,
                **kwargs))
    if len(covariates) > 0:
        # Evaluate each covariate in turn
        for c in covariates:
            evaluators.append(
                _evaluate_regression(
                    target,
                    kfolds,
                    ordinals=ordinals,
                    covariates=[c],
                    covariate_widget=covariate_widget,
                    kfold_widget=kfold_widget,
                    *args,
                    **kwargs))
        # Evaluate all covariates together, without and with the key
        # covariate if specified
        if with_aggregate_covariates is True:
            if len(covariates) > 1:
                evaluators.append(
                    _evaluate_regression(
                        target,
                        kfolds,
                        ordinals=ordinals,
                        covariates=covariates,
                        covariate_widget=covariate_widget,
                        kfold_widget=kfold_widget,
                        *args,
                        **kwargs))
            if predictor is not None:
                evaluators.append(
                    _evaluate_regression(
                        target,
                        kfolds,
                        ordinals=ordinals,
                        covariates=[
                            predictor] + covariates,
                        covariate_widget=covariate_widget,
                        kfold_widget=kfold_widget,
                        *args,
                        **kwargs))
    return baseline_df, evaluators


def tabulate_metrics(baseline_df, evaluators):
    '''Tabulate the prediction scores for baseline classifiers/regressions
    and fitted classifiers/regressions
    '''
    metrics_df = pd.concat(([] if baseline_df is None else [
                           baseline_df]) + [e.metrics_df for e in evaluators], ignore_index=True)
    metrics_df.Covariates = metrics_df.Covariates.apply(lambda c: ','.join(c))
    metrics_df.drop_duplicates(inplace=True)
    metrics_df.Covariates = metrics_df.Covariates.apply(lambda c: c.split(','))
    return metrics_df


def get_regressors_from_evaluators(evaluators):
    '''Get the best-fit classifiers/regressions corresponding to each target/
    covariate set in a list of evaluators
    '''
    regressors = []
    for e in evaluators:
        regressors.append(
            e.cross_validator(e.xy_generator, e.kfolds, e.estimator, e.params))
    return regressors
