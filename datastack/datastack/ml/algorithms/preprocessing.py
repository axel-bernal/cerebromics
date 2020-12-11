import numpy as np

# Import Quantile transform for backward compatibility
from datastack.cerebro.transform import Quantile as QuantileTransform  # @UnusedImport


def fix_outliers_ccn(samples, inplace=False, add_sex=True, setnan=True, logtransform=False, drop_outliers=False):
    if not inplace:
        samples = samples.copy()

    # outliers
    # more than 1 Y
    outliersY = (samples['ccn.chrY'] > 1.05)  # flag these as outliers
    # more than 2 X
    outliersX = (samples['ccn.chrX'] > 2.05)  # flag as outliers

    # contaminated samples
    X105 = (samples['ccn.chrX'] > 1.05)
    Y005 = (samples['ccn.chrY'] > 0.05)
    outliersXY = (X105 & Y005)  # outliers
    samples['ccn.contaminated'] = outliersXY

    outliers = (outliersY | outliersX | outliersXY) | (~X105 & ~Y005)
    samples['ccn.outliers'] = outliers
    males = ~outliers & ~X105 & Y005
    females = ~outliers & X105 & ~Y005
    samples['ccn.sex'] = np.zeros(samples.shape[0], dtype="|S10")
    samples.ix[females, 'ccn.sex'] = 'Female'
    samples.ix[males, 'ccn.sex'] = 'Male'
    samples['chrX'] = np.zeros(samples.shape[0])
    samples['chrY'] = np.zeros(samples.shape[0])
    samples.ix[males, 'chrY'] = samples.ix[males, 'ccn.chrY'] - 1.0
    samples.ix[females, 'chrX'] = (samples.ix[females, 'ccn.chrX'] - 2.0) / 2.0

    if logtransform:
        samples.ix[females, 'chrX'] = np.log((-samples.ix[females, 'chrX']) + 1e-3)
        samples.ix[males, 'chrY'] = np.log((-samples.ix[males, 'chrY']) + 1e-3)
        samples.ix[females, 'chrX'] -= samples.ix[females, 'chrX'].mean()
        samples.ix[males, 'chrY'] -= samples.ix[males, 'chrY'].mean()

    X095 = (samples['ccn.chrX'] < 0.95)
    X105 = (samples['ccn.chrX'] > 1.05)
    X150 = (samples['ccn.chrX'] > 1.50)
    X195 = (samples['ccn.chrX'] < 1.95)
    X205 = (samples['ccn.chrX'] > 2.05)

    Y050 = (samples['ccn.chrY'] < 0.50)
    Y005 = (samples['ccn.chrY'] > 0.05)
    Y095 = (samples['ccn.chrY'] < 0.95)
    Y105 = (samples['ccn.chrY'] > 1.05)

    samples['ccnX'] = samples['ccn.chrX'].copy()
    samples['ccnY'] = samples['ccn.chrY'].copy()

    samples.ix[~females, 'ccnX'] = np.nan
    samples.ix[~males, 'ccnY'] = np.nan

    if 0:
        # males
        samples.ix[(samples['ccnX'] < 1.5) & Y095, 'ccnY'] = 0.95
        samples.ix[(samples['ccnX'] < 1.5) & Y105, 'ccnY'] = 1.05
        samples.ix[(samples['ccnY'] > 0.5) & X095, 'ccnX'] = 0.95
        samples.ix[(samples['ccnY'] > 0.5) & X105, 'ccnX'] = 1.05
        # females
        samples.ix[X150 & X195, 'ccnX'] = 1.95
        samples.ix[X150 & X205, 'ccnX'] = 2.05
        samples.ix[Y050 & Y005, 'ccnY'] = 0.05

        if drop_outliers:
            samples = samples[~outliers]
    return samples
    #
    # mae_female = np.absolute(samples[['ccn.chrX', 'ccn.chrY']].values - np.array([2,0])[np.newaxis,:]).mean(1)
    # mae_male   = np.absolute(samples[['ccn.chrX', 'ccn.chrY']].values - np.array([1,1])[np.newaxis,:]).mean(1)
    # mse_female = ((samples[['ccn.chrX', 'ccn.chrY']].values - np.array([2,0])[np.newaxis,:])**2).mean(1)
    # mse_male   = ((samples[['ccn.chrX', 'ccn.chrY']].values - np.array([1,1])[np.newaxis,:])**2).mean(1)
    #
    # mae = pd.DataFrame({"female": mae_female, "male":mae_male}, index=samples.index)
    # mse = pd.DataFrame({"female": mse_female, "male":mse_male}, index=samples.index)
    # return mae, mse


def exclude(df, on, what):
    idx = ~df[on].isnull()
    for c in what:
        idx = idx & (df[on] != c)
    data = df.copy()[idx]
    return data


if __name__ == '__main__':
    import pylab as plt
    plt.ion()
    import seaborn as sns
    from postgenomic.telomeres.load_integration import load_integration_data

    from datastack.pipes_lib.datasource.settings import *
    qc_date = "08_01_2016"
    qc = load_integration_data(fn='/Users/clippert/Data/telomeres_GC/qc-export/%s/0-integration.csv' % qc_date)
    data_all = qc[qc[MASTER_INDEX]]

    data_all['is_genentech'] = data_all['Client'].copy()
    data_all.ix[data_all['is_genentech'] != 'Genentech', 'is_genentech'] = data_all.ix[data_all['is_genentech'] != 'Genentech', 'Chemistry']

    data = data_all.copy()
    if 0:
        # V2.5 chemistry samples have lower telomeres than V2 samples
        chemistries_exclude = ['V2.5']
        # Spector and Gleeson differ a lot between old and new data
        # V2 Genentech samples have lower telomeres than V2 samples
        studies_exclude = ['Spector', 'Gleeson', 'Genentech']
        data = exclude(data, CHEMISTRY, chemistries_exclude)
        data = exclude(data, CLIENT_ID, studies_exclude)
    else:
        # V2.5 chemistry samples have lower telomeres than V2 samples
        # Spector and Gleeson differ a lot between old and new data
        # V2 Genentech samples have lower telomeres than V2 samples
        studies_exclude = []  # 'Spector', 'Gleeson', 'Genentech']
        sex_exclude = ['Male']
        data = exclude(data_all, CHEMISTRY, chemistries_exclude)
        data = exclude(data, PHENO_GENDER, sex_exclude)
        idx1 = (data[CLIENT_ID] == 'Genentech') & (data[PHENO_GENDER] == 'Female')
        idx2 = (data[CLIENT_ID] != 'Genentech') & (data[PHENO_GENDER] == 'Female')

        # plot
        # ========================================
        graph = sns.jointplot(x=data[idx1]['ccn.chrX'], y=data[idx1]['ccn.chrY'], color='r', alpha=0.4)
        graph.x = data[idx2]['ccn.chrX']
        graph.y = data[idx2]['ccn.chrY']
        graph.plot_joint(plt.scatter, marker='.', c='b', s=50, alpha=0.4)
        plt.legend(['Genentech (N=%i)' % idx1.sum(), 'Other (N=%i)' % idx2.sum()])
        plt.xlim([0.9, 3.0])
        plt.ylim([-0.01, 0.2])
        plt.savefig("ccnXY_Genentech_V2.pdf")

    plot_cols = ['hli_calc_age_sample_taken', 'telomeres.Length', 'ccn.chrX', 'ccn.chrY', 'ccn.chrM']
    plot_cols = ['ccn.chrX', 'ccn.chrY']
    hue = 'Chemistry'
    hue = 'Client'
    hue = 'is_genentech'
    plot_data = data[[hue] + plot_cols]
    sns.pairplot(plot_data.dropna(), hue=hue)
