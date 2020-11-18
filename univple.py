# Universal pleasantness project from Majid lab

from copy import copy
import joblib
from matplotlib import lines, cm, colors
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pathlib
from pathlib import Path
import pickle
import pingouin as pg
import platform
import pyrfume
import pystan
from scipy.stats import norm, kendalltau, kstest
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from tqdm.auto import tqdm, trange
import warnings
from wurlitzer import sys_pipes
os.environ['NUMEXPR_MAX_THREADS'] = '8'

N_SUBJECTS = 283


def load_data(by='odor'):
    data = pd.read_csv('data/Universal Pleasantness.csv')
    data = data.set_index(['Group', 'Participant', 'OdorName']).unstack('OdorName')['Ranking'].astype(int)
    odorants = list(data)
    if by=='ranks':
        data = data.apply(np.argsort, axis=1)+1
        data.columns = ['1st', '2nd', '3rd'] + ['%dth' % x for x in range(4, 11)]
    return data, odorants


def shuffle_data(data, groups, how, random_state=0):
    # Shuffle ranks within each group.
    # A unique shuffle is generated for each culture, but all individuals within a group get the same shuffle
    n_individuals, n_odorants = data.shape
    n_groups = len(groups)
    data_sh = data.copy()
    if how == 'odors-within-culture':
        for i, group in enumerate(groups):
            group_index = data.index.get_level_values('Group') == group
            data_sh.loc[group_index] = data.loc[group_index].sample(frac=1, replace=False, axis=1, random_state=random_state+i).values
        #data_sh.index = data.index
        #shuffles = [np.argsort(np.random.randn(n_odorants)) for i in range(len(groups))]
        #data_sh = data.apply(lambda x: x[shuffles[groups.index(x.name[0])]].reset_index(drop=True), axis=1)
        #data_sh.columns = data.columns
    elif how == 'individuals':
        data_sh[:] = data.sample(frac=1, replace=False, random_state=random_state).values
        #data_sh.index = data.index
    elif how == 'odors':
        data_sh[:] = data.sample(frac=1, replace=False, random_state=random_state, axis=1).values
        #data_sh.index = data.index
    else:
        raise Exception(f"No such shuffle method: {how}")
    return data_sh


def get_groups(data):
    groups = list(data.index.get_level_values('Group').unique())
    group_ids = data.index.map(lambda x: groups.index(x[0])+1).values # Integer group IDs for each individual
    return groups, group_ids


def get_model_path(models_path: str, model_name: str,
                   compiled: bool = False, with_suffix: bool = False,
                   check_exists: bool=True) -> str:
    """Get a full model path for one model file.

    Args:
        models_path: Path to directory where models are stored.
        model_name: Name of the model (without .stan suffix).

    Returns:
        A full path to a Stan model file.
    """
    models_path = Path(models_path)
    if compiled:
        file_path = models_path / ('%s_%s_%s.stanc' %
                                   (model_name, platform.platform(),
                                    platform.python_version()))
    else:
        file_path = Path(models_path) / ('%s.stan' % model_name)
    if check_exists:
        assert file_path.is_file(), "No %s file found at %s" %\
            ('.stanc' if compiled else '.stan', file_path)
    if not with_suffix:
        file_path = file_path.with_suffix('')
    return file_path.resolve()


def load_or_compile_stan_model(model_name: str, models_path: str = '.',
                               force_recompile: bool = False,
                               verbose: bool = False):
    """Loads a compiled Stan model from disk or compiles it if does not exist.

    Args:
        model_name (str): Name of the stan model (i.e. model filename without the .stan suffix)
        force_recompile (bool, optional): [description]. Defaults to False.
        verbose (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    uncompiled_path = get_model_path(models_path, model_name, with_suffix=True)
    compiled_path = get_model_path(models_path, model_name,
                                   compiled=True, with_suffix=True, check_exists=False)
    stan_raw_last_mod_t = os.path.getmtime(uncompiled_path)
    try:
        stan_compiled_last_mod_t = os.path.getmtime(compiled_path)
    except FileNotFoundError:
        stan_compiled_last_mod_t = 0
    if force_recompile or (stan_compiled_last_mod_t < stan_raw_last_mod_t):
        models_path = str(Path(models_path).resolve())
        print(models_path)
        sm = pystan.StanModel(file=str(uncompiled_path), include_paths=[models_path])
        with open(compiled_path, 'wb') as f:
            pickle.dump(sm, f)
    else:
        if verbose:
            print("Loading %s from cache..." % model_name)
        with open(compiled_path, 'rb') as f:
            sm = pickle.load(f)
    return sm


def fit_model(model, d, warmup=5000, iter=20000):
    # The data that needs to be passed to the Stan model
    groups = list(d.index.get_level_values('Group').unique())
    data_ = {
        'n_odorants': 10, # How many odorants
        'n_individuals': d.shape[0], # How many individuals
        'n_groups': len(groups), # How many groups
        'group_id': d.index.map(lambda x: groups.index(x[0])+1).values, # Integer group IDs for each individual
        'ranks': d.iloc[:, -10:].values  # The last 10 columns of the dataframe, i.e. the ranking data
    }
    
    with sys_pipes(): # Used to pipe C-level output to the browser so I can see sampling messages.
        # Sample (fit) the model
        # None of these parameters matter except in the sense that sampling must proceed
        # slowly enough to get the answer without getting infinities.
        # This is all standard fare.  
        fit = model.sampling(data=data_, warmup=warmup, iter=iter, chains=4, control={'adapt_delta': 0.85, 'max_treedepth': 15})
        # You may see many warning messages but basically as long as nothing blows up and Rhat ~ 1 then it is OK.

    # Put the results into a Pandas dataframe
    samples = fit.to_dataframe()
    
    return fit, samples


def load_or_sample(model, data, name, use_cache=True, warmup=5000, iter=20000):
    name = 'samples_%s_%s_%s' % (name, joblib.hash(model), joblib.hash(data))
    path = pathlib.Path(name)
    if path.exists() and use_cache:
        samples = pd.read_csv(path)
    else:
        fit, samples = fit_model(model, data, warmup=warmup, iter=iter)
        samples.to_csv(path.name) # Save the fitted samples
    return samples
        
    
def plot_global_agreement(samples, odorants):
    # Check to see if chains (independent sampling runs) agree
    # This will be indiciated by each panel having 4 very similary (heavily overlapping) histograms
    fig, axes = plt.subplots(5, 2, sharex=True, figsize=(7, 6))
    for i, ax in enumerate(axes.flat):
        odorant_id = i + 1
        for chain in range(4):
            chain_samples = samples[samples['chain']==chain]
            global_odor_valence = chain_samples['mu_global[%d]' % odorant_id]
            # Plot the histogram of global valence samples for each odorant
            ax.hist(global_odor_valence, color='rgbk'[chain], alpha=0.5)
            ax.axes.get_yaxis().set_visible(False)
            ax.set_title(odorants[i])
    plt.tight_layout()
    
    
def get_order(odorants, samples):
    # Compute global mean valences for each odorant
    global_means = [samples['mu_global[%d]' % (j+1)].mean() for j in range(len(odorants))]
    # Get their order (by global mean descending) so we can plot all data in a common, sensible order
    order = np.argsort(global_means)[::-1]
    return order


def plot_global_means(groups, odorants, samples):
    # How do the odors differ in global valence?
    # Note that 0 is meaningless (it is not the transition between pleasant and unpleasant)
    # Results would be identical of a constant was added to all values
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    order = get_order(odorants, samples)
    m = np.array([samples['mu_global[%d]' % odorant].mean() for odorant in range(1, len(odorants)+1)])
    s = np.array([samples['mu_global[%d]' % odorant].std() for odorant in range(1, len(odorants)+1)])
    ax[0].errorbar(m[order], np.array(odorants)[order], xerr=s[order], fmt='o')
    ax[0].set_xlabel('Global Valence');
    
    for i, group in enumerate(groups):
        m = [samples['mu_group[%d,%d]' % (i+1, j+1)].mean() for j in order]
        s = [samples['mu_group[%d,%d]' % (i+1, j+1)].std() for j in order]
        ax[1].errorbar(m, range(len(odorants)), xerr=s, alpha=0.5, marker='o', label=group)
    ax[1].set_yticks(range(len(odorants)), [odorants[i] for i in order])
    ax[1].set_xlabel('Group Valence')
    ax[1].legend(loc=(1.04, 0))
    
    
# Create an individual level sigma which is the mean of the
# individual-level sigmas for each group
def plot_sigmas(samples, groups):
    samples['sigma_ind'] = samples[['sigma_ind[%d]' % i for i in range(1, len(groups)+1)]].median(axis=1)
    sigmas = {'$\sigma_{global}$': 'sigma_global',
              '$\sigma_{group}$': 'sigma_group',
              '$\sigma_{individual}$': 'sigma_ind'}
    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(3, 2)
    axl = [fig.add_subplot(gs[i, 0]) for i in range(len(sigmas))]
    for j in range(4): # Over chains
        s = samples[samples['chain']==j]
        for i, (name, var) in enumerate(sigmas.items()):
            axl[i].hist(s[var], bins=25, color='rgbk'[j], alpha=0.3)
            axl[i].axes.get_yaxis().set_visible(False)
            axl[i].set_xlabel(name)
            axl[i].set_xlim(0, 6)
    axr = fig.add_subplot(gs[:, 1])
    means = [samples[key].mean() for key in sigmas.values()]
    stds = [samples[key].std() for key in sigmas.values()]
    plt.bar(range(len(sigmas)), means, yerr=stds, capsize=9)
    plt.xticks(range(len(sigmas)), sigmas.keys())
    plt.tight_layout()
    
    
def new_init(model, fit):
    # Continue sampling from the last sample of the previous fit
    means = [fit.unconstrain_pars({key: value.mean(axis=0) for key, value in fit.extract().items()})]*4
    means = [{key: value.mean(axis=0) for key, value in fit.extract().items()}]*4
    return means


def mu_group_corr(samples, groups, odorants, transpose=True):
    mugs = samples[[x for x in samples if 'mu_group' in x]].values
    mugs = mugs.reshape(-1, len(groups), len(odorants))
    if transpose:
        rs = np.dstack([np.corrcoef(mug.T) for mug in mugs])  # Odorant vs Odorant correlation matrix for each sample
    else:
        rs = np.dstack([np.corrcoef(mug) for mug in mugs])  # Group vs Group correlation matrix for each sample
    return rs


def corr_heatmaps(samples, groups, odorants, transpose=True):
    sns.set(font_scale=1)
    fig, ax = plt.subplots(1, 2, figsize=(15, 4))
    plt.subplots_adjust(wspace=0.5)
    
    # Correlation matrices (group vs group) for each sample
    group_rs = mu_group_corr(samples, groups, odorants, transpose=True)
    # Mean of the inter-group correlation for each sample
    group_rs_mean = pd.DataFrame(group_rs.mean(axis=2), index=groups, columns=groups)
    sns.heatmap(group_rs_mean, vmin=-1, vmax=1, cmap='RdBu_r', cbar_kws={'label': 'R'}, ax=ax[0])
    
    # Correlation matrices (group vs group) for each sample
    odorant_rs = mu_group_corr(samples, groups, odorants, transpose=False)
    # Mean of the inter-group correlation for each sample
    odorant_rs_mean = pd.DataFrame(odorant_rs.mean(axis=2), index=odorants, columns=odorants)
    sns.heatmap(odorant_rs_mean, vmin=-1, vmax=1, cmap='RdBu_r', cbar_kws={'label': 'R'}, ax=ax[1])


def plot_var_explained(samples, groups, odorants):
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(10, 4))
    mugs = get_means(samples, groups, odorants, 'mu_group')
    pca = PCA()
    for i, X in enumerate([mugs.T, mugs]):
        pca.fit(X)
        n_components = len(pca.explained_variance_ratio_)
        explained = [0] + list(pca.explained_variance_ratio_.cumsum())
        ax[i].plot(range(1+n_components), explained, 'o-')
        ax[i].set_xlim(-0.2, 10)
        ax[i].set_ylim(-0.02, 1.02)
        ax[i].set_xlabel("Number of PCs (for %s)" % ('Groups' if i==0 else 'Odorants'))
    ax[0].set_ylabel("Cumulative Variance Explained")

    
def get_means(samples, groups, odorants, param):
    if param == 'mu_ind':
        index = range(N_SUBJECTS)
        columns = odorants
    elif param == 'mu_group':
        index = groups
        columns = odorants
    else:
        raise Exception("Could not handle param %s" % param)
    x = pd.DataFrame([[samples['%s[%d,%d]' % (param, j+1, i+1)].mean()
                       for i,_ in enumerate(columns)] for j,_ in enumerate(index)],
                     index=index, columns=columns)
    return x

    
def plot_all_individuals(samples, groups, group_ids, odorants):
    mugs = get_means(samples, groups, odorants, 'mu_group')
    muis = get_means(samples, groups, odorants, 'mu_ind')
    
    pca = PCA(n_components=2)
    mugs_pcs = pca.fit_transform(mugs)
    assert list(mugs.columns) == list(odorants)
    group_colors = plt.cm.rainbow(np.linspace(0, 1, len(groups)))
    individual_colors = [group_colors[group_ids[i]-1] for i in range(len(group_ids))]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for j, ax in enumerate(axes):
        # Plot all the groups as big dots
        for i, group in enumerate(groups):
            ax.scatter(*mugs_pcs[i, :].T,
                        c=group_colors[i].reshape(1, -1), s=70, label=group);

        # Plot all the individuals as small dots
        ax.scatter(*pca.transform(muis).T,
                    c=individual_colors, s=5, alpha=0.5);
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2');
        if j==1:
            ax.legend(loc=(1.04, 0));
        lim = 12 if j==0 else 4
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_title('Zoomed' if j==1 else 'Full-Scale')

    
def ranks_vs_values(samples, groups):
    data, odorants = load_data(by='odor')
    n_subjects = data.shape[0]
    mu_ind = get_means(samples, groups, odorants, 'mu_ind')
    rs_ind = np.zeros(n_subjects)
    for i in range(n_subjects):
        rs_ind[i] = np.corrcoef(data.iloc[i, :], mu_ind.iloc[i, :])[0, 1]
    plt.hist(rs_ind);
    plt.xlabel("Pearson correlation between rankings and valences")
    plt.ylabel("Number of subjects")
    
    
def ss(coords, clusters):
    """Sum of squares from cluster centers"""
    x = 0
    for cluster in clusters.values():
        # Sum of squared distances of each cluster member to the cluster mean
        sumsquares = (coords.loc[cluster].sub(coords.loc[cluster].mean(axis=0), axis=1)**2).sum().sum()
        x += sumsquares
    return x


def ss_null(coords, by, n=1000):
    sizes = [len(x) for x in by.values()]
    csizes = [0] + list(np.cumsum(sizes))
    result = []
    for i in range(n):
        shuffle = np.random.permutation(coords.index)
        by_shuffle = {i: shuffle[csizes[i]:csizes[i+1]] for i in range(len(sizes))}
        result.append(ss(coords, by_shuffle))
    return result


def get_supergroup_stats(samples, groups, odorants, supergroups):
    mu_group = get_means(samples, groups, odorants, 'mu_group')
    sg_scores = {kind: ss(mu_group, grouping) for kind, grouping in supergroups.items()}
    sg_scores_null = {kind: ss_null(mu_group, grouping, n=10000) for kind, grouping in supergroups.items()}
    supergroup_stats = pd.Series([(sg_scores_null[key] < sg_scores[key]).mean() for key in supergroups], index=supergroups.keys())
    return supergroup_stats


def plot_supergroups(samples, groups, odorants, supergroups, method='PCA'):
    mugs = get_means(samples, groups, odorants, 'mu_group')
    if method == 'PCA':
        pca = PCA(n_components=2)
        mugs_reduced = pca.fit_transform(mugs)
    elif method == 'MDS':
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            mds = MDS(n_components=2, dissimilarity='euclidean')
            mugs_reduced = mds.fit_transform(mugs)
    mugs_reduced = pd.DataFrame(mugs_reduced, index=mugs.index, columns=[1, 2])
    supergroup_nums = []
    fig, ax = plt.subplots(1, len(supergroups), figsize=(12, 4))
    plt.subplots_adjust(wspace=1.5)
    for i, (sg_label, sg) in enumerate(supergroups.items()):
        for j, (label, some_groups) in enumerate(sg.items()):
            c = 'rgbk'[j]
            for k, group in enumerate(some_groups):
                m = 'ovsp*'[k]
                ax[i].scatter(*mugs_reduced.loc[group].T, color=c, marker=m, label='%s (%s)' % (group, label))
        ax[i].legend(fontsize=9, loc=(1.04, 0))
        ax[i].set_title('By %s' % sg_label.title())
        ax[i].set_xlabel('Dim 1')
        ax[i].set_ylabel('Dim 2')
    plt.suptitle(method)
    

def plot_ind_corrs(data, samples, groups, odorants):
    mu_ind = get_means(samples, groups, odorants, 'mu_ind')
    mu_ind.index = data.index
    fig, axes = plt.subplots(4, 5, figsize=(18, 12))
    rs = {}
    for i, group in enumerate(groups):
        ax = axes.flat[i]
        rs[group] = mu_ind.loc[group].T.corr()
        sns.heatmap(rs[group], vmin=-1, vmax=1, cmap='RdBu_r', ax=ax)
        ax.set_title(group)
        if i % 5:
            ax.set_ylabel('')
        if i < 5:
            ax.set_xlabel('')
    for i, group in enumerate(groups):
        ax = axes.flat[i + len(groups)]
        n = rs[group].shape[0]
        x = rs[group].values[np.triu_indices(n, k=1)]
        ax.hist(x)
        ax.set_xlim(-1, 1)
        ax.set_title(group)
        if i % 5 == 0:
            ax.set_ylabel('# of pairs\nof individuals')
        if i >= 5:
            ax.set_xlabel('Correlation (R)')
        if group == 'Maniq':
            n_odorants = len(odorants)
            n_individuals = mu_ind.loc[group].shape[0]
            z_mean = 0
            z_se = 1/np.sqrt(n_odorants-3)
            rs_ = np.linspace(-0.999, 0.999, 10000)
            zs = np.arctanh(rs_)
            z_pdf = norm.pdf(zs, z_mean, z_se)
            ax.plot(rs_, z_pdf*n_individuals, 'r--')
            ks, p = kstest(x, 'norm', args=(z_mean, z_se))
            ax.set_title('%s (p=%.3g)' % (group, p))
    plt.tight_layout();

    
def compute_kendall_taus(raw_data, model_predictions):
    """Compute Kendall-Tau correlation for ranks between individuals, and between model predictions and individuals."""
    taus = pd.DataFrame(index=raw_data.index, columns=raw_data.index)
    for individual1 in tqdm(taus.index):
        for individual2 in taus.columns.drop(individual1):
            x = raw_data.loc[individual1]
            y = raw_data.loc[individual2]
            taus.loc[individual1, individual2] = kendalltau(x, y)[0]
    taus.loc[('Model', 1), :] = raw_data.apply(lambda x: kendalltau(model_predictions, x)[0], axis=1)
    return taus


def fig_kendall_tau(taus, groups):
    tau_stats = pd.DataFrame(index=groups)
    for group in groups:
        group_data = taus.loc[group]
        n = group_data.shape[0]
        tau_stats.loc[group, 'Culture'] = group_data[group].mean().mean()
        tau_stats.loc[group, 'Culture_sem'] = group_data[group].std().mean() / np.sqrt(n)
        tau_stats.loc[group, 'Universal'] = group_data.mean().mean()
        tau_stats.loc[group, 'Universal_sem'] = group_data.std().mean() / np.sqrt(n)
        tau_stats.loc[group, 'Model'] = taus.loc['Model', group].mean().mean()
        # Model is the same for every group
        tau_stats.loc[group, 'Model_sem'] = 0#taus.loc['Model', group].std().mean() / np.sqrt(n)

    sns.set(font_scale=1.4)
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    cmap = cm.get_cmap('jet')
    norm = colors.Normalize(vmin=0, vmax=len(groups)-1)
    for i, group in enumerate(groups):
        tau_stats.loc[[group]].plot.scatter(x='Culture', y='Universal', xerr='Culture_sem', yerr='Universal_sem',
                                             color=cmap(norm(i)), label=group, ax=ax[0])
    ax[0].plot([0, 1], [0, 1], '--')
    ax[0].set_xlim(0, 0.6)
    ax[0].set_ylim(0, 0.6)
    ax[0].legend(loc=4, fontsize=11)
    ax[0].set_title('Correlation between individual and...')
    for i, group in enumerate(groups):
        tau_stats.loc[[group]].plot.scatter(x='Culture', y='Model', xerr='Culture_sem', yerr='Model_sem',
                                             color=cmap(norm(i)), label=group, ax=ax[1])
    ax[1].plot([0, 1], [0, 1], '--')
    ax[1].set_xlim(0, 0.6)
    ax[1].set_ylim(0, 0.6)
    ax[1].legend(loc=4, fontsize=11)
    ax[1].set_title('Correlation between individual and...')
    
    
def fig_highest_lowest(ranked_data, groups, odorants):
    fig, ax = plt.subplots(2, 2, sharex=True, figsize=(15, 8))
    firsts = pd.DataFrame(index=range(1, 11), columns=groups)
    lasts = pd.DataFrame(index=range(1, 11), columns=groups)
    for group in groups:
        firsts[group] = ranked_data.loc[group, '1st'].value_counts()
        lasts[group] = ranked_data.loc[group, '10th'].value_counts()
    firsts = firsts.fillna(0)
    lasts = lasts.fillna(0)
    cmap = copy(cm.get_cmap('Reds'))
    cmap.set_bad('white')
    sns.heatmap(firsts / firsts.sum(), cmap=cmap, ax=ax[0, 0], cbar_kws={'label': 'p(first)'})
    ax[0, 0].set_yticklabels(odorants, rotation=0);
    sns.heatmap(lasts / lasts.sum(), cmap=cmap, ax=ax[0, 1], cbar_kws={'label': 'p(last)'})
    ax[0, 1].set_yticklabels(odorants, rotation=0)
    sns.heatmap(firsts.corr(), ax=ax[1, 0], cmap='RdBu_r', vmin=-1, vmax=1, cbar_kws={'label': 'r(first)'})
    sns.heatmap(lasts.corr(), ax=ax[1, 1], cmap='RdBu_r', vmin=-1, vmax=1, cbar_kws={'label': 'r(last)'})
    plt.tight_layout()
    
    
def anova_eta2(df, ax=None):
    z = df.stack().reset_index()
    z = z.rename(columns={0: 'Rank'})
    aov = pg.anova(dv='Rank', between=['Group', 'OdorName'], data=z, detailed=True, effsize='n2').set_index('Source')
    assert aov.loc['Group', 'n2'] < 0.0001
    aov = aov.drop('Group')
    aov.loc['Residual', 'n2'] = 1 - aov['n2'].sum()
    aov = aov.rename(index={'Residual': 'Individual', 'Group': 'Culture', 'OdorName': 'Odor', 'Group * OdorName': 'Culture x Odor'})
    return aov


def get_eta2(raw_data, n_shuffles=25):
    groups = raw_data.index.unique('Group')
    shuffle_types = ['odors-within-culture', 'individuals']
    aov = anova_eta2(raw_data)
    eta2_mean = pd.DataFrame(columns=pd.Index(['raw']+shuffle_types, name='Shuffle Type'))
    eta2_sd = pd.DataFrame(columns=eta2_mean.columns)
    eta2_mean['raw'] = aov['n2']
    eta2_sd['raw'] = aov['n2']*0
    for shuffle_type in tqdm(shuffle_types):
        eta2_vals = pd.DataFrame(columns=aov['n2'].index)
        for i in trange(n_shuffles, leave=False):
            key = 'shuffle-%s' % shuffle_type
            shuffled_data = shuffle_data(raw_data, groups, shuffle_type, random_state=i)
            aov = anova_eta2(shuffled_data)
            eta2_vals.loc[i] = aov['n2']
        eta2_mean[shuffle_type] = eta2_vals.mean()
        eta2_sd[shuffle_type] = eta2_vals.std()
    return eta2_mean, eta2_sd


def fig_eta2(eta2_mean, eta2_sd, simplify=False):
    """A figure for showing eta^2 for the various ANOVA model factors."""
    colors = ['black', 'forestgreen', 'goldenrod']
    labels = ['Data', 'Odors shuffled\nwithin cultures', 'Individuals shuffled\nacross cultures']
    maxx = eta2_mean.max().max()*1.1
    if simplify:
        eta2_mean = eta2_mean['raw']
        eta2_sd = eta2_sd['raw']
        colors = ['lightgreen', 'mediumpurple', 'sandybrown']
        labels = ['']
        figsize = (7, 3)
    else:
        figsize = (15, 5)
    #sns.set(font_scale=1.4)
    #sns.set_style('whitegrid')
    plt.figure(figsize=figsize)
    ax = eta2_mean.plot.barh(color=colors, yerr=eta2_sd, ax=plt.gca())
    ax.set_xlabel('$\eta^2$');
    ax.set_ylabel('')
    if simplify:
        ax.set_yticklabels(x.get_text().split(' x ')[0] for x in ax.get_yticklabels())
    ax.set_xlim(0, maxx)
    if not simplify:
        l = ax.legend(loc=(0.78, 0.2), fontsize=14)
        _ = [old_label.set_text(labels[i]) for i, old_label in enumerate(l.get_texts())]
    
    
def shuffled_variances(raw_data, odorants):
    groups = raw_data.index.unique('Group')
    variances = pd.DataFrame(0, index=odorants, columns=['total', 'across', 'within',
                                                         'total-odors-shuffle', 'across-individuals-shuffle',
                                                         'within-individuals-shuffle', 'total-odors-within-culture-shuffle',
                                                         'across-odors-within-culture-shuffle', 'within-odors-within-culture-shuffle'])
    shuffles = ['odors-within-culture', 'individuals', 'odors']
    n_shuffles = 100
    for i in range(n_shuffles):
        for shuffle in shuffles:
            data['shuffle-%s' % shuffle] = up.shuffle_data(data['raw'], groups, shuffle, random_state=i)
        variances['total'] += data['raw'].var()
        variances['across'] += data['raw'].groupby('Group').mean().var()
        variances['within'] += data['raw'].groupby('Group').var().mean()
        variances['total-odors-shuffle'] += data['shuffle-odors'].var()
        variances['across-individuals-shuffle'] += data['shuffle-individuals'].groupby('Group').mean().var()
        variances['within-individuals-shuffle'] += data['shuffle-individuals'].groupby('Group').var().mean()
        variances['total-odors-within-culture-shuffle'] += data['shuffle-odors-within-culture'].var()
        variances['across-odors-within-culture-shuffle'] += data['shuffle-odors-within-culture'].groupby('Group').mean().var()
        variances['within-odors-within-culture-shuffle'] += data['shuffle-odors-within-culture'].groupby('Group').var().mean()
    variances /= n_shuffles
    return variances


def get_model_predictions(odorants):
    # Load predictions from DREAM model
    model_predictions = pd.read_csv('data/dream_model_prediction.csv', header=1, index_col=0).drop('CID')

    # Mapping between PubChem IDs and molecule names
    cids = {379: 'Octanoic acid', 1183: 'Vanillin', 3314: 'Eugenol', 6054: '2-Phenylethanol', 6549: 'Linalool',
            7762: 'Ethyl butyrate', 8077: 'Diethyl disulfide', 10430: 'Isovaleric acid', 18827: '1-Octen-3-ol', 32594: '2-Isobutyl-3-methoxypyrazine'}

    # Join model predictions with molecule names and sort
    model_predictions.index = model_predictions.index.astype(int)
    model_predictions = model_predictions.join(pd.Series(cids, name='Name')).set_index('Name')
    # Use out-of-sample prediction
    model_predictions = model_predictions.loc[odorants, 'Predicted_OUT'].rank(ascending=False).astype(int)
    return model_predictions
