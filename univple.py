# Universal pleasantness project from Majid lab

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
import platform
import pyrfume
import pystan
from scipy.stats import norm, kstest
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
import warnings
from wurlitzer import sys_pipes
os.environ['NUMEXPR_MAX_THREADS'] = '8'

N_SUBJECTS = 283


def load_data(by='odor'):
    data = pyrfume.load_data('mainland_unpub/UniversalPleasantness/Universal Pleasantness.csv')
    data = data.set_index(['Group', 'Participant', 'OdorName']).unstack('OdorName')['Ranking'].astype(int)
    odorants = list(data)
    if by=='ranks':
        data = data.apply(np.argsort, axis=1)+1
        data.columns = ['1st', '2nd', '3rd'] + ['%dth' % x for x in range(4, 11)]
    return data, odorants


def shuffle_data(data, groups, how='within-group'):
    # Shuffle ranks within each group.
    # A unique shuffle is generated for each group, but all individuals within a group get the same shuffle
    n_odorants = data.shape[1]
    shuffles = [np.argsort(np.random.randn(n_odorants)) for i in range(len(groups))]
    data_sh = data.apply(lambda x: x[shuffles[groups.index(x.name[0])]].reset_index(drop=True), axis=1)
    data_sh.columns = data.columns
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
