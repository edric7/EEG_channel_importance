import copy

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import os
import seaborn as sn
import joblib
import scipy
import mne
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from confidenceinterval.bootstrap import bootstrap_ci
from data.data_handler_ceh import DataHandlerCeh


# ----------------------------------------------------------------------------------

def compute_confidence_intervals(datasets, ds_names, seed=0, n_resamples=100):
    print("n_resamples: ", n_resamples)

    methods = ['rf', 'mi', 'pi', 'abl', 'shap']
    accs_dict = {}
    model_idx = 0

    for idx, (ds, ds_name) in enumerate(zip(datasets, ds_names)):
        print(idx + 1, "/", len(ds_names))
        print(ds_name)

        num_chns = ds.num_channels_per_iteration

        results = []

        accs_dict[ds_name] = {}

        for m in methods:
            r = joblib.load(os.path.join(ds.result_path, f"models_{m}"))
            results.append((r, m))

            # iteration over importance algorithms
        for mr_list, method in tqdm(results):

            accs_dict[ds_name][method] = {}

            # iteration over number of channels
            for i, mr in enumerate(mr_list):
                num_chn = num_chns[i]

                accs_dict[ds_name][method][num_chn] = {}

                for mm in mr:
                    res = mm[0]

                    y_pred_all = []
                    y_true_all = []
                    accuracies = []

                    # iteration over folds
                    for y_pred, y_gt in zip(res.y_preds, res.y_gts):
                        y_pred_all += y_pred
                        y_true_all += y_gt
                        accuracies.append(sklearn.metrics.accuracy_score(y_pred, y_gt))

                    random_generator = np.random.default_rng(seed=seed)
                    mean, ci = bootstrap_ci(y_true=y_true_all,
                                            y_pred=y_pred_all,
                                            metric=sklearn.metrics.balanced_accuracy_score,
                                            confidence_level=0.95,
                                            n_resamples=n_resamples,
                                            method='bootstrap_bca',
                                            random_state=random_generator)

                    best = "best" if res.best else "worst"

                    accs_dict[ds_name][method][num_chn][best] = (mean, ci, accuracies, y_pred_all, y_true_all)

    return accs_dict

# ----------------------------------------------------------------------------------

def correlation_rankings(ax, dataset, title="", idx=0, horizontal=False):
    result_path = dataset.result_path
    rankings = []
    methods = ['rf', 'mi', 'pi', 'abl', 'shap']

    for m in methods:
        imp_path = os.path.join(dataset.result_path, f"importances_{m}")

        electrodes = joblib.load(imp_path)
        ranking = [(imp[0], i + 1) for i, imp in enumerate(electrodes)]
        ranking = sorted(ranking, key=lambda elem: elem[0])
        ranking = [b for a, b in ranking]
        rankings.append(ranking)

    rankings_str = ["RF", "MI", "PI", "AS", "SHAP"]

    print(np.array(rankings).shape)
    print(rankings_str)

    res = scipy.stats.spearmanr(rankings, axis=1, alternative='two-sided')

    ax_cm = ax

    if not horizontal:
        cbar = True
    else:
        cbar = True if idx == 2 else False

    cmap = sn.color_palette("vlag", as_cmap=True)
    g = sn.heatmap(res.correlation, fmt=".2f", annot=False, xticklabels=rankings_str, yticklabels=rankings_str,
                   center=0, ax=ax_cm, vmin=-1, vmax=1, cbar=cbar, cmap=cmap, cbar_kws={'fraction': 0.05})

    g.set_yticklabels(g.get_yticklabels(), rotation=90, fontsize=7)

    g.set_xticklabels(g.get_xticklabels(), rotation=0, fontsize=7)

    ax_cm.set_title(title)


# ----------------------------------------------------------------------------------

def correlation_rankings_folds(ax, dataset, idx=0):
    result_path = dataset.result_path
    methods = ['pi', 'abl', 'shap']

    plt.rcParams['font.size'] = str(6)

    names = ['PI', 'AS', 'SHAP']

    for j, (m, name) in enumerate(zip(methods, names)):
        if idx == 0:
            print(m, name)
            ax[j].set_title(name)

        rankings = []
        imp_path = os.path.join(dataset.result_path, f"importances_{m}")
        imps_per_fold = joblib.load(imp_path + "_per_fold")

        rankings_str = []
        for k, imps in enumerate(imps_per_fold):
            imps = sorted(imps, key=lambda elem: elem[1])

            ranking = [(imp[0], i + 1) for i, imp in enumerate(imps)]
            ranking = sorted(ranking, key=lambda elem: elem[0])
            ranking = [b for a, b in ranking]

            rankings.append(ranking)
            rankings_str.append(str(k + 1))

        res = scipy.stats.spearmanr(rankings, axis=1, alternative='two-sided')

        ax_cm = ax[j]

        if j == 2:
            cbar = True
        else:
            cbar = False

        cmap = sn.color_palette("vlag", as_cmap=True)

        sn.heatmap(res.correlation, annot=False, fmt=".2f", xticklabels=False,
                   yticklabels=False, center=0, ax=ax_cm, vmin=-1, vmax=1, cbar=cbar, cmap=cmap,
                   cbar_kws={'fraction': 0.05})


# ----------------------------------------------------------------------------------

def topo_plot_map(ax, dataset, fontsize=10, idx=0, norm="1", threshold=0.25, horizontal=False, contours=6):
    ch_names = dataset.channel_names

    freq_map = {"theta": r"$\theta$", 'delta': r"$\delta$", "alpha": r"$\alpha$",
                "beta": r"$\beta$", "lower gamma": r"$\gamma_1$", "upper gamma": r"$\gamma_2$", "all": "",
                "gamma": r"$\gamma$", "beta1": r"$\beta_1$", "beta2": r"$\beta_2$", "beta3": r"$\beta_3$",
                "beta4": r"$\beta_4$", "gamma1": r"$\gamma_1$", "gamma2": r"$\gamma_2$", "gamma3": r"$\gamma_3$",
                "gamma4": r"$\gamma_4$"
                }

    # Load electrodes rankings
    electrodes_mi = joblib.load(os.path.join(dataset.result_path, "importances_mi"))
    electrodes_rf = joblib.load(os.path.join(dataset.result_path, "importances_rf"))
    electrodes_pi = joblib.load(os.path.join(dataset.result_path, "importances_pi"))
    electrodes_shap = joblib.load(os.path.join(dataset.result_path, "importances_shap"))
    electrodes_abl = joblib.load(os.path.join(dataset.result_path, "importances_abl"))

    freq_rf = {ch: freq_map[f] for ch, f, _ in electrodes_rf}
    freq_mi = {ch: freq_map[f] for ch, f, _ in electrodes_mi}
    freq_pi = {ch: freq_map[f] for ch, f, _ in electrodes_pi}
    freq_shap = {ch: freq_map[f] for ch, f, _ in electrodes_shap}
    freq_abl = {ch: freq_map[f] for ch, f, _ in electrodes_abl}

    electrodes_rf = [(ch, v) for ch, _, v in electrodes_rf]
    electrodes_mi = [(ch, v) for ch, _, v in electrodes_mi]
    electrodes_pi = [(ch, v) for ch, _, v in electrodes_pi]
    electrodes_abl = [(ch, v) for ch, _, v in electrodes_abl]
    electrodes_shap = [(ch, v) for ch, _, v in electrodes_shap]

    # order the scores according to ch_names
    electrodes_rf_ = [(ch, (dict(electrodes_rf))[ch]) for ch in ch_names]
    electrodes_mi_ = [(ch, (dict(electrodes_mi))[ch]) for ch in ch_names]
    electrodes_pi_ = [(ch, (dict(electrodes_pi))[ch]) for ch in ch_names]
    electrodes_abl_ = [(ch, (dict(electrodes_abl))[ch]) for ch in ch_names]
    electrodes_shap_ = [(ch, (dict(electrodes_shap))[ch]) for ch in ch_names]

    # order the scores according to ch_names
    freq_rf_ = [(ch, freq_rf[ch]) for ch in ch_names]
    freq_mi_ = [(ch, freq_mi[ch]) for ch in ch_names]
    freq_pi_ = [(ch, freq_pi[ch]) for ch in ch_names]
    freq_abl_ = [(ch, freq_abl[ch]) for ch in ch_names]
    freq_shap_ = [(ch, freq_shap[ch]) for ch in ch_names]

    freq_dict = {"RF":freq_rf_, "MI":freq_mi_, "PI":freq_pi_,"AS":freq_abl_, "SHAP":freq_shap_}

    # get the values
    vals_rf = [val for ch, val in electrodes_rf_]
    vals_mi = [val for ch, val in electrodes_mi_]
    vals_pi = [val for ch, val in electrodes_pi_]
    vals_abl = [val for ch, val in electrodes_abl_]
    vals_shap = [val for ch, val in electrodes_shap_]

    vals = [["RF", vals_rf], ["MI", vals_mi], ["PI", vals_pi], ["AS", vals_abl], ["SHAP", vals_shap]]

    for i, (_, v) in enumerate(vals):
        v = np.array(v)
        v = v - np.min(v)

        if norm == '1':
            norm_ = np.sum(np.abs(v))
        elif norm == '2':
            norm_ = np.sqrt(np.dot(v,v))
        else:  # norm == 'max':
            norm_ = np.max(np.abs(v))

        vals[i][1] = v/norm_
        # print("sum", np.sum(vals[i][1]))

    info = mne.create_info(ch_names, sfreq=120, ch_types='eeg')
    info.set_montage('standard_1020')

    for i, (m, v) in enumerate(vals):
        freqs = freq_dict[m]
        plt.rcParams['font.size'] = str(fontsize)

        ch_names_ = []

        for j, (v_, ch) in enumerate(zip(v, ch_names)):
            if v_ > threshold:
                if m != "AS":
                    ch_names_.append(ch + freqs[j][1])
                else:
                    ch_names_.append(ch)
            else:
                ch_names_.append("")

        if horizontal:
            mne.viz.plot_topomap(v, info, names=ch_names_, vlim=(0, 1), axes=ax[idx][i], show=False, contours=contours)
        else:
            mne.viz.plot_topomap(v, info, names=ch_names_, vlim=(0, 1), axes=ax[i][idx], show=False, contours=contours)

# ----------------------------------------------------------------------------------


def plot_importance_from_datasets(datasets, method, xlabel, file_format, figsize=(5, 4)):
    for ds in datasets:
        print(ds)
        imp_path = os.path.join(ds.result_path, f"importances_{method}")
        best_electrodes = joblib.load(imp_path)
        save_path = os.path.join(ds.plots_path, f"importance_barchart_{method}.{file_format}")

        x_acc = [t[2] for t in best_electrodes]
        ticks_acc = [t[0] for t in best_electrodes]

        fig = plt.figure(figsize=figsize)
        plt.barh(ticks_acc, x_acc, color='maroon', height=0.8)

        plt.ylabel("Electrodes")
        plt.xlabel(xlabel)

        if save_path:
            plt.savefig(save_path)

        plt.show()


# ----------------------------------------------------------------------------------


def plot_importances_from_dataset(ax, dataset, stacked=True, width=1, ylim=None, num_chn=None, fontsize=12, norm="2",
                                  threshold=0.25):

    freq_map = {"theta": r"$\theta$", 'delta': r"$\delta$", "alpha": r"$\alpha$",
                "beta": r"$\beta$", "lower gamma": r"$\gamma_1$", "upper gamma": r"$\gamma_2$", "all": "",
                "gamma": r"$\gamma$", "beta1": r"$\beta_1$", "beta2": r"$\beta_2$", "beta3": r"$\beta_3$",
                "beta4": r"$\beta_4$", "gamma1": r"$\gamma_1$", "gamma2": r"$\gamma_2$", "gamma3": r"$\gamma_3$",
                "gamma4": r"$\gamma_4$"
                }

    ch_names = dataset.channel_names

    # Load electrodes rankings
    # list of triples: [(ch, freq, imp), ...]
    electrodes_rf = joblib.load(os.path.join(dataset.result_path, "importances_rf"))
    electrodes_mi = joblib.load(os.path.join(dataset.result_path, "importances_mi"))
    electrodes_pi = joblib.load(os.path.join(dataset.result_path, "importances_pi"))
    electrodes_shap = joblib.load(os.path.join(dataset.result_path, "importances_shap"))
    electrodes_abl = joblib.load(os.path.join(dataset.result_path, "importances_abl"))

    freq_rf = {ch: freq_map[f] for ch, f, _ in electrodes_rf}
    freq_mi = {ch: freq_map[f] for ch, f, _ in electrodes_mi}
    freq_pi = {ch: freq_map[f] for ch, f, _ in electrodes_pi}
    freq_shap = {ch: freq_map[f] for ch, f, _ in electrodes_shap}
    freq_abl = {ch: freq_map[f] for ch, f, _ in electrodes_abl}

    electrodes_rf = {ch: v for ch, _, v in electrodes_rf}
    electrodes_mi = {ch: v for ch, _, v in electrodes_mi}
    electrodes_pi = {ch: v for ch, _, v in electrodes_pi}
    electrodes_abl = {ch: v for ch, _, v in electrodes_abl}
    electrodes_shap = {ch: v for ch, _, v in electrodes_shap}

    # order the scores according to ch_names
    electrodes_rf_ = [(ch, electrodes_rf[ch]) for ch in ch_names]
    electrodes_mi_ = [(ch, electrodes_mi[ch]) for ch in ch_names]
    electrodes_pi_ = [(ch, electrodes_pi[ch]) for ch in ch_names]
    electrodes_abl_ = [(ch, electrodes_abl[ch]) for ch in ch_names]
    electrodes_shap_ = [(ch, electrodes_shap[ch]) for ch in ch_names]

    # order the scores according to ch_names
    freq_rf_ = [(ch, freq_rf[ch]) for ch in ch_names]
    freq_mi_ = [(ch, freq_mi[ch]) for ch in ch_names]
    freq_pi_ = [(ch, freq_pi[ch]) for ch in ch_names]
    freq_abl_ = [(ch, freq_abl[ch]) for ch in ch_names]
    freq_shap_ = [(ch, freq_shap[ch]) for ch in ch_names]

    # ---------------------------------------

    # get the values
    vals_rf = [val for ch, val in electrodes_rf_]
    vals_mi = [val for ch, val in electrodes_mi_]
    vals_pi = [val for ch, val in electrodes_pi_]
    vals_abl = [val for ch, val in electrodes_abl_]
    vals_shap = [val for ch, val in electrodes_shap_]

    vals = [["RF", vals_rf], ["MI", vals_mi], ["PI", vals_pi], ["SHAP", vals_shap], ["AS", vals_abl]]
    data = {}

    for m, v in vals:
        v = np.array(v)
        v = v - np.min(v)

        if norm == '1':
            norm_ = np.sum(np.abs(v))
        elif norm == '2':
            norm_ = np.sqrt(np.dot(v,v))
        else:  # norm == 'max':
            norm_ = np.max(np.abs(v))

        data[m] = v/norm_

    values = np.array([v for _, v in data.items()])
    values = np.sum(values, axis=0)
    indices = np.argsort(values)

    ch_names = [ch_names[i] for i in indices]

    if num_chn is not None:
        if num_chn > len(ch_names):
            num_chn = len(ch_names)

        num_chn_ = len(ch_names) - num_chn
        ch_names = ch_names[num_chn_:]

    for k, v in data.items():
        data[k] = [v[i] for i in indices]
        if num_chn is not None:
            data[k] = data[k][num_chn_:]

    freq_rf_ = np.array(freq_rf_)[indices]
    freq_rf_ = [f for _, f in freq_rf_]

    freq_mi_ = np.array(freq_mi_)[indices]
    freq_mi_ = [f for _, f in freq_mi_]

    freq_pi_ = np.array(freq_pi_)[indices]
    freq_pi_ = [f for _, f in freq_pi_]

    freq_abl_ = np.array(freq_abl_)[indices]
    freq_abl_ = [f for _, f in freq_abl_]

    freq_shap_ = np.array(freq_shap_)[indices]
    freq_shap_ = [f for _, f in freq_shap_]

    freqs = [freq_rf_, freq_mi_, freq_pi_, freq_shap_, freq_abl_]

    methods = ["RF","MI","PI","SHAP","AS"]

    index = pd.Index(ch_names, name='')

    df = pd.DataFrame(data, index=index)
    df.plot(kind='bar', stacked=stacked, width=width, ax=ax)

    if ylim is not None:
        ax.set_ylim((0, ylim))

    ax.legend(fontsize=10)

    if num_chn is not None:
        num_chn_ = len(dataset.channel_names) - num_chn

        for i, f in enumerate(freqs):
            freqs[i] = freqs[i][num_chn_:]

    for idx, c in enumerate(ax.containers):
        labels = freqs[idx]
        vals = data[methods[idx]]

        for k, v in enumerate(vals):
            if v < threshold:
                labels[k] = ""

        ax.bar_label(c, labels=labels, label_type='center', color="white", fontsize=fontsize)

