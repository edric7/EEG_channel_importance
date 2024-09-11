import math
import copy as cp
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import os
import seaborn as sn
import random
import joblib
import data
import utils
import scipy
import mne
import pandas as pd
import warnings
from tqdm import tqdm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# ----------------------------------------------------------------------------------

class ModelResult:
    
    def __init__(self, models, model_name):
        self.models = models
        self.model_name = model_name
        
        self.y_preds = []
        self.y_gts = []
        
        self.chns = []
        
# ----------------------------------------------------------------------------------

def run_model_training_random(data_str, ch_names, num_channels, window_len, step_size, models, model_path, independent, save_res=True, n_splits=10, iterations=2, 
                              ref_type=None, whole_data_path=None, raw_list_path=None, sample_len_gme=6, offset_gme=3):
    sample_len = sample_len_gme
    offset = offset_gme
    model_results = []

    for i in num_channels:
        print("...............")
        print("num ch", i)

        model_result = None

        for j in range(0, iterations):
            print("- iter", j)

            random.shuffle(ch_names)
            print(ch_names[:i])
            
            if ref_type == "avgRefAfter":
                warnings.warn("avgRefAfter is not yet implemented")
                #X, y, folds, feature_names = data.prepare_data(raw_list_path, use_ref, sample_len, window_len, offset, step_size,
                #                                   include_channels=ch_names[:i], independent=independent, n_splits=n_splits)
            else:
                X, y, folds, feature_names = joblib.load(whole_data_path)
                X, feature_names = data.exclude_channels_from_X(X, feature_names, ch_names, ch_names[i:])

            model_name = "RandomForest"
            model = RandomForestClassifier(random_state=0)

            y_gts, y_preds, trained_models = _cross_val_predict(X, y, folds, model, False)

            print(">", len(y_gts), len(y_preds))

            if model_result is None:
                model_result = ModelResult(trained_models, model_name)

            if isinstance(y_preds, list): model_result.y_preds += y_preds
            else: model_result.y_preds += y_preds.tolist()

            if isinstance(y_gts, list): model_result.y_gts += y_gts
            else: model_result.y_gts += y_gts.tolist()

            print(">>", len(model_result.y_preds), len(model_result.y_preds))

        model_results.append([model_result])

    if save_res:
        print("save to ", model_path)
        joblib.dump(model_results, model_path)

    return model_results



def run_model_training(data_str, ch_names, num_channels, window_len, step_size, models, model_path, best_electrodes, independent, 
                           save_res=True, n_splits=10, ref_type=None, whole_data_path=None, raw_list_path=None, sample_len_gme=6, offset_gme=3):
    
    sample_len = sample_len_gme
    offset = offset_gme
    
    try:
        best_electrodes = [a for a,b in best_electrodes]
    except:
        best_electrodes = [a for a,b,_ in best_electrodes]
        
    if not models:
        models = [
            (RandomForestClassifier(random_state=0), "RandomForest"), 
            (LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'), "LDA"),
        ]

    model_results = []

    for i in num_channels:
         
        print(ref_type)
        if ref_type == "avgRefAfter":
            warnings.warn("avgRefAfter is not yet implemented")
                              
            if data_str == 'gme':
                pass
                #X, y, folds, feature_names = data.prepare_data(raw_list_path, use_ref, sample_len, window_len, offset, step_size,
                #                                   include_channels=best_electrodes[:i], independent=independent, n_splits=n_splits)
            else:
                warnings.warn("No no ref data preparation for IEA data")
        else:
            X, y, folds, feature_names = joblib.load(whole_data_path)
            print("select:", best_electrodes[:i]) 
            
            X, feature_names = data.exclude_channels_from_X(X, feature_names, ch_names, best_electrodes[i:])

        model_res = train_and_run_models(X, y, folds, models=models, show_progress=False)    

        for mr in model_res:
            mr.chns = best_electrodes[:i]

        model_results.append(model_res)

    if save_res:
        print("save to ", model_path)
        joblib.dump(model_results, model_path)

    return model_results


# ----------------------------------------------------------------------------------


def train_and_run_models(X, y, folds_indices, models=None, show_progress=False):
  
    # default list of classifiers to evaluate
    if not models:
        models = [
            (RandomForestClassifier(random_state=0), "RandomForest"), 
            (LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'), "LDA"),
        ]
        
    model_results = []
    
    # iterate over the classifier, execute a cross validation and store the results
    for i, c in enumerate(models):
        if show_progress:
            print(f"{i+1}/{len(models)}", c[1])
        
        model_name = c[1]
        model = c[0]
        
        y_gts, y_preds, trained_models = _cross_val_predict(X, y, folds_indices, model, show_progress)
        
        model_result = ModelResult(trained_models, model_name)
        
        if isinstance(y_preds, list):
            model_result.y_preds = y_preds
        else:
            model_result.y_preds = y_preds.tolist()
        
        if isinstance(y_gts, list):
            model_result.y_gts = y_gts
        else:
            model_result.y_gts = y_gts.tolist()
        
        model_results.append(model_result)
        
    return model_results

# ----------------------------------------------------------------------------------

def plot_results(model_results, save_path=None, average='macro', width_per_subplot=4, show=False, num_chns=0, file_format=None):
    
    if len(model_results) > 1: cols = 2
    else: cols = 1
    rows = int(math.ceil(len(model_results) / cols))
    
    fig_cm, ax_cm = plt.subplots(rows, cols)
    
    fmt = "{:.2f}"
    
    acc_mins = []
    acc_maxs = []
    acc_means = []
    
    rec_mins = []
    rec_maxs = []
    rec_means = []
    
    prec_mins = []
    prec_maxs = []
    prec_means = []
    
    fsc_mins = []
    fsc_maxs = []
    fsc_means = []
    
    model_names = []
 
    for i, res in enumerate(model_results):
        model_name = res.model_name
        
        px, py = utils.get_1d_as_n_m(i, cols)
 
        all_gts = []
        all_preds = []
        accs = []
        fscs = []
        precs = []
        recs = []

        for y_pred, y_gt in zip(res.y_preds, res.y_gts):

            all_gts += y_gt
            all_preds += y_pred
            
            accs.append(sklearn.metrics.accuracy_score(y_gt, y_pred))
            fscs.append(sklearn.metrics.f1_score(y_gt, y_pred, average=average))
            recs.append(sklearn.metrics.recall_score(y_gt, y_pred, average=average, zero_division=0))
            precs.append(sklearn.metrics.precision_score(y_gt, y_pred, average=average, zero_division=0))
    
        mean_acc = np.mean(accs)
        min_acc = np.min(accs)
        max_acc = np.max(accs)
        
        mean_rec = np.mean(recs)
        min_rec = np.min(recs)
        max_rec = np.max(recs)
        
        mean_prec = np.mean(precs)
        min_prec = np.min(precs)
        max_prec = np.max(precs)
        
        mean_fsc = np.mean(fscs)
        min_fsc = np.min(fscs)
        max_fsc = np.max(fscs)
        
        acc_mins.append(mean_acc - min_acc)
        acc_maxs.append(max_acc - mean_acc)
        acc_means.append(mean_acc)
        
        rec_mins.append(mean_rec - min_rec)
        rec_maxs.append(max_rec - mean_rec)
        rec_means.append(mean_rec)
        
        prec_mins.append(mean_prec - min_prec)
        prec_maxs.append(max_prec - mean_prec)
        prec_means.append(mean_prec)
        
        fsc_mins.append(mean_fsc - min_fsc)
        fsc_maxs.append(max_fsc - mean_fsc)
        fsc_means.append(mean_fsc)
        
        print(model_name)
        print(f"\tAccuracy: {fmt.format(mean_acc)} ({fmt.format(min_acc)}, {fmt.format(max_acc)})")
        print(f"\tF-Score: {fmt.format(mean_fsc)} ({fmt.format(min_fsc)}, {fmt.format(max_fsc)}) (unweighted avg)")
        #print(f"\tPrecision (average='{average}'): {fmt.format(mean_prec)} ({fmt.format(min_prec)}, {fmt.format(max_prec)})")
        #print(f"\tRecall (average='{average}'): {fmt.format(mean_rec)} ({fmt.format(min_rec)}, {fmt.format(max_rec)})")

        model_names.append(model_name)
            
        # add confusion matrices to the axes
        if rows > 1: axis =  ax_cm[px][py]
        elif cols > 1: axis = ax_cm[py]
        else: axis = ax_cm
                
        plot_confusion_matrix(axis, all_gts, all_preds, [0,1], model_name=model_name)

        
    # plot the confusion matrices
    width = cols * width_per_subplot
    height = rows * width_per_subplot
    fig_cm.set_size_inches(width, height)
    
    if save_path: 
        plt.savefig(os.path.join(save_path, f"cm_{num_chns}.{file_format}"))
    if show:
        plt.show()
    
    plt.close()
        
    """
    rows = 1
    cols = 2

    # plot errorbar for accuracy and F-Score
    fig_sc, ax_sc = plt.subplots(rows, cols)

    model_names = [mr.model_name for mr in model_results]

    if rows > 1:
    
        plot_errorbar(ax_sc[0][0], model_names, acc_means, acc_mins, acc_maxs, "Accuracy")
        plot_errorbar(ax_sc[0][1], model_names, fsc_means, fsc_mins, fsc_maxs, "F-Score")
        
    else:
        plot_errorbar(ax_sc[0], model_names, acc_means, acc_mins, acc_maxs, "Accuracy")
        plot_errorbar(ax_sc[1], model_names, fsc_means, fsc_mins, fsc_maxs, "F-Score")
        
        
    #plot_errorbar(ax_sc[1][0], model_names, prec_means, prec_mins, prec_maxs, "Precision")
    #plot_errorbar(ax_sc[1][1], model_names, rec_means, rec_mins, rec_maxs, "Recall")

    width = cols * width_per_subplot
    height = rows * width_per_subplot
    fig_sc.set_size_inches(width, height)
    
    if save_path:
        plt.savefig(os.path.join(save_path, f"metrics_{num_chns}.{file_format}"))
    
    if show:
        plt.show()
    plt.close()
    """
    
# ----------------------------------------------------------------------------------

def _cross_val_predict(X, y, folds_indices, model, show_progress):
    """
    Executes a cross validation and returns additional information which the sklearn CV does not.

    Args:
        model: the classifier
        folds: the folds for the cross validation
        X: data 
        y: labels
    """

    # create arrays and lists to store the results
    y_gts = [] # np.empty([0], dtype=int)
    y_preds = [] # np.empty([0], dtype=int)
    
    y = np.array(y)
    trained_models = []
    
    # iterate over each fold, fit the model, predict and store the results
    idx = 0
    
    if show_progress:
        container = tqdm(folds_indices)
    else: 
        container = folds_indices
    
    for train_ind, test_ind, subj_idx in container:
        train_X, test_X = X[train_ind], X[test_ind]
        train_y, test_y = y[train_ind], y[test_ind]
        
        model_ = cp.deepcopy(model)
        model_.fit(train_X, train_y)
        
        preds = model_.predict(test_X)
        
        trained_models.append((model_, idx))    
        
        y_gts.append(test_y.tolist()) 
        y_preds.append(preds.tolist()) 
        
        idx += 1
    
    return y_gts, y_preds, trained_models

# ----------------------------------------------------------------------------------

def plot_confusion_matrix(axis, actual_classes, predicted_classes, sorted_labels, model_name=""):
    """
    plots a confusion matrix to the given axis
    """

    # plots a confusion matrix
    axis.set_title(model_name)
    matrix = confusion_matrix(actual_classes, predicted_classes, labels=sorted_labels)
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    
    sn.heatmap(matrix, annot=True, fmt=".2f", xticklabels=sorted_labels, yticklabels=sorted_labels, cmap="Blues", ax=axis, vmin=0, vmax=1)
    axis.set_xlabel('Predicted')
    axis.set_ylabel('Actual')

# ----------------------------------------------------------------------------------

def plot_errorbar(axis, model_names, means, lower, upper, title):
    """
    plots the scores with min, max and average in an errorplot
    """
    col = (1.0, 0.0, 0.0, 0.5)
    axis.errorbar(model_names, means, yerr = [lower, upper], fmt='.k', ecolor=col, elinewidth=3, capsize=10)
    axis.set_ylim(0,1.01)
    axis.set_title(title)

# ----------------------------------------------------------------------------------

def plot_importances(best_electrodes, xlabel, title=None, save_path=None, figsize=(5, 4)):
      
    x_acc = [t[2] for t in best_electrodes]
    ticks_acc = [t[0] for t in best_electrodes]

    fig = plt.figure(figsize=figsize)
    plt.barh(ticks_acc, x_acc, color ='maroon', height = 0.8)
    
    plt.ylabel("Electrodes")
    plt.xlabel(xlabel)
    
    if title:
        plt.title(f"Effect of removing electrode on accuracy")
    
    if save_path: 
        plt.savefig(save_path)
    
    plt.show()
    
# ----------------------------------------------------------------------------------

def line_plot_over_methods(ax, data_str, param, num_channels, file_format, vmin='auto', vmax='auto', title="", idx=0, horizontal=False):

    methods = ['rf','mi','pi','abl','shap','random']
    results = []
    scores_dict = {}
    result_path = param['paths']['plots']['all'] 
    
    vmin_auto = True if vmin=='auto' else False
    vmax_auto = True if vmax=='auto' else False
    
    for m in methods:
        r = joblib.load(param['paths']['res'][m])
        
        results.append((r,m))
        scores_dict[m] = {}
        scores_dict[m]['acc'] = []
        scores_dict[m]['fsc'] = []
    
    # iteration over importance algorithms
    for mr_list, method in results:

        # iteration over number of channels
        for mr in mr_list:

            res = mr[0]

            accs = []
            fscs = []

            # iteration over folds
            for y_pred, y_gt in zip(res.y_preds, res.y_gts):
                accs.append(sklearn.metrics.accuracy_score(y_pred, y_gt))
                fscs.append(sklearn.metrics.f1_score(y_pred, y_gt, average='macro'))

            scores_dict[method]['acc'].append(np.mean(accs))
            scores_dict[method]['fsc'].append(np.mean(fscs))

    for alg in scores_dict:
        for m in scores_dict[alg]:
            scores_dict[alg][m].reverse()

    # num_channels.reverse()
    x = num_channels
    x = [str(x_) for x_ in x]
    x.reverse()
    
    # fig, ax = plt.subplots(1) # , figsize=(6,6))

    # ax[0].set_title(title)
    ax.set_title(title)
    
    for i, m in enumerate(['acc']): # , 'fsc']):

        axis = ax # [i]

        axis.plot(x, scores_dict['rf'][m], label="RF", marker='o')
        axis.plot(x, scores_dict['mi'][m], label="MI", marker='o')
        axis.plot(x, scores_dict['pi'][m], label="PI", marker='o')
        axis.plot(x, scores_dict['abl'][m], label="AS", marker='o')
        axis.plot(x, scores_dict['shap'][m], label="SHAP", marker='o')
        axis.plot(x, scores_dict['random'][m], label="Random", marker='o', linestyle='dashed')

        vals = scores_dict['rf'][m] + scores_dict['mi'][m] + scores_dict['pi'][m] + scores_dict['abl'][m] + scores_dict['shap'][m] + scores_dict['random'][m]
        
        if vmin_auto:
            vmin = min(vals) - 0.01
        
        if vmax_auto:
            vmax = max(vals) + 0.01
        
        axis.set_xticks(x)
        axis.set_ylim((vmin,vmax))
        if m == 'acc': 
            axis.set_ylabel("Accuracy")
        else: 
            axis.set_ylabel("F-Score")

        axis.set_xlabel("Number of Electrodes")
        if horizontal: 
            if idx == 2:
                axis.legend(loc=(1.01, 0.33))
        else:
            axis.legend(loc=(1.01, 0.45))

                
        axis.grid()

    #plt.savefig(os.path.join(result_path, f"lineplot_best.{file_format}"), bbox_inches='tight')
    #plt.show()
    
# ----------------------------------------------------------------------------------

def correlation_rankings(ax, param, file_format, width_per_subplot=3, title="", idx=0):
    
    result_path = param['paths']['plots']['all']
    rankings = []
    methods = ['rf','mi','pi','abl','shap']
    
    for m in methods:
        imp_path = param['paths']['imp'][m]

        electrodes = joblib.load(imp_path)
        ranking = [(imp[0], i+1) for i, imp in enumerate(electrodes)]
        ranking = sorted(ranking, key=lambda elem: elem[0])
        ranking = [b for a,b in ranking]
        rankings.append(ranking)

    rankings_str = ["RF", "MI", "PI", "AS", "SHAP"] # list(map(str.upper, methods))

    
    print(np.array(rankings).shape)
    print(rankings_str)
    
    res = scipy.stats.spearmanr(rankings, axis=1, alternative='two-sided')

    #cols = 2
    #rows = 1
    ax_cm = ax
    # fig_cm, ax_cm = plt.subplots(1, 2)

    # width_per_subplot = 5
    #width = cols * width_per_subplot
    #height = rows * width_per_subplot
    #fig_cm.set_size_inches(width, height)

    cbar = True
    if idx == 2:
        cbar = True
        # sns.heatmap(df, cbar=False)
    
    cmap = sn.color_palette("vlag", as_cmap=True)
    sn.heatmap(res.correlation, annot=True, fmt=".2f", xticklabels=rankings_str, yticklabels=rankings_str, center=0, ax=ax_cm, vmin=-1, vmax=1, cbar=cbar, cmap=cmap)
    # sn.heatmap(res.pvalue, annot=True, fmt=".2f", xticklabels=rankings_str, yticklabels=rankings_str, cmap="Blues", ax=ax_cm[1], vmin=0, vmax=1)

    # plt.title("Spearman")
    ax_cm.set_title(title)
    # ax_cm[1].set_title("P-Value")
    #plt.savefig(os.path.join(result_path, f"spearman.{file_format}"))
    #plt.show()

# ----------------------------------------------------------------------------------

def correlation_rankings_folds(ax, param, file_format, width_per_subplot=3, title="", idx=0):
    
    result_path = param['paths']['plots']['all']
    methods = ['pi','abl','shap']
    
    plt.rcParams['font.size'] = str(12)
    
    names = ['Permutation Importance', 'Ablation Study', 'SHAP']
    
    j = -1
    for m, name in zip(methods,names): 
        j += 1
        ax[j].set_title(name)
        
        print("------------", m)
        rankings = []
        imp_path = param['paths']['imp'][m]
        imps_per_fold = joblib.load(imp_path + "_per_fold")
        
        rankings_str = []
        for idx, imps in enumerate(imps_per_fold):
            
            imps = sorted(imps, key=lambda elem:elem[1])
                    
            ranking = [(imp[0], i+1) for i, imp in enumerate(imps)]
            ranking = sorted(ranking, key=lambda elem: elem[0])
            ranking = [b for a,b in ranking]
            
            rankings.append(ranking)
            rankings_str.append(str(idx+1))

        res = scipy.stats.spearmanr(rankings, axis=1, alternative='two-sided')
        
        ax_cm = ax[j]

        cbar = True
        if idx == 2:
            cbar = True
            
        cmap = sn.color_palette("vlag", as_cmap=True)

        sn.heatmap(res.correlation, annot=False, fmt=".2f", xticklabels=rankings_str, 
                   yticklabels=rankings_str, center=0, ax=ax_cm, vmin=-1, vmax=1, cbar=cbar, cmap=cmap)
        
    
# ----------------------------------------------------------------------------------

def topo_plot_map(ax, param, file_format, figsize=(10,6), fontsize=10, title="", idx=0):

    
    plt.rcParams['font.size'] = str(15)
    
    whole_data_path = param['paths']['whole']
    raw_list_path = param['paths']['raw']
    result_path = param['paths']['plots']['all']
    ch_names = param['ch_names']
    num_channels = param['num_channels']
    window_len = param['window_length']
    step_size = param['step_size']
    data_type = param['data_type']
    
    # Load electrodes rankings
    electrodes_mi = joblib.load(param['paths']['imp']['mi'])
    electrodes_rf = joblib.load(param['paths']['imp']['rf'])
    electrodes_pi = joblib.load(param['paths']['imp']['pi'])
    electrodes_shap = joblib.load(param['paths']['imp']['shap'])
    electrodes_abl = joblib.load(param['paths']['imp']['abl'])
    
       
    electrodes_rf = [(ch, v) for ch,_,v in electrodes_rf]
    electrodes_mi = [(ch, v) for ch,_,v in electrodes_mi]
    electrodes_pi = [(ch, v) for ch,_,v in electrodes_pi]
    electrodes_abl = [(ch, v) for ch,_,v in electrodes_abl]
    electrodes_shap = [(ch, v) for ch,_,v in electrodes_shap]

    
    # order the scores according to ch_names
    electrodes_rf_ = [(ch, (dict(electrodes_rf))[ch]) for ch in ch_names]
    electrodes_mi_ = [(ch, (dict(electrodes_mi))[ch]) for ch in ch_names]
    electrodes_pi_ = [(ch, (dict(electrodes_pi))[ch]) for ch in ch_names]
    electrodes_abl_ = [(ch, (dict(electrodes_abl))[ch]) for ch in ch_names]
    electrodes_shap_ = [(ch, (dict(electrodes_shap))[ch]) for ch in ch_names]

    # get the values
    vals_rf = [val for ch, val in electrodes_rf_]
    vals_mi = [val for ch, val in electrodes_mi_]
    vals_pi = [val for ch, val in electrodes_pi_]
    vals_abl = [val for ch, val in electrodes_abl_]
    vals_shap = [val for ch, val in electrodes_shap_]

    vals = [["RF",vals_rf], ["MI", vals_mi], ["PI", vals_pi], ["AS", vals_abl], ["SHAP", vals_shap]]

    for i, (_,v) in enumerate(vals):
        
        min_v = min(v)
        if min_v > 0: min_v = 0
        
        v = [v_-min_v for v_ in v]
        v = np.array([v])
        v = sklearn.preprocessing.normalize(v)
        vals[i][1] = v[0]
    
    if data_type == 'iea':
        print(ch_names)
        info = mne.create_info(ch_names, sfreq=500, ch_types=['eeg']*len(ch_names))
        info.set_montage('standard_1020')
    else:
        raws_filtered_path = f"results/gme_data/raws_filtered"
        raw_list = joblib.load(raws_filtered_path)
        raw = raw_list[0][0].copy()
        info = raw.info

    # fig, ax = plt.subplots(2,3, figsize=figsize)
    
    for i, (m,v) in enumerate(vals):
        # x, y = utils.get_1d_as_n_m(i, cols=3)
        if idx == 0:
            ax[i].title.set_text(m)
        
        #font = {'size': 12}
        #matplotlib.rc('font', **font)
        
        plt.rcParams['font.size'] = str(fontsize)

        mne.viz.plot_topomap(v, info, names=ch_names, show_names=True, vmin=0, vmax=1, axes=ax[i], show=False)

        
    # plt.savefig(os.path.join(result_path, f"topo.{file_format}"))
    # ax[1][1].set_visible(False)
    # ax[1][2].set_visible(False)
    # plt.show()
    
# ----------------------------------------------------------------------------------

def plot_importance_from_param(params, method, xlabel, file_format, figsize=(10,8)):
    
    for p in params:
        utils.print_param(p)
        # if p['load']: continue
        
        data_type = p['data_type']
        save_path = p['paths']['plots'][method]

        imp_path = p['paths']['imp'][method]
        save_path = os.path.join(save_path, f"importance_barchart.{file_format}")

        best_electrodes = joblib.load(imp_path)      
        plot_importances(best_electrodes, xlabel=xlabel, save_path=save_path, figsize=figsize)

# ----------------------------------------------------------------------------------

def train_models_from_param(params, method, random=False, iterations=10):

    for p in params:
        utils.print_param(p)
        if p['load']: continue
        ref_type = p['ref_type']
        independent = p['independent']
        raw_list_path = p['paths']['raw']
        
        model_path = p['paths']['res'][method]
        whole_data_path = p['paths']['whole']
        raw_data_path = p['paths']['raw']
        ch_names = p['ch_names']
        num_channels = p['num_channels']
        window_len = p['window_length']
        step_size = p['step_size']
        n_splits = p['n_splits']
        
        if not random:
        
            imp_path = p['paths']['imp'][method]
            best_electrodes = joblib.load(imp_path)   
            electrode_names = [b[0] for b in best_electrodes]

            run_model_training(data, ch_names, num_channels, window_len, step_size, None, model_path, best_electrodes, independent, 
                                save_res=True, n_splits=n_splits, ref_type=ref_type, whole_data_path=whole_data_path, raw_list_path=raw_list_path)    
        else:
            run_model_training_random(data, ch_names, num_channels, window_len, step_size, None, model_path, independent, 
                                      n_splits=n_splits, iterations=iterations, ref_type=ref_type, whole_data_path=whole_data_path, 
                                      raw_list_path=None, sample_len_gme=6, offset_gme=3)
    
# ----------------------------------------------------------------------------------

def plot_classification_results_from_param(params, method, file_format):

    for p in params:
        utils.print_param(p)
        if p['load']: continue

        plots_path = p['paths']['plots'][method]
        model_path = p['paths']['res'][method]

        model_results = joblib.load(model_path)
        num_chns = p['num_channels']
        
        for i, res in zip(num_chns, model_results):
            print("-"*20, "Num Electrodes:", i, "-"*20, "\n")
            print("electrodes:",  res[0].chns, "\n")

            plot_results(res, width_per_subplot=4, save_path=plots_path, num_chns=i, file_format=file_format)

# ----------------------------------------------------------------------------------

def plot_importances_from_param(ax, param, file_format, stacked=True, width=1, figsize=(10, 6), ylim=None, num_chn=None):
    whole_data_path = param['paths']['whole']
    raw_list_path = param['paths']['raw']
    result_path = param['paths']['plots']['all']
    ch_names = param['ch_names']
    num_channels = param['num_channels']
    window_len = param['window_length']
    step_size = param['step_size']
    data_type = param['data_type']

    freq_map = {"theta": r"$\theta$", 'delta':r"$\delta$", "alpha": r"$\alpha$", 
                "beta": r"$\beta$", "gamma": r"$\gamma$", "all": ""}

    # Load electrodes rankings
    electrodes_rf = joblib.load(param['paths']['imp']['rf'])
    electrodes_mi = joblib.load(param['paths']['imp']['mi'])
    electrodes_pi = joblib.load(param['paths']['imp']['pi'])
    electrodes_shap = joblib.load(param['paths']['imp']['shap'])
    electrodes_abl = joblib.load(param['paths']['imp']['abl'])

    freq_rf = {ch: freq_map[f] for ch,f,_ in electrodes_rf}
    freq_mi = {ch: freq_map[f] for ch,f,_ in electrodes_mi}
    freq_pi = {ch: freq_map[f] for ch,f,_ in electrodes_pi}
    freq_shap = {ch: freq_map[f] for ch,f,_ in electrodes_shap}
    freq_abl = {ch: freq_map[f] for ch,f,_ in electrodes_abl}

    electrodes_rf = {ch: v for ch,_,v in electrodes_rf}
    electrodes_mi = {ch: v for ch,_,v in electrodes_mi}
    electrodes_pi = {ch: v for ch,_,v in electrodes_pi}
    electrodes_abl = {ch: v for ch,_,v in electrodes_abl}
    electrodes_shap = {ch: v for ch,_,v in electrodes_shap}

    
      
    names = ["rf", "Mi", "pi", "abl", "shap"]
    freqss = [freq_rf, freq_mi, freq_pi, freq_abl, freq_shap]
    
    cols = []
    
    for n, freqs in zip(names, freqss):  
        # print(".........")
        # print(n)
        string = ""
        
        col = []
        for idx, f in enumerate(freqs): 
            if idx == 12:
                break
            string += f"& {f} {freq_rf[f]}"
            col.append(f"& {f} {freq_rf[f]}")
            
        cols.append(col)
        #print(string + r"\\")
        
    cols = np.array(cols)
    # cols = np.transpose(cols)
    
    for row in cols:
        string = []
        result = ''.join(v for v in row)
        print(result + r"\\")
        
    print(".........................................")
    
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

    vals = [["RF",vals_rf], ["MI", vals_mi], ["PI", vals_pi], ["SHAP", vals_shap], ["AS", vals_abl]]
    data = {}
    
    # shift values above 0, then normalize
    for i, (l,v) in enumerate(vals):
        min_v = min(v)
        
        if min_v > 0: 
            min_v = 0
        
        v = [v_-min_v for v_ in v]
        v = np.array([v])
        v = np.squeeze(sklearn.preprocessing.normalize(v))
        data[l] = v
        
    values = np.array([v for _,v in data.items()])
    values = np.sum(values, axis=0)
            
    indices = np.argsort(values)
    
    ch_names = [ch_names[i] for i in indices]
        
    if num_chn is not None:
        if num_chn > len(ch_names):
            num_chn = len(ch_names)
            
        num_chn_ = len(ch_names) - num_chn
        ch_names = ch_names[num_chn_:]
        
    for k,v in data.items():
        data[k] = [v[i] for i in indices]
        if num_chn != None:
            data[k] = data[k][num_chn_:]
            
    
    freq_rf_ = np.array(freq_rf_)[indices]
    freq_rf_ = [f for _,f in freq_rf_]
    
    freq_mi_ = np.array(freq_mi_)[indices]
    freq_mi_ = [f for _,f in freq_mi_]
    
    freq_pi_ = np.array(freq_pi_)[indices]
    freq_pi_ = [f for _,f in freq_pi_]
    
    freq_abl_ = np.array(freq_abl_)[indices]
    freq_abl_ = [f for _,f in freq_abl_]
    
    freq_shap_ = np.array(freq_shap_)[indices]
    freq_shap_ = [f for _,f in freq_shap_]

    freqs = [freq_rf_, freq_mi_, freq_pi_, freq_shap_, freq_abl_]  

    index = pd.Index(ch_names, name='Channels')

    df = pd.DataFrame(data, index=index)
    df.plot(kind='bar', stacked=stacked, width=width, ax=ax)
    if ylim is not None:
        ax.set_ylim((0, ylim))
        
    ax.set_ylabel('Normalized Importance Scores')
        
    if num_chn is not None:
        num_chn_ = len(param['ch_names']) - num_chn
        
        for i, f in enumerate(freqs):
            freqs[i] = freqs[i][num_chn_:]
        
    for idx, c in enumerate(ax.containers):
        labels = freqs[idx]
 
        # remove the labels parameter if it's not needed for customized labels
        ax.bar_label(c, labels=labels, label_type='center', color="white", fontsize=15)

    return cols