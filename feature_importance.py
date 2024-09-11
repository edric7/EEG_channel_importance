import numpy as np
import sklearn
import mne
import pandas as pd
import shap
import joblib
import data
import evaluation
import warnings
import functools
import time

from sklearn.inspection import permutation_importance
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

# ----------------------------------------------------------------------------------

def timer(func):
    # https://stackoverflow.com/questions/14452145/how-to-measure-time-taken-between-lines-of-code-in-python
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print("Finished {} in {} secs".format(repr(func.__name__), round(run_time, 3)))
        return value, run_time

    return wrapper

# ----------------------------------------------------------------------------------

def get_electrodes(feature_names):
    """
    Returns which electrode is used in the feature names given
    """

    electrodes = set()
    for fn in feature_names:
        en = fn.split('_')
        electrodes.add(en[0])
    return electrodes

# ----------------------------------------------------------------------------------

def print_electrodes(ch_names):
    """
    prints a scalp with the location electrodes which belong to the desired channels
    from: https://stackoverflow.com/questions/58783695/how-can-i-plot-a-montage-in-python-mne-using-a-specified-set-of-eeg-channels
    """

    montage = mne.channels.make_standard_montage('standard_1020')
    ind = [i for (i, channel) in enumerate(montage.ch_names) if channel in ch_names]
    montage_new = montage.copy()

    # Keep only the desired channels
    montage_new.ch_names = [montage.ch_names[x] for x in ind]
    kept_channel_info = [montage.dig[x+3] for x in ind]

    # Keep the first three rows
    montage_new.dig = montage.dig[0:3]+kept_channel_info
    montage_new.plot()

# ----------------------------------------------------------------------------------

def _aggregate_scores(score, feature_names, plot, title, agg_type='max'):
    
    feat_importances = pd.Series(score, index=feature_names)
    
    zipped = list(zip(score, feature_names))
    zipped = sorted(zipped, key=lambda x: x[1], reverse=True)
    
    ch_freq_dict = {}
    
    # get max scores over channels
    channel_scores = {}
    for imp, name in zipped:       
        ch = name.split('_')[0]
        freq = name.split('_')[1]
        
        if agg_type == 'max':
            if ch in channel_scores:
                if imp > channel_scores[ch][1]:
                    channel_scores[ch] = (freq, imp)
                    # ch_freq_dict[ch] = (freq, imp)
            else:
                channel_scores[ch] = (freq, imp)
        else:
            # only add positive importances
            imp_ = imp if imp >= 0 else 0
            if ch in channel_scores: channel_scores[ch] += imp_
            else: channel_scores[ch] = imp_
                
    electrodes = []
    for k,(f,v) in channel_scores.items():
        electrodes.append((k,f,v))
        
    # print(ch_freq_dict)

    best_electrodes = sorted(electrodes, key=lambda x: x[2], reverse=True)
    # best_electrodes = sorted(electrodes, key=lambda x: x[1], reverse=True)
    
    print(best_electrodes)
    return list(best_electrodes)

# ----------------------------------------------------------------------------------

@timer
def get_importances_rf_importance(X, y, feature_names, plot=False, agg_type='max'):
    rf = RandomForestClassifier(random_state=0)
    rf.fit(X,y)
    importances = rf.feature_importances_
    return _aggregate_scores(importances, feature_names, plot, "RandomForest Feature Importance", agg_type=agg_type)

# ----------------------------------------------------------------------------------

@timer
def get_importances_mutual_information(X, y, feature_names, plot=False, random_state=0, agg_type='max'):
    score = mutual_info_classif(X, y, random_state=random_state)
    return _aggregate_scores(score, feature_names, plot, "Mutual Information Importance", agg_type=agg_type)

# ----------------------------------------------------------------------------------  

@timer
def get_importances_permutation_importance(models, X, y, folds, feature_names, n_repeats, plot=False, random_state=0, agg_type='max'):

    importances = None
    imps_per_fold = []
     
    for m in models:
        model = m[0]
        idx = m[1]
        
        train_ind, test_ind, _ = folds[idx]
        y = np.array(y)
        
        train_X, test_X = X[train_ind], X[test_ind]
        train_y, test_y = y[train_ind], y[test_ind]
        
        res = permutation_importance(model, test_X, test_y, n_repeats=n_repeats, random_state=0)
        
        if importances is None:
            importances = np.array(res.importances_mean)
        else: 
            importances += np.array(res.importances_mean)
        
        
        imps_per_fold.append(res.importances_mean)
    
    importances_avg = importances / len(folds)
    agg_scores = _aggregate_scores(importances_avg, feature_names, plot, "Permutation Importance", agg_type=agg_type)
    
    agg_imps_per_fold = []
    
    for imps in imps_per_fold:
        agg_imps_per_fold.append(_aggregate_scores(imps, feature_names, plot, "Permutation Importance", agg_type=agg_type))

    return agg_scores, agg_imps_per_fold

# ----------------------------------------------------------------------------------

@timer
def get_importances_shap(X, y, folds_indices, feature_names, plot=False, random_state=0, agg_type='max', test=False, use_old_alg=False, early_stop=None):

    y = np.array(y)
    shap.initjs()

    importances_sum = None 
    shap_list = []
    
    imps_per_fold = []
    
    if early_stop is None: reps = len(folds_indices)
    else: reps = early_stop
    
    for train_ind, test_ind, _ in tqdm(folds_indices[:early_stop]):
        
        X_train, X_test = X[train_ind], X[test_ind]
        y_train, y_test = y[train_ind], y[test_ind]
        
        if test:
            warnings.warn(">> TEST MODE IS ACTIVATED; ONLY USING SUBSET OF DATA")
            X_train = X_train[:10]
            y_train = y_train[:10]
            X_test = X_test[:10]
            y_test = y_test[:10]
        
        X_train_ = pd.DataFrame(X_train)
        X_train_.columns = feature_names
        X_test_ = pd.DataFrame(X_test)
        X_test_.columns = feature_names
        
        # use a RandomForestRegressor
        rforest = sklearn.ensemble.RandomForestRegressor(n_estimators=1000, max_depth=None, min_samples_split=2, random_state=random_state)
        rforest.fit(X_train_, y_train)
        
        # explain all the predictions in the test set
        explainer = shap.TreeExplainer(rforest)
        shap_values = explainer.shap_values(X_test_)
        
        shap_list.append(shap_values)
        
        imps_per_fold.append(np.sum(np.abs(shap_values), axis=0))
        
        if plot:
            shap.summary_plot(shap_values, X_test_)
            shap.force_plot(explainer.expected_value, shap_values, X_test_)

            
            
    shap_concats = np.concatenate(shap_list)
    importances = np.sum(np.abs(shap_concats), axis=0)
    
    
    agg_score = _aggregate_scores(importances, feature_names, plot, "Shap Importance", agg_type='max')
    agg_imps_per_fold = []
    for imps in imps_per_fold:
        agg_imps_per_fold.append(_aggregate_scores(imps, feature_names, plot, "Permutation Importance", agg_type=agg_type))
    
    return agg_score, agg_imps_per_fold
    
# ----------------------------------------------------------------------------------

@timer
def get_ablation_importance(models, ch_names, raw_list_path, sample_len, offset, window_len, step_size, independent, plot=False, random_state=0, 
                            ref_type=None, whole_data_path=None):
    
    X, y, folds, feature_names = joblib.load(whole_data_path)
    y = np.array(y)
    
    imps_per_fold = []
    
    idx = -1
    for train_ind, test_ind, subj_idx in folds:
        idx += 1
        print(f"---------- fold: {idx+1}")
        
        train_X, test_X = X[train_ind], X[test_ind]
        train_y, test_y = y[train_ind], y[test_ind]
    
        importances = []
    
        # all electrodes
        model = RandomForestClassifier(random_state=0) 
        model.fit(train_X, train_y)
        preds = model.predict(test_X)
        acc_all = sklearn.metrics.accuracy_score(preds, test_y)

        for ch in ch_names:
            exclude_chn = [ch]
            
            X_train_select, fns_select = data.exclude_channels_from_X(train_X, feature_names, ch_names, exclude_chn)
            X_test_select, _ = data.exclude_channels_from_X(test_X, feature_names, ch_names, exclude_chn)

            model = RandomForestClassifier(random_state=0) 
            model.fit(X_train_select, train_y)
                        
            preds = model.predict(X_test_select)
            acc_select = sklearn.metrics.accuracy_score(preds, test_y)
            
            acc_decrease = acc_all - acc_select
            print("fold:", idx+1, ", exclude:", ch, ", decrease in acc:", acc_decrease)
            
            importances.append(acc_decrease)
        
        imps_per_fold.append(importances)
        
    tmp = np.array(imps_per_fold)
    avg_channel_scores = np.average(np.array(imps_per_fold), axis=0)
    avg_channel_scores = [(ch,'all', sc) for ch,sc in zip(ch_names, avg_channel_scores)]
    
    print(avg_channel_scores)
    
    sorted_acc = sorted(avg_channel_scores, key=lambda x: x[2], reverse=True)
    
    # sorted_acc = [()]

    imps_per_fold_ = []
    for imps in imps_per_fold:
        tuples = [(ch,sc) for ch,sc in zip(ch_names, imps)]
        sorted_tps = sorted(tuples, key=lambda x: x[1], reverse=True)
        
        imps_per_fold_.append(sorted_tps)
    
    return sorted_acc, imps_per_fold_


  

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

def get_ablation_importance1(models, X, y, folds, fns, ch_names, raw_list_path, sample_len, offset, window_len, 
                            step_size, independent, plot=False, random_state=0, ref_type=None, whole_data_path=None):

    model_results = evaluation.train_and_run_models(X, y, folds, models, show_progress=False)
    
    y_preds_all = []
    y_gts_all = []
    
    acc_sum = 0
    fsc_sum = 0
    count = 0
    for y_pred, y_gt in zip(model_results[0].y_preds, model_results[0].y_gts):
        acc_sum += sklearn.metrics.accuracy_score(y_pred, y_gt)
        fsc_sum += sklearn.metrics.f1_score(y_pred, y_gt)
        count += 1
    
    acc_all = acc_sum/count
    fsc_all = fsc_sum/count
    
    print("Accuracy all electrodes:", acc_all)
    print("Fscore all electrodes:", fsc_all)
    
    
    
    
    
    
    
    electrode_scores = []
    print("Electrodes: ", ch_names)

    for i, ch in enumerate(ch_names):
        print("\n", "-"*10, "Exclude: ", ch, f"({i+1}/{len(ch_names)})", "-"*10, "\n")

        exclude_chn = [ch]
        
        if ref_type == "avgRefAfter":
            warnings.warn("avgRefAfter is not yet implemented")
            #X, y, folds, feature_names = data.prepare_data(raw_list_path, sample_len, offset, window_len, step_size,
            #                                               exclude_channels=exclude_chn, independent=independent, n_splits=n_splits)
        else:
            X, y, folds, feature_names = joblib.load(whole_data_path)
            X, feature_names = data.exclude_channels_from_X(X, feature_names, ch_names, exclude_chn)

        model_results = evaluation.train_and_run_models(X, y, folds, models, show_progress=False)
        
        y_preds = []
        y_gts = []

        for y_pred, y_gt in zip(model_results[0].y_preds, model_results[0].y_gts):
            y_preds += y_pred
            y_gts += y_gt

        acc = sklearn.metrics.accuracy_score(y_preds, y_gts)
        fsc = sklearn.metrics.f1_score(y_preds, y_gts)
        
        print("Acc: ", acc, ", Fsc: ", fsc)
        print("Decrease in accuracy: ", acc_all-acc)
        print("Decrease in F-Score: ", fsc_all-fsc)

        electrode_scores.append((ch, acc_all-acc, fsc_all-fsc))    

    sorted_acc = sorted(electrode_scores, key=lambda x: x[1], reverse=True)
    sorted_fsc = sorted(electrode_scores, key=lambda x: x[2], reverse=True)

    return sorted_acc    


  
