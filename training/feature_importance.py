import numpy as np
import sklearn
import pandas as pd
import copy as cp
import shap
import joblib
import math
import warnings

from training import train
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

# ----------------------------------------------------------------------------------


def _aggregate_scores(score, feature_names, agg_type='max'):
    
    zipped = list(zip(score, feature_names))
    zipped = sorted(zipped, key=lambda x: x[1], reverse=True)
    
    # get max scores over channels
    channel_scores = {}
    for imp, name in zipped:       
        ch = name.split('_')[0]
        freq = name.split('_')[1]
        
        if agg_type == 'max':
            if ch in channel_scores:
                if imp > channel_scores[ch][1]:
                    channel_scores[ch] = (freq, imp)
            else:
                channel_scores[ch] = (freq, imp)
        else:
            # only add positive importances
            imp_ = imp if imp >= 0 else 0

            if ch in channel_scores:
                channel_scores[ch] += imp_
            else:
                channel_scores[ch] = imp_
                
    electrodes = []
    for k, (f, v) in channel_scores.items():
        electrodes.append((k, f, v))

    best_electrodes = sorted(electrodes, key=lambda x: x[2], reverse=True)

    return list(best_electrodes)

# ----------------------------------------------------------------------------------


def get_importances_rf_importance(x, y, feature_names, agg_type='max'):
    n_estimators = int(math.sqrt(x.shape[1]))
    rf = RandomForestClassifier(random_state=0, n_estimators=n_estimators)
    rf.fit(x, y)
    return _aggregate_scores(rf.feature_importances_, feature_names, agg_type=agg_type)

# ----------------------------------------------------------------------------------


def get_importances_mutual_information(x, y, feature_names, random_state=0, agg_type='max'):
    score = mutual_info_classif(x, y, random_state=random_state)
    return _aggregate_scores(score, feature_names, agg_type=agg_type)

# ----------------------------------------------------------------------------------  


def get_importances_permutation_importance(models, x, y, folds, feature_names, n_repeats, random_state=0,
                                           agg_type='max'):

    importances = None
    imps_per_fold = []
     
    for m in tqdm(models):
        model = m[0]
        idx = m[1]
        
        train_ind, test_ind, _ = folds[idx]
        y = np.array(y)
        
        train_x, test_x = x[train_ind], x[test_ind]
        train_y, test_y = y[train_ind], y[test_ind]
        
        res = permutation_importance(model, test_x, test_y, n_repeats=n_repeats, random_state=random_state)
        
        if importances is None:
            importances = np.array(res.importances_mean)
        else: 
            importances += np.array(res.importances_mean)

        imps_per_fold.append(res.importances_mean)
    
    importances_avg = importances / len(folds)
    agg_scores = _aggregate_scores(importances_avg, feature_names, agg_type=agg_type)
    
    agg_imps_per_fold = []
    
    for imps in imps_per_fold:
        agg_imps_per_fold.append(_aggregate_scores(imps, feature_names, agg_type=agg_type))

    return agg_scores, agg_imps_per_fold

# ----------------------------------------------------------------------------------


def get_importances_shap(x, y, folds_indices, feature_names, random_state=0, agg_type='max', test=False, model=None):

    y = np.array(y)
    shap.initjs()

    shap_list = []
    imps_per_fold = []

    if test:
        warnings.warn(">> TEST MODE IS ACTIVATED; ONLY USING SUBSET OF DATA")
            
    use_rf_reg = True if model is None else False

    for train_ind, test_ind, _ in tqdm(folds_indices):
        
        x_train, x_test = x[train_ind], x[test_ind]
        y_train, y_test = y[train_ind], y[test_ind]
        
        if test:
            x_train = x_train[:10]
            y_train = y_train[:10]
            x_test = x_test[:10]

        x_train_ = pd.DataFrame(x_train)
        x_train_.columns = feature_names
        
        x_test_ = pd.DataFrame(x_test)
        x_test_.columns = feature_names

        if use_rf_reg:
            num_feat = int(math.sqrt(x_train_.shape[1]))
            model = sklearn.ensemble.RandomForestRegressor(n_estimators=num_feat, max_depth=None, min_samples_split=2,
                                                           random_state=random_state)
            model.fit(x_train_, y_train)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(x_test_)
        else:
            model_cpy = cp.deepcopy(model)
            model_cpy.fit(x_train_.to_numpy(), y_train)
            explainer = shap.SamplingExplainer(model_cpy.predict_proba, x_train_)
            shap_values = explainer.shap_values(x, npermutations=10, main_effects=False, error_bounds=False,
                                                batch_evals=True, silent=False)

        shap_list.append(shap_values)
        imps_per_fold.append(np.sum(np.abs(shap_values), axis=0))
        
    shap_concats = np.concatenate(shap_list)
    importances = np.sum(np.abs(shap_concats), axis=0)
    
    agg_score = _aggregate_scores(importances, feature_names, agg_type='max')
    agg_imps_per_fold = []
    for imps in imps_per_fold:
        agg_imps_per_fold.append(_aggregate_scores(imps, feature_names, agg_type=agg_type))
    
    return agg_score, agg_imps_per_fold
    
# ----------------------------------------------------------------------------------


def get_ablation_importance(ch_names, model, folds_path=None):
    
    x, y, folds, feature_names = joblib.load(folds_path)
    y = np.array(y)

    imps_per_fold = []
    
    idx = -1
    for train_ind, test_ind, subj_idx in folds:
        idx += 1

        train_x, test_x = x[train_ind], x[test_ind]
        train_y, test_y = y[train_ind], y[test_ind]
    
        importances = []


        # all electrodes
        model_ = clone(model)

        model_.fit(train_x, train_y)
        preds = model_.predict(test_x)
        acc_all = sklearn.metrics.accuracy_score(preds, test_y)

        for ch in ch_names:
            exclude_chn = [ch]
            
            x_train_select, fns_select = train.exclude_channels_from_X(train_x, feature_names, ch_names, exclude_chn)
            x_test_select, _ = train.exclude_channels_from_X(test_x, feature_names, ch_names, exclude_chn)

            # model = RandomForestClassifier(random_state=0)
            model_ = clone(model)
            model_.fit(x_train_select, train_y)
                        
            preds = model_.predict(x_test_select)
            acc_select = sklearn.metrics.accuracy_score(preds, test_y)
            
            acc_decrease = acc_all - acc_select
            print("fold:", idx+1, ", exclude:", ch, ", decrease in acc:", acc_decrease)
            
            importances.append(acc_decrease)
        
        imps_per_fold.append(importances)
        
    avg_channel_scores = np.average(np.array(imps_per_fold), axis=0)
    avg_channel_scores = [(ch, 'all', sc) for ch, sc in zip(ch_names, avg_channel_scores)]
    sorted_acc = sorted(avg_channel_scores, key=lambda xx: xx[2], reverse=True)

    imps_per_fold_ = []
    for imps in imps_per_fold:
        tuples = [(ch, sc) for ch, sc in zip(ch_names, imps)]
        sorted_tps = sorted(tuples, key=lambda xx: xx[1], reverse=True)
        
        imps_per_fold_.append(sorted_tps)
    
    return sorted_acc, imps_per_fold_

# ----------------------------------------------------------------------------------


  
