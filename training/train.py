import copy as cp
import numpy as np
import os
import joblib
from tqdm import tqdm

# ----------------------------------------------------------------------------------

class ModelResult:

    def __init__(self, models, model_name):
        self.models = models
        self.model_name = model_name

        self.y_preds = []
        self.y_gts = []

        self.chns = []
        self.best = None
        self.ds_name = None

    def __str__(self):
        output = ""
        output += "Modelname: " + self.model_name + "\n"
        output += "Num Channels: " + str(len(self.chns)) + "\n"
        output += "Channels: " + str(self.chns) + "\n"
        best_str = "Best" if self.best else "Worst"
        output += best_str + "\n"
        output += self.ds_name

        return output

# ----------------------------------------------------------------------------------

def train_models_from_datasets(datasets, ds_names, method):
    for ds, ds_name in zip(datasets, ds_names):
        print(ds_name)

        model_path = os.path.join(ds.result_path, f"models_{method}")
        imp_path = os.path.join(ds.result_path, f"importances_{method}")
        best_electrodes = joblib.load(imp_path)

        folds_path = os.path.join(ds.result_path, "folds_list")
        save_res = True

        try:
            best_electrodes = [a for a, b in best_electrodes]
        except:
            best_electrodes = [a for a, b, _ in best_electrodes]

        worst_electrodes = list(reversed(best_electrodes))

        model_results = []

        for i in ds.num_channels_per_iteration:

            mr_list = []
            for best in [True, False]:
                print("-" * 20)

                print("best" if best else "worst")

                if best:
                    electrodes = best_electrodes
                else:
                    electrodes = worst_electrodes

                X, y, folds, feature_names = joblib.load(folds_path)

                print("select:", electrodes[:i])

                X, feature_names = exclude_channels_from_X(X, feature_names, ds.channel_names, electrodes[i:])
                model_res = train_and_run_models(X, y, folds, models=ds.models, show_progress=False)

                for mr in model_res:
                    print("num chns", len(electrodes[:i]))
                    mr.chns = electrodes[:i]
                    mr.best = best
                    mr.ds_name = ds_name
                    print(ds_name)

                mr_list.append(model_res)

            model_results.append(mr_list)

        if save_res:
            print("save to ", model_path)
            joblib.dump(model_results, model_path)

        return model_results

# ----------------------------------------------------------------------------------

def train_and_run_models(X, y, folds_indices, models=None, show_progress=False):
    model_results = []

    # iterate over the classifier, execute a cross validation and store the results
    for i, c in enumerate(models):
        print(i)

        if show_progress:
            print(f"{i + 1}/{len(models)}", c[1])

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

def _cross_val_predict(X, y, folds_indices, model, show_progress):
    """
    Executes a cross validation and returns additional information which the sklearn CV does not.
    """

    # create arrays and lists to store the results
    y_gts = []
    y_preds = []

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

def exclude_channels_from_X(X, feature_names, ch_names, exclude_channels):
    parts = [fn.split("_")[0] for fn in feature_names]
    mask = [(False if p in exclude_channels else True) for p in parts]
    indices = [i for i, b in enumerate(mask) if b]

    X_new = X[:, indices]
    feature_names_new = feature_names[indices]

    return X_new, feature_names_new

# ----------------------------------------------------------------------------------
