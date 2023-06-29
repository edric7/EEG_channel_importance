from mne_icalabel import label_components
import matplotlib.pyplot as plt
import numpy as np
import mne
import os
import copy as cp


# ----------------------------------------------------------------------------------

def standardize_data(data):
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    standardized_data = (data - mean) / std
    return standardized_data

# ----------------------------------------------------------------------------------
def apply_automatic_ICA_artifact_removal(raw_list, icas, verbose=0):
    filtered_ica = []

    for (raw, idx), ica in zip(raw_list, icas):
        raw.load_data()

        ic_labels = label_components(raw, ica, method="iclabel")

        labels = ic_labels["labels"]
        proba = ic_labels["y_pred_proba"]

        print(proba)

        exclude_idx = [idx for idx, label in enumerate(labels) if label not in ["brain", "other"]]

        if verbose:
            print("Excluding these ICA components:", [labels[i] for i in exclude_idx])

        ica.apply(raw, exclude=exclude_idx)

        filtered_ica.append((raw, idx))

    return filtered_ica

# ------------------------------------------------------------------------------------------

def compute_ICAs(raw_list, result_path, plot=False):
    print("create icas")

    icas = []

    for i, (raw, idx) in enumerate(raw_list):

        ica = mne.preprocessing.ICA(
            # n_components=15,
            max_iter="auto",
            method="infomax",
            random_state=0,
            fit_params=dict(extended=True),
        )

        ica.fit(raw)
        icas.append(ica)

        fig = plt.figure(figsize=(8,8))
        if plot:
            ica.plot_components(inst=raw, picks=range(22), show=False)
        plt.savefig(os.path.join(result_path, f"ica_comps_raw_gme_{i}_{idx}.png"))
        plt.show()

    return icas

# ----------------------------------------------------------------------------------

def cut_data_into_windows(X, y, window_length, step_size):
    X_new = []
    y_new = []
    step = int(step_size*window_length)

    for sample, label in zip(X,y):
        position = 0

        while((position+window_length) < X.shape[2]):
            sample_new = sample[:,position:(position+window_length)]
            position += step
            X_new.append(sample_new)
            y_new.append(label)

    return np.array(X_new), y_new
    
# ----------------------------------------------------------------------------------

def create_psd_band_features(x, ch_names, include_features=None, mask=None, bands=None, band_names=None,
                             exclude_channels=None):
    """
    Creates power spectral density features for frequency bands.

    Args:

    """

    if exclude_channels is None:
        exclude_channels = []

    if bands is None:
        # bounds of the frequency bands
        bands = [(1, 4), (4, 8), (8, 14), (14, 30), (30, 45), (45, 60)]
        band_names = ["delta", "theta", "alpha", "beta", "lower gamma", "upper gamma"]

    if mask is None:
        mask = [
            1,  # mean
            1,  # max
            1,  # min
            1,  # std
            1,  # ptp
            1,  # median
        ]

    # create empty nested lists, so it's easier to add the new samples
    x_all = []
    for _ in x:
        sample = []
        for i in range(0, len(ch_names)-len(exclude_channels)):
            sample.append([])
        x_all.append(cp.copy(sample))

    x_feature_names = []
    for _ in x:
        sample = []
        for i in range(0, len(ch_names)-len(exclude_channels)):
            sample.append([])
        x_feature_names.append(cp.copy(sample))

    # go over each frequency band, calculate statistical values and add them to the new samples
    for band, name in zip(bands, band_names):
        x_band = mne.time_frequency.psd_array_multitaper(x, 120, fmin=band[0], fmax=band[1])

        for i, sample in enumerate(x_band[0]):
            idx = 0

            for j, ch in enumerate(sample):
                ch_name = ch_names[j]

                if ch_name in exclude_channels:
                    continue

                mean = np.mean(ch)
                max_ = np.max(ch)
                min_ = np.min(ch)
                std_ = np.std(ch)
                ptp_ = np.ptp(ch)
                median_ = np.median(ch)

                if mask[0]:
                    x_all[i][idx].append(mean)
                if mask[1]:
                    x_all[i][idx].append(max_)
                if mask[2]:
                    x_all[i][idx].append(min_)
                if mask[3]:
                    x_all[i][idx].append(std_)
                if mask[4]:
                    x_all[i][idx].append(ptp_)
                if mask[5]:
                    x_all[i][idx].append(median_)

                if mask[0]:
                    x_feature_names[i][idx].append(ch_name + "_" + name + "_mean")
                if mask[1]:
                    x_feature_names[i][idx].append(ch_name + "_" + name + "_max")
                if mask[2]:
                    x_feature_names[i][idx].append(ch_name + "_" + name + "_min")
                if mask[3]:
                    x_feature_names[i][idx].append(ch_name + "_" + name + "_std")
                if mask[4]:
                    x_feature_names[i][idx].append(ch_name + "_" + name + "_ptp")
                if mask[5]:
                    x_feature_names[i][idx].append(ch_name + "_" + name + "_median")

                idx += 1

    x_all_ = np.array(x_all)
    x_feature_names_ = np.array(x_feature_names)

    x_flattened = []
    for sample in x_all_:
        x_flattened.append(cp.copy(sample.flatten()))
    x_flattened = np.array(x_flattened)

    x_flattened_fn = []
    for sample in x_feature_names_:
        x_flattened_fn.append(cp.copy(sample.flatten()))
    x_flattened_fn = np.array(x_flattened_fn)

    feature_labels = x_flattened_fn[0]

    if include_features:
        features_indices_to_select = [i for i, val in enumerate(feature_labels) if val in set(include_features)]
    else:
        features_indices_to_select = None

    if features_indices_to_select:
        x_selected = x_flattened[:, features_indices_to_select]
        feature_labels = (np.array(feature_labels)[features_indices_to_select]).tolist()
    else:
        x_selected = x_flattened

    feature_labels = np.array(feature_labels.tolist())

    return x_selected, feature_labels

# ----------------------------------------------------------------------------------
