import copy as cp
import pyxdf
import numpy as np
import sklearn
import mne
import pandas as pd
import os
import joblib
import utils
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def load_raw_data(data_path, read_from_xdf):
   
    raw_path = os.path.join(data_path, "raw_list")

    file_list = [
        # ("proband_003_haupt1.xdf", 3), #  -> Fehler: Kein Marker Stream
        ("proband_003_haupt2.xdf", 3),
        ("proband_004_haupt1.xdf", 4),
        ("proband_004_haupt2.xdf", 4),
        ("proband_005_haupt1.xdf", 5),
        ("proband_005_haupt2.xdf", 5),
        ("proband_006_haupt1.xdf", 6),
        ("proband_006_haupt2.xdf", 6),
        ("proband_007_haupt1.xdf", 7),
        ("proband_007_haupt2.xdf", 7),
        ("proband_008_haupt1.xdf", 8),
        ("proband_008_haupt2.xdf", 8),
        ("proband_009_haupt1.xdf", 9),
        ("proband_009_haupt2.xdf", 9),
        ("proband_010_haupt1.xdf", 10),
        ("proband_010_haupt2.xdf", 10),
        ("proband_011_haupt1.xdf", 11),
        ("proband_011_haupt2.xdf", 11),
        ("sub-daniel-haupt_ses-S001_task-Default_run-001_eeg.xdf", 12),
        ("sub-daniel-haupt-2_ses-S001_task-Default_run-001_eeg.xdf", 12), 
        ("Haupt1_sub-P001_ses-S001_task-Default_run-001_eeg.xdf", 13),
        ("Haupt2_sub-P001_ses-S001_task-Default_run-001_eeg.xdf", 13),
    ]

    print("load xdf into raw_list")
    raw_list = _load_xdf_into_raw(file_list, data_path)

    return raw_list

# -----------------------------------------------------------------------------------------------

def _load_xdf_into_raw(file_list, data_path):
    """
    Given a list of filenames, this method loads the data of the files into Raw-Objects
    and returns them

    """

    raw_list = []

    for filename, idx in file_list:
        path_xdf = os.path.join(data_path, filename)
        
        streams, header = pyxdf.load_xdf(path_xdf)
        data_matrix, data_timestamps, channel_labels, streamToPosMapping = _load_stream_data(streams)

        stream_info = streams[0]['info']
        fs = float(stream_info['nominal_srate'][0])
        fs = 500
        info = mne.create_info(channel_labels, fs, ch_types='eeg')                  

        data_reshaped = data_matrix.transpose()
        data_reshaped = data_reshaped[:32,:]

        # get the markers and the ground truths
        marker_timestamps, ground_truths = _get_marker_and_labels_from_stream(streams, False, streamToPosMapping)

        # create the stim channel
        stim_data = _create_stim_channel(data_timestamps, marker_timestamps, ground_truths)

        raw = mne.io.RawArray(data=data_reshaped, info=info)

        # add the stim channel
        _add_stim_to_raw(raw, stim_data)

        montage =  mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage)
        
        raw._filenames = [filename]
        raw_list.append((raw, idx))
        
    return raw_list

# -----------------------------------------------------------------------------------------------

def _load_stream_data(streams, stream_name='ActiChamp-0'):
    """
    Loads the recorded data, timestamps and channel information of a specific
    stream in a xdf file.

    Args:
        streams :
            Streams which should be used, has to be loaded before with _load_stream()
        stream_name (str):
            Name of the stream in the xdf file to be loaded

    Returns:
        data_matrix, data_timestamps, channel_labels

    """
    
    # find all stream names in the file
    streamToPosMapping = {}
    for pos in range(0, len(streams)):
        stream = streams[pos]['info']['name']
        streamToPosMapping[stream[0]] = pos
    
    # raise an error if the searched stream_name is not existing
    if stream_name not in streamToPosMapping.keys():
        raise ValueError(
            'Stream ' + str(stream_name) + ' not found in xdf file. '
                                           'Found streams are: '
                                           '\n' + str(
                streamToPosMapping.keys()))
        
    # Read the channel labels of the stream
    channel_labels = []
    try:
        for channel in streams[streamToPosMapping[stream_name]]['info']['desc'][0]['channels'][0]['channel']:
            channel_labels.append(channel['label'][0])
    except TypeError:
        # no channel information could be found!
        pass

    # Read the data and timestamps
    data_matrix = streams[streamToPosMapping[stream_name]]['time_series']
    data_timestamps = streams[streamToPosMapping[stream_name]]['time_stamps']
    
    return data_matrix, data_timestamps, channel_labels, streamToPosMapping

# ---------------------------------------------------------------------------------------

def _get_marker_and_labels_from_stream(streams, exclude_trainings_trials, streamToPosMapping=None):
    """
    Gets the markers and ground truths from the stream

    Args:
        streams: The streamfs from the xdf-files
        exclude_trainings_trials: True to exclude the training trials
    """

    marker_timestamps = []
    marker_names = []
    ground_truths = []
    durations = []
    start = None
    end = None
    go = False
        
    
    idx = streamToPosMapping['ssvepMarkerStream']
    stream = streams[idx]

    # list of strings, draw one vertical line for each marker
    for timestamp, marker in zip(stream['time_stamps'], stream['time_series']):

        if marker[0] == 'phase_start' and marker[1] == 'Phase: Run Classification':
            go = True

        if go or (not exclude_trainings_trials):
            if marker[0] == "ground_truth":
                ground_truths.append(marker[1])    

            elif marker[0] == "classification_start":      
                if start is None:
                    start = timestamp
                # marker_timestamps.append(timestamp)
                marker_names.append(marker[0])
            
            elif marker[0] == "classification_end":
                if end is None:
                    end = timestamp
                
                marker_timestamps.append((start, end))
                durations.append(end-start)
                start = None
                end = None
    
    return marker_timestamps, ground_truths

# -----------------------------------------------------------------------------------------------

def _create_stim_channel(data_timestamps, marker_timestamps, ground_truths):

    mts = [m[0] for m in marker_timestamps]
    stim = []
    
    idx = 0
    for t in data_timestamps:   
                
        if len(mts) > 0 and t >= mts[0]:
            ground_truth = ground_truths[idx] 
            stim.append(int(ground_truth)+1)
            mts.pop(0)
            idx += 1
        else:
            stim.append(0)

    return np.array([stim])

# -----------------------------------------------------------------------------------------------

def _add_stim_to_raw(raw, stim_data):
    """
    Adds the stim channel to the Raw-Object
    """
    
    # remove it if theres already a stim channel present (to rerun sections in notebook)
    if "STI" in raw.info["ch_names"]: 
        raw = raw.drop_channels("STI")
        
    info = mne.create_info(['STI'], raw.info['sfreq'], ['stim'])
    stim_raw = mne.io.RawArray(stim_data, info)
    raw.add_channels([stim_raw], force_update_info=True) 

# ------------------------------------------------------------------------------------------

def _create_epochs_objects(raw_list, sample_length=6, offset=3):
    """
    Create Epochs-objects from a list of Raw objects (or RawArray objects)
    """

    # has to be slightly negative for some reason
    tmin = offset
    # duration of trial in seconds
    tmax = tmin + sample_length
    epochs_list = []

    # stores the names of the files from Raw that failed to be converted
    files_failed = []

    # go through each Raw and create Epochs, also extracts and formats the labels
    for raw, subj_nr in raw_list:      
        # print(raw._filenames[0], subj_nr)

        # get the events and try to create Epoch
        events = mne.find_events(raw, stim_channel="STI", initial_event="True", output="onset")
        epochs = mne.Epochs(raw, events, tmin=offset, tmax=tmax, baseline=None)     
        
        epochs.load_data()
        epochs = epochs.drop_channels("STI")

        # extract the labels from the events
        y = (epochs.events[:, 2] - 2).astype(np.int64)

        ch_names = epochs.info['ch_names']

        # adjust the labels
        for idx, i in enumerate(y):
            if i == 0: y[idx] = 0
            else: y[idx] = 1

        epochs_list.append((epochs, y, subj_nr))

    return epochs_list

# -----------------------------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------------------------

def create_folds_GME(epochs_list, window_len_sec=None, step_size=1, ch_names=None, scale=False, independent=True, n_splits=10):
                    
    X_stacked_list = []
    y_stacked = []
    
    indices_per_subject = []
    
    start1 = 0
    end1 = 0
    
    sum_ = 0
    sampling_rate = 500
        
    for epochs, y, subj_idx in epochs_list:
        
        X = epochs.get_data()
        X_windowed = []
    
        if window_len_sec is not None:
            window_length = int(window_len_sec * sampling_rate)
            X_windowed, y_windowed = cut_data_into_windows(X, y, window_length, step_size)
        else:
            X_windowed = X
            y_windowed = y
        
        # <--------------------------------------------------------------------------------------------------------------------------------------------------
        X_windowed, y_windowed = sklearn.utils.shuffle(X_windowed, y_windowed)
        
        X_new, scaler, feature_names = \
            create_psd_band_features(X_windowed, y_windowed, mask=None, include_features=None, ch_names=ch_names, exclude_channels=None, scale=scale)
    
        X_stacked_list.append(X_new)
        y_stacked += y_windowed
    
        start1 = end1
        end1 = start1 + len(y_windowed)
    
        subj_indices = list(range(start1, end1))
        indices_per_subject.append((subj_indices, subj_idx))
        
    X_stacked = np.concatenate(X_stacked_list)
    
    if independent:

        subjects = []
        for _, subj_idx in indices_per_subject:
            if subj_idx not in subjects:
                subjects.append(subj_idx)

        folds = []

        for test_subj in subjects:
            train_indices = []
            test_indices = []

            test_subjects = []
            train_subjects = []

            for indices, subj in indices_per_subject: 

                if subj == test_subj:
                    test_indices += indices
                    test_subjects.append(test_subj)
                else:
                    train_indices += indices
                    train_subjects.append(subj)

            folds.append((train_indices, test_indices, test_subj))
    else:
        X_stacked, y_stacked = sklearn.utils.shuffle(X_stacked, y_stacked)

        folds = []
        
        skf = sklearn.model_selection.StratifiedKFold(n_splits=n_splits)
        
        for train_indices, test_indices in skf.split(X_stacked, y_stacked):
            folds.append((train_indices, test_indices, "mixed subjects"))
    
    return X_stacked, y_stacked, folds, feature_names

# ------------------------------------------------------------------------------------------

def create_folds_IEA(ba_data_path=None, window_len_sec=4, independent=True, n_splits=10, use_ica=False):

    # ---- load files and select channels ----
    
    if use_ica:
        warnings.warn("ICA not implemented for the IEA data set")
        
    if ba_data_path is None:
        ba_data_path = r"C:\Users\hendr\Documents\GitProjects\EEG Data\iea_experiment"

    # load epochs
    subjects=["01", "16", "22", "34", "35", "36", "51", "63", "72", "77", "79", "87", "94"]

    objs = []
    verbose = 0
    file = "epo"
    
    # ch_rename_dict = {'CZ':'Cz', 'FP2':'Fp2', 'OZ':'Oz'}
    
    for nr in subjects:
        subj_path = os.path.join(ba_data_path, f"iea_{nr}")
        path = os.path.join(subj_path, f"iea_{nr}-{file}.fif")
        epo = mne.read_epochs(path, verbose=verbose)
        # epo.rename_channels(ch_rename_dict)
        objs.append((epo, f"{nr}_{file}"))

    chns_count = dict([('Cz',0), ('Fp2',0), ('F3',0), ('F4',0), ('FT7',0), ('C3',0), ('C4',0), ('Fp1', 0), 
                       ('FT8',0), ('P3',0), ('PZ',0), ('P4',0), ('PO7',0), ('PO8',0), ('Oz',0), ('FZ', 0)])

    for epo, name in objs:
        if "P4-0" in epo.info['ch_names']: mne.rename_channels(epo.info, {'P4-0': 'P4'})
        if "CZ" in epo.info['ch_names']: mne.rename_channels(epo.info, {'CZ': 'Cz'})
        if "FP2" in epo.info['ch_names']: mne.rename_channels(epo.info, {'FP2': 'Fp2'})
        if "FP1" in epo.info['ch_names']: mne.rename_channels(epo.info, {'FP1': 'Fp1'})
        if "OZ" in epo.info['ch_names']: mne.rename_channels(epo.info, {'OZ': 'Oz'})
        

        for ch in epo.info['ch_names']:
            try: chns_count[ch] += 1
            except: print(ch, "not in dict")

    select = []
    for i, ch in enumerate(chns_count):
        if chns_count[ch]/len(objs) == 1:
            # print(ch, chns_count[ch], chns_count[ch]/len(objs))
            select.append(ch)    

    # ---- preprocess epochs ----
        
    chns = select
    start = 0
    end = 0

    X_stacked_list = []
    y_stacked = []
    indices_per_subject = []

    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print(select)

    
    for epo, name in objs:
        # is already filtered
        # epo.filter(l_freq=1, h_freq=30)
          
        print(epo.info['ch_names'])
        
        epo = epo.pick_channels(select)
        epo = epo.apply_baseline((1, 1.5))

        # epo.plot()
        # print(sorted(epo.info['ch_names']))

        print(epo.info)
        
        
        
        X = epo.get_data() 
        y = (epo.events[:, 2] - 2).astype(np.int64)

        # translate labels
        for idx, i in enumerate(y):
            if i == 5: y[idx] = 1
            else: y[idx] = 0     

        # cut beginning and end
        X_offset = []
        for i, x in enumerate(X):
            X_offset.append(x[:, 251:-250])
        X_offset = np.array(X_offset)

        # cut data into windows
        X_windowed = []
        y_windowed = []
        fs = 500

        for i, x in enumerate(X_offset):
            step = fs*window_len_sec

            for idx in range(0, X_offset.shape[2], step):
                X_windowed.append(x[:, idx:idx+step])
                y_windowed.append(y[i])

        X_windowed = np.array(X_windowed)

        # shuffle the subject data
        X_windowed, y_windowed = sklearn.utils.shuffle(X_windowed, y_windowed)

        # range of indices for subject
        start = end
        end = end + len(y_windowed)

        # create psd features
        X_new, scaler, feature_names = \
            create_psd_band_features(X_windowed, y_windowed, mask=None, include_features=None, ch_names=select, exclude_channels=None, scale=False)

        X_stacked_list.append(X_new)
        y_stacked += y_windowed

        subj_indices = list(range(start, end))
        indices_per_subject.append((subj_indices, name))


    X_stacked = np.concatenate(X_stacked_list)

    print(X_stacked.shape)
    print(len(y_stacked))    

    # ---- create folds ----
    
    indepenent = True

    if independent:
        subjects = []
        for _, subj_idx in indices_per_subject:
            if subj_idx not in subjects:
                subjects.append(subj_idx)

        folds = []

        print(subjects)

        for test_subj in subjects:
            train_indices = []
            test_indices = []

            test_subjects = []
            train_subjects = []

            for indices, subj in indices_per_subject: 

                if subj == test_subj:
                    test_indices += indices
                    test_subjects.append(test_subj)
                else:
                    train_indices += indices
                    train_subjects.append(subj)

            folds.append((train_indices, test_indices, test_subj))
    else:
        X_stacked, y_stacked = sklearn.utils.shuffle(X_stacked, y_stacked)

        folds = []

        skf = sklearn.model_selection.StratifiedKFold(n_splits=n_splits)

        for train_indices, test_indices in skf.split(X_stacked, y_stacked):
            folds.append((train_indices, test_indices, "mixed subjects"))
            
    return X_stacked, y_stacked, folds, feature_names

# ------------------------------------------------------------------------------------------

def create_epochs(raw_list, use_average_ref=True, sample_len=6, offset=3, include_channels=None, exclude_channels=None, downsample=False, use_ica=False):

    for i, (raw, idx) in enumerate(raw_list):
        
        if include_channels is not None:
            include_channels.append("STI")
            raw = raw.pick_channels(include_channels)
        elif exclude_channels is not None:
            raw.drop_channels(exclude_channels)

        if use_average_ref:
            if include_channels is None or len(include_channels) > 1:
                print(">> using avg rereferencing")
                raw = raw.copy().set_eeg_reference(ref_channels='average')
            else:
                print("Referencing was skipped because only 1 channel is included")
        
        #else:
        #    # Re-Referencing to the reference electrode of choice
        #    if 'Cz' in raw.info['ch_names']:
        #        raw.set_eeg_reference(ref_channels=['Cz'])
        #    # remove Cz because its used as reference
        #    if 'Cz' in raw.info['ch_names']:
        #        raw.drop_channels(['Cz'])
  
        raw_list[i] = (raw, idx)

    epochs_list = _create_epochs_objects(raw_list, sample_len, offset)
    return epochs_list

# ------------------------------------------------------------------------------------------

def create_psd_band_features(X, y, ch_names, include_features=None, scaler=None, mask=None, bands=None, band_names=None, exclude_channels=[], scale=False):
    """
    Creates power spectral density features for frequency bands.

    Args:
        X,y: data and labels
        ch_names: Names of the channels
        features_indices_to_select: list of indices of features that are to be selected
        scaler: StandardScaler
        mask: list of ones and zeros to change which statistical features are calculated
        bands: the frequency bands as a list of tuples
        band_names: the names of the frequency bands (e.g. alpha, beta, ..)
    """   

    if exclude_channels is None:
        exclude_channels = []
    
    #if scale:
    #    X, _ = standardize_data(X_subj)
    #else:
    #    X = X_subj
    
    if bands is None:
        # bounds of the frequency bands
        
        # print(">>> FREQUENCY BAND GAMMA HAS BEEN EXCLUDED FOR DATASET B")
        # bands = [(1,4), (4,8), (8,14), (14,30)] # , (30,45)]
        # band_names = ["delta", "theta", "alpha", "beta"] # , "gamma"]
        
        bands = [(1,4), (4,8), (8,14), (14,30), (30,45)]
        band_names = ["delta", "theta", "alpha", "beta", "gamma"]
        
    if mask is None:
        mask = [
            1, # mean
            1, # max
            1, # min
            1, # std
            1, # ptp
            1, # median
        ]

    # create empty nested lists so its easier to add the new samples
    feature_labels = []
    X_all = []
    for s in X:
        sample = []
        for i in range(0, len(ch_names)-len(exclude_channels)):
            sample.append([])
        X_all.append(cp.copy(sample))
    
    X_feature_names = []
    for s in X:
        sample = []
        for i in range(0, len(ch_names)-len(exclude_channels)):
            sample.append([])
        X_feature_names.append(cp.copy(sample))
    
    # go over each frequency band, calculate statistical values and add them to the new samples
    for band, name in zip(bands, band_names):
        X_band = mne.time_frequency.psd_array_multitaper(X, 500, fmin=band[0], fmax=band[1])
        # laut paper: welch
        #X_band = mne.time_frequency.psd_array_welch(X, 500, fmin=band[0], fmax=band[1])
        
        for i, sample in enumerate(X_band[0]):
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

                if mask[0]: X_all[i][idx].append(mean)
                if mask[1]: X_all[i][idx].append(max_)
                if mask[2]: X_all[i][idx].append(min_)
                if mask[3]: X_all[i][idx].append(std_)
                if mask[4]: X_all[i][idx].append(ptp_)
                if mask[5]: X_all[i][idx].append(median_)
           
                if mask[0]: X_feature_names[i][idx].append(ch_name + "_" + name + "_mean")
                if mask[1]: X_feature_names[i][idx].append(ch_name + "_" + name + "_max")
                if mask[2]: X_feature_names[i][idx].append(ch_name + "_" + name + "_min")
                if mask[3]: X_feature_names[i][idx].append(ch_name + "_" + name + "_std")
                if mask[4]: X_feature_names[i][idx].append(ch_name + "_" + name + "_ptp")
                if mask[5]: X_feature_names[i][idx].append(ch_name + "_" + name + "_median")

                idx += 1
    
    X_all_ = np.array(X_all)
    X_feature_names_ = np.array(X_feature_names)
        
    X_flattened = []
    for sample in X_all_:
        X_flattened.append(cp.copy(sample.flatten()))
    X_flattened = np.array(X_flattened)

    X_flattened_fn = []
    for sample in X_feature_names_:
        X_flattened_fn.append(cp.copy(sample.flatten()))
    X_flattened_fn = np.array(X_flattened_fn)
    
    feature_labels = X_flattened_fn[0]
        
    if include_features:
        features_indices_to_select = [i for i, val in enumerate(feature_labels) if val in set(include_features)]
    else:
        features_indices_to_select = None
    
    if features_indices_to_select:
        X_selected = X_flattened[:, features_indices_to_select]
        feature_labels = (np.array(feature_labels)[features_indices_to_select]).tolist()
    else:
        X_selected = X_flattened
        
    return X_selected, scaler, feature_labels

# ----------------------------------------------------------------------------------

def create_time_freq_features(X, X_psd):
    """
    Creates a combination of time series features and spectral features
    Applies PCA to both of them and stacks them together
    
    Args:
        X: The (untouched) data
        X_psd: PSD data from X, like from create_psd_band_features(..)
    """

    # PCA für den Zeitbereich
    X_t = []
    for sample in X:
        ch_features = []
        for channel in sample:
            ch_features += channel.flatten().tolist()
        X_t.append(ch_features)
    X_t = pd.DataFrame(X_t)

    # standardisieren, da Distanzmaß verwendet wird
    X_t = sklearn.preprocessing.scale(X_t)

    pca = PCA(n_components=22)
    pca.fit(X_t)
    X_t_pca = pca.transform(X_t)

    cum_exp_var = []
    var_exp = 0
    for i in pca.explained_variance_ratio_:
        var_exp += i
        cum_exp_var.append(var_exp)

    # PCA für Frequenzbereich
    pca = PCA(n_components=10)
    pca.fit(X_psd)
    X_f_pca = pca.transform(X_psd)

    cum_exp_var = []
    var_exp = 0
    for i in pca.explained_variance_ratio_:
        var_exp += i
        cum_exp_var.append(var_exp)

    # stack time and freq domain 
    X_pca = np.hstack((X_t_pca, X_f_pca))
    
    return X_pca

# ----------------------------------------------------------------------------------

def exclude_channels_from_X(X, feature_names, ch_names, exclude_channels):
    parts = [fn.split("_")[0] for fn in feature_names]
    mask = [(False if p in exclude_channels else True) for p in parts] 
    indices = [i for i, b in enumerate(mask) if b]
    
    X_new = X[:, indices]
    feature_names_new = feature_names[indices]

    return X_new, feature_names_new
    
# ----------------------------------------------------------------------------------

def load_and_preprocess_data_from_param(params, ba_data_path):

    for p in params:
        data_type = p['data_type']
        independent = p['independent']
        ref_type = p['ref_type']
        window_len = p['window_length']
        step_size = p['step_size']
        agg_type = p['agg']
        n_splits = p['n_splits']
        whole_data_path = p['paths']['whole']
        raw_list_path = p['paths']['raw']
        load_data = p['load']
        use_ica = p['use_ica']

        sample_len = 6
        offset = 3
        utils.print_param(p)

        if ref_type == "avgRefBefore":
            use_ref = True
        else:
            use_ref = False

        if data_type == 'iea':
            if not load_data:
                X, y, folds, fns = create_folds_IEA(ba_data_path, window_len_sec=window_len,
                                                    independent=independent, n_splits=n_splits, use_ica=use_ica)
                joblib.dump((X, y, folds, fns), whole_data_path)
            else:
                X, y, folds, fns = joblib.load(whole_data_path)
            p['data'] = (X, y, folds, fns)

        elif data_type == 'gme':

            if not load_data:
                print(">> using this data:", raw_list_path)
                X, y, folds, fns = prepare_data(raw_list_path, use_ref, sample_len, offset, window_len, step_size,
                                                independent=independent, n_splits=n_splits, use_ica=use_ica)
                joblib.dump((X, y, folds, fns), whole_data_path)
            else:
                X, y, folds, fns = joblib.load(whole_data_path)
            p['data'] = (X, y, folds, fns)

# ------------------------------------------------------------------------------------------

def prepare_data(raw_list_path, use_ref, sample_len, window_len, offset, step_size, include_channels=None, 
                 exclude_channels=None, downsample=False, independent=True, n_splits=10, use_ica=False):
     
    raw_list = joblib.load(raw_list_path)
    
    epochs_list = create_epochs(raw_list, use_ref, sample_len, offset, include_channels=include_channels, 
                                exclude_channels=exclude_channels, downsample=downsample, use_ica=use_ica)
    
    epochs_, _, _ = epochs_list[0]
    ch_names = epochs_.info['ch_names']

    X, y, folds, feature_names = create_folds_GME(
        epochs_list, window_len_sec=window_len, step_size=step_size, ch_names=ch_names, scale=False, independent=independent, 
        n_splits=n_splits)
    
    return X, y, folds, feature_names

# ------------------------------------------------------------------------------------------