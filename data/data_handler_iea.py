import numpy as np
import mne
import os
import joblib
import data.utils as utils
from tqdm import tqdm
from data.data_handler import DataHandler


class DataHandlerIEA(DataHandler):
         
    def __init__(self, models, window_length, step_size, use_ica, base_result_path, data_path, test=False,
                 avg_ref=True, resample=False, sampling_rate=None, ds_name="", scale=False, all_chns=False):

        super(DataHandlerIEA, self).__init__(use_ica, base_result_path, sampling_rate, test, resample, avg_ref, ds_name,
                                             step_size, window_length)

        self.channel_names = ['Cz', 'Fp2', 'F3', 'FT7', 'C3', 'C4', 'FT8', 'P3', 'P4', 'PO7', 'PO8', 'Oz']
        self.num_channels_per_iteration = [1, 2, 4, 8, 12]

        self.data_path = data_path

        self.events_list = None
        self.num_classes = 2
        self.num_data_points = None
        self.models = models
        self.scale = scale
        self.all_chns = all_chns
        if self.all_chns: self.num_channels_per_iteration = [i+1 for i in range(0, 12)]

    # ---------------------------------------------------------------------------------------

    def get_num_data_points(self):
        if self.num_data_points is None:
            folds_list = self.get_folds(filename_suffix="_raw")
            self.num_data_points = folds_list[0].shape[-1]
            del folds_list
        return self.num_data_points

    # ---------------------------------------------------------------------------------------

    def get_num_classes(self):
        return self.num_classes

    # ---------------------------------------------------------------------------------------

    def get_channel_names(self):
        return self.channel_names

    # ---------------------------------------------------------------------------------------

    def get_num_channels(self):
        return len(self.get_channel_names())

    # ---------------------------------------------------------------------------------------

    def load_raws(self, verbose=0):
        
        filenames = ["01", "16", "22", "34", "35", "36", "51", "63", "72", "77", "79", "87", "94"]

        if self.test:
            filenames = filenames[:2]
        
        for idx, fn in enumerate(filenames):

            filenames[idx] = \
                (os.path.join(self.data_path, f"iea_{fn}", f"iea_{fn}-raw.fif"),
                 os.path.join(self.data_path, f"iea_{fn}", f"iea_{fn}-eve.fif"))

        raw_list = []
        events_list = []
        
        for idx, (fn_raw, fn_eve) in enumerate(filenames):
            raw = mne.io.read_raw_fif(fn_raw, preload=True, verbose=verbose)
            events = mne.read_events(fn_eve)
            raw_list.append((raw, idx))
            events_list.append(events)

        self.events_list = events_list
        return raw_list

    # ---------------------------------------------------------------------------------------    

    def preprocess_raws(self, raw_list, verbose=0, plot=False):

        events_resampled = []

        for (raw, idx), events in tqdm(zip(raw_list, self.events_list)):
            if verbose: 
                print("Before preprocessing\n")
                print(raw.info)
                print(raw.info.ch_names)
            
            if plot and idx == 0:
                raw.plot()
            
            # select channels
            if "P4-0" in raw.info['ch_names']:
                mne.rename_channels(raw.info, {'P4-0': 'P4'})
            if "CZ" in raw.info['ch_names']:
                mne.rename_channels(raw.info, {'CZ': 'Cz'})
            if "FP2" in raw.info['ch_names']:
                mne.rename_channels(raw.info, {'FP2': 'Fp2'})
            if "FP1" in raw.info['ch_names']:
                mne.rename_channels(raw.info, {'FP1': 'Fp1'})
            if "OZ" in raw.info['ch_names']:
                mne.rename_channels(raw.info, {'OZ': 'Oz'})

            raw.pick_channels(self.channel_names)
                        
            # notch filter
            raw.notch_filter(np.arange(50, 201, 50), fir_design='firwin')

            raw.filter(1, 60)

            if self.resample:
                if verbose:
                    print("resampling to ", self.sampling_rate)
                raw, events = raw.resample(self.sampling_rate, events=events)
                events_resampled.append(events)
            else:
                events_resampled.append(events)

            # average ref
            if self.avg_ref:
                raw = raw.set_eeg_reference(ref_channels="average")

            if self.scale:
                raw = raw.apply_function(utils.standardize_data, channel_wise=False)

            if verbose:
                print("After preprocessing\n")
                print(raw.info)
                print(".......................")
            
            if plot and idx == 0:
                raw.plot()

        self.events_list = events_resampled

        return raw_list

    # ---------------------------------------------------------------------------------------    
    
    def create_epochs(self, raw_list, verbose=0, plot=False):
        epochs_list = []
        event_id = {'internal': 7, 'external': 5}

        # Define the time window to extract for each epoch
        tmin, tmax = 1.0, 13.0
        
        for (raw, idx), events in tqdm(zip(raw_list, self.events_list)):

            epochs = mne.Epochs(raw, events=events, event_id=event_id,
                                tmin=tmin, tmax=tmax, 
                                baseline=None, preload=True)
            
            y = (epochs.events[:, 2]).astype(np.int64)

            # 5 to 1, 7 to 0
            y = [0 if val == 7 else 1 for val in y]
            
            if verbose: 
                print(epochs.info)
                print(epochs)
                print("These events were dropped:")
                print(epochs.drop_log)
            if plot:
                epochs.plot()
                
            epochs_list.append((epochs, y, idx))
                        
        epochs_list = epochs_list
        joblib.dump(epochs_list, self.epochs_path)

        return epochs_list

    # ------------------------------------------------------------------------------------------


    def create_folds(self, epochs_list, file_name_suffix=""):
        X_stacked_list = []
        y_stacked = []
        feature_names = None
        indices_per_subject = []

        end = 0
        subjects = []

        for epochs, y, subj_idx in epochs_list:

            subjects.append(subj_idx)
            X = epochs.get_data()

            ch_names = epochs.info['ch_names']

            if self.window_length is not None:
                window_length = int(self.window_length * self.sampling_rate)
                X, y = utils.cut_data_into_windows(X, y, window_length, self.step_size)

            X, feature_names = utils.create_psd_band_features(X, mask=None, include_features=None,
                                                                 ch_names=ch_names, exclude_channels=None)

            X_stacked_list.append(X)
            y_stacked += list(y)

            start = end
            end = start + len(y)

            subj_indices = list(range(start, end))
            indices_per_subject.append((subj_indices, subj_idx))

        X_stacked = np.concatenate(X_stacked_list)

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

        folds_list = (X_stacked, y_stacked, folds, feature_names)
        # joblib.dump(self.folds_list, self.folds_path + file_name_suffix)

        filename = f"folds_list{file_name_suffix}"
        path = os.path.join(self.result_path, filename)
        joblib.dump(folds_list, path)

        return X_stacked, y_stacked, folds, feature_names

    # ---------------------------------------------------------------------------------------    
    
    def _load_files_into_epochs(self, iea_data_path):
        
        # load epochs
        subjects = ["01", "16", "22", "34", "35", "36", "51", "63", "72", "77", "79", "87", "94"]

        if self.test: 
            subjects = subjects[:3]
            
        self.epochs = []
        verbose = 0
        file = "epo"

        self.file_names = []
        
        for nr in subjects:
            subj_path = os.path.join(iea_data_path, f"iea_{nr}")
            path = os.path.join(subj_path, f"iea_{nr}-{file}.fif")
            epo = mne.read_epochs(path, verbose=verbose)
            self.file_names.append(f"{nr}_{file}")
            self.epochs.append(epo)

        chns_count = dict([('Cz', 0), ('Fp2', 0), ('F3', 0), ('F4', 0), ('FT7', 0), ('C3', 0), ('C4', 0), ('Fp1', 0),
                           ('FT8', 0), ('P3', 0), ('PZ', 0), ('P4', 0), ('PO7', 0), ('PO8', 0), ('Oz', 0), ('FZ', 0)])

        for epo in self.epochs:
            if "P4-0" in epo.info['ch_names']:
                mne.rename_channels(epo.info, {'P4-0': 'P4'})
            if "CZ" in epo.info['ch_names']:
                mne.rename_channels(epo.info, {'CZ': 'Cz'})
            if "FP2" in epo.info['ch_names']:
                mne.rename_channels(epo.info, {'FP2': 'Fp2'})
            if "FP1" in epo.info['ch_names']:
                mne.rename_channels(epo.info, {'FP1': 'Fp1'})
            if "OZ" in epo.info['ch_names']:
                mne.rename_channels(epo.info, {'OZ': 'Oz'})

            for ch in epo.info['ch_names']:
                try:
                    chns_count[ch] += 1
                except:
                    print(ch, "not in dict")

        select = []
        for i, ch in enumerate(chns_count):
            if chns_count[ch]/len(self.epochs) == 1:
                select.append(ch)    
        
        for epo in self.epochs:
            epo = epo.pick_channels(select)
            epo.apply_baseline((1, 1.5))
