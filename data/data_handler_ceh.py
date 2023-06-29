import os
import mne
import joblib
import numpy as np
from tqdm import tqdm
import data.utils as utils
from data.data_handler import DataHandler


class DataHandlerCeh(DataHandler):
    
    def __init__(self, models, window_length, step_size, use_ica, base_result_path, sampling_rate, data_path,
                 test=False, resample=True, use_divergent_convergent_labels=True, avg_ref=True, ds_name="", scale=False,
                 all_chns=False):

        super(DataHandlerCeh, self).__init__(use_ica, base_result_path, sampling_rate, test, resample, avg_ref, ds_name,
                                             step_size, window_length)

        if use_divergent_convergent_labels:
            self.event_id = {'divergent_internal': 1, 'convergent_internal': 2,
                             'divergent_external': 3, 'convergent_external': 4}
        else:
            self.event_id = {'internal': 0, 'external': 1}

        self.num_channels_per_iteration = [1, 2, 4, 8, 16, 19]

        self.models = models
        self.num_data_points = None

        self.channel_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8',
                              'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2']

        self.data_path = data_path

        self.use_divergent_convergent_labels = use_divergent_convergent_labels
        self.num_classes = 4 if self.use_divergent_convergent_labels else 2
        self.filenames = None
        self.events_list = None
        self._prepare_paths()
        self.scale = scale
        self.all_chns = all_chns
        if self.all_chns: self.num_channels_per_iteration = [i+1 for i in range(0, 19)]

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
        if self.channel_names is None:
            epochs_list = self.get_epochs()
            self.channel_names = epochs_list[0][0].info['ch_names']
        return self.channel_names

    # ---------------------------------------------------------------------------------------

    def get_num_channels(self):
        return len(self.get_channel_names())

    # ---------------------------------------------------------------------------------------

    def load_raws(self, verbose=0):
        
        filenames = ["AH20W", "AM17F", "AM27F", "AW02S", "BW16S", "CA03K", "CD17W", "CW31L",
                     # "DG09J",
                     "EC01Z", "ED18J", "EJ21S", "EM20Z", "FG21Z", "GA17W", "GM18Z", 
                     "GW02S", "GW20W", "HM29J", "IH05Z", "KJ27K", "KM22W", "KW15W", "LK29K",
                     "MA03S", "MG10J", "MJ16J", "MJ18S", "MP22S", "RP06K", "RW05W", "SA08K",
                     "SB08S", "SU31W", "VG26L", "ZZ22S"]
                
        if self.test:
            filenames = filenames[:6]

        paths = []
        
        for idx, fn in enumerate(filenames):
            paths.append(os.path.join(self.data_path, fn + ".vhdr"))

        raw_list = []

        for idx, fn in enumerate(filenames):
            raw = mne.io.read_raw_brainvision(paths[idx], preload=True)
            raw_list.append((raw, idx))

        self.filenames = filenames
        return raw_list
    
    # ---------------------------------------------------------------------------------------
    
    def preprocess_raws(self, raw_list, verbose=0, plot=False):
        events_list = []

        for (raw, idx) in tqdm(raw_list):
            if verbose: 
                print(raw.info)

            # notch filter
            raw.notch_filter(np.arange(50, 201, 50), fir_design='firwin')

            # MNE Tut: Removing slow drifts makes for more stable regression coefficients. 
            # Make sure to apply the same filter to both EEG and EOG channels!
            raw.filter(1, 60)

            #if self.avg_ref:
            #    raw = raw.set_eeg_reference(ref_channels='average')
            events, _ = mne.events_from_annotations(raw, verbose=False)

            # downsample data and events
            if self.resample:
                if verbose:
                    print("resample to ", self.sampling_rate)
                raw, events = raw.resample(self.sampling_rate, events=events)

            events_list.append(events)
            
            # Set EOG channels as EOG in raw.info
            eog_chns = ['EOG1', 'EOG2', 'EOG3']
            eog_map = {ch: 'eog' for ch in eog_chns}
            raw.set_channel_types(eog_map)
            
            ref_chns = ["Ref2"]
            
            raw = raw.drop_channels(ref_chns)
            raw = raw.set_eeg_reference(ref_channels="average")

            if self.scale:
                raw = raw.apply_function(utils.standardize_data, channel_wise=False)

            raw.info.set_montage('easycap-M1')

            raw_list[idx] = (raw.drop_channels(eog_chns), idx)
            
            if verbose:
                print(raw.info)
            if plot:
                raw.plot()

        self.events_list = events_list

        return raw_list

    # ------------------------------------------------------------------------------------------
    
    def _filter_events(self, events_list, verbose=0, plot=False):
        """
        Stimuli

        S2 = Start of resting phase (120s duration)  
        S3 = end of resting phase  
        S4 = start of divergent thinking block  
        S5 = start of convergent thinking block  
        S6 = start fixation (each trial; 5s duration)  
        S14 = blank screen between fixation end and item presentation (500ms duration)  
        S7 = item presentation (500ms* or 20s duration)  
        S8* = stimulus masking in 50% of trials (does not occur in every trial; 19.5s duration)  
        S9 = start of response phase (6s duration)  
        S10 = end of response phase and start of inter-trial-interval (3s duration + length of drift check)  
        S17 = end of experiment  
        
        -- Example sequence: -- 
        
        15822 convergent        <-- start of block 

        18129 fixation          <-- start of trial
        18734 blank_screen
        18799 item_presentation <-- |
        21203 start_response    <-- | no simulus_masking, so external attention
        21938 end_response

        22498 fixation
        23102 blank_screen
        23168 item_presentation
        23232 stimulus_masking  <-- indicates internal attention
        25576 start_response
        26311 end_response

        """
        
        # extracts the events for the classes 
        
        # Define the event codes and their meanings
        event_id = {
            '2_resting_start': 2,     '3_resting_end': 3,      '4_divergent': 4,
            '5_convergent': 5,        '6_fixation': 6,         '14_blank_screen': 14,
            '7_item_presentation': 7, '8_stimulus_masking': 8, '9_start_response': 9,
            '10_end_response': 10,    '17_end_exp': 17,        '99999_misc1': 99999,
            '10001_misc2': 10001
        }
        
        rev_event_id = {
            2: 'resting_start',   3: 'resting_end',   4: 'divergent',         5: 'convergent',
            6: 'fixation',       14: 'blank_screen',  7: 'item_presentation', 8: 'stimulus_masking',
            9: 'start_response', 10: 'end_response', 17: 'end_exp',           99999: 'misc1',
            10001: '10001_misc2'
           }
        
        # event_inv = {v: k for k, v in self.event_id.items()}
        new_event_id = self.event_id

        filtered_events_list = []
        
        for i, events in enumerate(events_list):
            label = None
            new_events = []
            
            if plot: 
                print("_"*20)
                print(i, self.filenames[i])
                mne.viz.plot_events(events, verbose=True, event_id=self.event_id, on_missing='ignore')
            
            idx = 0
            
            while idx < len(events) and events[idx][2] != event_id['17_end_exp']:

                if self.use_divergent_convergent_labels:
                    if events[idx][2] == event_id['5_convergent']:
                        label = 'internal'
                    elif events[idx][2] == event_id['4_divergent']:
                        label = 'external'                
                    
                if events[idx][2] == event_id['6_fixation']:
                    idx += 1
                    
                    if events[idx][2] == event_id['14_blank_screen']:
                        idx += 1

                        if events[idx][2] == event_id['7_item_presentation']:

                            start_time = events[idx][0]
                    
                            if verbose and i == 0:
                                print("start", start_time/1000.0)

                            if events[idx+1][2] == event_id['8_stimulus_masking']:
                                idx += 1
                                
                                if not self.use_divergent_convergent_labels:
                                    label = "internal"  
                            else:
                                if not self.use_divergent_convergent_labels:
                                    label = "external"  
                            
                            if events[idx+1][2] == event_id['9_start_response']:
                                end_time = events[idx+1][0]
                                
                                duration = (end_time-start_time)/1000.0

                                if verbose and i == 0:
                                    print("end", end_time/1000.0, "\nduration: ", duration, "\n")
                                
                                idx += 1

                            new_events.append([start_time, 0, new_event_id[label]])
                                
                        else:
                            print(f"ERROR: Expected 7 (item_presentation) but got {events[idx][2]}")
                    else:
                        print(f"ERROR: Expected 14 (blank_screen) but got {events[idx][2]}")
                else: 
                    if verbose:
                        print(f"skipped: {rev_event_id[events[idx][2]]}")

                idx += 1
            
            if plot: 
                mne.viz.plot_events(new_events, verbose=True, sfreq=self.sampling_rate, event_id=new_event_id,
                                    on_missing='ignore')
            
            filtered_events_list.append(new_events)

        return filtered_events_list

    # ---------------------------------------------------------------------------------------

    def create_epochs(self, raw_list, verbose=0, plot=False):
        epochs_list = []

        self.events_list = self._filter_events(self.events_list, verbose, plot)

        # Define the time window to extract for each epoch
        
        tmin, tmax = 3.0, 18.0

        for (raw, idx), events in zip(raw_list, self.events_list):

            if verbose:
                print(events)
            
            epochs = mne.Epochs(raw, events=events, event_id=self.event_id, tmin=tmin, tmax=tmax, preload=True,
                                baseline=(None, None))
            
            y = (epochs.events[:, 2]).astype(np.int64)
            
            if verbose: 
                print(epochs.info)
                print(epochs)
                print("These events were dropped:")
                print(epochs.drop_log)
            if plot:
                epochs.plot()
            epochs_list.append((epochs, y, idx))
                        
        joblib.dump(epochs_list, self.epochs_path)
        return epochs_list
    
    # -----------------------------------------------------------------------------------------------

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

        filename = f"folds_list{file_name_suffix}"
        path = os.path.join(self.result_path, filename)
        joblib.dump(folds_list, path)

        return X_stacked, y_stacked, folds, feature_names

    # -----------------------------------------------------------------------------------------------
