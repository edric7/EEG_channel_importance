import os
import mne
import pyxdf
import joblib
import numpy as np
import data.utils as utils
from data.data_handler import DataHandler


class DataHandlerGME(DataHandler):
     
    def __init__(self, models, window_length, step_size, use_ica, base_result_path, data_path, test=False,
                 avg_ref=True, resample=False, sampling_rate=500, filename_suffix="", offset=3, ds_name="", scale=False,
                 all_chns=False):

        super().__init__(use_ica, base_result_path, sampling_rate, test, resample, avg_ref, ds_name, step_size, window_length)

        self.channel_names = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 
                              'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 
                              'FC2', 'F4', 'F8', 'Fp2']

        self.num_channels_per_iteration = [1, 2, 4, 8, 16, 32]

        self.models = models
        self.sample_length = 6
        self.offset = offset

        self.data_path = data_path

        self.sampling_rate = sampling_rate
            
        self.num_classes = 2
        self.num_data_points = None
        self.events_list = None
        self.filename_suffix = filename_suffix
        self.scale = scale
        self.all_chns = all_chns

        if self.all_chns:
            self.num_channels_per_iteration = [i+1 for i in range(0, 32)]


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

        file_list = [
            # ("proband_003_haupt1.xdf", 3), #  -> Error: No Marker Stream
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

        if self.test:
            file_list = file_list[:2]
            
        raw_list = self._load_xdf_into_raw(file_list, self.data_path)
        return raw_list

    # ------------------------------------------------------------------------------------------

    def preprocess_raws(self, raw_list, verbose=0, plot=False):
        filtered = []
        events_list = []

        for raw, idx in raw_list:

            picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
            raw.filter(1, 60)
            raw.notch_filter(np.arange(50, 201, 50), picks=picks, fir_design='firwin')
            
            if self.avg_ref:
                raw = raw.set_eeg_reference(ref_channels='average')

            if self.scale:
                raw = raw.apply_function(utils.standardize_data, channel_wise=False)

            events = mne.find_events(raw, stim_channel="STI", initial_event=True, output="onset")

            if self.resample:
                if verbose:
                    print("resampling to ", self.sampling_rate)
                raw, events = raw.resample(self.sampling_rate, events=events)

            filtered.append((raw, idx))
            events_list.append(events)

        self.events_list = events_list

        return filtered

    # ------------------------------------------------------------------------------------------

    def create_epochs(self, raw_list, verbose=0, plot=0):

        tmin = self.offset
        tmax = tmin + self.sample_length
        epochs_list = []

        # go through each Raw and create Epochs
        for (raw, subj_nr), events in zip(raw_list, self.events_list):

            epochs = mne.Epochs(raw, events, tmin=tmin, tmax=tmax, baseline=None)

            epochs.load_data()
            try:
                epochs = epochs.drop_channels("STI")
            except:
                print("drop_channels STI did not work")

            epochs_list.append((epochs, subj_nr))

        epochs_list = epochs_list

        epochs_list_ = []

        # merge epochs from same subjects, also extracts and formats the labels
        for i in range(0, len(epochs_list)-1):
            subj = epochs_list[i][1]
            subj_next = epochs_list[i+1][1]

            epochs = epochs_list[i][0]
            epochs_next = epochs_list[i+1][0]

            if i == 0:
                y = self._adjust_epochs_labels(epochs)
                epochs_list_.append((epochs, y, subj))
            elif subj == subj_next:
                epochs_new = mne.concatenate_epochs([epochs, epochs_next])
                y = self._adjust_epochs_labels(epochs_new)
                epochs_list_.append((epochs_new, y, subj))

        joblib.dump(epochs_list_, self.epochs_path)

        return epochs_list_

    # ------------------------------------------------------------------------------------------

    def _adjust_epochs_labels(self, epochs):

        # extract the labels from the events
        y = (epochs.events[:, 2] - 2).astype(np.int64)

        # adjust the labels
        for idx, j in enumerate(y):
            if j == 0:
                y[idx] = 0
            else:
                y[idx] = 1

        return y

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

        filename = f"folds_list{file_name_suffix}"
        path = os.path.join(self.result_path, filename)
        joblib.dump(folds_list, path)

        return X_stacked, y_stacked, folds, feature_names

    # -----------------------------------------------------------------------------------------------

    def _load_xdf_into_raw(self, file_list, data_path):
        """
        Given a list of filenames, this method loads the data of the files into Raw-Objects
        and returns them

        """

        raw_list = []

        for filename, idx in file_list:
            path_xdf = os.path.join(data_path, filename)

            streams, header = pyxdf.load_xdf(path_xdf)
            data_matrix, data_timestamps, channel_labels, stream_to_pos_mapping = self._load_stream_data(streams)

            # stream_info = streams[0]['info']
            # fs = float(stream_info['nominal_srate'][0])
            fs = 500
            info = mne.create_info(channel_labels, fs, ch_types='eeg')                  

            data_reshaped = data_matrix.transpose()
            data_reshaped = data_reshaped[:32, :]

            # get the markers and the ground truths
            marker_timestamps, ground_truths = self._get_marker_and_labels_from_stream(streams, False,
                                                                                       stream_to_pos_mapping)

            # create the stim channel
            stim_data = self._create_stim_channel(data_timestamps, marker_timestamps, ground_truths)

            raw = mne.io.RawArray(data=data_reshaped, info=info)

            # add the stim channel
            self._add_stim_to_raw(raw, stim_data)

            montage = mne.channels.make_standard_montage('standard_1020')
            raw.set_montage(montage)

            raw._filenames = [filename]
            raw_list.append((raw, idx))

        return raw_list

    # -----------------------------------------------------------------------------------------------

    @staticmethod
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
        stream_to_pos_mapping = {}
        for pos in range(0, len(streams)):
            stream = streams[pos]['info']['name']
            stream_to_pos_mapping[stream[0]] = pos

        # raise an error if the searched stream_name is not existing
        if stream_name not in stream_to_pos_mapping.keys():
            raise ValueError(
                'Stream ' + str(stream_name) + ' not found in xdf file. '
                                               'Found streams are: '
                                               '\n' + str(
                    stream_to_pos_mapping.keys()))

        # Read the channel labels of the stream
        channel_labels = []
        try:
            for channel in streams[stream_to_pos_mapping[stream_name]]['info']['desc'][0]['channels'][0]['channel']:
                channel_labels.append(channel['label'][0])
        except TypeError:
            # no channel information could be found!
            pass

        # Read the data and timestamps
        data_matrix = streams[stream_to_pos_mapping[stream_name]]['time_series']
        data_timestamps = streams[stream_to_pos_mapping[stream_name]]['time_stamps']

        return data_matrix, data_timestamps, channel_labels, stream_to_pos_mapping

    # ---------------------------------------------------------------------------------------

    @staticmethod
    def _get_marker_and_labels_from_stream(streams, exclude_trainings_trials, stream_to_pos_mapping=None):
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

        idx = stream_to_pos_mapping['ssvepMarkerStream']
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

    @staticmethod
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

    @staticmethod
    def _add_stim_to_raw(raw, stim_data):
        """
        Adds the stim channel to the Raw-Object
        """

        # remove it if there is already a stim channel present (to rerun sections in notebook)
        if "STI" in raw.info["ch_names"]: 
            raw = raw.drop_channels("STI")

        info = mne.create_info(['STI'], raw.info['sfreq'], ['stim'])
        stim_raw = mne.io.RawArray(stim_data, info)
        raw.add_channels([stim_raw], force_update_info=True) 

    # -----------------------------------------------------------------------------------------------
