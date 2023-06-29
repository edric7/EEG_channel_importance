from abc import abstractmethod
import os
import joblib


class DataHandler:

    def __init__(self, use_ica, base_result_path, sampling_rate, test, resample, avg_ref, ds_name, step_size,
                 window_length):

        self.channel_names = None
        self.test = test
        self.ds_name = ds_name
        self.window_length = window_length
        self.step_size = step_size
        self.use_ica = use_ica
        self.resample = resample
        self.baseline_correction = False
        self.independent = True
        self.use_ica = use_ica
        self.base_result_path = base_result_path
        self.sampling_rate = sampling_rate
        self.avg_ref = avg_ref
        self._prepare_paths()

    # ---------------------------------------------------------------------------------------

    def __str__(self):
        output = "............\n"
        output += f"{self.ds_name} Data Handler\n"
        output += "\tchannels: " + str(self.channel_names) + "\n"
        output += "\ttest_mode: " + str(self.test) + "\n"
        output += "\twindow_length: " + str(self.window_length) + "\n"
        output += "\tstep_size: " + str(self.step_size) + "\n"
        output += "\tuse_ica: " + str(self.use_ica) + "\n"
        output += "\tbaseline correction: " + str(self.baseline_correction) + "\n"
        output += "\tresampled: " + str(self.resample) + "\n"
        output += "\tsampling_rate: " + str(self.sampling_rate) + "\n"

        return output

    # ---------------------------------------------------------------------------------------

    def _prepare_paths(self):
        ica_str = '_ica' if self.use_ica else ""
        wl_str = str(int(1000*self.window_length))
        test_str = "_test" if self.test else ""
        blc_str = "_blc" if self.baseline_correction else ""

        self.result_path = os.path.join(self.base_result_path, f"results/{self.ds_name}_data")
        self.result_path = os.path.join(self.result_path,
                                        f"resample_{self.resample}_sr_{self.sampling_rate}_wl_{wl_str}_step_" +
                                        f"{int(self.step_size*100)}{ica_str}{blc_str}" +
                                        f"{test_str}")

        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        self.epochs_path = os.path.join(self.result_path, "epochs_list")
        self.plots_path = os.path.join(self.result_path, "plots")
        self.icas_path = os.path.join(self.result_path, "icas")
        self.model_path = os.path.join(self.result_path, "model")

        if not os.path.exists(self.plots_path):
            os.makedirs(self.plots_path)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

    # ---------------------------------------------------------------------------------------

    def get_folds(self, filename_suffix=""):
        folds_path = os.path.join(self.result_path, "folds_list" + filename_suffix)
        folds_list = joblib.load(folds_path)
        return folds_list

    # ---------------------------------------------------------------------------------------

    def get_epochs(self):
        return joblib.load(self.epochs_path)


    # ---------------------------------------------------------------------------------------

    @abstractmethod
    def get_num_classes(self):
        pass

    # ---------------------------------------------------------------------------------------

    @abstractmethod
    def get_channel_names(self):
        pass

    # ---------------------------------------------------------------------------------------

    @abstractmethod
    def get_num_data_points(self):
        pass

    # ---------------------------------------------------------------------------------------

    @abstractmethod
    def get_num_channels(self):
        pass

    # ---------------------------------------------------------------------------------------

    @abstractmethod
    def load_raws(self, verbose=0):
        pass

    # ---------------------------------------------------------------------------------------

    @abstractmethod
    def preprocess_raws(self, raw_list, verbose=0, plot=False):
        pass

    # ---------------------------------------------------------------------------------------

    @abstractmethod
    def create_epochs(self, raw_list, verbose=0, plot=False):
        pass

    # -----------------------------------------------------------------------------------------------

    @abstractmethod
    def create_folds(self, epochs_list, file_name_suffix=""):
        pass

    # -----------------------------------------------------------------------------------------------
