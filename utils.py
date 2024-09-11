import os

# ----------------------------------------------------------------------------------

def get_1d_as_n_m(index, cols):
    """
    Given a number columns and an index, calculate where in a grid of (n, cols)
    the index lands in terms of two coordinates
    """
    a = int(index / cols)
    b = index % cols
    return (a, b)

# ----------------------------------------------------------------------------------

def print_param(param):
    p = param
    data_type = p['data_type']
    independent = p['independent']
    ref_type = p['ref_type']
    use_ica = p['use_ica']
    window_len = p['window_length']
    step_size = p['step_size']
    agg_type = p['agg']
    n_splits = p['n_splits']
    whole_data_path = p['paths']['whole']
    raw_list_path = p['paths']['raw']
    load_data = p['load']    

    print("-"*40, "\ndata:", data_type, ", independent:", independent, ", win_len:", window_len, ", step_size", step_size, 
          ", agg:", agg_type, ", n_splits:", n_splits, ", ref type:", ref_type, ", use ica:", use_ica)
    
# ----------------------------------------------------------------------------------

def prepare_params(params, test, base_result_path):

    for p in params: 
        p['paths'] = {}

        data_type = p['data_type']
        independent = p['independent']
        ref_type = p['ref_type']
        window_len = p['window_length']
        agg_type = p['agg']
        n_splits = p['n_splits']
        step_size = p['step_size']
        use_ica = p['use_ica']

        if data_type == 'iea':
            p['ch_names'] = ['Cz', 'Fp2', 'F3', 'FT7', 'C3', 'C4', 'FT8', 'P3', 'P4', 'PO7', 'PO8', 'Oz']
            
            if test: p['num_channels'] = [1,2,4]
            else: p['num_channels'] = [1,2,4,8,12]
        else: 
            p['ch_names'] = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 
                              'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 
                              'FC2', 'F4', 'F8', 'Fp2']

            if test: p['num_channels'] = [1,2,4]
            else: p['num_channels'] = [1,2,4,8,16,32]

        ica_str      = '_ica' if use_ica else "" 
        ind_str      = "ind" if independent else "dep"
        n_splits_str = '' if independent else f"_{n_splits}"
        wl_str   = str(int(1000*window_len))

        result_path = os.path.join(base_result_path, f"results/{data_type}_data")
        result_path = os.path.join(result_path, f"{agg_type}_{wl_str}_{int(step_size*100)}_{ind_str}{n_splits_str}_{ref_type}{ica_str}")

        if not os.path.exists(result_path): 
            os.makedirs(result_path)

        if data_type == 'iea':
            p['paths']['raw'] = None
        else:
            fn = "raws_filtered_ica" if use_ica else "raws_filtered"
            p['paths']['raw'] = os.path.join(f"results/gme_data", fn)
            p['paths']['epo'] = os.path.join(result_path, "epochs_list")

        p['paths']['imp'] = {}
        p['paths']['res'] = {}
        p['paths']['plots'] = {}
        p['paths']['whole'] = os.path.join(result_path, "data_all_electrodes")

        methods = ['rf', 'mi', 'pi', 'abl', 'shap', 'random','all']
        for s in methods: 
            p['paths']['imp'][s] = os.path.join(result_path, f"imp_{s}")  
            p['paths']['res'][s] = os.path.join(result_path, f"model_results_{s}")  
            p['paths']['plots'][s] = os.path.join(result_path, f"plots_{s}")  

            if not os.path.exists(p['paths']['plots'][s]): 
                os.makedirs(p['paths']['plots'][s])               
                
# ----------------------------------------------------------------------------------

