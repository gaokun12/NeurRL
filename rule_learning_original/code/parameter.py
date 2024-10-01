def return_parameter(task_name):
    if task_name == 'gun':
        default_label = 1 
        default_negative = 0
        label_name = 'label'
        time_event_name = ['event'] # there is no event in the column data 
        time_series_name = ['t'] # this indicate all columns are time series data
        body_windows_length = 150 # hour as unit 
        future_window_length = 0 # hour as unit default 0
    elif 'sgh' in task_name:
        default_label = True
        default_negative = False
        label_name = 'label'
        time_series_name = ['glucose'] # continues variate name is called glucose 
        time_event_name = ['insulin'] # event variable name is called insulin
        body_windows_length = 24 # hour as unit 
        future_window_length = 24 # hour as unit default 0
    elif 'mimic' in task_name:
        default_label = True
        default_negative = False
        label_name = 'label'
        time_series_name = ['glucose'] # continues variate name is called glucose 
        time_event_name = ['insulin'] # event variable name is called insulin
        body_windows_length = 24 # hour as unit 
        future_window_length = 24 # hour as unit default 0
    elif 'coffee' in task_name:
        default_label = 1 
        default_negative = 0
        label_name = 'label'
        time_event_name = ['event'] # there is no event in the column data 
        time_series_name = ['t'] # this indicate all columns are time series data
        body_windows_length = None # hour as unit 
        future_window_length = 0 # hour as unit default 0
    elif 'ucr' in task_name:
        default_label = 1
        default_negative = 0
        label_name = 'label'
        time_event_name = ['event'] # there is no event in the column data 
        time_series_name = ['t'] # this indicate all columns are time series data
        body_windows_length = None # hour as unit 
        future_window_length = 0 # hour as unit default 0
    else:
        raise ValueError('task_name is not valid')
    return default_label, default_negative, label_name, time_event_name, time_series_name, body_windows_length, future_window_length 