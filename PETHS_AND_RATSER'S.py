""" To RUN this Python File"""
""" python  kaniyar_Narayana_Iyengar_Anirudh_Iyengar_HW2.py  """
""" You would visualize 3 plots each for cue and Movements """

#### Importing required Libraries
import numpy as np
import h5py
from scipy.io import loadmat
import matplotlib.pyplot as plt

#### Loading Both neural and behaviorData
annots = loadmat('data\BME_526_HW2_NeuralData.mat')
neural =annots['Channels']['Chan2'][0][0]

with h5py.File('data/BME_526_HW2_BehaviorData.mat', 'r') as f:
    keys = list(f.keys())  # Get list of keys in the HDF5 file

    # Create a dictionary to hold the loaded data
    dummy_beh = {}
    for key in keys:
        dummy_beh[key] = np.array(f[key]) 

#### Considering Required neurons and channels
neuron_idx = [2,1,1]
channel_indices = [3, 54, 94]

event_name_map = {
    'i_times': 'Index',
    'm_times': 'Middle',
    't_times': 'Thumb',
    'im_times': 'Index - Middle',
    'ti_times': 'Thumb - Index',
    'tm_times': 'Thumb - Middle',
    'tim_times': 'TIM'
}

#### Functions to align spkie times , Calculate peth and Plot peth
def align_spike_times(spike_times, event_times):
    # Subtract each event time from every spike time to align spike times relative to events
    # The new axis is added to event_times to allow broadcasting for element-wise subtraction
    aligned_spike_times = spike_times - event_times[:, np.newaxis] 
    return aligned_spike_times

def calculate_peth(spike_times):
    time_window = [-1, 1] # Time window for PETH
    bin_size = 0.05 # the bin size for the histogram
    
    edges = np.arange(time_window[0], time_window[1] + bin_size, bin_size) # The result is a histogram of spike times binned according to the specified edges
    peth, _ = np.histogram(spike_times, bins=edges)
    peth = peth / bin_size # Normalize the PETH
    peth = peth / len(spike_times) # Normalize the PETH 
    
    return peth

def plot_peth(peth,ax,threshold=None):
    time_window = [-1, 1]
    bin_size = 0.05
    
    time_vector = np.arange(time_window[0], time_window[1], bin_size)
    ax.plot(time_vector, peth)
    ax.fill_between(time_vector, peth + 0.3 * np.std(peth, axis=0), peth - 0.3 * np.std(peth, axis=0), color='grey', alpha=0.5) # Filling The Standard Deviation on waveform
    if threshold is not None:
        ax.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Firing Rate')
    ax.set_xlim(time_window)
    ax.legend()
    ax.grid(True)

#### Raster Plot And PETH for "the times of the cue being displayed"
# Assuming neuron_idx, channel_indices, and dummy_beh are defined
thresholds = {}
for i, neu_idx in enumerate(neuron_idx):
    neural = annots['Channels'][f'Chan{channel_indices[i]}'][0][0]
    ap_times = neural[neural[:, 1] == neu_idx, 2]
    
    # Create subplots for raster plots and PETHs
    fig, axs = plt.subplots(2, len(dummy_beh), figsize=(25, 10), sharex=True)
    
    # Plot raster plots in the first row
    for j, event_type in enumerate(dummy_beh.keys()):
        axs[0, j].set_title(event_name_map[event_type])
        axs[0, j].set_ylabel('Trials')
        axs[0, j].set_xticks([-1, 0, 1])  
        axs[0, j].set_xticklabels([-1, 0, 1])
        for event in range(0,1):# 1st coloumn
            start_time = -1
            end_time = 1
            for n in range(len(dummy_beh[event_type])):
                aligned_ap_times = ap_times - dummy_beh[event_type][n, event] # Calculate the spike times aligned to the current trial's event time (1st coloumn)
                ap_index = np.where((aligned_ap_times >= start_time) & (aligned_ap_times <= end_time))[0] # Find the indices of spike times falling within the specified time window
                trial_ap_times = aligned_ap_times[ap_index]
                trial_indices = [n] * len(trial_ap_times) # Create indices for the current trial to use for plotting
                axs[0, j].scatter(trial_ap_times, trial_indices, color='k', s=5)
    
    # Plot PETHs in the second row
    for j, event_type in enumerate(dummy_beh.keys()):
        axs[1, j].set_title(event_name_map[event_type])
        axs[1, j].set_ylabel('Firing Rate')
        axs[1, j].set_xlabel('Time (s)')
        axs[1, j].set_xticks([-1, 0, 1])  # Set x-axis ticks
        axs[1, j].set_xticklabels([-1, 0, 1])
        neuron1_ap_itimes_cue_aligned = align_spike_times(ap_times,dummy_beh[event_type][:,1]) # recive the aligned spike times
        peth = calculate_peth(neuron1_ap_itimes_cue_aligned) # calculate the PETH required for plotting
        

        threshold = np.max(peth) * 0.7  # Finding the Thresold
        thresholds[(neu_idx, event_type)] = threshold 
        plot_peth(peth,axs[1,j], threshold=threshold) #Plot the PETH Plot
        
    fig.suptitle(f'Neuron {neu_idx} - Channel {channel_indices[i]} for cue')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

#### Raster Plot And PETH for "the times the subject moved"
    
# Assuming neuron_idx, channel_indices, and dummy_beh are defined
thresholds = {}
for i, neu_idx in enumerate(neuron_idx):
    neural = annots['Channels'][f'Chan{channel_indices[i]}'][0][0]
    ap_times = neural[neural[:, 1] == neu_idx, 2]
    
    # Create subplots for raster plots and PETHs
    fig, axs = plt.subplots(2, len(dummy_beh), figsize=(25, 10), sharex=True)
    
    # Plot raster plots in the first row
    for j, event_type in enumerate(dummy_beh.keys()):
        axs[0, j].set_title(event_name_map[event_type])
        axs[0, j].set_ylabel('Trials')
        axs[0, j].set_xticks([-1, 0, 1])  # Set x-axis ticks
        axs[0, j].set_xticklabels([-1, 0, 1])
        for event in range(1,2): #To use only 2nd coloumn of the event 
            start_time = -1
            end_time = 1
            for n in range(len(dummy_beh[event_type])):
                aligned_ap_times = ap_times - dummy_beh[event_type][n, event] # Calculate the spike times aligned to the current trial's event time (2nd coloumn)
                ap_index = np.where((aligned_ap_times >= start_time) & (aligned_ap_times <= end_time))[0] # Find the indices of spike times falling within the specified time window
                trial_ap_times = aligned_ap_times[ap_index]
                trial_indices = [n] * len(trial_ap_times) # Create indices for the current trial to use for plotting
                axs[0, j].scatter(trial_ap_times, trial_indices, color='k', s=5)
    
    # Plot PETHs in the second row
    for j, event_type in enumerate(dummy_beh.keys()):
        axs[1, j].set_title(event_name_map[event_type])
        axs[1, j].set_ylabel('Firing Rate')
        axs[1, j].set_xlabel('Time (s)')
        axs[1, j].set_xticks([-1, 0, 1])  # Set x-axis ticks
        axs[1, j].set_xticklabels([-1, 0, 1])
        neuron1_ap_itimes_cue_aligned = align_spike_times(ap_times,dummy_beh[event_type][:,1]) # recive the aligned spike times
        peth = calculate_peth(neuron1_ap_itimes_cue_aligned) # calculate the PETH required for plotting
        

        threshold = np.max(peth) * 0.7  # Finding the Thresold
        thresholds[(neu_idx, event_type)] = threshold 
        plot_peth(peth,axs[1,j], threshold=threshold) #Plot the PETH Plot
    
       
    fig.suptitle(f'Neuron {neu_idx} - Channel {channel_indices[i]} for Movement')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


"""1. - a. What is the field of the data (think in terms of response fields)?
    The data fields consist of the behavior data variables ('i_times', 'im_times', 'm_times', 't_times', 'ti_times', 'tim_times', 'tm_times') representing different finger movements. Each variable contains cue and movement times.
    
    - b. What is the response field for each of the neurons that you analyzed?
    Each neuron's response field is determined by the combination of the channel number and neuron index within the neural data. 

    2. - a. Choose three specific neurons for your analysis and describe why they would be useful in decoding the movements?
    We'll select three neurons **[2,1,1]** that exhibit distinct firing patterns in response to different finger movements. These neurons could have response fields aligned with specific movements, making them useful for decoding.

    - b. From the PETH plots for these three neurons, what would be a good action potential firing rate threshold to use for detecting a specific movement? Would the threshold be the same or different for each neuron?
    From the PETH plots, we'll observe the firing rates during different finger movements. A good threshold would be above the baseline firing rate but below the peak firing rate. The threshold might vary for each neuron and the event type.

    - c. Plot this threshold on your PETHs. 
    The plots above were generated with a threshold applied to the PETHs, allowing for the visualization of their relationship with the firing rate and the identification of instances when the firing rate exceeds the threshold at specific times."""

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# Load Electrode Map Data
electrode_map_data = sio.loadmat('data/ElectrodeMap.mat')['MapStruct']

cerebus_channel_map = electrode_map_data['CerebusChannel'][0][0]
tdt_channel_map = electrode_map_data['TDTChannel'][0][0]
