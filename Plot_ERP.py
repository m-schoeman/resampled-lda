import mne
from matplotlib import pyplot as plt
from sklearn.datasets import make_spd_matrix
import seaborn as sns
import warnings
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
warnings.simplefilter(action='ignore', category=RuntimeWarning)

mne.set_log_level('WARNING')

def get_single_run_data(file_path, reject_non_iid = False):
    non_scalp_channels = ['EOGvu', 'x_EMGl', 'x_GSR', 'x_Respi', 'x_Pulse', 'x_Optic'] 
    raw = mne.io.read_raw_brainvision(file_path, misc=non_scalp_channels, preload=True)
    raw.set_montage('standard_1020')
    if reject_non_iid:
        raw.set_annotations(raw.annotations[7:85])
    return raw


def preprocessed_to_epoch(preprocessed_data, decimate=10, baseline_ival=(-.2, 0)):
    raw_stim_ids = {"Stimulus/S  1": 1, "Stimulus/S 21": 2} 
    class_ids = {"Target": 2, "Non-target": 1}
    reject = dict()
    # reject = dict(eeg=60 * 1e-6)
    events = mne.events_from_annotations(preprocessed_data, event_id=raw_stim_ids)[0]
    epo_data = mne.Epochs(preprocessed_data, events, event_id=class_ids,
                          baseline=baseline_ival, decim=decimate,
                          reject=reject, proj=False, preload=True)
    return epo_data

def add_color_spans(ax, color_spans, color='grey'):
    for i, cs in enumerate(color_spans):
        even = (i + 1) % 2 == 0
        ax.axvspan(cs[0], cs[1], alpha=0.15 if even else 0.3, facecolor=color, edgecolor=None)

path = 'Root to single dataset, e.g. /subject1/Oddball_Run_1_Trial_016_SOA_0.162.vhdr'
raw = get_single_run_data(path)


eeg_data = raw

filter_band = (1.5, 16)
eeg_data.filter(filter_band[0], filter_band[1], method='iir')
epo_data = preprocessed_to_epoch(raw)

plot_channels = ['Cz']
channel_styles = ['-', '--']
cp = sns.color_palette()


evo_t = epo_data['Target'].average(picks=plot_channels)
evo_nt = epo_data['Non-target'].average(picks=plot_channels)

color_spans = [[0.10, 0.14], [0.14,0.17], [0.17,0.20], [0.20,0.23], [0.23,0.27], [0.27,0.30], [0.30,0.35], [0.35,0.41], 
               [0.41,0.45], [0.45,0.50]]

fig, ax = plt.subplots(1, 1, facecolor='white', figsize=(9, 6))

for ch_i, ch in enumerate(plot_channels):
    ax.plot(evo_t.times, evo_t.data[ch_i, :]*1000000,
            linestyle=channel_styles[ch_i], color=cp[1], label=f'{ch} Target')
    ax.plot(evo_nt.times, evo_nt.data[ch_i, :]*1000000,
            linestyle=channel_styles[ch_i], color=cp[0], label=f'{ch} Non-target')

ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude (µV)')
ax.set_title('ERP at channel Cz with averaged time intervals')
ax.legend()
add_color_spans(ax, color_spans)
fig.show()
