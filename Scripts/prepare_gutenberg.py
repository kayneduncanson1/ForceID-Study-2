import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PrePro import trim_filt_norm

"""This script was used to characterise and prepare Gutenberg. It first reads metadata and transforms it to a suitable
format for use in other scripts (e.g., main.py). It also reads, trims, filters, and time normalises raw measures. The
final two pre-processing steps (standardising measures via z-score normalisation and combining measures from left and
right stance into a single input representation) are conducted in the run_expt function that is called in main.py."""

# Get ID labels on a per-session basis (for certain subsets in the dataset, participants did more than one session):
labels_orig_by_session = pd.read_csv('./Datasets/gb-all/spreadsheets/GRF_metadata.csv',
                                     usecols=[1]).values.squeeze()
# Get other metadata on a per-session basis:
session_nos_orig_by_session = pd.read_csv('./Datasets/gb-all/spreadsheets/GRF_metadata.csv',
                                          usecols=[2]).values.squeeze()
sex_by_session = pd.read_csv('./Datasets/gb-all/spreadsheets/GRF_metadata.csv', usecols=[5]).values.squeeze()
age_by_session = pd.read_csv('./Datasets/gb-all/spreadsheets/GRF_metadata.csv', usecols=[6]).values.squeeze()
height_by_session = pd.read_csv('./Datasets/gb-all/spreadsheets/GRF_metadata.csv', usecols=[7]).values.squeeze()
mass_by_session = pd.read_csv('./Datasets/gb-all/spreadsheets/GRF_metadata.csv', usecols=[9]).values.squeeze()

ids_orig, counts_sessions_by_id = np.unique(labels_orig_by_session, return_counts=True)
# Get arr of unique ID labels, starting from the total no. IDs across ForceID-A and GaitRec:
ids = np.arange(193 + 211, 193 + 211 + ids_orig.shape[0])
indices_sessions_where_id_changes = np.cumsum(counts_sessions_by_id)

# Get dataset numbers, ID labels, and session numbers on a per-trial basis, as well as trial numbers:
datasets = pd.read_csv('./Datasets/gb-all/spreadsheets/GRF_F_ML_RAW_left.csv', usecols=[0]).values.squeeze()
labels_orig = pd.read_csv('./Datasets/gb-all/spreadsheets/GRF_F_ML_RAW_left.csv',
                          usecols=[1]).values.squeeze()
session_nos_orig = pd.read_csv('./Datasets/gb-all/spreadsheets/GRF_F_ML_RAW_left.csv', usecols=[2]).values.squeeze()
trial_nos_orig = pd.read_csv('./Datasets/gb-all/spreadsheets/GRF_F_ML_RAW_left.csv', usecols=[3]).values.squeeze()

counts_samples = np.unique(labels_orig, return_counts=True)[1]

labels = np.concatenate([np.linspace(ids[i], ids[i], counts_samples[i]) for i in range(ids.shape[0])]).astype(np.int64)

# Get other metadata on a per-trial basis..
# Initialise variables:
session_nos = np.zeros_like(session_nos_orig)
sexes = np.zeros_like(session_nos_orig).astype(np.float64)
ages = np.zeros_like(session_nos_orig).astype(np.float64)
heights = np.zeros_like(session_nos_orig).astype(np.float64)
masses = np.zeros_like(session_nos_orig).astype(np.float64)

height_by_id = []
mass_by_id = []

counts_samples_by_session_all = []
counts_sessions_max = np.max(counts_sessions_by_id)

# For each ID:
for i in range(indices_sessions_where_id_changes.shape[0]):

    # Get their subset of metadata from each session:
    if i == 0:

        session_nos_subset = session_nos_orig_by_session[:indices_sessions_where_id_changes[i]]
        sex_subset = np.array([sex_by_session[0]] * counts_sessions_by_id[i]) # Only measured in first session
        age_subset = np.array([age_by_session[0]] * counts_sessions_by_id[i]) # Only measured in first session
        # Height and mass were measured in each session, so get the mean across sessions...
        # Height in units of metres and rounded to two decimal places. Mass rounded to one decimal place:
        height_subset = np.array([np.round(np.mean(height_by_session[:indices_sessions_where_id_changes[i]]) / 100,
                                           decimals=2)] * counts_sessions_by_id[i])
        mass_subset = np.array([np.round(np.mean(mass_by_session[:indices_sessions_where_id_changes[i]]),
                                         decimals=1)] * counts_sessions_by_id[i])
        height_by_id.append(height_subset[0])
        mass_by_id.append(mass_subset[0])

    else:

        session_nos_subset = session_nos_orig_by_session[indices_sessions_where_id_changes[i - 1]:
                                                         indices_sessions_where_id_changes[i]]

        sex_subset = np.array([sex_by_session[indices_sessions_where_id_changes[i - 1]]] * counts_sessions_by_id[i])
        age_subset = np.array([age_by_session[indices_sessions_where_id_changes[i - 1]]] * counts_sessions_by_id[i])

        height_subset = np.array([np.round(np.mean(height_by_session[indices_sessions_where_id_changes[i - 1]:
                                                                     indices_sessions_where_id_changes[i]]) / 100,
                                           decimals=2)] * counts_sessions_by_id[i])
        mass_subset = np.array([np.round(np.mean(mass_by_session[indices_sessions_where_id_changes[i - 1]:
                                                                 indices_sessions_where_id_changes[i]]),
                                         decimals=1)] * counts_sessions_by_id[i])
        height_by_id.append(height_subset[0])
        mass_by_id.append(mass_subset[0])

    counts_samples_by_session = np.zeros((counts_sessions_max,))

    # For each session:
    for idx, session_no in enumerate(session_nos_subset):

        # Get the indices of samples from the session:
        indices_samples = np.asarray(session_nos_orig == session_no).nonzero()[0]

        # Make new single digit session numbers:
        session_nos[indices_samples] = idx + 1

        counts_samples_by_session[idx] = indices_samples.shape[0]

        sexes[indices_samples] = sex_subset[idx]
        ages[indices_samples] = age_subset[idx]
        heights[indices_samples] = height_subset[idx]
        masses[indices_samples] = mass_subset[idx]

    counts_samples_by_session_all.append(counts_samples_by_session)

speeds = np.array(['P'] * labels.shape[0]) # 'P' = self-selected 'preferred' speed.
footwear = np.zeros_like(labels) # 0 = barefoot (all participants walked barefoot).

height_by_id = np.array(height_by_id)
mass_by_id = np.array(mass_by_id)

# Create an arr of unique trial identifiers that each contain ID label, session no., trial no. and speed category:
trial_names = np.array(['_'.join(['GB', format(labels[i], '04d'), format(session_nos[i], '02d'),
                                  format(trial_nos_orig[i], '02d'), speeds[i]])
                        for i in range(labels.shape[0])])

# Save metadata so that it can be loaded in other scripts:
np.save('./Datasets/gb-all/objects/ids.npy', ids)
np.save('./Datasets/gb-all/objects/counts_samples.npy', counts_samples)
np.save('./Datasets/gb-all/objects/labels.npy', labels)
np.save('./Datasets/gb-all/objects/session_nos.npy', session_nos)
np.save('./Datasets/gb-all/objects/trial_nos.npy', trial_nos_orig)
np.save('./Datasets/gb-all/objects/speeds.npy', speeds)
np.save('./Datasets/gb-all/objects/sexes.npy', sexes)
np.save('./Datasets/gb-all/objects/ages.npy', ages)
np.save('./Datasets/gb-all/objects/heights.npy', heights)
np.save('./Datasets/gb-all/objects/masses.npy', masses)
np.save('./Datasets/gb-all/objects/heights_by_id.npy', height_by_id)
np.save('./Datasets/gb-all/objects/masses_by_id.npy', mass_by_id)
np.save('./Datasets/gb-all/objects/footwear.npy', footwear)
np.save('./Datasets/gb-all/objects/trial_names.npy', trial_names)

# Read raw force platform measures from csv files and convert to shape (N, C, L)...
# Key
# ---
# N = no. samples
# C = no. channels (directional components)
# L = length i.e., no frames
# fx = GRF x (medio-lateral) direction
# fy = GRF y (antero-posterior) direction
# fz = GRF z (vertical) direction
# cx = COP x direction
# cy = COP y direction
# r = right stance
# l = left stance
# ---
fx_r_raw = np.expand_dims(pd.read_csv('./Datasets/gb-all/spreadsheets/GRF_F_ML_RAW_right.csv')
                          .fillna(0).values[:, 4:], axis=1)
fy_r_raw = np.expand_dims(pd.read_csv('./Datasets/gb-all/spreadsheets/GRF_F_AP_RAW_right.csv')
                          .fillna(0).values[:, 4:], axis=1)
fz_r_raw = np.expand_dims(pd.read_csv('./Datasets/gb-all/spreadsheets/GRF_F_V_RAW_right.csv')
                          .fillna(0).values[:, 4:], axis=1)
cx_r_raw = np.expand_dims(pd.read_csv('./Datasets/gb-all/spreadsheets/GRF_COP_ML_RAW_right.csv')
                          .fillna(0).values[:, 4:], axis=1)
cy_r_raw = np.expand_dims(pd.read_csv('./Datasets/gb-all/spreadsheets/GRF_COP_AP_RAW_right.csv')
                          .fillna(0).values[:, 4:], axis=1)
fx_l_raw = np.expand_dims(pd.read_csv('./Datasets/gb-all/spreadsheets/GRF_F_ML_RAW_left.csv')
                          .fillna(0).values[:, 4:], axis=1)
fy_l_raw = np.expand_dims(pd.read_csv('./Datasets/gb-all/spreadsheets/GRF_F_AP_RAW_left.csv')
                          .fillna(0).values[:, 4:], axis=1)
fz_l_raw = np.expand_dims(pd.read_csv('./Datasets/gb-all/spreadsheets/GRF_F_V_RAW_left.csv')
                          .fillna(0).values[:, 4:], axis=1)
cx_l_raw = np.expand_dims(pd.read_csv('./Datasets/gb-all/spreadsheets/GRF_COP_ML_RAW_left.csv')
                          .fillna(0).values[:, 4:], axis=1)
cy_l_raw = np.expand_dims(pd.read_csv('./Datasets/gb-all/spreadsheets/GRF_COP_AP_RAW_left.csv')
                          .fillna(0).values[:, 4:], axis=1)

# Concatenate directional components from each stance side along the channel axis:
sigs_all_r_raw = np.concatenate((fx_r_raw, fy_r_raw, fz_r_raw, cx_r_raw, cy_r_raw), axis=1)
sigs_all_l_raw = np.concatenate((fx_l_raw, fy_l_raw, fz_l_raw, cx_l_raw, cy_l_raw), axis=1)

# If running script for the first time, we recommend saving all raw measures from each stance side:
np.save('./Datasets/gb-all/objects/sigs_all_r_raw.npy', sigs_all_r_raw)
np.save('./Datasets/gb-all/objects/sigs_all_l_raw.npy', sigs_all_l_raw)
# That way, they can be loaded efficiently on subsequent runs of the script. The lines below can be uncommented and...
# the lines above that read raw measures from csv files can be commented out:
# sigs_all_r_raw = np.load('./Datasets/gb-all/objects/sigs_all_r_raw.npy', allow_pickle=True)
# sigs_all_l_raw = np.load('./Datasets/gb-all/objects/sigs_all_l_raw.npy', allow_pickle=True)

# Initialise variables required to trim, filter, and time normalise measures from each stance side:
fz_thresh = 50  # Newtons. Vertical GRF magnitude threshold for defining the stance phase.
interp_len = 100 # frames. The no. frames that the measures will be time normalised to via linear interpolation.
cutoff_freq = 30  # Hz. Cut-off frequency for Butterworth low-pass filter.
sampling_rate = 250  # Hz. Original sampling rate/frequency.

sigs_all_r_pro = trim_filt_norm(sigs_all_r_raw, fz_thresh, cutoff_freq, sampling_rate, interp_len, ds='not_fi-all')
sigs_all_l_pro = trim_filt_norm(sigs_all_l_raw, fz_thresh, cutoff_freq, sampling_rate, interp_len, ds='not_fi-all')

# Plot measures from a given directional component and stance side for n random IDs:
indices_samples_where_id_changes = np.cumsum(counts_samples)
n_random_ids_to_plot = 5 # <=30
stance_side = 'r' # Options: 'r' or 'l'
channel = 2 # 0 = Fx, 1 = Fy, 2 = Fz, 3 = Cx, 4 = Cy.
indices_random_ids = random.sample(np.arange(ids.shape[0]), k=n_random_ids_to_plot)

for idx in indices_random_ids:

    if stance_side == 'r':

        [plt.plot(sigs_all_r_pro[j, channel], lw=1, color='black', alpha=0.3)
         for j in range(indices_samples_where_id_changes[idx - 1], indices_samples_where_id_changes[idx])]

    else:

        [plt.plot(sigs_all_l_pro[j, channel], lw=1, color='black', alpha=0.3)
         for j in range(indices_samples_where_id_changes[idx - 1], indices_samples_where_id_changes[idx])]

    plt.show()

# Save all trimmed, filtered, and time normalised measures as a single object to be loaded in other scripts:
np.save('./Datasets/gb-all/objects/sigs_all_r_pro.npy', sigs_all_r_pro)
np.save('./Datasets/gb-all/objects/sigs_all_l_pro.npy', sigs_all_l_pro)
