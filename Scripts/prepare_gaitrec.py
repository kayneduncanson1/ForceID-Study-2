import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PrePro import trim_filt_norm
from sklearn import preprocessing

"""This script was used to characterise and prepare both subsets of GaitRec. The primary subset contained shod walking
from healthy controls (referred as 'gr-sho' in strings, '..._shod' in variables, and GaitRec-S in the UMAP subsection
of the results). The secondary subset was both barefoot and shod walking from healthy controls ('gr-all' string and
GaitRec-M in the UMAP section of the results). The script reads metadata and transforms it to a suitable format for use
in other scripts (e.g., main.py). It also reads, trims, filters, and time normalises the raw measures. The final two
pre-processing steps (standardising measures via z-score normalisation and combining measures from left and right stance
into a single input representation) are conducted in the run_expt function that is called in main.py."""

pathology_labels = pd.read_csv('./Datasets/gr-all/spreadsheets/GRF_metadata.csv',
                               usecols=[2]).values.squeeze()
indices_healthy_controls = np.asarray(pathology_labels == 'HC').nonzero()[0] # 'HC' = Healthy Control

# Get ID labels on a per-session basis (each participant did 3-6 sessions on the same day):
labels_orig_by_session = pd.read_csv('./Datasets/gr-all/spreadsheets/GRF_metadata.csv',
                                     usecols=[0]).values.squeeze()[indices_healthy_controls]
# Get other metadata on a per-session basis:
session_nos_orig_by_session = pd.read_csv('./Datasets/gr-all/spreadsheets/GRF_metadata.csv',
                                          usecols=[1]).values.squeeze()[indices_healthy_controls]
sex_by_session = pd.read_csv('./Datasets/gr-all/spreadsheets/GRF_metadata.csv',
                             usecols=[4]).values.squeeze()[indices_healthy_controls]
age_by_session = pd.read_csv('./Datasets/gr-all/spreadsheets/GRF_metadata.csv',
                             usecols=[5]).values.squeeze()[indices_healthy_controls]
height_by_session = pd.read_csv('./Datasets/gr-all/spreadsheets/GRF_metadata.csv',
                                usecols=[6]).values.squeeze()[indices_healthy_controls]
mass_by_session = pd.read_csv('./Datasets/gr-all/spreadsheets/GRF_metadata.csv',
                              usecols=[8]).values.squeeze()[indices_healthy_controls]
# Footwear (0 = barefoot, 1 = shod in personal footwear):
footwear_by_session = pd.read_csv('./Datasets/gr-all/spreadsheets/GRF_metadata.csv',
                                  usecols=[11]).values.squeeze()[indices_healthy_controls]
# Self-selected speed (1 = slower than preferred, 2 = preferred, 3 = faster than preferred):
speeds_orig_by_session = pd.read_csv('./Datasets/gr-all/spreadsheets/GRF_metadata.csv',
                                     usecols=[13]).values.squeeze()[indices_healthy_controls]
session_dates = pd.read_csv('./Datasets/gr-all/spreadsheets/GRF_metadata.csv',
                            usecols=[16]).values.squeeze()[indices_healthy_controls]

ids_orig, counts_sessions_by_id = np.unique(labels_orig_by_session, return_counts=True)
# Get arr of ID unique labels, starting from the total no. IDs in ForceID-A:
ids = np.arange(193, 193 + ids_orig.shape[0])
indices_sessions_where_id_changes = np.cumsum(counts_sessions_by_id)

# Confirmed that each participant completed all of their sessions on the same date:
session_dates_by_id = np.split(session_dates, indices_sessions_where_id_changes[:-1])
counts_unique_session_dates = np.array([np.unique(item).shape[0] for item in session_dates_by_id])
counts_multiple_dates = np.asarray(counts_unique_session_dates != 1).nonzero()[0].shape[0]
counts_single_date = np.asarray(counts_unique_session_dates == 1).nonzero()[0].shape[0]

# For healthy controls, get ID labels and session numbers on a per-trial basis, as well as trial numbers:
labels_orig = pd.read_csv('./Datasets/gr-all/spreadsheets/GRF_F_ML_RAW_left.csv',
                          usecols=[0]).values.squeeze()
indices_labels_healthy_controls = np.concatenate([np.asarray(labels_orig == id_).nonzero()[0] for id_ in ids_orig])
labels_orig_hc = labels_orig[indices_labels_healthy_controls]
session_nos_orig = pd.read_csv('./Datasets/gr-all/spreadsheets/GRF_F_ML_RAW_left.csv',
                               usecols=[1]).values.squeeze()[indices_labels_healthy_controls]
trial_nos_orig = pd.read_csv('./Datasets/gr-all/spreadsheets/GRF_F_ML_RAW_left.csv',
                             usecols=[2]).values.squeeze()[indices_labels_healthy_controls]

counts_samples = np.unique(labels_orig_hc, return_counts=True)[1]

labels = np.concatenate([np.linspace(ids[i], ids[i], counts_samples[i]) for i in range(ids.shape[0])]).astype(np.int64)

# Get other metadata on a per-trial basis...
# Initialise variables:
session_nos = np.zeros_like(session_nos_orig)
speeds = np.array([''] * session_nos_orig.shape[0])
footwear = np.zeros_like(session_nos_orig)
sexes = np.zeros_like(session_nos_orig).astype(np.float64)
ages = np.zeros_like(session_nos_orig).astype(np.float64)
heights = np.zeros_like(session_nos_orig).astype(np.float64)
masses = np.zeros_like(session_nos_orig).astype(np.float64)

height_by_id = []
mass_by_id = []

counts_bf_sessions_by_id = []
counts_shod_sessions_by_id = []
counts_samples_by_session_all = []
counts_sessions_max = np.max(counts_sessions_by_id)

# For each ID:
for i in range(indices_sessions_where_id_changes.shape[0]):

    # Get their subset of metadata from each session:
    if i == 0:

        session_nos_subset = session_nos_orig_by_session[:indices_sessions_where_id_changes[i]]
        speeds_subset = speeds_orig_by_session[:indices_sessions_where_id_changes[i]]
        footwear_subset = footwear_by_session[:indices_sessions_where_id_changes[i]]
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

        speeds_subset = speeds_orig_by_session[indices_sessions_where_id_changes[i - 1]:
                                               indices_sessions_where_id_changes[i]]

        footwear_subset = footwear_by_session[indices_sessions_where_id_changes[i - 1]:
                                              indices_sessions_where_id_changes[i]]

        session_dates_subset = session_dates[indices_sessions_where_id_changes[i - 1]:
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

    # Get counts of bf and shod sessions and append to list:
    counts_bf_sessions_by_id.append(np.shape(np.asarray(footwear_subset == 0).nonzero()[0]))
    counts_shod_sessions_by_id.append(np.shape(np.asarray(footwear_subset == 1).nonzero()[0]))

    # Get indices that would sort session numbers in ascending order:
    indices_sort_session_nos_subset = np.argsort(session_nos_subset)

    # Sort session numbers, speeds, and footwear based on session order:
    session_nos_subset_sorted = session_nos_subset[indices_sort_session_nos_subset]
    speeds_subset_sorted = speeds_subset[indices_sort_session_nos_subset]
    footwear_subset_sorted = footwear_subset[indices_sort_session_nos_subset]

    counts_samples_by_session = np.zeros((counts_sessions_max,))

    # For each session:
    for idx, session_no in enumerate(session_nos_subset_sorted):

        # Get the indices of samples from the session:
        indices_samples = np.asarray(session_nos_orig == session_no).nonzero()[0]

        # Make new single digit session numbers:
        session_nos[indices_samples] = idx + 1

        counts_samples_by_session[idx] = indices_samples.shape[0]

        # Make speed labels instead of numbers:
        if speeds_subset_sorted[idx] == 1:

            speeds[indices_samples] = 'S'

        elif speeds_subset_sorted[idx] == 2:

            speeds[indices_samples] = 'P'

        else: # 3

            speeds[indices_samples] = 'F'

        footwear[indices_samples] = footwear_subset_sorted[idx]
        sexes[indices_samples] = sex_subset[idx]
        ages[indices_samples] = age_subset[idx]
        heights[indices_samples] = height_subset[idx]
        masses[indices_samples] = mass_subset[idx]

    counts_samples_by_session_all.append(counts_samples_by_session)

height_by_id = np.array(height_by_id)
mass_by_id = np.array(mass_by_id)

# Create an arr of unique trial identifiers that each contain ID label, session no., trial no. and speed category:
trial_names = np.array(['_'.join(['GR', format(labels[i], '04d'), format(session_nos[i], '02d'),
                                  format(trial_nos_orig[i], '02d'), speeds[i]])
                        for i in range(labels.shape[0])])

# Get metadata for shod subset:
indices_shod = np.asarray(footwear == 1).nonzero()[0]
labels_shod = labels[indices_shod]
session_nos_shod = session_nos[indices_shod]
footwear_shod = np.ones_like(labels_shod)
trial_nos_orig_shod = trial_nos_orig[indices_shod]
speeds_shod = speeds[indices_shod]
sexes_shod = sexes[indices_shod]
ages_shod = ages[indices_shod]
heights_shod = heights[indices_shod]
masses_shod = masses[indices_shod]

ids_shod, counts_samples_shod = np.unique(labels_shod, return_counts=True)
indices_samples_where_id_changes_shod = np.unique(labels_shod, return_index=True)[1][1:]

# Convert session numbers from absolute (over whole ds) to relative (over just shod subset):
le = preprocessing.LabelEncoder()
session_nos_by_id_shod_temp = np.split(session_nos_shod, indices_samples_where_id_changes_shod)
session_nos_by_id_shod = [le.fit_transform(item) + 1 for item in session_nos_by_id_shod_temp]
session_nos_relative_shod = np.concatenate(session_nos_by_id_shod)

# Create an arr of unique trial identifiers that each contain ID label, session no., trial no. and speed category:
trial_names_shod = np.array(['_'.join(['GR', format(labels_shod[i], '04d'), format(session_nos_relative_shod[i], '02d'),
                                       format(trial_nos_orig_shod[i], '02d'), speeds_shod[i]])
                             for i in range(labels_shod.shape[0])])

height_by_id_shod = []
mass_by_id_shod = []

for idx, height in enumerate(height_by_id):

    if ids[idx] in ids_shod:

        height_by_id_shod.append(height)
        mass_by_id_shod.append(mass_by_id[idx])

height_by_id_shod = np.array(height_by_id_shod)
mass_by_id_shod = np.array(mass_by_id_shod)

# Save metadata so that it can be loaded in other scripts:
np.save('./Datasets/gr-all/objects/ids.npy', ids)
np.save('./Datasets/gr-all/objects/counts_samples.npy', counts_samples)
np.save('./Datasets/gr-all/objects/labels.npy', labels)
np.save('./Datasets/gr-all/objects/session_nos.npy', session_nos)
np.save('./Datasets/gr-all/objects/trial_nos.npy', trial_nos_orig)
np.save('./Datasets/gr-all/objects/speeds.npy', speeds)
np.save('./Datasets/gr-all/objects/footwear.npy', footwear)
np.save('./Datasets/gr-all/objects/sexes.npy', sexes)
np.save('./Datasets/gr-all/objects/ages.npy', ages)
np.save('./Datasets/gr-all/objects/heights.npy', heights)
np.save('./Datasets/gr-all/objects/masses.npy', masses)
np.save('./Datasets/gr-all/objects/heights_by_id.npy', height_by_id)
np.save('./Datasets/gr-all/objects/masses_by_id.npy', mass_by_id)
np.save('./Datasets/gr-all/objects/trial_names.npy', trial_names)

np.save('./Datasets/gr-sho/objects/indices.npy', indices_shod)
np.save('./Datasets/gr-sho/objects/ids.npy', ids_shod)
np.save('./Datasets/gr-sho/objects/counts_samples.npy', counts_samples_shod)
np.save('./Datasets/gr-sho/objects/labels.npy', labels_shod)
np.save('./Datasets/gr-sho/objects/session_nos.npy', session_nos_shod)
np.save('./Datasets/gr-sho/objects/trial_nos.npy', trial_nos_orig_shod)
np.save('./Datasets/gr-sho/objects/speeds.npy', speeds_shod)
np.save('./Datasets/gr-sho/objects/sexes.npy', sexes_shod)
np.save('./Datasets/gr-sho/objects/ages.npy', ages_shod)
np.save('./Datasets/gr-sho/objects/heights.npy', heights_shod)
np.save('./Datasets/gr-sho/objects/masses.npy', masses_shod)
np.save('./Datasets/gr-sho/objects/heights_by_id.npy', height_by_id_shod)
np.save('./Datasets/gr-sho/objects/masses_by_id.npy', mass_by_id_shod)
np.save('./Datasets/gr-sho/objects/footwear.npy', footwear_shod)
np.save('./Datasets/gr-sho/objects/trial_names.npy', trial_names_shod)

# Also save indices of samples from barefoot walking for later use:
indices_bf = np.asarray(footwear == 0).nonzero()[0]

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
fx_r_raw = np.expand_dims(pd.read_csv('./Datasets/gr-all/spreadsheets/GRF_F_ML_RAW_right.csv')
                      .fillna(0).values[indices_labels_healthy_controls, 3:], axis=1)
fy_r_raw = np.expand_dims(pd.read_csv('./Datasets/gr-all/spreadsheets/GRF_F_AP_RAW_right.csv')
                      .fillna(0).values[indices_labels_healthy_controls, 3:], axis=1)
fz_r_raw = np.expand_dims(pd.read_csv('./Datasets/gr-all/spreadsheets/GRF_F_V_RAW_right.csv')
                      .fillna(0).values[indices_labels_healthy_controls, 3:], axis=1)
cx_r_raw = np.expand_dims(pd.read_csv('./Datasets/gr-all/spreadsheets/GRF_COP_ML_RAW_right.csv')
                      .fillna(0).values[indices_labels_healthy_controls, 3:], axis=1)
cy_r_raw = np.expand_dims(pd.read_csv('./Datasets/gr-all/spreadsheets/GRF_COP_AP_RAW_right.csv')
                      .fillna(0).values[indices_labels_healthy_controls, 3:], axis=1)
fx_l_raw = np.expand_dims(pd.read_csv('./Datasets/gr-all/spreadsheets/GRF_F_ML_RAW_left.csv')
                      .fillna(0).values[indices_labels_healthy_controls, 3:], axis=1)
fy_l_raw = np.expand_dims(pd.read_csv('./Datasets/gr-all/spreadsheets/GRF_F_AP_RAW_left.csv')
                      .fillna(0).values[indices_labels_healthy_controls, 3:], axis=1)
fz_l_raw = np.expand_dims(pd.read_csv('./Datasets/gr-all/spreadsheets/GRF_F_V_RAW_left.csv')
                      .fillna(0).values[indices_labels_healthy_controls, 3:], axis=1)
cx_l_raw = np.expand_dims(pd.read_csv('./Datasets/gr-all/spreadsheets/GRF_COP_ML_RAW_left.csv')
                      .fillna(0).values[indices_labels_healthy_controls, 3:], axis=1)
cy_l_raw = np.expand_dims(pd.read_csv('./Datasets/gr-all/spreadsheets/GRF_COP_AP_RAW_left.csv')
                      .fillna(0).values[indices_labels_healthy_controls, 3:], axis=1)

# Concatenate directional components from each stance side along the channel axis:
sigs_all_r_raw = np.concatenate((fx_r_raw, fy_r_raw, fz_r_raw, cx_r_raw, cy_r_raw), axis=1)
sigs_all_l_raw = np.concatenate((fx_l_raw, fy_l_raw, fz_l_raw, cx_l_raw, cy_l_raw), axis=1)

# If running script for the first time, we recommend saving all raw measures from each stance side:
np.save('./Datasets/gr-all/objects/sigs_all_r_raw.npy', sigs_all_r_raw)
np.save('./Datasets/gr-all/objects/sigs_all_l_raw.npy', sigs_all_l_raw)
# That way, they can be loaded efficiently on subsequent runs of the script. The lines below can be uncommented and...
# the lines above that read raw measures from csv files can be commented out:
# sigs_all_r_raw = np.load('./Datasets/gr-all/objects/sigs_all_r_raw.npy', allow_pickle=True)
# sigs_all_l_raw = np.load('./Datasets/gr-all/objects/sigs_all_l_raw.npy', allow_pickle=True)

# Initialise variables required to trim, filter, and time normalise measures from each stance side:
fz_thresh = 50  # Newtons. Vertical GRF magnitude threshold for defining the stance phase.
interp_len = 100 # frames. The no. frames that the measures will be time normalised to via linear interpolation.
cutoff_freq = 30  # Hz. Cut-off frequency for Butterworth low-pass filter.
sampling_rate = 250  # Hz. Original sampling rate/frequency.

# Trim, filter, and time normalise measures from each stance side (see function definition for details on ds param...
# 'not_fi-all' means any dataset other than ForceID-A):
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
np.save('./Datasets/gr-all/objects/sigs_all_r_pro.npy', sigs_all_r_pro)
np.save('./Datasets/gr-all/objects/sigs_all_l_pro.npy', sigs_all_l_pro)

