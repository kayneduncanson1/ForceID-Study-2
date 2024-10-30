import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PrePro import trim_filt_norm
from Utils import load_channel_raw

"""This script was used to characterise and prepare the private version of ForceID-A. It first reads metadata and
transforms it to a suitable format for use in other scripts (e.g., main.py). It also reads, trims, filters, and time
normalises raw measures. The final two pre-processing steps (standardising measures via z-score normalisation and
combining measures from left and right stance into a single input representation) are conducted in the run_expt function
that is called in main.py."""

# Convert data files from xlsx to csv format so that they can be read more efficiently. This section can be commented
# out after first use...
# Key
# ---
# fx = GRF x (medio-lateral) direction
# fy = GRF y (antero-posterior) direction
# fz = GRF z (vertical) direction
# cx = COP x direction
# cy = COP y direction
# fp1 = force platform 1
# fp2 = force platform 2
# ---
metadata = pd.read_excel('./Datasets/fi-all/spreadsheets/Metadata.xlsx')
fx_fp1_raw = pd.read_excel('./Datasets/fi-all/spreadsheets/Fx_FP1_raw.xlsx')
fy_fp1_raw = pd.read_excel('./Datasets/fi-all/spreadsheets/Fy_FP1_raw.xlsx')
fz_fp1_raw = pd.read_excel('./Datasets/fi-all/spreadsheets/Fz_FP1_raw.xlsx')
cx_fp1_raw = pd.read_excel('./Datasets/fi-all/spreadsheets/Cx_FP1_raw.xlsx')
cy_fp1_raw = pd.read_excel('./Datasets/fi-all/spreadsheets/Cy_FP1_raw.xlsx')
fx_fp2_raw = pd.read_excel('./Datasets/fi-all/spreadsheets/Fx_FP2_raw.xlsx')
fy_fp2_raw = pd.read_excel('./Datasets/fi-all/spreadsheets/Fy_FP2_raw.xlsx')
fz_fp2_raw = pd.read_excel('./Datasets/fi-all/spreadsheets/Fz_FP2_raw.xlsx')
cx_fp2_raw = pd.read_excel('./Datasets/fi-all/spreadsheets/Cx_FP2_raw.xlsx')
cy_fp2_raw = pd.read_excel('./Datasets/fi-all/spreadsheets/Cy_FP2_raw.xlsx')

metadata.to_csv('./Datasets/fi-all/spreadsheets/Metadata.csv')
fx_fp1_raw.to_csv('./Datasets/fi-all/spreadsheets/Fx_FP1_raw.csv')
fy_fp1_raw.to_csv('./Datasets/fi-all/spreadsheets/Fy_FP1_raw.csv')
fz_fp1_raw.to_csv('./Datasets/fi-all/spreadsheets/Fz_FP1_raw.csv')
cx_fp1_raw.to_csv('./Datasets/fi-all/spreadsheets/Cx_FP1_raw.csv')
cy_fp1_raw.to_csv('./Datasets/fi-all/spreadsheets/Cy_FP1_raw.csv')
fx_fp2_raw.to_csv('./Datasets/fi-all/spreadsheets/Fx_FP2_raw.csv')
fy_fp2_raw.to_csv('./Datasets/fi-all/spreadsheets/Fy_FP2_raw.csv')
fz_fp2_raw.to_csv('./Datasets/fi-all/spreadsheets/Fz_FP2_raw.csv')
cx_fp2_raw.to_csv('./Datasets/fi-all/spreadsheets/Cx_FP2_raw.csv')
cy_fp2_raw.to_csv('./Datasets/fi-all/spreadsheets/Cy_FP2_raw.csv')

# Get arr of unique ID labels:
ids = pd.read_csv('./Datasets/fi-all/spreadsheets/Metadata.csv', usecols=[1]).values.squeeze()
# Get other metadata on a per-ID basis:
age_by_id = pd.read_csv('./Datasets/fi-all/spreadsheets/Metadata.csv', usecols=[2]).values.squeeze()
sex_by_id = pd.read_csv('./Datasets/fi-all/spreadsheets/Metadata.csv', usecols=[4]).values.squeeze()
# All participants did two sessions. Get mass, height, and footwear categories from each session:
mass_by_id_s1 = pd.read_csv('./Datasets/fi-all/spreadsheets/Metadata.csv', usecols=[5]).values
mass_by_id_s2 = pd.read_csv('./Datasets/fi-all/spreadsheets/Metadata.csv', usecols=[6]).values
height_by_id_s1 = pd.read_csv('./Datasets/fi-all/spreadsheets/Metadata.csv', usecols=[7]).values
height_by_id_s2 = pd.read_csv('./Datasets/fi-all/spreadsheets/Metadata.csv', usecols=[8]).values
footwear_category_by_id_s1 = pd.read_csv('./Datasets/fi-all/spreadsheets/Metadata.csv', usecols=[9]).values.squeeze()
footwear_category_by_id_s2 = pd.read_csv('./Datasets/fi-all/spreadsheets/Metadata.csv', usecols=[10]).values.squeeze()

# Convert sexes from string to int:
indices_females = np.asarray(sex_by_id == 'F').nonzero()[0]
indices_males = np.asarray(sex_by_id == 'M').nonzero()[0]
sex_by_id[indices_females] = 0
sex_by_id[indices_males] = 1

# Get mean mass and height across sessions and round them:
height_by_id = np.round(np.mean(np.concatenate((height_by_id_s1, height_by_id_s2), axis=1), axis=1), decimals=2)
mass_by_id = np.round(np.mean(np.concatenate((mass_by_id_s1, mass_by_id_s2), axis=1), axis=1), decimals=1)

# Some metadata is already available on a per-trial basis, so read it in:
labels = pd.read_csv('./Datasets/fi-all/spreadsheets/Fx_FP1_raw.csv', usecols=[1]).values.squeeze()
session_nos = pd.read_csv('./Datasets/fi-all/spreadsheets/Fx_FP1_raw.csv', usecols=[2]).values.squeeze()
trial_nos = pd.read_csv('./Datasets/fi-all/spreadsheets/Fx_FP1_raw.csv', usecols=[3]).values.squeeze()
speeds = pd.read_csv('./Datasets/fi-all/spreadsheets/Fx_FP1_raw.csv', usecols=[4]).values.squeeze()
stance_sides_fp1 = pd.read_csv('./Datasets/fi-all/spreadsheets/Fx_FP1_raw.csv', usecols=[5])\
    .values.squeeze()
stance_sides_fp2 = pd.read_csv('./Datasets/fi-all/spreadsheets/Fx_FP2_raw.csv', usecols=[5])\
    .values.squeeze()

counts_samples = np.unique(labels, return_counts=True)[1]

# Get remaining metadata on a per-trial basis:
sexes = []
ages = []
heights = []
masses = []
footwear_categories_s1 = []
footwear_categories_s2 = []

for i in range(ids.shape[0]):

    sexes.append(np.linspace(sex_by_id[i], sex_by_id[i], counts_samples[i]))
    ages.append(np.linspace(age_by_id[i], age_by_id[i], counts_samples[i]))
    heights.append(np.linspace(height_by_id[i], height_by_id[i], counts_samples[i]))
    masses.append(np.linspace(mass_by_id[i], mass_by_id[i], counts_samples[i]))
    footwear_categories_s1.append([footwear_category_by_id_s1[i]] * counts_samples[i])
    footwear_categories_s2.append([footwear_category_by_id_s2[i]] * counts_samples[i])

sexes = np.concatenate(sexes).astype(np.float64)
ages = np.concatenate(ages).astype(np.float64)
heights = np.concatenate(heights).astype(np.float64)
masses = np.concatenate(masses).astype(np.float64)
footwear_categories_s1 = np.concatenate(footwear_categories_s1)
footwear_categories_s2 = np.concatenate(footwear_categories_s2)
footwear = np.ones_like(labels) # 1 = shod (personal footwear).

# Create an arr of unique trial identifiers that each contain ID label, session no., trial no. and speed category:
trial_names = np.array(['_'.join(['FI', format(labels[i], '04d'), format(session_nos[i], '02d'),
                                  format(trial_nos[i], '02d'), speeds[i]]) for i in range(labels.shape[0])])

# Save metadata so that it can be loaded in other scripts:
np.save('./Datasets/fi-all/objects/ids.npy', ids)
np.save('./Datasets/fi-all/objects/counts_samples.npy', counts_samples)
np.save('./Datasets/fi-all/objects/labels.npy', labels)
np.save('./Datasets/fi-all/objects/session_nos.npy', session_nos)
np.save('./Datasets/fi-all/objects/trial_nos.npy', trial_nos)
np.save('./Datasets/fi-all/objects/speeds.npy', speeds)
np.save('./Datasets/fi-all/objects/footwear.npy', footwear)
np.save('./Datasets/fi-all/objects/sexes.npy', sexes)
np.save('./Datasets/fi-all/objects/ages.npy', ages)
np.save('./Datasets/fi-all/objects/heights.npy', heights)
np.save('./Datasets/fi-all/objects/masses.npy', masses)
np.save('./Datasets/fi-all/objects/heights_by_id.npy', height_by_id)
np.save('./Datasets/fi-all/objects/masses_by_id.npy', mass_by_id)
np.save('./Datasets/fi-all/objects/trial_names.npy', trial_names)


# Read raw force platform measures from csv files and convert to shape (N, C, L), where...
# N = no. samples
# C = no. channels (directional components)
# L = length i.e., no frames.
fx_fp1_raw = np.expand_dims(pd.read_csv('./Datasets/fi-all/spreadsheets/Fx_FP1_raw.csv')
                            .fillna(0).values[:, 6:], axis=1)
fy_fp1_raw = np.expand_dims(pd.read_csv('./Datasets/fi-all/spreadsheets/Fy_FP1_raw.csv')
                            .fillna(0).values[:, 6:], axis=1)
fz_fp1_raw = np.expand_dims(pd.read_csv('./Datasets/fi-all/spreadsheets/Fz_FP1_raw.csv')
                            .fillna(0).values[:, 6:], axis=1)
cx_fp1_raw = np.expand_dims(pd.read_csv('./Datasets/fi-all/spreadsheets/Cx_FP1_raw.csv')
                            .fillna(0).values[:, 6:], axis=1)
cy_fp1_raw = np.expand_dims(pd.read_csv('./Datasets/fi-all/spreadsheets/Cy_FP1_raw.csv')
                            .fillna(0).values[:, 6:], axis=1)
fx_fp2_raw = np.expand_dims(pd.read_csv('./Datasets/fi-all/spreadsheets/Fx_FP2_raw.csv')
                            .fillna(0).values[:, 6:], axis=1)
fy_fp2_raw = np.expand_dims(pd.read_csv('./Datasets/fi-all/spreadsheets/Fy_FP2_raw.csv')
                            .fillna(0).values[:, 6:], axis=1)
fz_fp2_raw = np.expand_dims(pd.read_csv('./Datasets/fi-all/spreadsheets/Fz_FP2_raw.csv')
                            .fillna(0).values[:, 6:], axis=1)
cx_fp2_raw = np.expand_dims(pd.read_csv('./Datasets/fi-all/spreadsheets/Cx_FP2_raw.csv')
                            .fillna(0).values[:, 6:], axis=1)
cy_fp2_raw = np.expand_dims(pd.read_csv('./Datasets/fi-all/spreadsheets/Cy_FP2_raw.csv')
                            .fillna(0).values[:, 6:], axis=1)

# Concatenate directional components from each force platform along the channel axis:
sigs_all_fp1_raw = np.concatenate((fx_fp1_raw, fy_fp1_raw, fz_fp1_raw, cx_fp1_raw, cy_fp1_raw), axis=1)
sigs_all_fp2_raw = np.concatenate((fx_fp2_raw, fy_fp2_raw, fz_fp2_raw, cx_fp2_raw, cy_fp2_raw), axis=1)

# Pad the objects to a fixed length of 3500:
sigs_all_fp1_pad = np.pad(sigs_all_fp1_raw, ((0, 0), (0, 0), (0, 3500 - sigs_all_fp1_raw.shape[2])))
sigs_all_fp2_pad = np.pad(sigs_all_fp2_raw, ((0, 0), (0, 0), (0, 3500 - sigs_all_fp2_raw.shape[2])))

# Assign measures to stance sides (r = right, l = left):
sigs_all_r_raw = np.array([sigs_all_fp1_pad[i] if stance_sides_fp1[i] == 'R' else sigs_all_fp2_pad[i]
                           for i in range(stance_sides_fp1.shape[0])], dtype=np.float64)
sigs_all_l_raw = np.array([sigs_all_fp1_pad[i] if stance_sides_fp1[i] == 'L' else sigs_all_fp2_pad[i]
                           for i in range(stance_sides_fp1.shape[0])], dtype=np.float64)

# Set variables that are no longer required to None for efficiency:
sigs_all_fp1_raw = None
sigs_all_fp2_raw = None
sigs_all_fp1_pad = None
sigs_all_fp2_pad = None

# Flip Fx and Cx such that their orientation is the same as in other datasets:
sigs_all_r_raw[:, 0, :] = sigs_all_r_raw[:, 0, :] * -1
sigs_all_r_raw[:, 3, :] = sigs_all_r_raw[:, 3, :] * -1
sigs_all_l_raw[:, 0, :] = sigs_all_l_raw[:, 0, :] * -1
sigs_all_l_raw[:, 3, :] = sigs_all_l_raw[:, 3, :] * -1

# Convert COP units from mm to m:
sigs_all_r_raw[:, 3:, :] = sigs_all_r_raw[:, 3:, :] / 1000
sigs_all_l_raw[:, 3:, :] = sigs_all_l_raw[:, 3:, :] / 1000

# If running script for the first time, save all raw measures from each stance side:
np.save('./Datasets/fi-all/objects/sigs_all_r_raw.npy', sigs_all_r_raw)
np.save('./Datasets/fi-all/objects/sigs_all_l_raw.npy', sigs_all_l_raw)
# That way, they can be loaded efficiently on subsequent runs of the script. The lines below can be uncommented and...
# the lines above that read raw measures from csv files can be commented out:
# sigs_all_r_raw = np.load('./Datasets/fi-all/objects/sigs_all_r_raw.npy', allow_pickle=True)
# sigs_all_l_raw = np.load('./Datasets/fi-all/objects/sigs_all_l_raw.npy', allow_pickle=True)

# Initialise variables required to trim, filter, and time normalise measures from each stance side:
fz_thresh = 50  # Newtons. Vertical GRF magnitude threshold for defining the stance phase.
interp_len = 100 # frames. The no. frames that the measures will be time normalised to via linear interpolation.
cutoff_freq = 30  # Hz. Cut-off frequency for Butterworth low-pass filter.
sampling_rate = 2000  # Hz. Original sampling rate/frequency.

sigs_all_r_pro = trim_filt_norm(sigs_all_r_raw, fz_thresh, cutoff_freq, sampling_rate, interp_len, ds='fi-all')
sigs_all_l_pro = trim_filt_norm(sigs_all_l_raw, fz_thresh, cutoff_freq, sampling_rate, interp_len, ds='fi-all')

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
np.save('./Datasets/fi-all/objects/sigs_all_r_pro.npy', sigs_all_r_pro)
np.save('./Datasets/fi-all/objects/sigs_all_l_pro.npy', sigs_all_l_pro)


