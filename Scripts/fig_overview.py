import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

"""This script was used to create the subplots that are included in the study overview figure for 'Dataset A' and 
'Dataset B'. Because the figure is a fictitious illustration, the signals in the subplots were sourced from the same
dataset (private version of ForceID-A)."""

plot_dsA = True # If False, generates subplots for 'dsB' instead.

labels = pd.read_csv('./Datasets/fi-all/spreadsheets/Cx_FP1_raw.csv', usecols=[1]).values.squeeze()
counts_samples = np.unique(labels, return_counts=True)[1]
indices_samples_where_id_changes = np.cumsum(counts_samples)

# Load all trimmed, filtered, and time normalised measures from each stance side as a single object:
sigs_all_r_pro = np.load('./Datasets/fi-all/objects/sigs_all_r_pro.npy', allow_pickle=True)
sigs_all_l_pro = np.load('./Datasets/fi-all/objects/sigs_all_l_pro.npy', allow_pickle=True)

indices_ids_dsA = [29, 108, 125, 151]
indices_ids_dsB = [32, 64, 115, 111]

indices_samples_dsA = [np.arange(indices_samples_where_id_changes[idx - 1], indices_samples_where_id_changes[idx])
                       for idx in indices_ids_dsA]
indices_samples_dsB = [np.arange(indices_samples_where_id_changes[idx - 1], indices_samples_where_id_changes[idx])
                       for idx in indices_ids_dsB]

plt.rcParams["font.family"] = "Times New Roman"
mpl.rcParams['mathtext.default'] = 'regular'

colors = ['#006BA4', '#FF800E', '#ABABAB', '#595959']

fig = plt.figure(tight_layout=True)
gs = fig.add_gridspec(1, 2)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

for i in range(len(indices_samples_dsA)):

    if plot_dsA:

        [ax1.plot(sigs_all_r_pro[k, 2, :], lw=1, c=colors[i], alpha=0.3) for k in indices_samples_dsA[i]]
        [ax2.plot(sigs_all_l_pro[k, 2, :], lw=1, c=colors[i], alpha=0.3) for k in indices_samples_dsA[i]]

    else:

        [ax1.plot(sigs_all_r_pro[k, 2, :], lw=1, c=colors[i], alpha=0.3) for k in indices_samples_dsB[i]]
        [ax2.plot(sigs_all_l_pro[k, 2, :], lw=1, c=colors[i], alpha=0.3) for k in indices_samples_dsB[i]]

ax1.set_xlim(0, 100)
ax2.set_xlim(0, 100)

ax1.set_ylim(0, 1650)
ax2.set_ylim(0, 1650)

ax1.set_xticks([])
ax2.set_xticks([])
ax1.set_xticklabels([])
ax2.set_xticklabels([])

ax1.set_yticks([])
ax2.set_yticks([])
ax1.set_yticklabels([])
ax2.set_yticklabels([])

if plot_dsA:

    plt.savefig('./Figures/fig_overview_dsA.pdf', dpi=1200)

else:

    plt.savefig('./Figures/fig_overview_dsB.pdf', dpi=1200)

plt.show()
