import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from Utils import get_full_metadata, load_list

"""This script was used to create the figure of distributions in demographic attributes (age, sex, height, and mass)."""

# Specify whether to plot the distributions of demographic attributes in the balanced version of each dataset or the...
# original (unbalanced) version:
plot_bal_datasets = True

# Get metadata for each dataset on a per-trial (i.e., sample) basis:
trial_names_fi, labels_fi, ids_fi, counts_samples_fi, sexes_fi, ages_fi, heights_fi, masses_fi, footwear_fi, speeds_fi\
    = get_full_metadata('fi-all')
trial_names_gr, labels_gr, ids_gr, counts_samples_gr, sexes_gr, ages_gr, heights_gr, masses_gr, footwear_gr, speeds_gr\
    = get_full_metadata('gr-sho')
trial_names_gb, labels_gb, ids_gb, counts_samples_gb, sexes_gb, ages_gb, heights_gb, masses_gb, footwear_gb, speeds_gb\
    = get_full_metadata('gb-all')
trial_names_ai, labels_ai, ids_ai, counts_samples_ai, sexes_ai, ages_ai, heights_ai, masses_ai, footwear_ai, speeds_ai\
    = get_full_metadata('ai-all')

# Get indices of samples where the ID changes:
indices_samples_where_id_changes_fi = np.cumsum(counts_samples_fi)
indices_samples_where_id_changes_gr = np.cumsum(counts_samples_gr)
indices_samples_where_id_changes_gb = np.cumsum(counts_samples_gb)
indices_samples_where_id_changes_ai = np.cumsum(counts_samples_ai)

# Get the first age and sex measure for each individual (representing the session one measure, as samples are sorted...
# by session in ascending order):
ages_fi_s1_temp = []
ages_gr_s1_temp = []
ages_gb_s1_temp = []
ages_ai_s1_temp = []

sexes_fi_s1_temp = []
sexes_gr_s1_temp = []
sexes_gb_s1_temp = []
sexes_ai_s1_temp = []

for idx in indices_samples_where_id_changes_fi[:-1]:

    ages_fi_s1_temp.append(ages_fi[idx])
    ages_gr_s1_temp.append(ages_gr[idx])
    ages_gb_s1_temp.append(ages_gb[idx])
    ages_ai_s1_temp.append(ages_ai[idx])

    sexes_fi_s1_temp.append(sexes_fi[idx])
    sexes_gr_s1_temp.append(sexes_gr[idx])
    sexes_gb_s1_temp.append(sexes_gb[idx])
    sexes_ai_s1_temp.append(sexes_ai[idx])

ages_fi_s1 = [ages_fi[0]] + ages_fi_s1_temp
ages_gr_s1 = [ages_gr[0]] + ages_gr_s1_temp
ages_gb_s1 = [ages_gb[0]] + ages_gb_s1_temp
ages_ai_s1 = [ages_ai[0]] + ages_ai_s1_temp
sexes_fi_s1 = [sexes_fi[0]] + sexes_fi_s1_temp
sexes_gr_s1 = [sexes_gr[0]] + sexes_gr_s1_temp
sexes_gb_s1 = [sexes_gb[0]] + sexes_gb_s1_temp
sexes_ai_s1 = [sexes_ai[0]] + sexes_ai_s1_temp

# Get mean mass and height across sessions for each ID:
heights_means_fi = np.load('./Datasets/fi-all/objects/heights_by_id.npy', allow_pickle=True)
heights_means_gr = np.load('./Datasets/gr-sho/objects/heights_by_id.npy', allow_pickle=True)
heights_means_gb = np.load('./Datasets/gb-all/objects/heights_by_id.npy', allow_pickle=True)
heights_means_ai = np.load('./Datasets/ai-all/objects/heights_by_id.npy', allow_pickle=True)

masses_means_fi = np.load('./Datasets/fi-all/objects/masses_by_id.npy', allow_pickle=True)
masses_means_gr = np.load('./Datasets/gr-sho/objects/masses_by_id.npy', allow_pickle=True)
masses_means_gb = np.load('./Datasets/gb-all/objects/masses_by_id.npy', allow_pickle=True)
masses_means_ai = np.load('./Datasets/ai-all/objects/masses_by_id.npy', allow_pickle=True)

trial_names_unbal = np.concatenate((trial_names_fi, trial_names_gr, trial_names_gb, trial_names_ai))
ages_unbal = np.concatenate((ages_fi_s1, ages_gr_s1, ages_gb_s1, ages_ai_s1))
sexes_unbal = np.concatenate((sexes_fi_s1, sexes_gr_s1, sexes_gb_s1, sexes_ai_s1))
heights_unbal = np.concatenate((heights_means_fi, heights_means_gr, heights_means_gb, heights_means_ai))
masses_unbal = np.concatenate((masses_means_fi, masses_means_gr, masses_means_gb, masses_means_ai))


# Get metadata for the balanced version of each dataset...
# First, get the trial names for the balanced datasets:
results_path = './Results/main'
trial_names_bal_fi = np.concatenate(load_list(os.path.join(results_path, 'trial_names_by_id_shuff_fi-all.txt')))
trial_names_bal_gr = np.concatenate(load_list(os.path.join(results_path, 'trial_names_by_id_shuff_gr-sho.txt')))
trial_names_bal_gb = np.concatenate(load_list(os.path.join(results_path, 'trial_names_by_id_shuff_gb-all.txt')))
trial_names_bal_ai = np.concatenate(load_list(os.path.join(results_path, 'trial_names_by_id_shuff_ai-all.txt')))

trial_names_bal = np.concatenate((trial_names_bal_fi,
                                  trial_names_bal_gr,
                                  trial_names_bal_gb,
                                  trial_names_bal_ai))

# Get the indices of the trial names for the balanced datasets in the trial names for the unbalanced datasets:
indices_trial_names_bal = np.concatenate([np.asarray(trial_names_unbal == name).nonzero()[0]
                                          for name in trial_names_bal])

# Use these indices to get the demographic measures for the balanced datasets and then take every 10th value to get the
# measures on a per-ID basis (there were 10 samples per ID):
ages_bal = ages_unbal[indices_trial_names_bal][::10]
sexes_bal = sexes_unbal[indices_trial_names_bal][::10]
heights_bal = heights_unbal[indices_trial_names_bal][::10]
masses_bal = masses_unbal[indices_trial_names_bal][::10]


# Generate a figure showing the distributions of each demographic attribute in each dataset...
# Set overall fig params:
plt.rcParams["font.family"] = "Times New Roman"
plt.style.use('tableau-colorblind10')
fig = plt.figure(constrained_layout=True, figsize=(7.0, 4.85))
gs = fig.add_gridspec(2, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

# Set params for bar chart of sex distribution:
x = np.arange(4)  # the label locations
width = 0.2  # the width of the bars

# Create Kernel Density Estimation (KDE) plots for age, mass, and height as well as a bar chart for sex (exclude nans):
if plot_bal_datasets:

    # Because each balanced ds contains 185 IDs, slices of size 185 were indexed via hard-coding to get measures...
    # from each ds:
    sns.kdeplot(ages_bal[555:][~np.isnan(ages_bal[555:])], ax=ax1, label='AIST')
    sns.kdeplot(ages_bal[370:555][~np.isnan(ages_bal[370:555])], ax=ax1, label='Gutenberg')
    sns.kdeplot(ages_bal[185:370], ax=ax1, label='GaitRec')
    sns.kdeplot(ages_bal[:185], ax=ax1, label='ForceID-A')

    sns.kdeplot(heights_bal[555:], ax=ax2, label='AIST')
    sns.kdeplot(heights_bal[370:555][~np.isnan(heights_bal[370:555])], ax=ax2, label='Gutenberg')
    sns.kdeplot(heights_bal[185:370], ax=ax2, label='GaitRec')
    sns.kdeplot(heights_bal[:185], ax=ax2, label='ForceID-A')

    sns.kdeplot(masses_bal[555:], ax=ax3, label='AIST')
    sns.kdeplot(masses_bal[370:555], ax=ax3, label='Gutenberg')
    sns.kdeplot(masses_bal[185:370], ax=ax3, label='GaitRec')
    sns.kdeplot(masses_bal[:185], ax=ax3, label='ForceID-A')

    count_female_bal = [np.asarray(sexes_bal[555:] == 0).nonzero()[0].shape[0],
                        np.asarray(sexes_bal[370:555] == 0).nonzero()[0].shape[0],
                        np.asarray(sexes_bal[185:370] == 0).nonzero()[0].shape[0],
                        np.asarray(sexes_bal[:185] == 0).nonzero()[0].shape[0]]
    count_male_bal = [np.asarray(sexes_bal[555:] == 1).nonzero()[0].shape[0],
                      np.asarray(sexes_bal[370:555] == 1).nonzero()[0].shape[0],
                      np.asarray(sexes_bal[185:370] == 1).nonzero()[0].shape[0],
                      np.asarray(sexes_bal[:185] == 1).nonzero()[0].shape[0]]

    rects1 = ax4.bar(x - (width / 2), count_female_bal, width, label='Female')
    rects2 = ax4.bar(x + (width / 2), count_male_bal, width, label='Male')

else:

    sns.kdeplot(ages_ai_s1[~np.isnan(ages_ai_s1)], ax=ax1, label='AIST')
    sns.kdeplot(ages_gb_s1[~np.isnan(ages_gb_s1)], ax=ax1, label='Gutenberg')
    sns.kdeplot(ages_gr_s1, ax=ax1, label='GaitRec')
    sns.kdeplot(ages_fi_s1, ax=ax1, label='ForceID-A')

    sns.kdeplot(heights_means_ai, ax=ax2, label='AIST')
    sns.kdeplot(heights_means_gb[~np.isnan(heights_means_gb)], ax=ax2, label='Gutenberg')
    sns.kdeplot(heights_means_gr, ax=ax2, label='GaitRec')
    sns.kdeplot(heights_means_fi, ax=ax2, label='ForceID-A')

    sns.kdeplot(masses_means_ai, ax=ax3, label='AIST')
    sns.kdeplot(masses_means_gb, ax=ax3, label='Gutenberg')
    sns.kdeplot(masses_means_gr, ax=ax3, label='GaitRec')
    sns.kdeplot(masses_means_fi, ax=ax3, label='ForceID-A')

    count_female_unbal = [np.asarray(sexes_ai_s1 == 0).nonzero()[0].shape[0],
                          np.asarray(sexes_gb_s1 == 0).nonzero()[0].shape[0],
                          np.asarray(sexes_gr_s1 == 0).nonzero()[0].shape[0],
                          np.asarray(sexes_fi_s1 == 0).nonzero()[0].shape[0]]
    count_male_unbal = [np.asarray(sexes_ai_s1 == 1).nonzero()[0].shape[0],
                        np.asarray(sexes_gb_s1 == 1).nonzero()[0].shape[0],
                        np.asarray(sexes_gr_s1 == 1).nonzero()[0].shape[0],
                        np.asarray(sexes_fi_s1 == 1).nonzero()[0].shape[0]]
    count_unknown_unbal = [0, 0, 3, 0] # Sex was reported as unknown for 3 participants in the unbal version of GaitRec.

    rects1 = ax4.bar(x - width, count_female_unbal, width, label='Female')
    rects2 = ax4.bar(x, count_male_unbal, width, label='Male')
    rects3 = ax4.bar(x + width, count_unknown_unbal, width, label='Unknown')

ax1.tick_params(labelsize=11)
ax2.tick_params(labelsize=11)
ax3.tick_params(labelsize=11)
ax4.tick_params(labelsize=10)

ax1.set_xlim(0, 100)
ax1.set_xticks(np.arange(0, 101, 20))
ax1.set_ylim(0, 0.1)
ax1.set_yticks(np.arange(0, 0.11, 0.02))

ax2.set_xlim(1.2, 2.2)
ax2.set_xticks(np.arange(1.2, 2.21, 0.2))
ax2.set_ylim(0, 5)
ax2.set_yticks(np.arange(0, 6, 1))

ax3.set_xlim(0, 170)
ax3.set_xticks(np.arange(0, 171, 34))
ax3.set_ylim(0, 0.05)
ax3.set_yticks(np.arange(0, 0.06, 0.01))

ax4.set_xticks(x, ['AIST', 'Gutenberg', 'GaitRec', 'ForceID-A'])

ax1.set_xlabel("Age (y)", size=11, labelpad=5)
ax2.set_xlabel("Height (m)", size=11, labelpad=5)
ax3.set_xlabel("Mass (kg)", size=11, labelpad=5)
ax4.set_xlabel('Dataset', size=11, labelpad=5)

ax1.set_ylabel("Density", size=11, labelpad=5)
ax2.set_ylabel("Density", size=11, labelpad=5)
ax3.set_ylabel("Density", size=11, labelpad=5)

ax1.legend(prop={'size': 9})
ax4.legend(prop={'size': 9})

fig.supylabel('Count', size=11)

if plot_bal_datasets:

    ax4.set_ylim(0, 150)
    ax4.set_yticks(np.arange(0, 151, 30))
    ax4.set_ylabel('Count', size=11, labelpad=-1)

    plt.savefig('./Figures/fig_demographics_bal.pdf', dpi=1200)
    plt.savefig('./Figures/fig_demographics_bal.png', dpi=1200)
    plt.savefig('./Figures/fig_demographics_bal.svg', dpi=1200)

else:

    ax4.set_ylim(0, 250)
    ax4.set_yticks(np.arange(0, 251, 50))
    ax4.set_ylabel('Count', size=11, labelpad=0)

    plt.savefig('./Figures/fig_demographics_unbal.pdf', dpi=1200)
    plt.savefig('./Figures/fig_demographics_unbal.png', dpi=1200)
    plt.savefig('./Figures/fig_demographics_unbal.svg', dpi=1200)

plt.show()
