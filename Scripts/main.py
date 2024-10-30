import numpy as np
import os
import random
import itertools
from sklearn.model_selection import KFold
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from pytorch_metric_learning import miners, losses, distances
from Losses import EvalHard
from TrainEval import run_expt
from Utils import set_seed, get_base_metadata, balance_ids_in_ds, gen_cv_folds, save_list, load_list

"""This script was used to run the experiments. It contains sections for different dataset configurations as the
experiments were completed in batches (e.g., commented out lines 344 onwards to run same-dataset configurations first)."""

# Set random seed to maximise reproducibility:
g = set_seed()

# List dataset configuration methods...
# Nomenclature
# ------------
# Strings are formatted as ['no. datasets-config method-additional detail'].
# 'one' = same-dataset part I.
# 'aug' = mixed-dataset part II, where the training set was 'augmented' with at least one other dataset.
# 'cro-r' = cross-dataset part IIIa (the 'r' indicates that training, validation, and test sets were 'restricted' to be
# the same size as in part I).
# 'cro' = cross-dataset part IIIb-d.
# 'umap' = different variant of mixed-dataset configuration that was implemented for the UMAP analysis. Namely, the
# training, validation, and test sets all contained a uniform distribution of IDs from all four datasets.
# ------------

config_methods_one_ds = ['one------'] # one_ds = one dataset
config_methods_two_ds = ['two-cro-r',
                         'two-cro--',
                         'two-aug--']
# When using three+ datasets, the mixed-dataset and cross-dataset configurations were run separately to avoid
# excessively long sections of code and enable reasonable run times (given our compute resources). E.g., see lines
# 615-790 below:
config_methods_thr_ds_aug = ['thr-aug--']
config_methods_thr_ds_cro = ['thr-cro--']

config_methods_fou_ds_aug = ['fou-aug--']
config_methods_fou_ds_cro = ['fou-cro--']
config_methods_fou_ds_umap = ['fou-umap-']


# Now list specific dataset configurations...
# Dataset nomenclature
# --------------------
# 'fi-all' = ForceID-A (private version)
# 'gr-sho' = GaitRec (shod walking from healthy controls)
# 'gb-all' = Gutenberg
# 'ai-all' = AIST
# --------------------
configs_one_ds = [['fi-all'],
                  ['gr-sho'],
                  ['gb-all'],
                  ['ai-all']]

# *For the rest of the dataset configuration lists, the positions of the items in the sublists matter.*
# - Mixed-dataset configurations (part IIa) - both datasets are used for training and the first dataset is used for
# validation and testing. For example, ['gr-sho', 'fi-all'] means using gr-sho and fi-all for training then gr-sho
# for validation and testing.
# - Cross-dataset configurations (part IIIa-b), the first dataset is used for training and validation, and the
# second dataset is used for testing. For example, ['gr-sho', 'fi-all'] means using gr-sho for training and validation
# then fi-all for testing.
configs_two_ds = [['gr-sho', 'fi-all'],
                  ['gb-all', 'fi-all'],
                  ['ai-all', 'fi-all'],
                  ['fi-all', 'gr-sho'],
                  ['gb-all', 'gr-sho'],
                  ['ai-all', 'gr-sho'],
                  ['fi-all', 'gb-all'],
                  ['gr-sho', 'gb-all'],
                  ['ai-all', 'gb-all'],
                  ['fi-all', 'ai-all'],
                  ['gr-sho', 'ai-all'],
                  ['gb-all', 'ai-all']]

# Mixed-dataset (part IIb) - all three datasets are used for training, and the first dataset is used for validation and
# testing:
configs_thr_ds_aug = [['fi-all', 'gr-sho', 'gb-all'],
                      ['fi-all', 'gb-all', 'ai-all'],
                      ['fi-all', 'gr-sho', 'ai-all'],
                      ['gr-sho', 'fi-all', 'gb-all'],
                      ['gr-sho', 'gb-all', 'ai-all'],
                      ['gr-sho', 'fi-all', 'ai-all'],
                      ['gb-all', 'fi-all', 'gr-sho'],
                      ['gb-all', 'gr-sho', 'ai-all'],
                      ['gb-all', 'fi-all', 'ai-all'],
                      ['ai-all', 'fi-all', 'gr-sho'],
                      ['ai-all', 'fi-all', 'gb-all'],
                      ['ai-all', 'gr-sho', 'gb-all']]

# Cross-dataset (part IIIc) - the first two datasets are used for training and validation, and the third dataset is used
# for testing:
configs_thr_ds_cro = [['gr-sho', 'gb-all', 'fi-all'],
                      ['gb-all', 'ai-all', 'fi-all'],
                      ['gr-sho', 'ai-all', 'fi-all'],
                      ['fi-all', 'gb-all', 'gr-sho'],
                      ['fi-all', 'ai-all', 'gr-sho'],
                      ['gb-all', 'ai-all', 'gr-sho'],
                      ['fi-all', 'gr-sho', 'gb-all'],
                      ['fi-all', 'ai-all', 'gb-all'],
                      ['gr-sho', 'ai-all', 'gb-all'],
                      ['fi-all', 'gr-sho', 'ai-all'],
                      ['fi-all', 'gb-all', 'ai-all'],
                      ['gr-sho', 'gb-all', 'ai-all']]

# Mixed-dataset (part IIc) - all four datasets are used for training, and the first dataset is used for validation and
# testing:
configs_fou_ds_aug = [['fi-all', 'gr-sho', 'gb-all', 'ai-all'],
                      ['gr-sho', 'fi-all', 'gb-all', 'ai-all'],
                      ['gb-all', 'fi-all', 'gr-sho', 'ai-all'],
                      ['ai-all', 'fi-all', 'gr-sho', 'gb-all']]

# Cross-dataset (part IIId) - the first three datasets are used for training and validation, and the fourth dataset is
# used for testing:
configs_fou_ds_cro = [['gr-sho', 'gb-all', 'ai-all', 'fi-all'],
                      ['fi-all', 'gb-all', 'ai-all', 'gr-sho'],
                      ['fi-all', 'gr-sho', 'ai-all', 'gb-all'],
                      ['fi-all', 'gr-sho', 'gb-all', 'ai-all']]

# Extra mixed-dataset configurations for UMAP analysis:
configs_fou_ds_umap = [['fi-all', 'gr-all', 'gb-all', 'ai-all'],
                       ['fi-all', 'gr-sho', 'gb-all', 'ai-all']]

# List architectures to include...
# Architecture nomenclature
# -------------------------
# F = fully-connected layer
# C = convolutional layer
# LU = LSTM layer (uni-directional)
# LB = LSTM layer (bi-directional)
# T = transformer encoder layer
# The strings are the same length, e.g. '1C1T-2F' = 1 convolutional layer, 1 transformer encoder layer, and 2
# fully-connected layers (respectively).
# -------------------------
archs = ['-----1F', '-----2F', '1C---1F', '1C---2F', '3C---1F',
         '3C---2F', '3C1LU1F', '3C1LU2F', '3C1LB1F', '3C1LB2F',
         '1C1T-1F', '1C1T-2F', '3C1T-1F', '3C1T-2F', '--1T-1F',
         '--1T-2F', '--2T-1F', '--2T-2F', '--3T-1F', '--3T-2F']


# Initialise global settings/params:
results_path = './Results/main'
batch_sizes = [128, 256, 512]
epochs = 1000
ptm = True # If using PyTorch Metric Learning package, set this to True.
single_input = True # Not used in this study. Can optionally have two separate inputs that are fused at the output level
miner = miners.BatchHardMiner(distance=distances.LpDistance(normalize_embeddings=False, p=2, power=1))
m = 0.3 # Margin for triplet margin loss.
criterion_tr = losses.TripletMarginLoss(margin=m, swap=False, smooth_loss=True,
                                        distance=distances.LpDistance(normalize_embeddings=False, p=2, power=1))
criterion_opt = None # If using a loss function (i.e., criterion) that requires an optimiser, initialise it here.
criterion_eval = EvalHard(margin=m)
cuda_tr = True
cuda_va = True
cuda_te = True
n_ids_bal = 185 # No. IDs to include in the balanced version of each dataset.
n_samples_per_id_bal = 10 # No. samples per ID to include in the balanced version of each dataset.


# Get list of all experimental settings for same-dataset configurations (part I):
expt_settings_one_ds = list(itertools.product(*[config_methods_one_ds, configs_one_ds, archs, batch_sizes]))
# Loop through all same-dataset configuration experiments:
for expt_idx, (config_method, config, arch, bs) in enumerate(expt_settings_one_ds):

    # We sometimes used conditionals like this to manage CPU and GPU memory when using architectures with a large number
    # of parameters. If using, comment out the cuda_tr, cuda_va, and cuda_te variables at lines 158-160 above:
    # if arch == '3C1T-2F' or arch in ['--1T-1F', '--1T-2F', '--2T-1F', '--2T-2F', '--3T-1F']:
    #
    #     cuda_tr = True
    #     cuda_va = False
    #     cuda_te = False
    #
    # elif arch == '--3T-2F':
    #
    #     cuda_tr = False
    #     cuda_va = True
    #     cuda_te = True
    #
    # else:
    #
    #     cuda_tr = True
    #     cuda_va = True
    #     cuda_te = True

    # Get basic metadata. (Config[i] accesses the i-th dataset in the configuration. In this case, there is only one
    # dataset.):
    trial_names, labels, ids, counts_samples = get_base_metadata(config[0])

    # Load dataset objects containing trimmed, filtered, and time normalised GRF and COP measures:
    if config[0] == 'gr-sho':

        indices_shod = np.load('./Datasets/gr-sho/objects/indices.npy',
                               allow_pickle=True)
        sigs_all_r = np.load('./Datasets/%s/objects/sigs_all_r_pro.npy' % config[0],
                             allow_pickle=True)[indices_shod]
        sigs_all_l = np.load('./Datasets/%s/objects/sigs_all_l_pro.npy' % config[0],
                             allow_pickle=True)[indices_shod]

    else:

        sigs_all_r = np.load('./Datasets/%s/objects/sigs_all_r_pro.npy' % config[0],
                             allow_pickle=True)
        sigs_all_l = np.load('./Datasets/%s/objects/sigs_all_l_pro.npy' % config[0],
                             allow_pickle=True)

    # First balance the no. IDs in the dataset compared to others:
    trial_names, counts_samples, indices_sort_labels = balance_ids_in_ds(trial_names,
                                                                         labels,
                                                                         ids,
                                                                         counts_samples,
                                                                         n_ids_bal)
    sigs_all_r = sigs_all_r[indices_sort_labels]
    sigs_all_l = sigs_all_l[indices_sort_labels]

    # Reset these variables to save memory as they are no longer needed:
    labels = None
    ids = None

    # Get trial names on a per-ID basis:
    indices_samples_where_id_changes = np.cumsum(counts_samples)[:-1]
    trial_names_by_id = np.split(trial_names, indices_samples_where_id_changes)

    # See if trial names for training, validation, and test sets have already been created and saved:
    trial_names_tr_check = os.path.join(results_path, 'trial_names_tr_by_id_%s_%s.txt' % (config_method, config[0]))
    trial_names_va_check = os.path.join(results_path, 'trial_names_va_by_id_%s_%s.txt' % (config_method, config[0]))
    trial_names_te_check = os.path.join(results_path, 'trial_names_te_by_id_%s_%s.txt' % (config_method, config[0]))

    if not os.path.isfile(trial_names_tr_check)\
            and not os.path.isfile(trial_names_va_check)\
            and not os.path.isfile(trial_names_te_check):

        # Select 10 random samples per ID (to finish balancing the dataset):
        trial_names_by_id_shuff = [random.sample(list(arr), k=n_samples_per_id_bal) for arr in trial_names_by_id]

        # Save trial names for the balanced version of the dataset:
        save_list(os.path.join(results_path, 'trial_names_by_id_shuff_%s_%s.txt' % (config_method, config[0])),
                  trial_names_by_id_shuff)

        # Allocate IDs into training, validation, and test sets for five-fold cross-validation (5-CV) using trial names:
        trial_names_tr_by_id, trial_names_va_by_id, trial_names_te_by_id = \
            gen_cv_folds(np.array(trial_names_by_id_shuff, dtype=object))

        # Save trial names for training, validation, and test sets:
        save_list(os.path.join(results_path, 'trial_names_tr_by_id_%s_%s.txt' % (config_method, config[0])),
                  trial_names_tr_by_id)
        save_list(os.path.join(results_path, 'trial_names_va_by_id_%s_%s.txt' % (config_method, config[0])),
                  trial_names_va_by_id)
        save_list(os.path.join(results_path, 'trial_names_te_by_id_%s_%s.txt' % (config_method, config[0])),
                  trial_names_te_by_id)

    elif os.path.isfile(trial_names_tr_check)\
            and os.path.isfile(trial_names_va_check)\
            and os.path.isfile(trial_names_te_check):

        trial_names_tr_by_id = load_list(os.path.join(results_path, 'trial_names_tr_by_id_%s_%s.txt' % (config_method,
                                                                                                        config[0])))
        trial_names_va_by_id = load_list(os.path.join(results_path, 'trial_names_va_by_id_%s_%s.txt' % (config_method,
                                                                                                        config[0])))
        trial_names_te_by_id = load_list(os.path.join(results_path, 'trial_names_te_by_id_%s_%s.txt' % (config_method,
                                                                                                        config[0])))

    else:

        raise Exception('trial_names_by_id files should exist in a set of three (tr, va, and te) for a given ds. '
                        'Check for error.')

    # Generic function to run a given experiment (i.e., dataset configuration over 5-CV) and return results:
    times, mod_checks, losses_tr_all, losses_va_all, losses_te_all, accs_tr_all, accs_va_all, accs_te_all, embs_va_all,\
        embs_te_all = run_expt(trial_names, sigs_all_r, sigs_all_l, trial_names_tr_by_id, trial_names_va_by_id,
                               trial_names_te_by_id, n_samples_per_id_bal, bs, arch, epochs, criterion_tr,
                               criterion_eval, ptm, miner, criterion_opt, cuda_tr, cuda_va, cuda_te, single_input)

    # Quick check of summary accuracy stats:
    acc_tr = np.round(np.mean([np.max(item) for item in accs_tr_all]) * 100, decimals=2)
    acc_va = np.round(np.mean([np.max(item) for item in accs_va_all]) * 100, decimals=2)
    acc_te = np.round(np.mean(accs_te_all) * 100, decimals=2)

    # Save results:
    save_list(os.path.join(results_path, 'losses_tr_%s_%s_%s_%s.txt' % (config_method, config[0], arch, bs)),
              losses_tr_all)
    save_list(os.path.join(results_path, 'losses_va_%s_%s_%s_%s.txt' % (config_method, config[0], arch, bs)),
              losses_va_all)
    save_list(os.path.join(results_path, 'losses_te_%s_%s_%s_%s.txt' % (config_method, config[0], arch, bs)),
              losses_te_all)
    save_list(os.path.join(results_path, 'accs_tr_%s_%s_%s_%s.txt' % (config_method, config[0], arch, bs)),
              accs_tr_all)
    save_list(os.path.join(results_path, 'accs_va_%s_%s_%s_%s.txt' % (config_method, config[0], arch, bs)),
              accs_va_all)
    save_list(os.path.join(results_path, 'accs_te_%s_%s_%s_%s.txt' % (config_method, config[0], arch, bs)),
              accs_te_all)
    torch.save(embs_va_all, os.path.join(results_path, 'embs_va_%s_%s_%s_%s.pth' % (config_method, config[0],
                                                                                    arch, bs)))
    torch.save(embs_te_all, os.path.join(results_path, 'embs_te_%s_%s_%s_%s.pth' % (config_method, config[0],
                                                                                    arch, bs)))
    torch.save(mod_checks, os.path.join(results_path, 'mod_checks_%s_%s_%s_%s.pth' % (config_method, config[0],
                                                                                      arch, bs)))

    # Generate and save figure of accuracy and loss during model training and validation:
    fig = plt.figure()
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])

    [ax1.plot(item, color='#1f77b4', lw=1, label='Tr') if idx == 0 else ax1.plot(item, color='#1f77b4')
     for idx, item in enumerate(accs_tr_all)]
    [ax1.plot(item, color='#f77f0e', lw=1, label='Va') if idx == 0 else ax1.plot(item, color='#f77f0e')
     for idx, item in enumerate(accs_va_all)]

    [ax2.plot(item, color='#1f77b4', lw=1, label='Tr') if idx == 0 else ax2.plot(item, color='#1f77b4')
     for idx, item in enumerate(losses_tr_all)]
    [ax2.plot(item, color='#f77f0e', lw=1, label='Va') if idx == 0 else ax2.plot(item, color='#f77f0e')
     for idx, item in enumerate(losses_va_all)]

    ax1.set_ylim(0, 1)
    ax2.set_ylim(0, 3)

    ax1.set_yticks(np.arange(0, 1.1, 0.2))
    ax2.set_yticks(np.arange(0, 3.1, 0.6))

    legend_ax1 = ax1.legend(prop={'size': 10}, frameon=False)
    legend_ax2 = ax2.legend(prop={'size': 10}, frameon=False)

    ax1.tick_params(labelsize=10)
    ax2.tick_params(labelsize=10)

    ax1.set_ylabel('Accuracy', size=12, labelpad=6)
    ax2.set_ylabel('Loss', size=12, labelpad=6)

    plt.xlabel('Epoch', size=12, labelpad=6)
    fig.tight_layout()

    plt.savefig(os.path.join(results_path, 'accloss_%s_%s_%s_%s.pdf' % (config_method, config[0], arch, bs)), dpi=1200)
    plt.savefig(os.path.join(results_path, 'accloss_%s_%s_%s_%s.svg' % (config_method, config[0], arch, bs)), dpi=1200)
    plt.savefig(os.path.join(results_path, 'accloss_%s_%s_%s_%s.png' % (config_method, config[0], arch, bs)), dpi=1200)
    plt.close()


# Get list of all experimental settings using two datasets (part IIa and part IIIa-b):
expt_settings_two_ds = list(itertools.product(*[config_methods_two_ds, configs_two_ds, archs, batch_sizes]))
# Loop through all experiments using two datasets. Note that the experiments from here on out in this script require...
# part I to be completed because the trial names for the balanced versions of each dataset as well as training, ...
# validation, and test sets from part I are loaded:
for expt_idx, (config_method, config, arch, bs) in enumerate(expt_settings_two_ds):

    # if arch == '3C1T-2F' or (arch == '--3T-2F' and bs == 512):
    #
    #     cuda_tr = True
    #     cuda_va = False
    #     cuda_te = False
    #
    # else:
    #
    #     cuda_tr = True
    #     cuda_va = True
    #
    #     if config_method == 'two-cro--':
    #
    #         cuda_te = False
    #
    #     else:
    #
    #         cuda_te = True

    trial_names_ds1, labels_ds1, ids_ds1, counts_samples_ds1 = get_base_metadata(config[0])
    trial_names_ds2, labels_ds2, ids_ds2, counts_samples_ds2 = get_base_metadata(config[1])

    sigs_all_r = []
    sigs_all_l = []

    for dataset in config:

        if dataset == 'gr-sho':

            indices_shod = np.load('./Datasets/gr-sho/objects/indices.npy',
                                   allow_pickle=True)
            sigs_all_r.append(np.load('./Datasets/%s/objects/sigs_all_r_pro.npy' % dataset,
                                      allow_pickle=True)[indices_shod])
            sigs_all_l.append(np.load('./Datasets/%s/objects/sigs_all_l_pro.npy' % dataset,
                                      allow_pickle=True)[indices_shod])

        else:

            sigs_all_r_temp = np.load('./Datasets/%s/objects/sigs_all_r_pro.npy' % dataset,
                                      allow_pickle=True)
            sigs_all_l_temp = np.load('./Datasets/%s/objects/sigs_all_l_pro.npy' % dataset,
                                      allow_pickle=True)

            sigs_all_r.append(sigs_all_r_temp)
            sigs_all_l.append(sigs_all_l_temp)

    sigs_all_r_temp = None
    sigs_all_l_temp = None

    sigs_all_r_ds1 = sigs_all_r[0]
    sigs_all_r_ds2 = sigs_all_r[1]
    sigs_all_l_ds1 = sigs_all_l[0]
    sigs_all_l_ds2 = sigs_all_l[1]

    trial_names_ds1, counts_samples_ds1, indices_sort_labels_ds1 = balance_ids_in_ds(trial_names_ds1,
                                                                                     labels_ds1,
                                                                                     ids_ds1,
                                                                                     counts_samples_ds1,
                                                                                     n_ids_bal)
    trial_names_ds2, counts_samples_ds2, indices_sort_labels_ds2 = balance_ids_in_ds(trial_names_ds2,
                                                                                     labels_ds2,
                                                                                     ids_ds2,
                                                                                     counts_samples_ds2,
                                                                                     n_ids_bal)

    sigs_all_r_ds1 = sigs_all_r_ds1[indices_sort_labels_ds1]
    sigs_all_r_ds2 = sigs_all_r_ds2[indices_sort_labels_ds2]
    sigs_all_l_ds1 = sigs_all_l_ds1[indices_sort_labels_ds1]
    sigs_all_l_ds2 = sigs_all_l_ds2[indices_sort_labels_ds2]

    labels_ds1 = None
    labels_ds2 = None
    ids_ds1 = None
    ids_ds2 = None

    # Combine measures and trial names from each dataset. Overwrites previous 'sigs_all_[stance side]' lists as they...
    # are no longer needed:
    sigs_all_r = np.concatenate((sigs_all_r_ds1, sigs_all_r_ds2))
    sigs_all_l = np.concatenate((sigs_all_l_ds1, sigs_all_l_ds2))
    trial_names = np.concatenate((trial_names_ds1, trial_names_ds2))

    if config_method == 'two-cro-r':

        # Load trial names for training, validation, and test sets from part I:
        trial_names_tr_by_id = load_list(os.path.join(results_path, 'trial_names_tr_by_id_one------_%s.txt' %
                                                      config[0]))
        trial_names_va_by_id = load_list(os.path.join(results_path, 'trial_names_va_by_id_one------_%s.txt' %
                                                      config[0]))
        # (Notice that the test set comes from the second dataset (config[1]):
        trial_names_te_by_id = load_list(os.path.join(results_path, 'trial_names_te_by_id_one------_%s.txt' %
                                                      config[1]))

    elif config_method == 'two-aug--':

        trial_names_va_by_id = load_list(os.path.join(results_path, 'trial_names_va_by_id_one------_%s.txt' %
                                                      config[0]))
        trial_names_te_by_id = load_list(os.path.join(results_path, 'trial_names_te_by_id_one------_%s.txt' %
                                                      config[0]))

        trial_names_tr_check = os.path.join(results_path, 'trial_names_tr_by_id_%s_%s-%s.txt' %
                                            (config_method, config[0], config[1]))

        if not os.path.isfile(trial_names_tr_check):

            # Load the trial names for the training set from part I:
            trial_names_tr_by_id_ds1 = load_list(os.path.join(results_path,
                                                              'trial_names_tr_by_id_one------_%s.txt' %
                                                              config[0]))
            # Load the trial names for the balanced version of the second dataset (config[1]) from part I:
            trial_names_by_id_ds2_shuff = load_list(os.path.join(results_path,
                                                                 'trial_names_by_id_shuff_one------_%s.txt' %
                                                                 config[1]))

            # For the training set from each CV fold in part I, add the balanced version of the second dataset:
            trial_names_tr_by_id = [trial_names_tr_by_id_ds1[i] + trial_names_by_id_ds2_shuff
                                    for i in range(len(trial_names_tr_by_id_ds1))]

            save_list(os.path.join(results_path, 'trial_names_tr_by_id_%s_%s-%s.txt' %
                                   (config_method, config[0], config[1])), trial_names_tr_by_id)

        else:

            trial_names_tr_by_id = load_list(os.path.join(results_path, 'trial_names_tr_by_id_%s_%s-%s.txt' %
                                                          (config_method, config[0], config[1])))

    else: # 'two-cro--'

        trial_names_tr_check = os.path.join(results_path, 'trial_names_tr_by_id_%s_%s.txt' % (config_method, config[0]))
        trial_names_va_check = os.path.join(results_path, 'trial_names_va_by_id_%s_%s.txt' % (config_method, config[0]))

        if not os.path.isfile(trial_names_tr_check) and not os.path.isfile(trial_names_va_check):

            # Load the trial names for the balanced version of the first dataset (config[0]) from part I:
            trial_names_by_id_ds1_shuff = np.array(load_list(os.path.join(results_path,
                                                                          'trial_names_by_id_shuff_one------_%s.txt' %
                                                                          config[0])), dtype=object)

            # This time use the sklearn KFold class because only need to define training and validation sets for 5-CV:
            kf = KFold(n_splits=5)  # 80:20% ID distribution

            trial_names_tr_by_id = []
            trial_names_va_by_id = []

            for indices_fold_tr, indices_fold_va in kf.split(trial_names_by_id_ds1_shuff):

                trial_names_tr_by_id.append(list(trial_names_by_id_ds1_shuff[indices_fold_tr]))
                trial_names_va_by_id.append(list(trial_names_by_id_ds1_shuff[indices_fold_va]))

            save_list(os.path.join(results_path, 'trial_names_tr_by_id_%s_%s.txt' % (config_method, config[0])),
                      trial_names_tr_by_id)
            save_list(os.path.join(results_path, 'trial_names_va_by_id_%s_%s.txt' % (config_method, config[0])),
                      trial_names_va_by_id)

            trial_names_by_id_ds1_shuff = None  # To save memory

        elif os.path.isfile(trial_names_tr_check) and os.path.isfile(trial_names_va_check):

            trial_names_tr_by_id = load_list(os.path.join(results_path, 'trial_names_tr_by_id_%s_%s.txt' %
                                                          (config_method, config[0])))
            trial_names_va_by_id = load_list(os.path.join(results_path, 'trial_names_va_by_id_%s_%s.txt' %
                                                          (config_method, config[0])))

        else:

            raise Exception('For two-cro-- expts, trial_names_tr_by_id and trial_names_va_by_id should both exist for a'
                            'given dataset.')

        trial_names_te_check = os.path.join(results_path, 'trial_names_te_by_id_%s_%s.txt' % (config_method, config[1]))

        if not os.path.isfile(trial_names_te_check):

            # Load the trial names for the balanced version of the second dataset (config[1]) from part I:
            trial_names_te_by_id_fold = load_list(os.path.join(results_path,
                                                               'trial_names_by_id_shuff_one------_%s.txt' %
                                                               config[1]))

            # Use the same trial names for the test set across all five folds (i.e., repeat five times):
            trial_names_te_by_id = [trial_names_te_by_id_fold for i in range(len(trial_names_tr_by_id))]

            save_list(os.path.join(results_path, 'trial_names_te_by_id_%s_%s.txt' % (config_method, config[1])),
                      trial_names_te_by_id)

            trial_names_te_by_id_fold = None

        else:

            trial_names_te_by_id = load_list(os.path.join(results_path, 'trial_names_te_by_id_%s_%s.txt' %
                                                          (config_method, config[1])))

        # Remove data from unnecessary variables to save memory:
        trial_names_ds1 = None
        trial_names_ds2 = None
        counts_samples_ds1 = None
        counts_samples_ds2 = None
        indices_shod = None  # In case indices_shod was defined.
        sigs_all_l_ds1 = None
        sigs_all_r_ds1 = None
        sigs_all_l_ds2 = None
        sigs_all_r_ds2 = None

    times, mod_checks, losses_tr_all, losses_va_all, losses_te_all, accs_tr_all, accs_va_all, accs_te_all, embs_va_all,\
    embs_te_all = run_expt(trial_names, sigs_all_r, sigs_all_l, trial_names_tr_by_id, trial_names_va_by_id,
                           trial_names_te_by_id, n_samples_per_id_bal, bs, arch, epochs, criterion_tr, criterion_eval,
                           ptm, miner, criterion_opt, cuda_tr, cuda_va, cuda_te, single_input)

    save_list(os.path.join(results_path, 'losses_tr_%s_%s-%s_%s_%s.txt' %
                           (config_method, config[0], config[1], arch, bs)), losses_tr_all)
    save_list(os.path.join(results_path, 'losses_va_%s_%s-%s_%s_%s.txt' %
                           (config_method, config[0], config[1], arch, bs)), losses_va_all)
    save_list(os.path.join(results_path, 'losses_te_%s_%s-%s_%s_%s.txt' %
                           (config_method, config[0], config[1], arch, bs)), losses_te_all)
    save_list(os.path.join(results_path, 'accs_tr_%s_%s-%s_%s_%s.txt' %
                           (config_method, config[0], config[1], arch, bs)), accs_tr_all)
    save_list(os.path.join(results_path, 'accs_va_%s_%s-%s_%s_%s.txt' %
                           (config_method, config[0], config[1], arch, bs)), accs_va_all)
    save_list(os.path.join(results_path, 'accs_te_%s_%s-%s_%s_%s.txt' %
                           (config_method, config[0], config[1], arch, bs)), accs_te_all)
    torch.save(embs_va_all, os.path.join(results_path, 'embs_va_%s_%s-%s_%s_%s.pth' %
                                         (config_method, config[0], config[1], arch, bs)))
    torch.save(embs_te_all, os.path.join(results_path, 'embs_te_%s_%s-%s_%s_%s.pth' %
                                         (config_method, config[0], config[1], arch, bs)))

    fig = plt.figure()
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])

    [ax1.plot(item, color='#1f77b4', lw=1, label='Tr') if idx == 0 else ax1.plot(item, color='#1f77b4')
     for idx, item in enumerate(accs_tr_all)]
    [ax1.plot(item, color='#f77f0e', lw=1, label='Va') if idx == 0 else ax1.plot(item, color='#f77f0e')
     for idx, item in enumerate(accs_va_all)]

    [ax2.plot(item, color='#1f77b4', lw=1, label='Tr') if idx == 0 else ax2.plot(item, color='#1f77b4')
     for idx, item in enumerate(losses_tr_all)]
    [ax2.plot(item, color='#f77f0e', lw=1, label='Va') if idx == 0 else ax2.plot(item, color='#f77f0e')
     for idx, item in enumerate(losses_va_all)]

    ax1.set_ylim(0, 1)
    ax2.set_ylim(0, 3)

    ax1.set_yticks(np.arange(0, 1.1, 0.2))
    ax2.set_yticks(np.arange(0, 3.1, 0.6))

    legend_ax1 = ax1.legend(prop={'size': 10}, frameon=False)
    legend_ax2 = ax2.legend(prop={'size': 10}, frameon=False)

    ax1.tick_params(labelsize=10)
    ax2.tick_params(labelsize=10)

    ax1.set_ylabel('Accuracy', size=12, labelpad=6)
    ax2.set_ylabel('Loss', size=12, labelpad=6)

    plt.xlabel('Epoch', size=12, labelpad=6)
    fig.tight_layout()

    plt.savefig(os.path.join(results_path, 'accloss_%s_%s-%s_%s_%s.pdf' %
                             (config_method, config[0], config[1], arch, bs)), dpi=1200)
    plt.savefig(os.path.join(results_path, 'accloss_%s_%s-%s_%s_%s.svg' %
                             (config_method, config[0], config[1], arch, bs)), dpi=1200)
    plt.savefig(os.path.join(results_path, 'accloss_%s_%s-%s_%s_%s.png' %
                             (config_method, config[0], config[1], arch, bs)), dpi=1200)
    plt.close()


# Get list of all experimental settings for mixed-dataset configurations part IIb:
expt_settings_thr_ds_aug = list(itertools.product(*[config_methods_thr_ds_aug, configs_thr_ds_aug, archs, batch_sizes]))
# Loop through experiments for mixed-dataset configurations part IIb:
for expt_idx, (config_method, config, arch, bs) in enumerate(expt_settings_thr_ds_aug):

    trial_names_ds1, labels_ds1, ids_ds1, counts_samples_ds1 = get_base_metadata(config[0])
    trial_names_ds2, labels_ds2, ids_ds2, counts_samples_ds2 = get_base_metadata(config[1])
    trial_names_ds3, labels_ds3, ids_ds3, counts_samples_ds3 = get_base_metadata(config[2])

    sigs_all_r = []
    sigs_all_l = []

    for dataset in config:

        if dataset == 'gr-sho':

            indices_shod = np.load('./Datasets/gr-sho/objects/indices.npy',
                                   allow_pickle=True)
            sigs_all_r.append(np.load('./Datasets/%s/objects/sigs_all_r_pro.npy' % dataset,
                                      allow_pickle=True)[indices_shod])
            sigs_all_l.append(np.load('./Datasets/%s/objects/sigs_all_l_pro.npy' % dataset,
                                      allow_pickle=True)[indices_shod])

        else:

            sigs_all_r_temp = np.load('./Datasets/%s/objects/sigs_all_r_pro.npy' % dataset,
                                      allow_pickle=True)
            sigs_all_l_temp = np.load('./Datasets/%s/objects/sigs_all_l_pro.npy' % dataset,
                                      allow_pickle=True)

            sigs_all_r.append(sigs_all_r_temp)
            sigs_all_l.append(sigs_all_l_temp)

    sigs_all_r_temp = None
    sigs_all_l_temp = None

    sigs_all_r_ds1 = sigs_all_r[0]
    sigs_all_r_ds2 = sigs_all_r[1]
    sigs_all_r_ds3 = sigs_all_r[2]

    sigs_all_l_ds1 = sigs_all_l[0]
    sigs_all_l_ds2 = sigs_all_l[1]
    sigs_all_l_ds3 = sigs_all_l[2]

    trial_names_ds1, counts_samples_ds1, indices_sort_labels_ds1 = balance_ids_in_ds(trial_names_ds1,
                                                                                     labels_ds1,
                                                                                     ids_ds1,
                                                                                     counts_samples_ds1,
                                                                                     n_ids_bal)
    trial_names_ds2, counts_samples_ds2, indices_sort_labels_ds2 = balance_ids_in_ds(trial_names_ds2,
                                                                                     labels_ds2,
                                                                                     ids_ds2,
                                                                                     counts_samples_ds2,
                                                                                     n_ids_bal)
    trial_names_ds3, counts_samples_ds3, indices_sort_labels_ds3 = balance_ids_in_ds(trial_names_ds3,
                                                                                     labels_ds3,
                                                                                     ids_ds3,
                                                                                     counts_samples_ds3,
                                                                                     n_ids_bal)

    sigs_all_r_ds1 = sigs_all_r_ds1[indices_sort_labels_ds1]
    sigs_all_r_ds2 = sigs_all_r_ds2[indices_sort_labels_ds2]
    sigs_all_r_ds3 = sigs_all_r_ds3[indices_sort_labels_ds3]
    sigs_all_l_ds1 = sigs_all_l_ds1[indices_sort_labels_ds1]
    sigs_all_l_ds2 = sigs_all_l_ds2[indices_sort_labels_ds2]
    sigs_all_l_ds3 = sigs_all_l_ds3[indices_sort_labels_ds3]

    labels_ds1 = None
    labels_ds2 = None
    labels_ds3 = None
    ids_ds1 = None
    ids_ds2 = None
    ids_ds3 = None

    sigs_all_r = np.concatenate((sigs_all_r_ds1, sigs_all_r_ds2, sigs_all_r_ds3))
    sigs_all_l = np.concatenate((sigs_all_l_ds1, sigs_all_l_ds2, sigs_all_l_ds3))
    trial_names = np.concatenate((trial_names_ds1, trial_names_ds2, trial_names_ds3))

    trial_names_va_by_id = load_list(os.path.join(results_path, 'trial_names_va_by_id_one------_%s.txt' % config[0]))
    trial_names_te_by_id = load_list(os.path.join(results_path, 'trial_names_te_by_id_one------_%s.txt' % config[0]))

    trial_names_tr_check = os.path.join(results_path, 'trial_names_tr_by_id_%s_%s-%s-%s.txt' %
                                        (config_method, config[0], config[1], config[2]))

    if not os.path.isfile(trial_names_tr_check):

        # Load trial names for the training set of the first dataset from part I:
        trial_names_tr_by_id_ds1 = load_list(os.path.join(results_path, 'trial_names_tr_by_id_one------_%s.txt' %
                                                          config[0]))

        # Load trial names for the balanced versions of the second and third datasets from part I:
        trial_names_by_id_ds2_shuff = load_list(os.path.join(results_path, 'trial_names_by_id_shuff_%s.txt' %
                                                             config[1]))
        trial_names_by_id_ds3_shuff = load_list(os.path.join(results_path, 'trial_names_by_id_shuff_%s.txt' %
                                                             config[2]))

        # For the training set from each CV fold in part I, add the balanced version of the second and third datasets:
        trial_names_tr_by_id = [trial_names_tr_by_id_ds1[i] +
                                trial_names_by_id_ds2_shuff +
                                trial_names_by_id_ds3_shuff
                                for i in range(len(trial_names_tr_by_id_ds1))]

        save_list(os.path.join(results_path, 'trial_names_tr_by_id_%s_%s-%s-%s.txt' %
                               (config_method, config[0], config[1], config[2])), trial_names_tr_by_id)

    else:

        trial_names_tr_by_id = load_list(os.path.join(results_path, 'trial_names_tr_by_id_%s_%s-%s-%s.txt' %
                                                      (config_method, config[0], config[1], config[2])))

    times, mod_checks, losses_tr_all, losses_va_all, losses_te_all, accs_tr_all, accs_va_all, accs_te_all, embs_va_all,\
    embs_te_all = run_expt(trial_names, sigs_all_r, sigs_all_l, trial_names_tr_by_id, trial_names_va_by_id,
                           trial_names_te_by_id, n_samples_per_id_bal, bs, arch, epochs, criterion_tr, criterion_eval,
                           ptm, miner, criterion_opt, cuda_tr, cuda_va, cuda_te, single_input)

    save_list(os.path.join(results_path, 'losses_tr_%s_%s-%s-%s_%s_%s.txt' %
                           (config_method, config[0], config[1], config[2], arch, bs)), losses_tr_all)
    save_list(os.path.join(results_path, 'losses_va_%s_%s-%s-%s_%s_%s.txt' %
                           (config_method, config[0], config[1], config[2], arch, bs)), losses_va_all)
    save_list(os.path.join(results_path, 'losses_te_%s_%s-%s-%s_%s_%s.txt' %
                           (config_method, config[0], config[1], config[2], arch, bs)), losses_te_all)
    save_list(os.path.join(results_path, 'accs_tr_%s_%s-%s-%s_%s_%s.txt' %
                           (config_method, config[0], config[1], config[2], arch, bs)), accs_tr_all)
    save_list(os.path.join(results_path, 'accs_va_%s_%s-%s-%s_%s_%s.txt' %
                           (config_method, config[0], config[1], config[2], arch, bs)), accs_va_all)
    save_list(os.path.join(results_path, 'accs_te_%s_%s-%s-%s_%s_%s.txt' %
                           (config_method, config[0], config[1], config[2], arch, bs)), accs_te_all)
    torch.save(embs_va_all, os.path.join(results_path, 'embs_va_%s_%s-%s-%s_%s_%s.pth' %
                                         (config_method, config[0], config[1], config[2], arch, bs)))
    torch.save(embs_te_all, os.path.join(results_path, 'embs_te_%s_%s-%s-%s_%s_%s.pth' %
                                         (config_method, config[0], config[1], config[2], arch, bs)))
    # torch.save(mod_checks, os.path.join(results_path, 'mod_checks_%s_%s-%s-%s_%s_%s.pth' %
    #                                     (config_method, config[0], config[1], config[2], arch, bs)))

    fig = plt.figure()
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])

    [ax1.plot(item, color='#1f77b4', lw=1, label='Tr') if idx == 0 else ax1.plot(item, color='#1f77b4')
     for idx, item in enumerate(accs_tr_all)]
    [ax1.plot(item, color='#f77f0e', lw=1, label='Va') if idx == 0 else ax1.plot(item, color='#f77f0e')
     for idx, item in enumerate(accs_va_all)]

    [ax2.plot(item, color='#1f77b4', lw=1, label='Tr') if idx == 0 else ax2.plot(item, color='#1f77b4')
     for idx, item in enumerate(losses_tr_all)]
    [ax2.plot(item, color='#f77f0e', lw=1, label='Va') if idx == 0 else ax2.plot(item, color='#f77f0e')
     for idx, item in enumerate(losses_va_all)]

    ax1.set_ylim(0, 1)
    ax2.set_ylim(0, 3)

    ax1.set_yticks(np.arange(0, 1.1, 0.2))
    ax2.set_yticks(np.arange(0, 3.1, 0.6))

    legend_ax1 = ax1.legend(prop={'size': 10}, frameon=False)
    legend_ax2 = ax2.legend(prop={'size': 10}, frameon=False)

    ax1.tick_params(labelsize=10)
    ax2.tick_params(labelsize=10)

    ax1.set_ylabel('Accuracy', size=12, labelpad=6)
    ax2.set_ylabel('Loss', size=12, labelpad=6)

    plt.xlabel('Epoch', size=12, labelpad=6)
    fig.tight_layout()

    plt.savefig(os.path.join(results_path, 'accloss_%s_%s-%s-%s_%s_%s.pdf' %
                             (config_method, config[0], config[1], config[2], arch, bs)), dpi=1200)
    plt.savefig(os.path.join(results_path, 'accloss_%s_%s-%s-%s_%s_%s.svg' %
                             (config_method, config[0], config[1], config[2], arch, bs)), dpi=1200)
    plt.savefig(os.path.join(results_path, 'accloss_%s_%s-%s-%s_%s_%s.png' %
                             (config_method, config[0], config[1], config[2], arch, bs)), dpi=1200)
    plt.close()


# Get list of all experimental settings for cross-dataset configurations part IIIc:
expt_settings_thr_ds_cro = list(itertools.product(*[config_methods_thr_ds_cro, configs_thr_ds_cro, archs, batch_sizes]))
# Loop through experiments for cross-dataset configurations part IIIc:
for expt_idx, (config_method, config, arch, bs) in enumerate(expt_settings_thr_ds_cro):

    trial_names_ds1, labels_ds1, ids_ds1, counts_samples_ds1 = get_base_metadata(config[0])
    trial_names_ds2, labels_ds2, ids_ds2, counts_samples_ds2 = get_base_metadata(config[1])
    trial_names_ds3, labels_ds3, ids_ds3, counts_samples_ds3 = get_base_metadata(config[2])

    sigs_all_r = []
    sigs_all_l = []

    for dataset in config:

        if dataset == 'gr-sho':

            indices_shod = np.load('./Datasets/gr-sho/objects/indices.npy',
                                   allow_pickle=True)
            sigs_all_r.append(np.load('./Datasets/%s/objects/sigs_all_r_pro.npy' % dataset,
                                      allow_pickle=True)[indices_shod])
            sigs_all_l.append(np.load('./Datasets/%s/objects/sigs_all_l_pro.npy' % dataset,
                                      allow_pickle=True)[indices_shod])

        else:

            sigs_all_r_temp = np.load('./Datasets/%s/objects/sigs_all_r_pro.npy' % dataset,
                                      allow_pickle=True)
            sigs_all_l_temp = np.load('./Datasets/%s/objects/sigs_all_l_pro.npy' % dataset,
                                      allow_pickle=True)

            sigs_all_r.append(sigs_all_r_temp)
            sigs_all_l.append(sigs_all_l_temp)

    sigs_all_r_temp = None
    sigs_all_l_temp = None

    sigs_all_r_ds1 = sigs_all_r[0]
    sigs_all_r_ds2 = sigs_all_r[1]
    sigs_all_r_ds3 = sigs_all_r[2]

    sigs_all_l_ds1 = sigs_all_l[0]
    sigs_all_l_ds2 = sigs_all_l[1]
    sigs_all_l_ds3 = sigs_all_l[2]

    trial_names_ds1, counts_samples_ds1, indices_sort_labels_ds1 = balance_ids_in_ds(trial_names_ds1,
                                                                                     labels_ds1,
                                                                                     ids_ds1,
                                                                                     counts_samples_ds1,
                                                                                     n_ids_bal)
    trial_names_ds2, counts_samples_ds2, indices_sort_labels_ds2 = balance_ids_in_ds(trial_names_ds2,
                                                                                     labels_ds2,
                                                                                     ids_ds2,
                                                                                     counts_samples_ds2,
                                                                                     n_ids_bal)
    trial_names_ds3, counts_samples_ds3, indices_sort_labels_ds3 = balance_ids_in_ds(trial_names_ds3,
                                                                                     labels_ds3,
                                                                                     ids_ds3,
                                                                                     counts_samples_ds3,
                                                                                     n_ids_bal)

    sigs_all_r_ds1 = sigs_all_r_ds1[indices_sort_labels_ds1]
    sigs_all_r_ds2 = sigs_all_r_ds2[indices_sort_labels_ds2]
    sigs_all_r_ds3 = sigs_all_r_ds3[indices_sort_labels_ds3]
    sigs_all_l_ds1 = sigs_all_l_ds1[indices_sort_labels_ds1]
    sigs_all_l_ds2 = sigs_all_l_ds2[indices_sort_labels_ds2]
    sigs_all_l_ds3 = sigs_all_l_ds3[indices_sort_labels_ds3]

    labels_ds1 = None
    labels_ds2 = None
    labels_ds3 = None
    ids_ds1 = None
    ids_ds2 = None
    ids_ds3 = None

    sigs_all_r = np.concatenate((sigs_all_r_ds1, sigs_all_r_ds2, sigs_all_r_ds3))
    sigs_all_l = np.concatenate((sigs_all_l_ds1, sigs_all_l_ds2, sigs_all_l_ds3))
    trial_names = np.concatenate((trial_names_ds1, trial_names_ds2, trial_names_ds3))

    trial_names_tr_check = os.path.join(results_path, 'trial_names_tr_by_id_%s_%s-%s.txt' %
                                        (config_method, config[0], config[1]))
    trial_names_va_check = os.path.join(results_path, 'trial_names_va_by_id_%s_%s-%s.txt' %
                                        (config_method, config[0], config[1]))

    if not os.path.isfile(trial_names_tr_check) and not os.path.isfile(trial_names_va_check):

        # Load the trial names for the balanced versions of the first and second datasets from part I:
        trial_names_by_id_ds1_shuff = load_list(os.path.join(results_path,
                                                             'trial_names_by_id_shuff_one------_%s.txt' % config[0]))
        trial_names_by_id_ds2_shuff = load_list(os.path.join(results_path,
                                                             'trial_names_by_id_shuff_one------_%s.txt' % config[1]))

        # Combine the datasets and then shuffle the order of IDs in each dataset. This ensured that the training and...
        # validation sets contained a random mix of IDs from both datasets:
        trial_names_by_id_shuff = np.array(random.sample(trial_names_by_id_ds1_shuff +
                                                         trial_names_by_id_ds2_shuff,
                                                         k=len(trial_names_by_id_ds1_shuff) +
                                                           len(trial_names_by_id_ds2_shuff)), dtype=object)

        kf = KFold(n_splits=5)

        trial_names_tr_by_id = []
        trial_names_va_by_id = []

        for indices_fold_tr, indices_fold_va in kf.split(trial_names_by_id_shuff):

            trial_names_tr_by_id.append(list(trial_names_by_id_shuff[indices_fold_tr]))
            trial_names_va_by_id.append(list(trial_names_by_id_shuff[indices_fold_va]))

        save_list(os.path.join(results_path, 'trial_names_tr_by_id_%s_%s-%s.txt' %
                               (config_method, config[0], config[1])), trial_names_tr_by_id)
        save_list(os.path.join(results_path, 'trial_names_va_by_id_%s_%s-%s.txt' %
                               (config_method, config[0], config[1])), trial_names_va_by_id)

        # Clear variables that are no longer required to save memory:
        trial_names_by_id_ds1_shuff = None
        trial_names_by_id_ds2_shuff = None
        trial_names_by_id_shuff = None
        indices_fold_tr = None
        indices_fold_va = None

    elif os.path.isfile(trial_names_tr_check) and os.path.isfile(trial_names_va_check):

        trial_names_tr_by_id = load_list(os.path.join(results_path, 'trial_names_tr_by_id_%s_%s-%s.txt' %
                                                      (config_method, config[0], config[1])))
        trial_names_va_by_id = load_list(os.path.join(results_path, 'trial_names_va_by_id_%s_%s-%s.txt' %
                                                      (config_method, config[0], config[1])))

    else:

        raise Exception('For thr-cro expts, trial_names_tr_by_id and trial_names_va_by_id should both exist for a '
                        'given two ds combination.')

    trial_names_te_by_id = load_list(os.path.join(results_path, 'trial_names_te_by_id_two-cro--_%s.txt' % config[2]))

    # Clear variables that are no longer required to save memory:
    trial_names_ds1 = None
    trial_names_ds2 = None
    trial_names_ds3 = None
    counts_samples_ds1 = None
    counts_samples_ds2 = None
    counts_samples_ds3 = None
    indices_shod = None  # In case indices_shod was defined.
    sigs_all_l_ds1 = None
    sigs_all_r_ds1 = None
    sigs_all_l_ds2 = None
    sigs_all_r_ds2 = None
    sigs_all_l_ds3 = None
    sigs_all_r_ds3 = None

    times, mod_checks, losses_tr_all, losses_va_all, losses_te_all, accs_tr_all, accs_va_all, accs_te_all, embs_va_all,\
    embs_te_all = run_expt(trial_names, sigs_all_r, sigs_all_l, trial_names_tr_by_id, trial_names_va_by_id,
                           trial_names_te_by_id, n_samples_per_id_bal, bs, arch, epochs, criterion_tr, criterion_eval,
                           ptm, miner, criterion_opt, cuda_tr, cuda_va, cuda_te, single_input)

    save_list(os.path.join(results_path, 'losses_tr_%s_%s-%s-%s_%s_%s.txt' %
                           (config_method, config[0], config[1], config[2], arch, bs)), losses_tr_all)
    save_list(os.path.join(results_path, 'losses_va_%s_%s-%s-%s_%s_%s.txt' %
                           (config_method, config[0], config[1], config[2], arch, bs)), losses_va_all)
    save_list(os.path.join(results_path, 'losses_te_%s_%s-%s-%s_%s_%s.txt' %
                           (config_method, config[0], config[1], config[2], arch, bs)), losses_te_all)
    save_list(os.path.join(results_path, 'accs_tr_%s_%s-%s-%s_%s_%s.txt' %
                           (config_method, config[0], config[1], config[2], arch, bs)), accs_tr_all)
    save_list(os.path.join(results_path, 'accs_va_%s_%s-%s-%s_%s_%s.txt' %
                           (config_method, config[0], config[1], config[2], arch, bs)), accs_va_all)
    save_list(os.path.join(results_path, 'accs_te_%s_%s-%s-%s_%s_%s.txt' %
                           (config_method, config[0], config[1], config[2], arch, bs)), accs_te_all)
    torch.save(embs_va_all, os.path.join(results_path, 'embs_va_%s_%s-%s-%s_%s_%s.pth' %
                                         (config_method, config[0], config[1], config[2], arch, bs)))
    torch.save(embs_te_all, os.path.join(results_path, 'embs_te_%s_%s-%s-%s_%s_%s.pth' %
                                         (config_method, config[0], config[1], config[2], arch, bs)))
    # You might want to comment out the saving of model checkpoints because these will quickly chew up memory storage:
    torch.save(mod_checks, os.path.join(results_path, 'mod_checks_%s_%s-%s-%s_%s_%s.pth' %
                                        (config_method, config[0], config[1], config[2], arch, bs)))

    fig = plt.figure()
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])

    [ax1.plot(item, color='#1f77b4', lw=1, label='Tr') if idx == 0 else ax1.plot(item, color='#1f77b4')
     for idx, item in enumerate(accs_tr_all)]
    [ax1.plot(item, color='#f77f0e', lw=1, label='Va') if idx == 0 else ax1.plot(item, color='#f77f0e')
     for idx, item in enumerate(accs_va_all)]

    [ax2.plot(item, color='#1f77b4', lw=1, label='Tr') if idx == 0 else ax2.plot(item, color='#1f77b4')
     for idx, item in enumerate(losses_tr_all)]
    [ax2.plot(item, color='#f77f0e', lw=1, label='Va') if idx == 0 else ax2.plot(item, color='#f77f0e')
     for idx, item in enumerate(losses_va_all)]

    ax1.set_ylim(0, 1)
    ax2.set_ylim(0, 3)

    ax1.set_yticks(np.arange(0, 1.1, 0.2))
    ax2.set_yticks(np.arange(0, 3.1, 0.6))

    legend_ax1 = ax1.legend(prop={'size': 10}, frameon=False)
    legend_ax2 = ax2.legend(prop={'size': 10}, frameon=False)

    ax1.tick_params(labelsize=10)
    ax2.tick_params(labelsize=10)

    ax1.set_ylabel('Accuracy', size=12, labelpad=6)
    ax2.set_ylabel('Loss', size=12, labelpad=6)

    plt.xlabel('Epoch', size=12, labelpad=6)
    fig.tight_layout()

    plt.savefig(os.path.join(results_path, 'accloss_%s_%s-%s-%s_%s_%s.pdf' %
                             (config_method, config[0], config[1], config[2], arch, bs)), dpi=1200)
    plt.savefig(os.path.join(results_path, 'accloss_%s_%s-%s-%s_%s_%s.svg' %
                             (config_method, config[0], config[1], config[2], arch, bs)), dpi=1200)
    plt.savefig(os.path.join(results_path, 'accloss_%s_%s-%s-%s_%s_%s.png' %
                             (config_method, config[0], config[1], config[2], arch, bs)), dpi=1200)
    plt.close()


# Get list of all experimental settings for mixed-dataset configurations part IIc:
expt_settings_fou_ds_aug = list(itertools.product(*[config_methods_fou_ds_aug, configs_fou_ds_aug, archs, batch_sizes]))
# Loop through experiments for mixed-dataset configurations part IIc:
for expt_idx, (config_method, config, arch, bs) in enumerate(expt_settings_fou_ds_aug):

    trial_names_ds1, labels_ds1, ids_ds1, counts_samples_ds1 = get_base_metadata(config[0])
    trial_names_ds2, labels_ds2, ids_ds2, counts_samples_ds2 = get_base_metadata(config[1])
    trial_names_ds3, labels_ds3, ids_ds3, counts_samples_ds3 = get_base_metadata(config[2])
    trial_names_ds4, labels_ds4, ids_ds4, counts_samples_ds4 = get_base_metadata(config[3])

    sigs_all_r = []
    sigs_all_l = []

    for dataset in config:

        if dataset == 'gr-sho':

            indices_shod = np.load('./Datasets/gr-sho/objects/indices.npy',
                                   allow_pickle=True)
            sigs_all_r.append(np.load('./Datasets/%s/objects/sigs_all_r_pro.npy' % dataset,
                                      allow_pickle=True)[indices_shod])
            sigs_all_l.append(np.load('./Datasets/%s/objects/sigs_all_l_pro.npy' % dataset,
                                      allow_pickle=True)[indices_shod])

        else:

            sigs_all_r_temp = np.load('./Datasets/%s/objects/sigs_all_r_pro.npy' % dataset,
                                      allow_pickle=True)
            sigs_all_l_temp = np.load('./Datasets/%s/objects/sigs_all_l_pro.npy' % dataset,
                                      allow_pickle=True)

            sigs_all_r.append(sigs_all_r_temp)
            sigs_all_l.append(sigs_all_l_temp)

    sigs_all_r_temp = None
    sigs_all_l_temp = None

    sigs_all_r_ds1 = sigs_all_r[0]
    sigs_all_r_ds2 = sigs_all_r[1]
    sigs_all_r_ds3 = sigs_all_r[2]
    sigs_all_r_ds4 = sigs_all_r[3]

    sigs_all_l_ds1 = sigs_all_l[0]
    sigs_all_l_ds2 = sigs_all_l[1]
    sigs_all_l_ds3 = sigs_all_l[2]
    sigs_all_l_ds4 = sigs_all_l[3]

    trial_names_ds1, counts_samples_ds1, indices_sort_labels_ds1 = balance_ids_in_ds(trial_names_ds1,
                                                                                     labels_ds1,
                                                                                     ids_ds1,
                                                                                     counts_samples_ds1,
                                                                                     n_ids_bal)
    trial_names_ds2, counts_samples_ds2, indices_sort_labels_ds2 = balance_ids_in_ds(trial_names_ds2,
                                                                                     labels_ds2,
                                                                                     ids_ds2,
                                                                                     counts_samples_ds2,
                                                                                     n_ids_bal)
    trial_names_ds3, counts_samples_ds3, indices_sort_labels_ds3 = balance_ids_in_ds(trial_names_ds3,
                                                                                     labels_ds3,
                                                                                     ids_ds3,
                                                                                     counts_samples_ds3,
                                                                                     n_ids_bal)
    trial_names_ds4, counts_samples_ds4, indices_sort_labels_ds4 = balance_ids_in_ds(trial_names_ds4,
                                                                                     labels_ds4,
                                                                                     ids_ds4,
                                                                                     counts_samples_ds4,
                                                                                     n_ids_bal)

    sigs_all_r_ds1 = sigs_all_r_ds1[indices_sort_labels_ds1]
    sigs_all_r_ds2 = sigs_all_r_ds2[indices_sort_labels_ds2]
    sigs_all_r_ds3 = sigs_all_r_ds3[indices_sort_labels_ds3]
    sigs_all_r_ds4 = sigs_all_r_ds4[indices_sort_labels_ds4]

    sigs_all_l_ds1 = sigs_all_l_ds1[indices_sort_labels_ds1]
    sigs_all_l_ds2 = sigs_all_l_ds2[indices_sort_labels_ds2]
    sigs_all_l_ds3 = sigs_all_l_ds3[indices_sort_labels_ds3]
    sigs_all_l_ds4 = sigs_all_l_ds4[indices_sort_labels_ds4]

    labels_ds1 = None
    labels_ds2 = None
    labels_ds3 = None
    labels_ds4 = None
    ids_ds1 = None
    ids_ds2 = None
    ids_ds3 = None
    ids_ds4 = None

    sigs_all_r = np.concatenate((sigs_all_r_ds1, sigs_all_r_ds2, sigs_all_r_ds3, sigs_all_r_ds4))
    sigs_all_l = np.concatenate((sigs_all_l_ds1, sigs_all_l_ds2, sigs_all_l_ds3, sigs_all_l_ds4))
    trial_names = np.concatenate((trial_names_ds1, trial_names_ds2, trial_names_ds3, trial_names_ds4))

    trial_names_va_by_id = load_list(os.path.join(results_path, 'trial_names_va_by_id_one------_%s.txt' % config[0]))
    trial_names_te_by_id = load_list(os.path.join(results_path, 'trial_names_te_by_id_one------_%s.txt' % config[0]))

    trial_names_tr_check = os.path.join(results_path, 'trial_names_tr_by_id_%s_%s-%s-%s-%s.txt' %
                                        (config_method, config[0], config[1], config[2], config[3]))

    if not os.path.isfile(trial_names_tr_check):

        trial_names_tr_by_id_ds1 = load_list(os.path.join(results_path, 'trial_names_tr_by_id_one------_%s.txt' %
                                                          config[0]))

        trial_names_by_id_ds2_shuff = load_list(os.path.join(results_path, 'trial_names_by_id_shuff_one------_%s.txt' %
                                                             config[1]))
        trial_names_by_id_ds3_shuff = load_list(os.path.join(results_path, 'trial_names_by_id_shuff_one------_%s.txt' %
                                                             config[2]))
        trial_names_by_id_ds4_shuff = load_list(os.path.join(results_path, 'trial_names_by_id_shuff_one------_%s.txt' %
                                                             config[3]))

        trial_names_tr_by_id = [trial_names_tr_by_id_ds1[i] +
                                trial_names_by_id_ds2_shuff +
                                trial_names_by_id_ds3_shuff +
                                trial_names_by_id_ds4_shuff
                                for i in range(len(trial_names_tr_by_id_ds1))]

        save_list(os.path.join(results_path, 'trial_names_tr_by_id_%s_%s-%s-%s-%s.txt' %
                               (config_method, config[0], config[1], config[2], config[3])), trial_names_tr_by_id)

    else:

        trial_names_tr_by_id = load_list(os.path.join(results_path, 'trial_names_tr_by_id_%s_%s-%s-%s-%s.txt' %
                                                      (config_method, config[0], config[1], config[2], config[3])))

    times, mod_checks, losses_tr_all, losses_va_all, losses_te_all, accs_tr_all, accs_va_all, accs_te_all, embs_va_all,\
    embs_te_all = run_expt(trial_names, sigs_all_r, sigs_all_l, trial_names_tr_by_id, trial_names_va_by_id,
                           trial_names_te_by_id, n_samples_per_id_bal, bs, arch, epochs, criterion_tr, criterion_eval,
                           ptm, miner, criterion_opt, cuda_tr, cuda_va, cuda_te, single_input)

    save_list(os.path.join(results_path, 'losses_tr_%s_%s-%s-%s-%s_%s_%s.txt' %
                           (config_method, config[0], config[1], config[2], config[3], arch, bs)), losses_tr_all)
    save_list(os.path.join(results_path, 'losses_va_%s_%s-%s-%s-%s_%s_%s.txt' %
                           (config_method, config[0], config[1], config[2], config[3], arch, bs)), losses_va_all)
    save_list(os.path.join(results_path, 'losses_te_%s_%s-%s-%s-%s_%s_%s.txt' %
                           (config_method, config[0], config[1], config[2], config[3], arch, bs)), losses_te_all)
    save_list(os.path.join(results_path, 'accs_tr_%s_%s-%s-%s-%s_%s_%s.txt' %
                           (config_method, config[0], config[1], config[2], config[3], arch, bs)), accs_tr_all)
    save_list(os.path.join(results_path, 'accs_va_%s_%s-%s-%s-%s_%s_%s.txt' %
                           (config_method, config[0], config[1], config[2], config[3], arch, bs)), accs_va_all)
    save_list(os.path.join(results_path, 'accs_te_%s_%s-%s-%s-%s_%s_%s.txt' %
                           (config_method, config[0], config[1], config[2], config[3], arch, bs)), accs_te_all)
    torch.save(embs_va_all, os.path.join(results_path, 'embs_va_%s_%s-%s-%s-%s_%s_%s.pth' %
                                         (config_method, config[0], config[1], config[2], config[3], arch, bs)))
    torch.save(embs_te_all, os.path.join(results_path, 'embs_te_%s_%s-%s-%s-%s_%s_%s.pth' %
                                         (config_method, config[0], config[1], config[2], config[3], arch, bs)))
    torch.save(mod_checks, os.path.join(results_path, 'mod_checks_%s_%s-%s-%s-%s_%s_%s.pth' %
                                        (config_method, config[0], config[1], config[2], config[3], arch, bs)))

    fig = plt.figure()
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])

    [ax1.plot(item, color='#1f77b4', lw=1, label='Tr') if idx == 0 else ax1.plot(item, color='#1f77b4')
     for idx, item in enumerate(accs_tr_all)]
    [ax1.plot(item, color='#f77f0e', lw=1, label='Va') if idx == 0 else ax1.plot(item, color='#f77f0e')
     for idx, item in enumerate(accs_va_all)]

    [ax2.plot(item, color='#1f77b4', lw=1, label='Tr') if idx == 0 else ax2.plot(item, color='#1f77b4')
     for idx, item in enumerate(losses_tr_all)]
    [ax2.plot(item, color='#f77f0e', lw=1, label='Va') if idx == 0 else ax2.plot(item, color='#f77f0e')
     for idx, item in enumerate(losses_va_all)]

    ax1.set_ylim(0, 1)
    ax2.set_ylim(0, 3)

    ax1.set_yticks(np.arange(0, 1.1, 0.2))
    ax2.set_yticks(np.arange(0, 3.1, 0.6))

    legend_ax1 = ax1.legend(prop={'size': 10}, frameon=False)
    legend_ax2 = ax2.legend(prop={'size': 10}, frameon=False)

    ax1.tick_params(labelsize=10)
    ax2.tick_params(labelsize=10)

    ax1.set_ylabel('Accuracy', size=12, labelpad=6)
    ax2.set_ylabel('Loss', size=12, labelpad=6)

    plt.xlabel('Epoch', size=12, labelpad=6)
    fig.tight_layout()

    plt.savefig(os.path.join(results_path, 'accloss_%s_%s-%s-%s-%s_%s_%s.pdf' %
                             (config_method, config[0], config[1], config[2], config[3], arch, bs)), dpi=1200)
    plt.savefig(os.path.join(results_path, 'accloss_%s_%s-%s-%s-%s_%s_%s.svg' %
                             (config_method, config[0], config[1], config[2], config[3], arch, bs)), dpi=1200)
    plt.savefig(os.path.join(results_path, 'accloss_%s_%s-%s-%s-%s_%s_%s.png' %
                             (config_method, config[0], config[1], config[2], config[3], arch, bs)), dpi=1200)
    plt.close()


# Get list of all experimental settings for cross-dataset configurations part IIId:
expt_settings_fou_ds_cro = list(itertools.product(*[config_methods_fou_ds_cro, configs_fou_ds_cro, archs, batch_sizes]))
# Loop through experiments for cross-dataset configurations part IIId:
for expt_idx, (config_method, config, arch, bs) in enumerate(expt_settings_fou_ds_cro):

    trial_names_ds1, labels_ds1, ids_ds1, counts_samples_ds1 = get_base_metadata(config[0])
    trial_names_ds2, labels_ds2, ids_ds2, counts_samples_ds2 = get_base_metadata(config[1])
    trial_names_ds3, labels_ds3, ids_ds3, counts_samples_ds3 = get_base_metadata(config[2])
    trial_names_ds4, labels_ds4, ids_ds4, counts_samples_ds4 = get_base_metadata(config[3])

    sigs_all_r = []
    sigs_all_l = []

    for dataset in config:

        if dataset == 'gr-sho':

            indices_shod = np.load('./Datasets/gr-sho/objects/indices.npy',
                                   allow_pickle=True)
            sigs_all_r.append(np.load('./Datasets/%s/objects/sigs_all_r_pro.npy' % dataset,
                                      allow_pickle=True)[indices_shod])
            sigs_all_l.append(np.load('./Datasets/%s/objects/sigs_all_l_pro.npy' % dataset,
                                      allow_pickle=True)[indices_shod])

        else:

            sigs_all_r_temp = np.load('./Datasets/%s/objects/sigs_all_r_pro.npy' % dataset,
                                      allow_pickle=True)
            sigs_all_l_temp = np.load('./Datasets/%s/objects/sigs_all_l_pro.npy' % dataset,
                                      allow_pickle=True)

            sigs_all_r.append(sigs_all_r_temp)
            sigs_all_l.append(sigs_all_l_temp)

    sigs_all_r_temp = None
    sigs_all_l_temp = None

    sigs_all_r_ds1 = sigs_all_r[0]
    sigs_all_r_ds2 = sigs_all_r[1]
    sigs_all_r_ds3 = sigs_all_r[2]
    sigs_all_r_ds4 = sigs_all_r[3]

    sigs_all_l_ds1 = sigs_all_l[0]
    sigs_all_l_ds2 = sigs_all_l[1]
    sigs_all_l_ds3 = sigs_all_l[2]
    sigs_all_l_ds4 = sigs_all_l[3]

    trial_names_ds1, counts_samples_ds1, indices_sort_labels_ds1 = balance_ids_in_ds(trial_names_ds1,
                                                                                     labels_ds1,
                                                                                     ids_ds1,
                                                                                     counts_samples_ds1,
                                                                                     n_ids_bal)
    trial_names_ds2, counts_samples_ds2, indices_sort_labels_ds2 = balance_ids_in_ds(trial_names_ds2,
                                                                                     labels_ds2,
                                                                                     ids_ds2,
                                                                                     counts_samples_ds2,
                                                                                     n_ids_bal)
    trial_names_ds3, counts_samples_ds3, indices_sort_labels_ds3 = balance_ids_in_ds(trial_names_ds3,
                                                                                     labels_ds3,
                                                                                     ids_ds3,
                                                                                     counts_samples_ds3,
                                                                                     n_ids_bal)
    trial_names_ds4, counts_samples_ds4, indices_sort_labels_ds4 = balance_ids_in_ds(trial_names_ds4,
                                                                                     labels_ds4,
                                                                                     ids_ds4,
                                                                                     counts_samples_ds4,
                                                                                     n_ids_bal)

    sigs_all_r_ds1 = sigs_all_r_ds1[indices_sort_labels_ds1]
    sigs_all_r_ds2 = sigs_all_r_ds2[indices_sort_labels_ds2]
    sigs_all_r_ds3 = sigs_all_r_ds3[indices_sort_labels_ds3]
    sigs_all_r_ds4 = sigs_all_r_ds4[indices_sort_labels_ds4]

    sigs_all_l_ds1 = sigs_all_l_ds1[indices_sort_labels_ds1]
    sigs_all_l_ds2 = sigs_all_l_ds2[indices_sort_labels_ds2]
    sigs_all_l_ds3 = sigs_all_l_ds3[indices_sort_labels_ds3]
    sigs_all_l_ds4 = sigs_all_l_ds4[indices_sort_labels_ds4]

    labels_ds1 = None
    labels_ds2 = None
    labels_ds3 = None
    labels_ds4 = None
    ids_ds1 = None
    ids_ds2 = None
    ids_ds3 = None
    ids_ds4 = None

    sigs_all_r = np.concatenate((sigs_all_r_ds1, sigs_all_r_ds2, sigs_all_r_ds3, sigs_all_r_ds4))
    sigs_all_l = np.concatenate((sigs_all_l_ds1, sigs_all_l_ds2, sigs_all_l_ds3, sigs_all_l_ds4))
    trial_names = np.concatenate((trial_names_ds1, trial_names_ds2, trial_names_ds3, trial_names_ds4))

    trial_names_tr_check = os.path.join(results_path, 'trial_names_tr_by_id_%s_%s-%s-%s.txt' %
                                        (config_method, config[0], config[1], config[2]))
    trial_names_va_check = os.path.join(results_path, 'trial_names_va_by_id_%s_%s-%s-%s.txt' %
                                        (config_method, config[0], config[1], config[2]))

    if not os.path.isfile(trial_names_tr_check) and not os.path.isfile(trial_names_va_check):

        trial_names_by_id_ds1_shuff = load_list(os.path.join(results_path, 'trial_names_by_id_shuff_one------_%s.txt' %
                                                             config[0]))
        trial_names_by_id_ds2_shuff = load_list(os.path.join(results_path, 'trial_names_by_id_shuff_one------_%s.txt' %
                                                             config[1]))
        trial_names_by_id_ds3_shuff = load_list(os.path.join(results_path, 'trial_names_by_id_shuff_one------_%s.txt' %
                                                             config[2]))

        trial_names_by_id_shuff = np.array(random.sample(trial_names_by_id_ds1_shuff +
                                                         trial_names_by_id_ds2_shuff +
                                                         trial_names_by_id_ds3_shuff,
                                                         k=len(trial_names_by_id_ds1_shuff) +
                                                           len(trial_names_by_id_ds2_shuff) +
                                                           len(trial_names_by_id_ds3_shuff)), dtype=object)

        kf = KFold(n_splits=5)

        trial_names_tr_by_id = []
        trial_names_va_by_id = []

        for indices_fold_tr, indices_fold_va in kf.split(trial_names_by_id_shuff):

            trial_names_tr_by_id.append(list(trial_names_by_id_shuff[indices_fold_tr]))
            trial_names_va_by_id.append(list(trial_names_by_id_shuff[indices_fold_va]))

        save_list(os.path.join(results_path, 'trial_names_tr_by_id_%s_%s-%s-%s.txt' %
                               (config_method, config[0], config[1], config[2])), trial_names_tr_by_id)
        save_list(os.path.join(results_path, 'trial_names_va_by_id_%s_%s-%s-%s.txt' %
                               (config_method, config[0], config[1], config[2])), trial_names_va_by_id)

        trial_names_by_id_ds1_shuff = None
        trial_names_by_id_ds2_shuff = None
        trial_names_by_id_ds3_shuff = None
        trial_names_by_id_shuff = None
        indices_fold_tr = None
        indices_fold_va = None

    elif os.path.isfile(trial_names_tr_check) and os.path.isfile(trial_names_va_check):

        trial_names_tr_by_id = load_list(os.path.join(results_path, 'trial_names_tr_by_id_%s_%s-%s-%s.txt' %
                                                      (config_method, config[0], config[1], config[2])))
        trial_names_va_by_id = load_list(os.path.join(results_path, 'trial_names_va_by_id_%s_%s-%s-%s.txt' %
                                                      (config_method, config[0], config[1], config[2])))

    else:

        raise Exception('For fou-cro expts, trial_names_tr_by_id and trial_names_va_by_id should both exist for a '
                        'given three dataset combination.')

    trial_names_te_by_id = load_list(os.path.join(results_path, 'trial_names_te_by_id_two-cro--_%s.txt' % config[3]))

    trial_names_ds1 = None
    trial_names_ds2 = None
    trial_names_ds3 = None
    trial_names_ds4 = None
    counts_samples_ds1 = None
    counts_samples_ds2 = None
    counts_samples_ds3 = None
    counts_samples_ds4 = None
    indices_shod = None
    sigs_all_l_ds1 = None
    sigs_all_r_ds1 = None
    sigs_all_l_ds2 = None
    sigs_all_r_ds2 = None
    sigs_all_l_ds3 = None
    sigs_all_r_ds3 = None
    sigs_all_l_ds4 = None
    sigs_all_r_ds4 = None

    times, mod_checks, losses_tr_all, losses_va_all, losses_te_all, accs_tr_all, accs_va_all, accs_te_all, embs_va_all,\
    embs_te_all = run_expt(trial_names, sigs_all_r, sigs_all_l, trial_names_tr_by_id, trial_names_va_by_id,
                           trial_names_te_by_id, n_samples_per_id_bal, bs, arch, epochs, criterion_tr, criterion_eval,
                           ptm, miner, criterion_opt, cuda_tr, cuda_va, cuda_te, single_input)

    save_list(os.path.join(results_path, 'losses_tr_%s_%s-%s-%s-%s_%s_%s.txt' %
                           (config_method, config[0], config[1], config[2], config[3], arch, bs)), losses_tr_all)
    save_list(os.path.join(results_path, 'losses_va_%s_%s-%s-%s-%s_%s_%s.txt' %
                           (config_method, config[0], config[1], config[2], config[3], arch, bs)), losses_va_all)
    save_list(os.path.join(results_path, 'losses_te_%s_%s-%s-%s-%s_%s_%s.txt' %
                           (config_method, config[0], config[1], config[2], config[3], arch, bs)), losses_te_all)
    save_list(os.path.join(results_path, 'accs_tr_%s_%s-%s-%s-%s_%s_%s.txt' %
                           (config_method, config[0], config[1], config[2], config[3], arch, bs)), accs_tr_all)
    save_list(os.path.join(results_path, 'accs_va_%s_%s-%s-%s-%s_%s_%s.txt' %
                           (config_method, config[0], config[1], config[2], config[3], arch, bs)), accs_va_all)
    save_list(os.path.join(results_path, 'accs_te_%s_%s-%s-%s-%s_%s_%s.txt' %
                           (config_method, config[0], config[1], config[2], config[3], arch, bs)), accs_te_all)
    torch.save(embs_va_all, os.path.join(results_path, 'embs_va_%s_%s-%s-%s-%s_%s_%s.pth' %
                                         (config_method, config[0], config[1], config[2], config[3], arch, bs)))
    torch.save(embs_te_all, os.path.join(results_path, 'embs_te_%s_%s-%s-%s-%s_%s_%s.pth' %
                                         (config_method, config[0], config[1], config[2], config[3], arch, bs)))
    # torch.save(mod_checks, os.path.join(results_path, 'mod_checks_%s_%s-%s-%s-%s_%s_%s.pth' %
    #                                     (config_method, config[0], config[1], config[2], config[3], arch, bs)))

    fig = plt.figure()
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])

    [ax1.plot(item, color='#1f77b4', lw=1, label='Tr') if idx == 0 else ax1.plot(item, color='#1f77b4')
     for idx, item in enumerate(accs_tr_all)]
    [ax1.plot(item, color='#f77f0e', lw=1, label='Va') if idx == 0 else ax1.plot(item, color='#f77f0e')
     for idx, item in enumerate(accs_va_all)]

    [ax2.plot(item, color='#1f77b4', lw=1, label='Tr') if idx == 0 else ax2.plot(item, color='#1f77b4')
     for idx, item in enumerate(losses_tr_all)]
    [ax2.plot(item, color='#f77f0e', lw=1, label='Va') if idx == 0 else ax2.plot(item, color='#f77f0e')
     for idx, item in enumerate(losses_va_all)]

    ax1.set_ylim(0, 1)
    ax2.set_ylim(0, 3)

    ax1.set_yticks(np.arange(0, 1.1, 0.2))
    ax2.set_yticks(np.arange(0, 3.1, 0.6))

    legend_ax1 = ax1.legend(prop={'size': 10}, frameon=False)
    legend_ax2 = ax2.legend(prop={'size': 10}, frameon=False)

    ax1.tick_params(labelsize=10)
    ax2.tick_params(labelsize=10)

    ax1.set_ylabel('Accuracy', size=12, labelpad=6)
    ax2.set_ylabel('Loss', size=12, labelpad=6)

    plt.xlabel('Epoch', size=12, labelpad=6)
    fig.tight_layout()

    plt.savefig(os.path.join(results_path, 'accloss_%s_%s-%s-%s-%s_%s_%s.pdf' %
                             (config_method, config[0], config[1], config[2], config[3], arch, bs)), dpi=1200)
    plt.savefig(os.path.join(results_path, 'accloss_%s_%s-%s-%s-%s_%s_%s.svg' %
                             (config_method, config[0], config[1], config[2], config[3], arch, bs)), dpi=1200)
    plt.savefig(os.path.join(results_path, 'accloss_%s_%s-%s-%s-%s_%s_%s.png' %
                             (config_method, config[0], config[1], config[2], config[3], arch, bs)), dpi=1200)
    plt.close()


expt_settings_fou_ds_umap = list(itertools.product(*[config_methods_fou_ds_umap, configs_fou_ds_umap, archs,
                                                     batch_sizes]))
for expt_idx, (config_method, config, arch, bs) in enumerate(expt_settings_fou_ds_umap):

    cuda_va = False
    cuda_te = False

    trial_names_ds1, labels_ds1, ids_ds1, counts_samples_ds1 = get_base_metadata(config[0])
    trial_names_ds2, labels_ds2, ids_ds2, counts_samples_ds2 = get_base_metadata(config[1])
    trial_names_ds3, labels_ds3, ids_ds3, counts_samples_ds3 = get_base_metadata(config[2])
    trial_names_ds4, labels_ds4, ids_ds4, counts_samples_ds4 = get_base_metadata(config[3])

    sigs_all_r = []
    sigs_all_l = []

    for dataset in config:

        if dataset == 'gr-sho':

            indices_shod = np.load('./Datasets/gr-sho/objects/indices.npy',
                                   allow_pickle=True)
            sigs_all_r.append(np.load('./Datasets/%s/objects/sigs_all_r_pro.npy' % dataset,
                                      allow_pickle=True)[indices_shod])
            sigs_all_l.append(np.load('./Datasets/%s/objects/sigs_all_l_pro.npy' % dataset,
                                      allow_pickle=True)[indices_shod])

        else:

            sigs_all_r_temp = np.load('./Datasets/%s/objects/sigs_all_r_pro.npy' % dataset,
                                      allow_pickle=True)
            sigs_all_l_temp = np.load('./Datasets/%s/objects/sigs_all_l_pro.npy' % dataset,
                                      allow_pickle=True)

            sigs_all_r.append(sigs_all_r_temp)
            sigs_all_l.append(sigs_all_l_temp)

    sigs_all_r_temp = None
    sigs_all_l_temp = None

    sigs_all_r_ds1 = sigs_all_r[0]
    sigs_all_r_ds2 = sigs_all_r[1]
    sigs_all_r_ds3 = sigs_all_r[2]
    sigs_all_r_ds4 = sigs_all_r[3]

    sigs_all_l_ds1 = sigs_all_l[0]
    sigs_all_l_ds2 = sigs_all_l[1]
    sigs_all_l_ds3 = sigs_all_l[2]
    sigs_all_l_ds4 = sigs_all_l[3]

    trial_names_ds1, counts_samples_ds1, indices_sort_labels_ds1 = balance_ids_in_ds(trial_names_ds1,
                                                                                     labels_ds1,
                                                                                     ids_ds1,
                                                                                     counts_samples_ds1,
                                                                                     n_ids_bal)
    trial_names_ds2, counts_samples_ds2, indices_sort_labels_ds2 = balance_ids_in_ds(trial_names_ds2,
                                                                                     labels_ds2,
                                                                                     ids_ds2,
                                                                                     counts_samples_ds2,
                                                                                     n_ids_bal)
    trial_names_ds3, counts_samples_ds3, indices_sort_labels_ds3 = balance_ids_in_ds(trial_names_ds3,
                                                                                     labels_ds3,
                                                                                     ids_ds3,
                                                                                     counts_samples_ds3,
                                                                                     n_ids_bal)
    trial_names_ds4, counts_samples_ds4, indices_sort_labels_ds4 = balance_ids_in_ds(trial_names_ds4,
                                                                                     labels_ds4,
                                                                                     ids_ds4,
                                                                                     counts_samples_ds4,
                                                                                     n_ids_bal)

    sigs_all_r_ds1 = sigs_all_r_ds1[indices_sort_labels_ds1]
    sigs_all_r_ds2 = sigs_all_r_ds2[indices_sort_labels_ds2]
    sigs_all_r_ds3 = sigs_all_r_ds3[indices_sort_labels_ds3]
    sigs_all_r_ds4 = sigs_all_r_ds4[indices_sort_labels_ds4]

    sigs_all_l_ds1 = sigs_all_l_ds1[indices_sort_labels_ds1]
    sigs_all_l_ds2 = sigs_all_l_ds2[indices_sort_labels_ds2]
    sigs_all_l_ds3 = sigs_all_l_ds3[indices_sort_labels_ds3]
    sigs_all_l_ds4 = sigs_all_l_ds4[indices_sort_labels_ds4]

    labels_ds1 = None
    labels_ds2 = None
    labels_ds3 = None
    labels_ds4 = None
    ids_ds1 = None
    ids_ds2 = None
    ids_ds3 = None
    ids_ds4 = None

    sigs_all_r = np.concatenate((sigs_all_r_ds1, sigs_all_r_ds2, sigs_all_r_ds3, sigs_all_r_ds4))
    sigs_all_l = np.concatenate((sigs_all_l_ds1, sigs_all_l_ds2, sigs_all_l_ds3, sigs_all_l_ds4))
    trial_names = np.concatenate((trial_names_ds1, trial_names_ds2, trial_names_ds3, trial_names_ds4))

    trial_names_tr_by_id_ds1 = load_list(os.path.join(results_path, 'trial_names_tr_by_id_%s.txt' % config[0]))
    trial_names_va_by_id_ds1 = load_list(os.path.join(results_path, 'trial_names_va_by_id_%s.txt' % config[0]))
    trial_names_te_by_id_ds1 = load_list(os.path.join(results_path, 'trial_names_te_by_id_%s.txt' % config[0]))
    trial_names_tr_by_id_ds2 = load_list(os.path.join(results_path, 'trial_names_tr_by_id_%s.txt' % config[1]))
    trial_names_va_by_id_ds2 = load_list(os.path.join(results_path, 'trial_names_va_by_id_%s.txt' % config[1]))
    trial_names_te_by_id_ds2 = load_list(os.path.join(results_path, 'trial_names_te_by_id_%s.txt' % config[1]))
    trial_names_tr_by_id_ds3 = load_list(os.path.join(results_path, 'trial_names_tr_by_id_%s.txt' % config[2]))
    trial_names_va_by_id_ds3 = load_list(os.path.join(results_path, 'trial_names_va_by_id_%s.txt' % config[2]))
    trial_names_te_by_id_ds3 = load_list(os.path.join(results_path, 'trial_names_te_by_id_%s.txt' % config[2]))
    trial_names_tr_by_id_ds4 = load_list(os.path.join(results_path, 'trial_names_tr_by_id_%s.txt' % config[3]))
    trial_names_va_by_id_ds4 = load_list(os.path.join(results_path, 'trial_names_va_by_id_%s.txt' % config[3]))
    trial_names_te_by_id_ds4 = load_list(os.path.join(results_path, 'trial_names_te_by_id_%s.txt' % config[3]))

    trial_names_tr_by_id = []
    trial_names_va_by_id = []
    trial_names_te_by_id = []

    for i in range(len(trial_names_tr_by_id_ds1)):

        trial_names_tr_by_id.append(trial_names_tr_by_id_ds1[i] +
                                    trial_names_tr_by_id_ds2[i] +
                                    trial_names_tr_by_id_ds3[i] +
                                    trial_names_tr_by_id_ds4[i])
        trial_names_va_by_id.append(trial_names_va_by_id_ds1[i] +
                                    trial_names_va_by_id_ds2[i] +
                                    trial_names_va_by_id_ds3[i] +
                                    trial_names_va_by_id_ds4[i])
        trial_names_te_by_id.append(trial_names_te_by_id_ds1[i] +
                                    trial_names_te_by_id_ds2[i] +
                                    trial_names_te_by_id_ds3[i] +
                                    trial_names_te_by_id_ds4[i])

    save_list(os.path.join(results_path, 'trial_names_tr_by_id_%s_%s-%s-%s-%s.txt' %
                           (config_method, config[0], config[1], config[2], config[3])), trial_names_tr_by_id)
    save_list(os.path.join(results_path, 'trial_names_va_by_id_%s_%s-%s-%s-%s.txt' %
                           (config_method, config[0], config[1], config[2], config[3])), trial_names_va_by_id)
    save_list(os.path.join(results_path, 'trial_names_te_by_id_%s_%s-%s-%s-%s.txt' %
                           (config_method, config[0], config[1], config[2], config[3])), trial_names_te_by_id)

    times, mod_checks, losses_tr_all, losses_va_all, losses_te_all, accs_tr_all, accs_va_all, accs_te_all, embs_va_all,\
    embs_te_all = run_expt(trial_names, sigs_all_r, sigs_all_l, trial_names_tr_by_id, trial_names_va_by_id,
                           trial_names_te_by_id, n_samples_per_id_bal, bs, arch, epochs, criterion_tr, criterion_eval,
                           ptm, miner, criterion_opt, cuda_tr, cuda_va, cuda_te, single_input)

    save_list(os.path.join(results_path, 'losses_tr_%s_%s-%s-%s-%s_%s_%s.txt' %
                           (config_method, config[0], config[1], config[2], config[3], arch, bs)), losses_tr_all)
    save_list(os.path.join(results_path, 'losses_va_%s_%s-%s-%s-%s_%s_%s.txt' %
                           (config_method, config[0], config[1], config[2], config[3], arch, bs)), losses_va_all)
    save_list(os.path.join(results_path, 'losses_te_%s_%s-%s-%s-%s_%s_%s.txt' %
                           (config_method, config[0], config[1], config[2], config[3], arch, bs)), losses_te_all)
    save_list(os.path.join(results_path, 'accs_tr_%s_%s-%s-%s-%s_%s_%s.txt' %
                           (config_method, config[0], config[1], config[2], config[3], arch, bs)), accs_tr_all)
    save_list(os.path.join(results_path, 'accs_va_%s_%s-%s-%s-%s_%s_%s.txt' %
                           (config_method, config[0], config[1], config[2], config[3], arch, bs)), accs_va_all)
    save_list(os.path.join(results_path, 'accs_te_%s_%s-%s-%s-%s_%s_%s.txt' %
                           (config_method, config[0], config[1], config[2], config[3], arch, bs)), accs_te_all)
    torch.save(embs_va_all, os.path.join(results_path, 'embs_va_%s_%s-%s-%s-%s_%s_%s.pth' %
                                         (config_method, config[0], config[1], config[2], config[3], arch, bs)))
    torch.save(embs_te_all, os.path.join(results_path, 'embs_te_%s_%s-%s-%s-%s_%s_%s.pth' %
                                         (config_method, config[0], config[1], config[2], config[3], arch, bs)))
    torch.save(mod_checks, os.path.join(results_path, 'mod_checks_%s_%s-%s-%s-%s_%s_%s.pth' %
                                        (config_method, config[0], config[1], config[2], config[3], arch, bs)))

    fig = plt.figure()
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])

    [ax1.plot(item, color='#1f77b4', lw=1, label='Tr') if idx == 0 else ax1.plot(item, color='#1f77b4')
     for idx, item in enumerate(accs_tr_all)]
    [ax1.plot(item, color='#f77f0e', lw=1, label='Va') if idx == 0 else ax1.plot(item, color='#f77f0e')
     for idx, item in enumerate(accs_va_all)]

    [ax2.plot(item, color='#1f77b4', lw=1, label='Tr') if idx == 0 else ax2.plot(item, color='#1f77b4')
     for idx, item in enumerate(losses_tr_all)]
    [ax2.plot(item, color='#f77f0e', lw=1, label='Va') if idx == 0 else ax2.plot(item, color='#f77f0e')
     for idx, item in enumerate(losses_va_all)]

    ax1.set_ylim(0, 1)
    ax2.set_ylim(0, 3)

    ax1.set_yticks(np.arange(0, 1.1, 0.2))
    ax2.set_yticks(np.arange(0, 3.1, 0.6))

    legend_ax1 = ax1.legend(prop={'size': 10}, frameon=False)
    legend_ax2 = ax2.legend(prop={'size': 10}, frameon=False)

    ax1.tick_params(labelsize=10)
    ax2.tick_params(labelsize=10)

    ax1.set_ylabel('Accuracy', size=12, labelpad=6)
    ax2.set_ylabel('Loss', size=12, labelpad=6)

    plt.xlabel('Epoch', size=12, labelpad=6)
    fig.tight_layout()

    plt.savefig(os.path.join(results_path, 'accloss_%s_%s-%s-%s-%s_%s_%s.pdf' %
                             (config_method, config[0], config[1], config[2], config[3], arch, bs)), dpi=1200)
    plt.savefig(os.path.join(results_path, 'accloss_%s_%s-%s-%s-%s_%s_%s.svg' %
                             (config_method, config[0], config[1], config[2], config[3], arch, bs)), dpi=1200)
    plt.savefig(os.path.join(results_path, 'accloss_%s_%s-%s-%s-%s_%s_%s.png' %
                             (config_method, config[0], config[1], config[2], config[3], arch, bs)), dpi=1200)
    plt.close()
