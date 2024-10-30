import numpy as np
import os
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import torch
from pytorch_metric_learning import miners, losses, distances
from Losses import EvalHard
from TrainEval import run_expt_osa
from Utils import set_seed, get_base_metadata, balance_ids_in_ds, save_list, load_list

"""This script was used to run the occlusion sensitivity analysis and create associated figures that for the main text
and Supplementary Information."""

run_expts = False # This specifies whether to run the occlusion sensitivity experiments. The experiments must be
# conducted before the figures can be generated.
make_fig = True # This specifies whether to make an occlusion sensitivity figure.
fig_main_text = True # If False, generates one of the figures in the Supplementary Information.
save_fig = True
show_fig = False
window_size = 20 # 10, 20, or 40.

# For a sequence of length 200 (hard-coded), get the indices of upper and lower window limits:
indices_windows_lower = list(np.arange(0, 200 - window_size + 1, window_size / 2).astype(int))
indices_windows_upper = list(np.arange(window_size, 201, window_size / 2).astype(int))
windows = [[indices_windows_lower[i], indices_windows_upper[i]] for i in range(len(indices_windows_lower))]

channels = list(np.arange(5)) # 3D GRF and 2D COP
n_folds = 5 # number of CV folds

archs = ['-----2F']
config_methods_one_ds = ['one------']
batch_sizes = [512]
results_path = './Results/main'
n_ids_bal = 185 # No. IDs included in the balanced version of each dataset.
n_samples_per_id_bal = 10 # No. samples per ID included in the balanced version of each dataset.

if run_expts:

    g = set_seed()

    configs_one_ds = [['fi-all'],
                      ['gr-sho'],
                      ['gb-all'],
                      ['ai-all']]

    m = 0.3 # Margin for triplet margin loss.
    epochs = 1000
    ptm = True # If using PyTorch Metric Learning package, set this to True.
    single_input = True # Not used in this study. Can optionally have two separate inputs that are fused at the output
    # level.
    miner = miners.BatchHardMiner(distance=distances.LpDistance(normalize_embeddings=False, p=2, power=1))
    criterion_tr = losses.TripletMarginLoss(margin=m, swap=False, smooth_loss=True,
                                            distance=distances.LpDistance(normalize_embeddings=False, p=2, power=1))
    criterion_opt = None # If using a loss function (i.e., criterion) that requires an optimiser, initialise it here.
    criterion_eval = EvalHard(margin=m)
    cuda_tr = True
    cuda_va = True
    cuda_te = True

    # Get list of all experimental settings for same-dataset configurations (part I):
    expt_settings_one_ds_osa = list(itertools.product(*[config_methods_one_ds, configs_one_ds, archs, batch_sizes,
                                                        channels, windows]))
    # Loop through all same-dataset configuration experiments:
    for (config_method, config, arch, batch_size, channel, window) in expt_settings_one_ds_osa:

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

        trial_names, labels, ids, counts_samples = get_base_metadata(config[0])

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

        trial_names, counts_samples, indices_sort_labels = balance_ids_in_ds(trial_names,
                                                                             labels,
                                                                             ids,
                                                                             counts_samples,
                                                                             n_ids_bal)
        sigs_all_r = sigs_all_r[indices_sort_labels]
        sigs_all_l = sigs_all_l[indices_sort_labels]

        indices_samples_where_id_changes = np.cumsum(counts_samples)[:-1]
        trial_names_by_id = np.split(trial_names, indices_samples_where_id_changes)

        # These experiments require that Part I already be conducted in main.py. Load the trial names for training,
        # validation, and test sets from Part I:
        trial_names_tr_by_id = load_list(os.path.join(results_path, 'trial_names_tr_by_id_%s_%s.txt' % (config_method,
                                                                                                        config[0])))
        trial_names_va_by_id = load_list(os.path.join(results_path, 'trial_names_va_by_id_%s_%s.txt' % (config_method,
                                                                                                        config[0])))
        trial_names_te_by_id = load_list(os.path.join(results_path, 'trial_names_te_by_id_%s_%s.txt' % (config_method,
                                                                                                        config[0])))
        losses_te_all, accs_te_all, embs_te_all = run_expt_osa(results_path, config, trial_names, sigs_all_r,
                                                               sigs_all_l, channel, window, trial_names_tr_by_id,
                                                               trial_names_va_by_id, trial_names_te_by_id,
                                                               n_samples_per_id_bal, batch_size, arch, criterion_eval,
                                                               cuda_te, single_input)

        # Quick check of mean test accuracy over 5-CV:
        acc_te = np.round(np.mean(accs_te_all) * 100, decimals=2)

        save_list(os.path.join(results_path, 'losses_te_osa_%s_%s_%s_%s_%s_%s.txt' %
                               (config_method, config[0], arch, batch_size, channel, window)), losses_te_all)
        save_list(os.path.join(results_path, 'accs_te_osa_%s_%s_%s_%s_%s_%s.txt' %
                               (config_method, config[0], arch, batch_size, channel, window)), accs_te_all)
        torch.save(embs_te_all, os.path.join(results_path, 'embs_te_osa_%s_%s_%s_%s_%s_%s.pth' %
                                             (config_method, config[0], arch, batch_size, channel, window)))

if make_fig:

    plt.rcParams["font.family"] = "Times New Roman"
    mpl.rcParams['mathtext.default'] = 'regular'

    # Note that if run_expts = True, a variable is defined at lines 42-45 as:
    # configs_one_ds = [['fi-all'],
    #                   ['gr-sho'],
    #                   ['gb-all'],
    #                   ['ai-all']]
    # The variable below is intentionally different. It does not contain inner lists because it is indexed to get a
    # string directly for loading a file name (line 165):
    configs_one_ds = ['fi-all',
                      'gr-sho',
                      'gb-all',
                      'ai-all']

    expt_settings_one_ds = list(itertools.product(*[config_methods_one_ds, configs_one_ds, archs, batch_sizes]))
    expt_settings_one_ds_osa = list(itertools.product(*[config_methods_one_ds, configs_one_ds, archs, batch_sizes,
                                                        channels, windows]))

    # Get the file names (as identifiers) and associated test accuracy measures from Part I:
    fnames = []
    accs = []

    for (config_method, config, arch, batch_size) in expt_settings_one_ds:

        fname = 'accs_te_%s_%s_%s_%s.txt' % (config_method, config, arch, batch_size)
        acc = load_list('./Results/accs_losses/%s' % fname)
        fnames.append(fname)
        accs.append(np.mean(np.array(acc) * 100)) # Mean accuracy over 5-CV.

    fnames = np.array(fnames)
    accs = np.array(accs)

    # Get the file names and accuracy measures for experiments using batch size N_X = 512 on each dataset. Note that
    # the experiments from Part I must have been completed with this batch size:
    fnames_512_fi = []
    fnames_512_gr = []
    fnames_512_gb = []
    fnames_512_ai = []

    accs_512_fi = []
    accs_512_gr = []
    accs_512_gb = []
    accs_512_ai = []

    for idx, name in enumerate(fnames):

        if name.__contains__('512') and name.__contains__('fi-all'):

            fnames_512_fi.append(name)
            accs_512_fi.append(accs[idx])

        elif name.__contains__('512') and name.__contains__('gr-sho'):

            fnames_512_gr.append(name)
            accs_512_gr.append(accs[idx])

        elif name.__contains__('512') and name.__contains__('gb-all'):

            fnames_512_gb.append(name)
            accs_512_gb.append(accs[idx])

        elif name.__contains__('512') and name.__contains__('ai-all'):

            fnames_512_ai.append(name)
            accs_512_ai.append(accs[idx])

    accs_512_fi = np.array(accs_512_fi)
    accs_512_gr = np.array(accs_512_gr)
    accs_512_gb = np.array(accs_512_gb)
    accs_512_ai = np.array(accs_512_ai)

    # Get the file names (as identifiers) and associated test accuracy measures from the occlusion sensitivity
    # experiments:
    fnames_osa = []
    accs_osa = []

    for (config_method, config, arch, batch_size, channel, window) in expt_settings_one_ds_osa:

        fname = 'accs_te_osa_%s_%s_%s_%s_%s_%s.txt' % (config_method, config, arch, batch_size, channel, window)
        acc = load_list('./Results/accs_te_osa/%s' % fname)
        fnames_osa.append(fname)
        accs_osa.append(np.mean(np.array(acc) * 100)) # Mean accuracy over 5-CV.

    # Get the file names and accuracy measures for experiments using batch size N_X = 512 on each dataset. Note that
    # the occlusion sensitivity experiments must have been completed with this batch size:
    fnames_osa_512_fi = []
    fnames_osa_512_gr = []
    fnames_osa_512_gb = []
    fnames_osa_512_ai = []

    accs_osa_512_fi = []
    accs_osa_512_gr = []
    accs_osa_512_gb = []
    accs_osa_512_ai = []

    for idx, name in enumerate(fnames_osa):

        if name.__contains__('512') and name.__contains__('fi-all'):

            fnames_osa_512_fi.append(name)
            accs_osa_512_fi.append(accs_osa[idx])

        elif name.__contains__('512') and name.__contains__('gr-sho'):

            fnames_osa_512_gr.append(name)
            accs_osa_512_gr.append(accs_osa[idx])

        elif name.__contains__('512') and name.__contains__('gb-all'):

            fnames_osa_512_gb.append(name)
            accs_osa_512_gb.append(accs_osa[idx])

        elif name.__contains__('512') and name.__contains__('ai-all'):

            fnames_osa_512_ai.append(name)
            accs_osa_512_ai.append(accs_osa[idx])

    accs_osa_512_fi = np.array(accs_osa_512_fi)
    accs_osa_512_gr = np.array(accs_osa_512_gr)
    accs_osa_512_gb = np.array(accs_osa_512_gb)
    accs_osa_512_ai = np.array(accs_osa_512_ai)

    # Get the differences in accuracy with vs. without the windows occluded:
    accs_te_diffs_fi = accs_512_fi - accs_osa_512_fi
    accs_te_diffs_gr = accs_512_gr - accs_osa_512_gr
    accs_te_diffs_gb = accs_512_gb - accs_osa_512_gb
    accs_te_diffs_ai = accs_512_ai - accs_osa_512_ai

    # Split the objects by the no. channels to get the accuracy differences on a per-channel basis:
    accs_te_diffs_by_channel_fi = np.array(np.split(accs_te_diffs_fi, len(channels), axis=0))
    accs_te_diffs_by_channel_gr = np.array(np.split(accs_te_diffs_gr, len(channels), axis=0))
    accs_te_diffs_by_channel_gb = np.array(np.split(accs_te_diffs_gb, len(channels), axis=0))
    accs_te_diffs_by_channel_ai = np.array(np.split(accs_te_diffs_ai, len(channels), axis=0))

    # We want a value for the difference in accuracy at each frame in the 200 frame sequence. So, initialise objects
    # of shape (C, L, V), where C = no. channels, L = sequence length (hard-coded), and V = no. cross-validation folds:
    accs_te_diffs_by_channel_fi_ext = np.zeros((len(channels), 200, n_folds))
    accs_te_diffs_by_channel_gr_ext = np.zeros((len(channels), 200, n_folds))
    accs_te_diffs_by_channel_gb_ext = np.zeros((len(channels), 200, n_folds))
    accs_te_diffs_by_channel_ai_ext = np.zeros((len(channels), 200, n_folds))

    for idx, val in enumerate(np.arange(0, 200 - window_size + 1, window_size)):

        # Set the values for each window according to the difference in accuracy due to occluding that window:
        accs_te_diffs_by_channel_fi_ext[:, val:val + window_size] = accs_te_diffs_by_channel_fi[:, idx * 2][:, None]
        accs_te_diffs_by_channel_gr_ext[:, val:val + window_size] = accs_te_diffs_by_channel_gr[:, idx * 2][:, None]
        accs_te_diffs_by_channel_gb_ext[:, val:val + window_size] = accs_te_diffs_by_channel_gb[:, idx * 2][:, None]
        accs_te_diffs_by_channel_ai_ext[:, val:val + window_size] = accs_te_diffs_by_channel_ai[:, idx * 2][:, None]

    # Get basic metadata:
    trial_names_fi, labels_fi, ids_fi, counts_samples_fi = get_base_metadata('fi-all')
    trial_names_gr, labels_gr, ids_gr, counts_samples_gr = get_base_metadata('gr-sho')
    trial_names_gb, labels_gb, ids_gb, counts_samples_gb = get_base_metadata('gb-all')
    trial_names_ai, labels_ai, ids_ai, counts_samples_ai = get_base_metadata('ai-all')

    # Load signals from each dataset:
    sigs_all_r = []
    sigs_all_l = []

    for dataset in configs_one_ds:

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

    # Get signals on a per-dataset basis:
    sigs_all_r_fi = sigs_all_r[0]
    sigs_all_r_gr = sigs_all_r[1]
    sigs_all_r_gb = sigs_all_r[2]
    sigs_all_r_ai = sigs_all_r[3]

    sigs_all_l_fi = sigs_all_l[0]
    sigs_all_l_gr = sigs_all_l[1]
    sigs_all_l_gb = sigs_all_l[2]
    sigs_all_l_ai = sigs_all_l[3]

    sigs_to_plot = []

    for cv_fold in range(n_folds):

        # Get the trial names for the test set from each dataset in Part I:
        trial_names_te_fi =\
            np.concatenate(load_list(os.path.join(results_path, 'trial_names_te_by_id_one------_fi-all.txt'))[cv_fold])
        trial_names_te_gr =\
            np.concatenate(load_list(os.path.join(results_path, 'trial_names_te_by_id_one------_gr-sho.txt'))[cv_fold])
        trial_names_te_gb =\
            np.concatenate(load_list(os.path.join(results_path, 'trial_names_te_by_id_one------_gb-all.txt'))[cv_fold])
        trial_names_te_ai =\
            np.concatenate(load_list(os.path.join(results_path, 'trial_names_te_by_id_one------_ai-all.txt'))[cv_fold])

        # Get the indices of the test set trial names in the broader sets of trial names:
        indices_trial_names_te_fi = []
        indices_trial_names_te_gr = []
        indices_trial_names_te_gb = []
        indices_trial_names_te_ai = []

        for i in range(trial_names_te_fi.shape[0]):

            indices_trial_names_te_fi.append(np.asarray(trial_names_fi == trial_names_te_fi[i]).nonzero()[0])
            indices_trial_names_te_gr.append(np.asarray(trial_names_gr == trial_names_te_gr[i]).nonzero()[0])
            indices_trial_names_te_gb.append(np.asarray(trial_names_gb == trial_names_te_gb[i]).nonzero()[0])
            indices_trial_names_te_ai.append(np.asarray(trial_names_ai == trial_names_te_ai[i]).nonzero()[0])

        indices_trial_names_te_fi = np.concatenate(indices_trial_names_te_fi)
        indices_trial_names_te_gr = np.concatenate(indices_trial_names_te_gr)
        indices_trial_names_te_gb = np.concatenate(indices_trial_names_te_gb)
        indices_trial_names_te_ai = np.concatenate(indices_trial_names_te_ai)

        # Then use the indices to get the signals for the test set from each dataset in Part I:
        sigs_all_r_te_fi = sigs_all_r_fi[indices_trial_names_te_fi]
        sigs_all_r_te_gr = sigs_all_r_gr[indices_trial_names_te_gr]
        sigs_all_r_te_gb = sigs_all_r_gb[indices_trial_names_te_gb]
        sigs_all_r_te_ai = sigs_all_r_ai[indices_trial_names_te_ai]

        sigs_all_l_te_fi = sigs_all_l_fi[indices_trial_names_te_fi]
        sigs_all_l_te_gr = sigs_all_l_gr[indices_trial_names_te_gr]
        sigs_all_l_te_gb = sigs_all_l_gb[indices_trial_names_te_gb]
        sigs_all_l_te_ai = sigs_all_l_ai[indices_trial_names_te_ai]

        # Get the mean across all signals from each stance side:
        sigs_all_te_fi_mean = np.mean(np.concatenate((sigs_all_r_te_fi, sigs_all_l_te_fi), axis=2), axis=0)
        sigs_all_te_gr_mean = np.mean(np.concatenate((sigs_all_r_te_gr, sigs_all_l_te_gr), axis=2), axis=0)
        sigs_all_te_gb_mean = np.mean(np.concatenate((sigs_all_r_te_gb, sigs_all_l_te_gb), axis=2), axis=0)
        sigs_all_te_ai_mean = np.mean(np.concatenate((sigs_all_r_te_ai, sigs_all_l_te_ai), axis=2), axis=0)

        if fig_main_text:

            # Just plot ForceID-A and GaitRec during left stance:
            sigs_to_plot.append([sigs_all_te_fi_mean[:, 100:],
                                 sigs_all_te_gr_mean[:, 100:]])

        else:

            # Plot all four datasets across both stance sides:
            sigs_to_plot.append([sigs_all_te_fi_mean,
                                 sigs_all_te_gr_mean,
                                 sigs_all_te_gb_mean,
                                 sigs_all_te_ai_mean])

    sigs_to_plot = np.array(sigs_to_plot).transpose((1, 0, 2, 3))

    if fig_main_text:

        # Add a dataset axis to the object containing the differences in accuracy with vs without windows occluded:
        accs_te_diffs_to_plot = np.concatenate(
            (accs_te_diffs_by_channel_fi_ext[None, :, 100:],
             accs_te_diffs_by_channel_gr_ext[None, :, 100:])).transpose((0, 3, 1, 2))

        fig = plt.figure(constrained_layout=True, figsize=(3.5, 5.5))
        x = np.arange(1, 101).astype(float)

    else:

        accs_te_diffs_to_plot = np.concatenate(
            (accs_te_diffs_by_channel_fi_ext[None, :],
             accs_te_diffs_by_channel_gr_ext[None, :],
             accs_te_diffs_by_channel_gb_ext[None, :],
             accs_te_diffs_by_channel_ai_ext[None, :])).transpose((0, 3, 1, 2))

        fig = plt.figure(constrained_layout=True, figsize=(7.0, 9.5))
        x = np.arange(1, 201).astype(float)

    # Additional analysis to get differences in accuracy for certain directional components and sub-phases (included in
    # the OSA results text):
    # ----------------------------------------------------------------------------------------------
    accs_te_diffs_fz_ms = accs_te_diffs_to_plot[:, :, 2, 40:60]
    accs_te_diffs_fz_is = accs_te_diffs_to_plot[:, :, 2, :20]
    accs_te_diffs_fy_ts = accs_te_diffs_to_plot[:, :, 1, 80:]
    accs_te_diffs_fx_is = accs_te_diffs_to_plot[:, :, 0, :20]
    accs_te_diffs_fx_ts = accs_te_diffs_to_plot[:, :, 0, 80:]
    accs_te_diffs_cx_ts = accs_te_diffs_to_plot[:, :, 3, 80:]

    # Executed the following line-by-line in the console to get ranges of difference in accuracy:
    # np.min(accs_te_diffs_fz_ms)
    # np.max(accs_te_diffs_fz_ms)
    # np.min(accs_te_diffs_fz_is)
    # np.max(accs_te_diffs_fz_is)
    # np.min(accs_te_diffs_fy_ts)
    # np.max(accs_te_diffs_fy_ts)
    # np.min(accs_te_diffs_fx_is)
    # np.max(accs_te_diffs_fx_is)
    # np.min(accs_te_diffs_fx_ts)
    # np.max(accs_te_diffs_fx_ts)
    # np.min(accs_te_diffs_cx_ts)
    # np.max(accs_te_diffs_cx_ts)
    # --------------------------------------------------------------------------------------------

    gs = fig.add_gridspec(5, 1) # Five rows - one for each channel.

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[3, 0])
    ax5 = fig.add_subplot(gs[4, 0])

    if window_size in [10, 20]:

        norm = plt.Normalize(0, 4) # This was hard-coded based on the range of difference in accuracy across all windows

    else: # 40

        norm = plt.Normalize(0, 5.5)

    for idx, ds in enumerate(sigs_to_plot):

        for cv_fold in range(n_folds):

            color_arr = np.concatenate(accs_te_diffs_to_plot[idx, cv_fold])

            points1 = np.array([x, ds[cv_fold, 0] + (idx * 7 * 40) + (cv_fold * 40)]).T.reshape(-1, 1, 2) # Fx
            points2 = np.array([x, ds[cv_fold, 1] + (idx * 7 * 120) + (cv_fold * 120)]).T.reshape(-1, 1, 2) # Fy
            points3 = np.array([x, ds[cv_fold, 2] + (idx * 7 * 400) + (cv_fold * 400)]).T.reshape(-1, 1, 2) # Fz

            if fig_main_text:

                points4 = np.array([x, ds[cv_fold, 3] + (idx * 7 * 0.02) + (cv_fold * 0.02)]).T.reshape(-1, 1, 2)  # Cx

            else:

                if idx == 3 and cv_fold == 0:

                    points4 = np.array([x, ds[cv_fold, 3] + (idx * 7 * 0.0197)]).T.reshape(-1, 1, 2)

                else:

                    points4 = np.array([x, ds[cv_fold, 3] + (idx * 7 * 0.02) + (cv_fold * 0.02)]).T.reshape(-1, 1, 2)

            points5 = np.array([x, ds[cv_fold, 4] + (idx * 7 * 0.2) + (cv_fold * 0.2)]).T.reshape(-1, 1, 2) # Cy

            segments1 = np.concatenate([points1[:-1], points1[1:]], axis=1)
            segments2 = np.concatenate([points2[:-1], points2[1:]], axis=1)
            segments3 = np.concatenate([points3[:-1], points3[1:]], axis=1)
            segments4 = np.concatenate([points4[:-1], points4[1:]], axis=1)
            segments5 = np.concatenate([points5[:-1], points5[1:]], axis=1)

            lc1 = LineCollection(segments1, cmap='plasma', norm=norm)
            lc2 = LineCollection(segments2, cmap='plasma', norm=norm)
            lc3 = LineCollection(segments3, cmap='plasma', norm=norm)
            lc4 = LineCollection(segments4, cmap='plasma', norm=norm)
            lc5 = LineCollection(segments5, cmap='plasma', norm=norm)

            if fig_main_text:

                lc1.set_array(color_arr[:100])
                lc2.set_array(color_arr[100:200])
                lc3.set_array(color_arr[200:300])
                lc4.set_array(color_arr[300:400])
                lc5.set_array(color_arr[400:])

            else:

                lc1.set_array(color_arr[:200])
                lc2.set_array(color_arr[200:400])
                lc3.set_array(color_arr[400:600])
                lc4.set_array(color_arr[600:800])
                lc5.set_array(color_arr[800:])

            lc1.set_linewidth(2)
            lc2.set_linewidth(2)
            lc3.set_linewidth(2)
            lc4.set_linewidth(2)
            lc5.set_linewidth(2)

            line1 = ax1.add_collection(lc1)
            line2 = ax2.add_collection(lc2)
            line3 = ax3.add_collection(lc3)
            line4 = ax4.add_collection(lc4)
            line5 = ax5.add_collection(lc5)

    axs = [ax1, ax2, ax3, ax4, ax5]

    if fig_main_text:

        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='plasma'), ax=axs, aspect=40, pad=0.01)

        ax1.set_xlim(0, 101)
        ax2.set_xlim(0, 101)
        ax3.set_xlim(0, 101)
        ax4.set_xlim(0, 101)
        ax5.set_xlim(0, 101)

        ax1.set_xticks(np.arange(0, 101, 20))
        ax2.set_xticks(np.arange(0, 101, 20))
        ax3.set_xticks(np.arange(0, 101, 20))
        ax4.set_xticks(np.arange(0, 101, 20))
        ax5.set_xticks(np.arange(0, 101, 20))

        ax5.set_xticklabels(np.arange(0, 101, 20))

        ax1.set_ylim(-60, 560)
        ax2.set_ylim(-250, 1600)
        ax3.set_ylim(-500, 5700)
        ax4.set_ylim(-0.02, 0.26)
        ax5.set_ylim(-0.2, 2.65)

        plt.xlabel('Percent stance (%)', size=11, labelpad=3)

    else:

        if window_size == 40:

            cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='plasma'), ax=axs, aspect=45, pad=0.01,
                                ticks=np.arange(0, 5.6, 0.5))
        else:

            cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='plasma'), ax=axs, aspect=45, pad=0.01)

        ax1.set_xlim(0, 201)
        ax2.set_xlim(0, 201)
        ax3.set_xlim(0, 201)
        ax4.set_xlim(0, 201)
        ax5.set_xlim(0, 201)

        ax1.set_xticks(np.arange(0, 201, 20))
        ax2.set_xticks(np.arange(0, 201, 20))
        ax3.set_xticks(np.arange(0, 201, 20))
        ax4.set_xticks(np.arange(0, 201, 20))
        ax5.set_xticks(np.arange(0, 201, 20))

        ax1.set_xticklabels([])
        ax2.set_xticklabels([])
        ax3.set_xticklabels([])
        ax4.set_xticklabels([])
        ax5.set_xticklabels(np.arange(0, 201, 20))

        ax1.set_ylim(-60, 1150)
        ax2.set_ylim(-250, 3250)
        ax3.set_ylim(-500, 11250)
        ax4.set_ylim(-0.02, 0.55)
        ax5.set_ylim(-0.2, 5.4)

        plt.xlabel('Frame number', size=11, labelpad=3)

    cbar.set_label('Decrease in accuracy (%)', size=11, rotation=270, labelpad=15)

    ax1.set_yticks([])
    ax2.set_yticks([])
    ax3.set_yticks([])
    ax4.set_yticks([])
    ax5.set_yticks([])

    ax1.set_yticklabels([])
    ax2.set_yticklabels([])
    ax3.set_yticklabels([])
    ax4.set_yticklabels([])
    ax5.set_yticklabels([])

    ax1.set_ylabel('$GRF_x$', size=11, style='italic', labelpad=-1)
    ax2.set_ylabel('$GRF_y$', size=11, style='italic', labelpad=-1)
    ax3.set_ylabel('$GRF_z$', size=11, style='italic', labelpad=-1)
    ax4.set_ylabel('$COP_x$', size=11, style='italic', labelpad=-1)
    ax5.set_ylabel('$COP_y$', size=11, style='italic', labelpad=-1)

    if save_fig:

        if fig_main_text:

            plt.savefig('./Figures/fig_osa_main_%s.pdf' % window_size, dpi=1200)
            plt.savefig('./Figures/fig_osa_main_%s.png' % window_size, dpi=1200)
            plt.savefig('./Figures/fig_osa_main_%s.svg' % window_size, dpi=1200)

        else:

            plt.savefig('./Figures/fig_osa_supp_%s.pdf' % window_size, dpi=1200)
            plt.savefig('./Figures/fig_osa_supp_%s.png' % window_size, dpi=1200)
            plt.savefig('./Figures/fig_osa_supp_%s.svg' % window_size, dpi=1200)

    if show_fig:

        plt.show()

