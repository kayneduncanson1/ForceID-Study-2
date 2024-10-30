import numpy as np
import itertools
from Utils import load_list

"""This script was used to load and inspect recognition accuracy obtained on training, validation, and test sets in
all experiments. There are also sections that get the cumulative rank scores for batch sizes and architectures
(respectively) to rank them in terms of accuracy. These sections were applied to rank batch sizes and architectures on
the test set to obtain the performance results reported in the main text and Supplementary Information."""

# Specify which subset (training/validation/test) to analyse:
subsets = ['te'] # Options: 'tr', 'va', 'te', or a combination thereof.
# For cases where subsets = ['tr'], specify whether to get the maximum accuracy on the training set across all
# epochs or the accuracy on the training set at the epoch where the accuracy on the validation set was maximum:
summary_metric_tr = 'max_va_acc' # Options: 'max_tr_acc' or 'max_va_acc'
# Specify batch size(s) to analyse:
batch_sizes = ['128', '256', '512']
# Specify architecture(s) to analyse:
archs = np.array(['-----1F', '-----2F', '1C---1F', '1C---2F', '3C---1F',
                  '3C---2F', '3C1LU1F', '3C1LU2F', '3C1LB1F', '3C1LB2F',
                  '1C1T-1F', '1C1T-2F', '3C1T-1F', '3C1T-2F', '--1T-1F',
                  '--1T-2F', '--2T-1F', '--2T-2F', '--3T-1F', '--3T-2F'])

# List specific dataset configurations:
configs_one_ds = ['fi-all',
                  'gr-sho',
                  'gb-all',
                  'ai-all']
# Variable nomenclature: 'configs_[no. datasets]_ds_[config method]_[dataset used for testing]':
configs_two_ds_aug_fi = ['fi-all-gr-sho',
                         'fi-all-gb-all',
                         'fi-all-ai-all']
configs_two_ds_aug_gr = ['gr-sho-fi-all',
                         'gr-sho-gb-all',
                         'gr-sho-ai-all']
configs_two_ds_aug_gb = ['gb-all-fi-all',
                         'gb-all-gr-sho',
                         'gb-all-ai-all']
configs_two_ds_aug_ai = ['ai-all-fi-all',
                         'ai-all-gr-sho',
                         'ai-all-gb-all']
configs_thr_ds_aug_fi = ['fi-all-gr-sho-gb-all',
                         'fi-all-gb-all-ai-all',
                         'fi-all-gr-sho-ai-all']
configs_thr_ds_aug_gr = ['gr-sho-fi-all-gb-all',
                         'gr-sho-gb-all-ai-all',
                         'gr-sho-fi-all-ai-all']
configs_thr_ds_aug_gb = ['gb-all-fi-all-gr-sho',
                         'gb-all-gr-sho-ai-all',
                         'gb-all-fi-all-ai-all']
configs_thr_ds_aug_ai = ['ai-all-fi-all-gr-sho',
                         'ai-all-fi-all-gb-all',
                         'ai-all-gr-sho-gb-all']
configs_fou_ds_aug_fi = ['fi-all-gr-sho-gb-all-ai-all']
configs_fou_ds_aug_gr = ['gr-sho-fi-all-gb-all-ai-all']
configs_fou_ds_aug_gb = ['gb-all-fi-all-gr-sho-ai-all']
configs_fou_ds_aug_ai = ['ai-all-fi-all-gr-sho-gb-all']

configs_two_ds_cro_fi = ['gr-sho-fi-all',
                         'gb-all-fi-all',
                         'ai-all-fi-all']
configs_two_ds_cro_gr = ['fi-all-gr-sho',
                         'gb-all-gr-sho',
                         'ai-all-gr-sho']
configs_two_ds_cro_gb = ['fi-all-gb-all',
                         'gr-sho-gb-all',
                         'ai-all-gb-all']
configs_two_ds_cro_ai = ['fi-all-ai-all',
                         'gr-sho-ai-all',
                         'gb-all-ai-all']
configs_thr_ds_cro_fi = ['gr-sho-gb-all-fi-all',
                         'gb-all-ai-all-fi-all',
                         'gr-sho-ai-all-fi-all']
configs_thr_ds_cro_gr = ['fi-all-gb-all-gr-sho',
                         'fi-all-ai-all-gr-sho',
                         'gb-all-ai-all-gr-sho']
configs_thr_ds_cro_gb = ['fi-all-gr-sho-gb-all',
                         'fi-all-ai-all-gb-all',
                         'gr-sho-ai-all-gb-all']
configs_thr_ds_cro_ai = ['fi-all-gr-sho-ai-all',
                         'fi-all-gb-all-ai-all',
                         'gr-sho-gb-all-ai-all']
configs_fou_ds_cro_fi = ['gr-sho-gb-all-ai-all-fi-all']
configs_fou_ds_cro_gr = ['fi-all-gb-all-ai-all-gr-sho']
configs_fou_ds_cro_gb = ['fi-all-gr-sho-ai-all-gb-all']
configs_fou_ds_cro_ai = ['fi-all-gr-sho-gb-all-ai-all']

configs_all = [configs_one_ds,
               configs_two_ds_aug_fi,
               configs_two_ds_aug_gr,
               configs_two_ds_aug_gb,
               configs_two_ds_aug_ai,
               configs_thr_ds_aug_fi,
               configs_thr_ds_aug_gr,
               configs_thr_ds_aug_gb,
               configs_thr_ds_aug_ai,
               configs_fou_ds_aug_fi,
               configs_fou_ds_aug_gr,
               configs_fou_ds_aug_gb,
               configs_fou_ds_aug_ai,
               configs_two_ds_cro_fi,
               configs_two_ds_cro_gr,
               configs_two_ds_cro_gb,
               configs_two_ds_cro_ai,
               configs_thr_ds_cro_fi,
               configs_thr_ds_cro_gr,
               configs_thr_ds_cro_gb,
               configs_thr_ds_cro_ai,
               configs_fou_ds_cro_fi,
               configs_fou_ds_cro_gr,
               configs_fou_ds_cro_gb,
               configs_fou_ds_cro_ai]

expt_settings_one_ds = list(itertools.product(*[subsets, ['one------'],
                                                configs_one_ds, archs, batch_sizes]))
expt_settings_two_ds_aug_fi = list(itertools.product(*[subsets, ['two-aug--'],
                                                       configs_two_ds_aug_fi, archs, batch_sizes]))
expt_settings_two_ds_aug_gr = list(itertools.product(*[subsets, ['two-aug--'],
                                                       configs_two_ds_aug_gr, archs, batch_sizes]))
expt_settings_two_ds_aug_gb = list(itertools.product(*[subsets, ['two-aug--'],
                                                       configs_two_ds_aug_gb, archs, batch_sizes]))
expt_settings_two_ds_aug_ai = list(itertools.product(*[subsets, ['two-aug--'],
                                                       configs_two_ds_aug_ai, archs, batch_sizes]))
expt_settings_thr_ds_aug_fi = list(itertools.product(*[subsets, ['thr-aug--'],
                                                       configs_thr_ds_aug_fi, archs, batch_sizes]))
expt_settings_thr_ds_aug_gr = list(itertools.product(*[subsets, ['thr-aug--'],
                                                       configs_thr_ds_aug_gr, archs, batch_sizes]))
expt_settings_thr_ds_aug_gb = list(itertools.product(*[subsets, ['thr-aug--'],
                                                       configs_thr_ds_aug_gb, archs, batch_sizes]))
expt_settings_thr_ds_aug_ai = list(itertools.product(*[subsets, ['thr-aug--'],
                                                       configs_thr_ds_aug_ai, archs, batch_sizes]))
expt_settings_fou_ds_aug_fi = list(itertools.product(*[subsets, ['fou-aug--'],
                                                       configs_fou_ds_aug_fi, archs, batch_sizes]))
expt_settings_fou_ds_aug_gr = list(itertools.product(*[subsets, ['fou-aug--'],
                                                       configs_fou_ds_aug_gr, archs, batch_sizes]))
expt_settings_fou_ds_aug_gb = list(itertools.product(*[subsets, ['fou-aug--'],
                                                       configs_fou_ds_aug_gb, archs, batch_sizes]))
expt_settings_fou_ds_aug_ai = list(itertools.product(*[subsets, ['fou-aug--'],
                                                       configs_fou_ds_aug_ai, archs, batch_sizes]))

expt_settings_two_ds_cro_r_fi = list(itertools.product(*[subsets, ['two-cro-r'],
                                                         configs_two_ds_cro_fi, archs, batch_sizes]))
expt_settings_two_ds_cro_r_gr = list(itertools.product(*[subsets, ['two-cro-r'],
                                                         configs_two_ds_cro_gr, archs, batch_sizes]))
expt_settings_two_ds_cro_r_gb = list(itertools.product(*[subsets, ['two-cro-r'],
                                                         configs_two_ds_cro_gb, archs, batch_sizes]))
expt_settings_two_ds_cro_r_ai = list(itertools.product(*[subsets, ['two-cro-r'],
                                                         configs_two_ds_cro_ai, archs, batch_sizes]))
expt_settings_two_ds_cro_fi = list(itertools.product(*[subsets, ['two-cro--'],
                                                       configs_two_ds_cro_fi, archs, batch_sizes]))
expt_settings_two_ds_cro_gr = list(itertools.product(*[subsets, ['two-cro--'],
                                                       configs_two_ds_cro_gr, archs, batch_sizes]))
expt_settings_two_ds_cro_gb = list(itertools.product(*[subsets, ['two-cro--'],
                                                       configs_two_ds_cro_gb, archs, batch_sizes]))
expt_settings_two_ds_cro_ai = list(itertools.product(*[subsets, ['two-cro--'],
                                                       configs_two_ds_cro_ai, archs, batch_sizes]))
expt_settings_thr_ds_cro_fi = list(itertools.product(*[subsets, ['thr-cro--'],
                                                       configs_thr_ds_cro_fi, archs, batch_sizes]))
expt_settings_thr_ds_cro_gr = list(itertools.product(*[subsets, ['thr-cro--'],
                                                       configs_thr_ds_cro_gr, archs, batch_sizes]))
expt_settings_thr_ds_cro_gb = list(itertools.product(*[subsets, ['thr-cro--'],
                                                       configs_thr_ds_cro_gb, archs, batch_sizes]))
expt_settings_thr_ds_cro_ai = list(itertools.product(*[subsets, ['thr-cro--'],
                                                       configs_thr_ds_cro_ai, archs, batch_sizes]))
expt_settings_fou_ds_cro_fi = list(itertools.product(*[subsets, ['fou-cro--'],
                                                       configs_fou_ds_cro_fi, archs, batch_sizes]))
expt_settings_fou_ds_cro_gr = list(itertools.product(*[subsets, ['fou-cro--'],
                                                       configs_fou_ds_cro_gr, archs, batch_sizes]))
expt_settings_fou_ds_cro_gb = list(itertools.product(*[subsets, ['fou-cro--'],
                                                       configs_fou_ds_cro_gb, archs, batch_sizes]))
expt_settings_fou_ds_cro_ai = list(itertools.product(*[subsets, ['fou-cro--'],
                                                       configs_fou_ds_cro_ai, archs, batch_sizes]))

expt_settings_all = [expt_settings_one_ds,
                     expt_settings_two_ds_aug_fi,
                     expt_settings_two_ds_aug_gr,
                     expt_settings_two_ds_aug_gb,
                     expt_settings_two_ds_aug_ai,
                     expt_settings_thr_ds_aug_fi,
                     expt_settings_thr_ds_aug_gr,
                     expt_settings_thr_ds_aug_gb,
                     expt_settings_thr_ds_aug_ai,
                     expt_settings_fou_ds_aug_fi,
                     expt_settings_fou_ds_aug_gr,
                     expt_settings_fou_ds_aug_gb,
                     expt_settings_fou_ds_aug_ai,
                     expt_settings_two_ds_cro_r_fi,
                     expt_settings_two_ds_cro_r_gr,
                     expt_settings_two_ds_cro_r_gb,
                     expt_settings_two_ds_cro_r_ai,
                     expt_settings_two_ds_cro_fi,
                     expt_settings_two_ds_cro_gr,
                     expt_settings_two_ds_cro_gb,
                     expt_settings_two_ds_cro_ai,
                     expt_settings_thr_ds_cro_fi,
                     expt_settings_thr_ds_cro_gr,
                     expt_settings_thr_ds_cro_gb,
                     expt_settings_thr_ds_cro_ai,
                     expt_settings_fou_ds_cro_fi,
                     expt_settings_fou_ds_cro_gr,
                     expt_settings_fou_ds_cro_gb,
                     expt_settings_fou_ds_cro_ai]

# Get file names and measures of mean accuracy over 5-CV for the subsets of experiments defined above:
fnames_by_expt_subset = []
accs_by_expt_subset = []

for idx_expts_subset, expts_subset in enumerate(expt_settings_all):

    fnames = []
    accs = []

    for (data_subset, config_method, config, arch, batch_size) in expts_subset:

        if data_subset == 'te':

            fname = 'accs_te_%s_%s_%s_%s.txt' % (config_method, config, arch, batch_size)
            # Test accuracy obtained on each CV fold:
            acc = load_list('./Results/accs_losses/%s' % fname)

        else:

            fname_va = 'accs_va_%s_%s_%s_%s.txt' % (config_method, config, arch, batch_size)
            # Validation accuracy obtained on each epoch in each CV fold:
            acc_va = load_list('./Results/accs_losses/%s' % fname_va)

            if data_subset == 'va':

                fname = fname_va
                # Get the maximum accuracy obtained on the validation set across epochs in each CV fold:
                acc = [np.max(item) for item in acc_va]

            else:

                fname = 'accs_tr_%s_%s_%s_%s.txt' % (config_method, config, arch, batch_size)
                # Training accuracy obtained on each epoch in each CV fold:
                acc_tr = load_list('./Results/accs_losses/%s' % fname)

                acc = []

                for cv_fold in range(len(acc_tr)):

                    if summary_metric_tr == 'max_va_acc':

                        index_va_acc_max = np.argmax(acc_va[cv_fold])
                        acc.append(acc_tr[cv_fold][index_va_acc_max])

                    else: # 'max_tr_acc'

                        acc.append(np.max(acc_tr[cv_fold]))

        fnames.append(fname)
        accs.append(np.mean(np.array(acc) * 100))

    fnames_by_expt_subset.append(np.array(fnames))
    accs_by_expt_subset.append(np.array(accs))

# Get file names and measures of mean accuracy over 5-CV on a per-experiment basis:
fnames_all = np.concatenate(fnames_by_expt_subset)
accs_all = np.concatenate(accs_by_expt_subset)

# Get cumulative rank scores for batch sizes...
# First, get accuracy measures for each batch size:
fnames_128 = []
fnames_256 = []
fnames_512 = []

accs_128 = []
accs_256 = []
accs_512 = []

for idx, name in enumerate(fnames_all):

    if name.__contains__('128'):

        fnames_128.append(name)
        accs_128.append(accs_all[idx])

    elif name.__contains__('256'):

        fnames_256.append(name)
        accs_256.append(accs_all[idx])

    elif name.__contains__('512'):

        fnames_512.append(name)
        accs_512.append(accs_all[idx])

accs_128 = np.array(accs_128)
accs_256 = np.array(accs_256)
accs_512 = np.array(accs_512)

# Combine accuracy measures for each batch size along a new axis (0):
accs_by_bs = np.concatenate((accs_128[None, :],
                             accs_256[None, :],
                             accs_512[None, :]), axis=0)
# Get the indices that sort batch sizes for each experiment and flip such that accuracy would be sorted in descending
# order.
indices_sort_bs = np.flip(np.argsort(accs_by_bs.transpose(), axis=1), axis=1)

# For each batch size in each experiment, find where the index of that batch size was in the sorted indices to get its
# ranking. Then, add 1 to each ranking so that they range from 1-3 rather than 0-2:
bs_ranks = []

for indices_sort_bs_expt in indices_sort_bs:

    bs_ranks_expt = []

    for idx_bs in range(indices_sort_bs_expt.shape[0]):

        bs_ranks_expt.append(np.asarray(indices_sort_bs_expt == idx_bs).nonzero()[0] + 1)

    bs_ranks.append(np.array(bs_ranks_expt).squeeze())

bs_ranks = np.array(bs_ranks)
# Sum rankings across all experiments:
bs_ranks_sum = np.sum(bs_ranks, axis=0)
# Sort the cumulative ranking scores from lowest to highest:
indices_sort_bs_ranks_sum = np.argsort(bs_ranks_sum)
bs_ranks_sum_sorted = bs_ranks_sum[indices_sort_bs_ranks_sum]
# Get batch sizes sorted in order of overall ranking (top ranked batch size = lowest cumulative rank score):
bs_sorted = np.array(batch_sizes)[indices_sort_bs_ranks_sum]


# For the best batch size (512), get cumulative rank scores for architectures...
# First, get accuracy measures for each architecture:
fnames_512_1f = []
fnames_512_2f = []
fnames_512_1c1f = []
fnames_512_1c2f = []
fnames_512_3c1f = []
fnames_512_3c2f = []
fnames_512_3c1lu1f = []
fnames_512_3c1lu2f = []
fnames_512_3c1lb1f = []
fnames_512_3c1lb2f = []
fnames_512_1c1t1f = []
fnames_512_1c1t2f = []
fnames_512_3c1t1f = []
fnames_512_3c1t2f = []
fnames_512_1t1f = []
fnames_512_1t2f = []
fnames_512_2t1f = []
fnames_512_2t2f = []
fnames_512_3t1f = []
fnames_512_3t2f = []

accs_512_1f = []
accs_512_2f = []
accs_512_1c1f = []
accs_512_1c2f = []
accs_512_3c1f = []
accs_512_3c2f = []
accs_512_3c1lu1f = []
accs_512_3c1lu2f = []
accs_512_3c1lb1f = []
accs_512_3c1lb2f = []
accs_512_1c1t1f = []
accs_512_1c1t2f = []
accs_512_3c1t1f = []
accs_512_3c1t2f = []
accs_512_1t1f = []
accs_512_1t2f = []
accs_512_2t1f = []
accs_512_2t2f = []
accs_512_3t1f = []
accs_512_3t2f = []

for idx, name in enumerate(fnames_512):

    if name.__contains__('-----1F'):

        fnames_512_1f.append(name)
        accs_512_1f.append(accs_512[idx])

    elif name.__contains__('-----2F'):

        fnames_512_2f.append(name)
        accs_512_2f.append(accs_512[idx])

    elif name.__contains__('1C---1F'):

        fnames_512_1c1f.append(name)
        accs_512_1c1f.append(accs_512[idx])

    elif name.__contains__('1C---2F'):

        fnames_512_1c2f.append(name)
        accs_512_1c2f.append(accs_512[idx])

    elif name.__contains__('3C---1F'):

        fnames_512_3c1f.append(name)
        accs_512_3c1f.append(accs_512[idx])

    elif name.__contains__('3C---2F'):

        fnames_512_3c2f.append(name)
        accs_512_3c2f.append(accs_512[idx])

    elif name.__contains__('3C1LU1F'):

        fnames_512_3c1lu1f.append(name)
        accs_512_3c1lu1f.append(accs_512[idx])

    elif name.__contains__('3C1LU2F'):

        fnames_512_3c1lu2f.append(name)
        accs_512_3c1lu2f.append(accs_512[idx])

    elif name.__contains__('3C1LB1F'):

        fnames_512_3c1lb1f.append(name)
        accs_512_3c1lb1f.append(accs_512[idx])

    elif name.__contains__('3C1LB2F'):

        fnames_512_3c1lb2f.append(name)
        accs_512_3c1lb2f.append(accs_512[idx])

    elif name.__contains__('1C1T-1F'):

        fnames_512_1c1t1f.append(name)
        accs_512_1c1t1f.append(accs_512[idx])

    elif name.__contains__('1C1T-2F'):

        fnames_512_1c1t2f.append(name)
        accs_512_1c1t2f.append(accs_512[idx])

    elif name.__contains__('3C1T-1F'):

        fnames_512_3c1t1f.append(name)
        accs_512_3c1t1f.append(accs_512[idx])

    elif name.__contains__('3C1T-2F'):

        fnames_512_3c1t2f.append(name)
        accs_512_3c1t2f.append(accs_512[idx])

    elif name.__contains__('--1T-1F'):

        fnames_512_1t1f.append(name)
        accs_512_1t1f.append(accs_512[idx])

    elif name.__contains__('--1T-2F'):

        fnames_512_1t2f.append(name)
        accs_512_1t2f.append(accs_512[idx])

    elif name.__contains__('--2T-1F'):

        fnames_512_2t1f.append(name)
        accs_512_2t1f.append(accs_512[idx])

    elif name.__contains__('--2T-2F'):

        fnames_512_2t2f.append(name)
        accs_512_2t2f.append(accs_512[idx])

    elif name.__contains__('--3T-1F'):

        fnames_512_3t1f.append(name)
        accs_512_3t1f.append(accs_512[idx])

    elif name.__contains__('--3T-2F'):

        fnames_512_3t2f.append(name)
        accs_512_3t2f.append(accs_512[idx])

accs_512_1f = np.array(accs_512_1f)
accs_512_2f = np.array(accs_512_2f)
accs_512_1c1f = np.array(accs_512_1c1f)
accs_512_1c2f = np.array(accs_512_1c2f)
accs_512_3c1f = np.array(accs_512_3c1f)
accs_512_3c2f = np.array(accs_512_3c2f)
accs_512_3c1lu1f = np.array(accs_512_3c1lu1f)
accs_512_3c1lu2f = np.array(accs_512_3c1lu2f)
accs_512_3c1lb1f = np.array(accs_512_3c1lb1f)
accs_512_3c1lb2f = np.array(accs_512_3c1lb2f)
accs_512_1c1t1f = np.array(accs_512_1c1t1f)
accs_512_1c1t2f = np.array(accs_512_1c1t2f)
accs_512_3c1t1f = np.array(accs_512_3c1t1f)
accs_512_3c1t2f = np.array(accs_512_3c1t2f)
accs_512_1t1f = np.array(accs_512_1t1f)
accs_512_1t2f = np.array(accs_512_1t2f)
accs_512_2t1f = np.array(accs_512_2t1f)
accs_512_2t2f = np.array(accs_512_2t2f)
accs_512_3t1f = np.array(accs_512_3t1f)
accs_512_3t2f = np.array(accs_512_3t2f)

# Combine accuracy measures for each architecture along a new axis (0):
accs_512_by_arch = np.concatenate((accs_512_1f[None, :],
                                   accs_512_2f[None, :],
                                   accs_512_1c1f[None, :],
                                   accs_512_1c2f[None, :],
                                   accs_512_3c1f[None, :],
                                   accs_512_3c2f[None, :],
                                   accs_512_3c1lu1f[None, :],
                                   accs_512_3c1lu2f[None, :],
                                   accs_512_3c1lb1f[None, :],
                                   accs_512_3c1lb2f[None, :],
                                   accs_512_1c1t1f[None, :],
                                   accs_512_1c1t2f[None, :],
                                   accs_512_3c1t1f[None, :],
                                   accs_512_3c1t2f[None, :],
                                   accs_512_1t1f[None, :],
                                   accs_512_1t2f[None, :],
                                   accs_512_2t1f[None, :],
                                   accs_512_2t2f[None, :],
                                   accs_512_3t1f[None, :],
                                   accs_512_3t2f[None, :]),
                                  axis=0)

# Get the indices that sort architectures for each experiment and flip such that accuracy would be sorted in descending
# order.
indices_sort_archs = np.flip(np.argsort(accs_512_by_arch.transpose(), axis=1), axis=1)

# For each architecture in each experiment, find where the index of that architecture was in the sorted indices to get
# its ranking. Then, add 1 to each ranking so that they range from 1-20 rather than 0-19:
arch_ranks_all = []

for indices_sort_archs_config in indices_sort_archs:

    arch_ranks = []

    for idx_arch in range(indices_sort_archs_config.shape[0]):

        arch_ranks.append(np.asarray(indices_sort_archs_config == idx_arch).nonzero()[0] + 1)

    arch_ranks_all.append(np.array(arch_ranks).squeeze())

arch_ranks_all = np.array(arch_ranks_all)
# Sum rankings across all experiments:
arch_ranks_sum = np.sum(arch_ranks_all, axis=0)
# Sort the cumulative ranking scores from lowest to highest:
indices_sort_archs_ranks_sum = np.argsort(arch_ranks_sum)
arch_ranks_sum_sorted = arch_ranks_sum[indices_sort_archs_ranks_sum]
# Get architectures sorted in order of overall ranking (top ranked architecture = lowest cumulative rank score):
archs_sorted = archs[indices_sort_archs_ranks_sum]
archs_top_five = archs_sorted[:5]

# Get accuracy measures from top five archs in same-dataset (part I) and mixed-dataset (part II) configurations (rounded
# to two decimal places):
fnames_512_3c1t2f_same_mix = []
fnames_512_2f_same_mix = []
fnames_512_1c1t2f_same_mix = []
fnames_512_1c1t1f_same_mix = []
fnames_512_1c2f_same_mix = []

accs_512_3c1t2f_same_mix = []
accs_512_2f_same_mix = []
accs_512_1c1t2f_same_mix = []
accs_512_1c1t1f_same_mix = []
accs_512_1c2f_same_mix = []

for idx, name in enumerate(fnames_512_3c1t2f):

    if name.__contains__('one------') or name.__contains__('aug'):

        fnames_512_3c1t2f_same_mix.append(name)
        fnames_512_2f_same_mix.append(name)
        fnames_512_1c1t2f_same_mix.append(name)
        fnames_512_1c1t1f_same_mix.append(name)
        fnames_512_1c2f_same_mix.append(name)

        accs_512_3c1t2f_same_mix.append(np.round(accs_512_3c1t2f[idx], decimals=2))
        accs_512_2f_same_mix.append(np.round(accs_512_2f[idx], decimals=2))
        accs_512_1c1t2f_same_mix.append(np.round(accs_512_1c1t2f[idx], decimals=2))
        accs_512_1c1t1f_same_mix.append(np.round(accs_512_1c1t1f[idx], decimals=2))
        accs_512_1c2f_same_mix.append(np.round(accs_512_1c2f[idx], decimals=2))

# Get accuracy measures from top five archs in same-dataset (part I) and cross-dataset (part III) configurations
# (rounded to two decimal places):
fnames_512_3c1t2f_same_cro = []
fnames_512_2f_same_cro = []
fnames_512_1c1t2f_same_cro = []
fnames_512_1c1t1f_same_cro = []
fnames_512_1c2f_same_cro = []

accs_512_3c1t2f_same_cro = []
accs_512_2f_same_cro = []
accs_512_1c1t2f_same_cro = []
accs_512_1c1t1f_same_cro = []
accs_512_1c2f_same_cro = []

for idx, name in enumerate(fnames_512_3c1t2f):

    if name.__contains__('one------') or name.__contains__('cro'):

        fnames_512_3c1t2f_same_cro.append(name)
        fnames_512_2f_same_cro.append(name)
        fnames_512_1c1t2f_same_cro.append(name)
        fnames_512_1c1t1f_same_cro.append(name)
        fnames_512_1c2f_same_cro.append(name)

        accs_512_3c1t2f_same_cro.append(np.round(accs_512_3c1t2f[idx], decimals=2))
        accs_512_2f_same_cro.append(np.round(accs_512_2f[idx], decimals=2))
        accs_512_1c1t2f_same_cro.append(np.round(accs_512_1c1t2f[idx], decimals=2))
        accs_512_1c1t1f_same_cro.append(np.round(accs_512_1c1t1f[idx], decimals=2))
        accs_512_1c2f_same_cro.append(np.round(accs_512_1c2f[idx], decimals=2))