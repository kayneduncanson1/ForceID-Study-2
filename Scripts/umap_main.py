import numpy as np
import os
import torch
import umap.plot
import matplotlib.pyplot as plt
from Utils import set_seed, get_full_metadata, load_list, get_unique_unsorted, points
from PrePro import get_directional_components
from Models import TwoF
from sklearn.preprocessing import StandardScaler

"""This script was used to create the UMAP figure in the main text."""

g = set_seed()

show_fig = False
save_fig = True

# UMAP params:
n_neighbors = 100
min_dist = 0.0
n_components = 2
metric = 'euclidean'

# Initialise standard scaler to perform z-score normalisation later:
s_scaler = StandardScaler()

# Specify experimental params:
datasets = ['fi-all', 'gr-all', 'gr-sho', 'gb-all', 'ai-all']
arch = '-----2F'
bs = 512
cv_fold = 1 # Note that this is the index, so actually cv fold 2.
cuda = False
results_path = './Results/main'

# Specify the attributes that are used to colour code UMAP plots:
attributes_to_plot = ['Datasets', 'Footwear', 'Walking speed', 'Age (y)', 'Height (m)', 'Mass (kg)', 'Sex']

# Get all metadata for each dataset. (Note that the subset of GaitRec containing both shod and barefoot walking --
# gr_all -- is included. It will later be plotted and labelled 'GaitRec-M' for 'mixed' footwear.):
trial_names_fi_all, labels_fi_all, ids_fi_all, counts_samples_fi_all, sexes_fi_all, ages_fi_all, heights_fi_all,\
masses_fi_all, footwear_fi_all, speeds_fi_all = get_full_metadata(datasets[0])
trial_names_gr_all, labels_gr_all, ids_gr_all, counts_samples_gr_all, sexes_gr_all, ages_gr_all, heights_gr_all,\
masses_gr_all, footwear_gr_all, speeds_gr_all = get_full_metadata(datasets[1])
trial_names_gr_sho, labels_gr_sho, ids_gr_sho, counts_samples_gr_sho, sexes_gr_sho, ages_gr_sho, heights_gr_sho,\
masses_gr_sho, footwear_gr_sho, speeds_gr_sho = get_full_metadata(datasets[2])
trial_names_gb_all, labels_gb_all, ids_gb_all, counts_samples_gb_all, sexes_gb_all, ages_gb_all, heights_gb_all,\
masses_gb_all, footwear_gb_all, speeds_gb_all = get_full_metadata(datasets[3])
trial_names_ai_all, labels_ai_all, ids_ai_all, counts_samples_ai_all, sexes_ai_all, ages_ai_all, heights_ai_all,\
masses_ai_all, footwear_ai_all, speeds_ai_all = get_full_metadata(datasets[4])

# Get the signals from each dataset:
sigs_all_r = []
sigs_all_l = []

for dataset in datasets:

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

sigs_r_fi_all = sigs_all_r[0]
sigs_r_gr_all = sigs_all_r[1]
sigs_r_gr_sho = sigs_all_r[2]
sigs_r_gb_all = sigs_all_r[3]
sigs_r_ai_all = sigs_all_r[4]

sigs_l_fi_all = sigs_all_l[0]
sigs_l_gr_all = sigs_all_l[1]
sigs_l_gr_sho = sigs_all_l[2]
sigs_l_gb_all = sigs_all_l[3]
sigs_l_ai_all = sigs_all_l[4]

# Because gr_all and gr_sho contain many of the same trials, they need to be handled separately going forward:
sigs_r_with_gr_all = np.concatenate((sigs_r_fi_all,
                                     sigs_r_gr_all,
                                     sigs_r_gb_all,
                                     sigs_r_ai_all))
sigs_l_with_gr_all = np.concatenate((sigs_l_fi_all,
                                     sigs_l_gr_all,
                                     sigs_l_gb_all,
                                     sigs_l_ai_all))
trial_names_with_gr_all = np.concatenate((trial_names_fi_all,
                                          trial_names_gr_all,
                                          trial_names_gb_all,
                                          trial_names_ai_all))
sexes_with_gr_all = np.concatenate((sexes_fi_all,
                                    sexes_gr_all,
                                    sexes_gb_all,
                                    sexes_ai_all))
ages_with_gr_all = np.concatenate((ages_fi_all,
                                   ages_gr_all,
                                   ages_gb_all,
                                   ages_ai_all))
heights_with_gr_all = np.concatenate((heights_fi_all,
                                      heights_gr_all,
                                      heights_gb_all,
                                      heights_ai_all))
masses_with_gr_all = np.concatenate((masses_fi_all,
                                     masses_gr_all,
                                     masses_gb_all,
                                     masses_ai_all))
footwear_with_gr_all = np.concatenate((footwear_fi_all,
                                       footwear_gr_all,
                                       footwear_gb_all,
                                       footwear_ai_all))
speeds_with_gr_all = np.concatenate((speeds_fi_all,
                                     speeds_gr_all,
                                     speeds_gb_all,
                                     speeds_ai_all))

sigs_r_with_gr_sho = np.concatenate((sigs_r_fi_all,
                                     sigs_r_gr_sho,
                                     sigs_r_gb_all,
                                     sigs_r_ai_all))
sigs_l_with_gr_sho = np.concatenate((sigs_l_fi_all,
                                     sigs_l_gr_sho,
                                     sigs_l_gb_all,
                                     sigs_l_ai_all))
trial_names_with_gr_sho = np.concatenate((trial_names_fi_all,
                                          trial_names_gr_sho,
                                          trial_names_gb_all,
                                          trial_names_ai_all))
sexes_with_gr_sho = np.concatenate((sexes_fi_all,
                                    sexes_gr_sho,
                                    sexes_gb_all,
                                    sexes_ai_all))
ages_with_gr_sho = np.concatenate((ages_fi_all,
                                   ages_gr_sho,
                                   ages_gb_all,
                                   ages_ai_all))
heights_with_gr_sho = np.concatenate((heights_fi_all,
                                      heights_gr_sho,
                                      heights_gb_all,
                                      heights_ai_all))
masses_with_gr_sho = np.concatenate((masses_fi_all,
                                     masses_gr_sho,
                                     masses_gb_all,
                                     masses_ai_all))
footwear_with_gr_sho = np.concatenate((footwear_fi_all,
                                       footwear_gr_sho,
                                       footwear_gb_all,
                                       footwear_ai_all))
speeds_with_gr_sho = np.concatenate((speeds_fi_all,
                                     speeds_gr_sho,
                                     speeds_gb_all,
                                     speeds_ai_all))

trial_names_fou_umap_with_gr_all = load_list(os.path.join(results_path,
                                                          'trial_names_tr_by_id_fou-umap-_%s-%s-%s-%s.txt' %
                                                          (datasets[0],
                                                           datasets[1],
                                                           datasets[3],
                                                           datasets[4])))

trial_names_fou_umap_with_gr_sho = load_list(os.path.join(results_path,
                                                          'trial_names_tr_by_id_fou-umap-_%s-%s-%s-%s.txt' %
                                                          (datasets[0],
                                                           datasets[2],
                                                           datasets[3],
                                                           datasets[4])))

trial_names_fold_with_gr_all = np.concatenate(trial_names_fou_umap_with_gr_all[cv_fold])
trial_names_fold_with_gr_sho = np.concatenate(trial_names_fou_umap_with_gr_sho[cv_fold])

# Get the indices of trial names from the specified (second) CV fold within the broader sets of trial names:
indices_trial_names_bal_gr_all = np.concatenate([np.asarray(trial_names_with_gr_all == name).nonzero()[0]
                                                 for name in trial_names_fold_with_gr_all])
indices_trial_names_bal_gr_sho = np.concatenate([np.asarray(trial_names_with_gr_sho == name).nonzero()[0]
                                                 for name in trial_names_fold_with_gr_sho])

plt.rcParams["font.family"] = "Times New Roman"

fig = plt.figure(tight_layout=True, figsize=(7.0, 9.5))
colors = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1']

gs = fig.add_gridspec(len(attributes_to_plot), 2)

axs_with_gr_all = []
axs_with_gr_sho = []

for idx, attribute in enumerate(attributes_to_plot):

    if attribute == 'Datasets':

        attribute_binned_gr_all = ['AIST', 'Gutenberg', 'GaitRec-M', 'ForceID-A']
        attribute_binned_gr_sho = ['AIST', 'Gutenberg', 'GaitRec-S', 'ForceID-A']

    else:

        attribute_binned_gr_all = []

        # For the specified attribute(s), get the subset of measures pertaining to samples in the specified (second) CV
        # fold. Then bin them based on quartiles/categories:
        if attribute == 'Age (y)':

            for age in ages_with_gr_all[indices_trial_names_bal_gr_all]:

                if 12 <= age < 22:

                    attribute_binned_gr_all.append('12-21')

                elif 22 <= age < 27:

                    attribute_binned_gr_all.append('22-26')

                elif 27 <= age < 42:

                    attribute_binned_gr_all.append('27-41')

                elif 42 <= age <= 78:

                    attribute_binned_gr_all.append('42-78')

                elif np.isnan(age):

                    attribute_binned_gr_all.append('Unknown')

        elif attribute == 'Sex':

            for sex in sexes_with_gr_all[indices_trial_names_bal_gr_all]:

                if sex == 0:

                    attribute_binned_gr_all.append('Female')

                elif sex == 1:

                    attribute_binned_gr_all.append('Male')

                else:

                    attribute_binned_gr_all.append('Unknown')

        elif attribute == 'Height (m)':

            for height in heights_with_gr_all[indices_trial_names_bal_gr_all]:

                if 1.38 <= height < 1.63:

                    attribute_binned_gr_all.append('1.38-1.62')

                elif 1.63 <= height < 1.7:

                    attribute_binned_gr_all.append('1.63-1.69')

                elif 1.7 <= height < 1.78:

                    attribute_binned_gr_all.append('1.70-1.77')

                elif 1.78 <= height <= 1.99:

                    attribute_binned_gr_all.append('1.78-1.99')

                elif np.isnan(height):

                    attribute_binned_gr_all.append('Unknown')

        elif attribute == 'Mass (kg)':

            for mass in masses_with_gr_all[indices_trial_names_bal_gr_all]:

                if 34.0 <= mass < 57.0:

                    attribute_binned_gr_all.append('34.0-56.9')

                elif 57.0 <= mass < 66.0:

                    attribute_binned_gr_all.append('57.0-65.9')

                elif 66.0 <= mass < 76.6:

                    attribute_binned_gr_all.append('66.0-76.5')

                elif 76.6 <= mass <= 150.0:

                    attribute_binned_gr_all.append('76.6-150.0')

                elif np.isnan(mass):

                    attribute_binned_gr_all.append('Unknown')

        elif attribute == 'Footwear':

            for shod_bool in footwear_with_gr_all[indices_trial_names_bal_gr_all]:

                if shod_bool == 0:

                    attribute_binned_gr_all.append('Barefoot')

                elif shod_bool == 1:

                    attribute_binned_gr_all.append('Shod')

                else:

                    attribute_binned_gr_all.append('Unknown')

        else: # 'Walking speed'

            for speed in speeds_with_gr_all[indices_trial_names_bal_gr_all]:

                if speed == 'F':

                    attribute_binned_gr_all.append('Fast')

                elif speed == 'P':

                    attribute_binned_gr_all.append('Preferred')

                elif speed == 'S':

                    attribute_binned_gr_all.append('Slow')

                else:

                    attribute_binned_gr_all.append('Unknown')

        attribute_binned_gr_sho = []

        if attribute == 'Age (y)':

            for age in ages_with_gr_sho[indices_trial_names_bal_gr_sho]:

                if 12 <= age < 22:

                    attribute_binned_gr_sho.append('12-21')

                elif 22 <= age < 27:

                    attribute_binned_gr_sho.append('22-26')

                elif 27 <= age < 42:

                    attribute_binned_gr_sho.append('27-41')

                elif 42 <= age <= 78:

                    attribute_binned_gr_sho.append('42-78')

                elif np.isnan(age):

                    attribute_binned_gr_sho.append('Unknown')

        elif attribute == 'Sex':

            for sex in sexes_with_gr_sho[indices_trial_names_bal_gr_sho]:

                if sex == 0:

                    attribute_binned_gr_sho.append('Female')

                elif sex == 1:

                    attribute_binned_gr_sho.append('Male')

                else:

                    attribute_binned_gr_sho.append('Unknown')

        elif attribute == 'Height (m)':

            for height in heights_with_gr_sho[indices_trial_names_bal_gr_sho]:

                if 1.38 <= height < 1.63:

                    attribute_binned_gr_sho.append('1.38-1.62')

                elif 1.63 <= height < 1.7:

                    attribute_binned_gr_sho.append('1.63-1.69')

                elif 1.7 <= height < 1.78:

                    attribute_binned_gr_sho.append('1.70-1.77')

                elif 1.78 <= height <= 1.99:

                    attribute_binned_gr_sho.append('1.78-1.99')

                elif np.isnan(height):

                    attribute_binned_gr_sho.append('Unknown')

        elif attribute == 'Mass (kg)':

            for mass in masses_with_gr_sho[indices_trial_names_bal_gr_sho]:

                if 34.0 <= mass < 57.0:

                    attribute_binned_gr_sho.append('34.0-56.9')

                elif 57.0 <= mass < 66.0:

                    attribute_binned_gr_sho.append('57.0-65.9')

                elif 66.0 <= mass < 76.6:

                    attribute_binned_gr_sho.append('66.0-76.5')

                elif 76.6 <= mass <= 150.0:

                    attribute_binned_gr_sho.append('76.6-150.0')

                elif np.isnan(mass):

                    attribute_binned_gr_sho.append('Unknown')

        elif attribute == 'Footwear':

            for shod_bool in footwear_with_gr_sho[indices_trial_names_bal_gr_sho]:

                if shod_bool == 0:

                    attribute_binned_gr_sho.append('Barefoot')

                elif shod_bool == 1:

                    attribute_binned_gr_sho.append('Shod')

                else:

                    attribute_binned_gr_sho.append('Unknown')

        else: # 'Walking speed'

            for speed in speeds_with_gr_sho[indices_trial_names_bal_gr_sho]:

                if speed == 'F':

                    attribute_binned_gr_sho.append('Fast')

                elif speed == 'P':

                    attribute_binned_gr_sho.append('Preferred')

                elif speed == 'S':

                    attribute_binned_gr_sho.append('Slow')

                else:

                    attribute_binned_gr_sho.append('Unknown')

    attribute_binned_gr_all = np.array(attribute_binned_gr_all)
    attribute_binned_gr_sho = np.array(attribute_binned_gr_sho)

    sigs_r_bal_with_gr_all = sigs_r_with_gr_all[indices_trial_names_bal_gr_all]
    sigs_l_bal_with_gr_all = sigs_l_with_gr_all[indices_trial_names_bal_gr_all]
    sigs_r_bal_with_gr_sho = sigs_r_with_gr_sho[indices_trial_names_bal_gr_sho]
    sigs_l_bal_with_gr_sho = sigs_l_with_gr_sho[indices_trial_names_bal_gr_sho]

    if attribute == 'Datasets':

        labels_bal_with_gr_all = attribute_binned_gr_all.repeat(trial_names_fold_with_gr_all.shape[0] / 4)
        labels_bal_with_gr_sho = attribute_binned_gr_sho.repeat(trial_names_fold_with_gr_sho.shape[0] / 4)

        categories_bal_with_gr_all = get_unique_unsorted(labels_bal_with_gr_all)
        categories_bal_with_gr_sho = get_unique_unsorted(labels_bal_with_gr_sho)

    else:

        labels_bal_with_gr_all = attribute_binned_gr_all.copy()
        labels_bal_with_gr_sho = attribute_binned_gr_sho.copy()

        categories_bal_with_gr_all = np.unique(labels_bal_with_gr_all)
        categories_bal_with_gr_sho = np.unique(labels_bal_with_gr_sho)

    sigs_bal_with_gr_all = np.concatenate((sigs_r_bal_with_gr_all, sigs_l_bal_with_gr_all), axis=0)
    sigs_bal_with_gr_sho = np.concatenate((sigs_r_bal_with_gr_sho, sigs_l_bal_with_gr_sho), axis=0)

    fx_with_gr_all, fy_with_gr_all, fz_with_gr_all, cx_with_gr_all, cy_with_gr_all =\
        get_directional_components(sigs_bal_with_gr_all)
    fx_with_gr_sho, fy_with_gr_sho, fz_with_gr_sho, cx_with_gr_sho, cy_with_gr_sho =\
        get_directional_components(sigs_bal_with_gr_sho)

    fx_stsd_with_gr_all = np.split(np.expand_dims(s_scaler.fit_transform(fx_with_gr_all), axis=1), 2)
    fy_stsd_with_gr_all = np.split(np.expand_dims(s_scaler.fit_transform(fy_with_gr_all), axis=1), 2)
    fz_stsd_with_gr_all = np.split(np.expand_dims(s_scaler.fit_transform(fz_with_gr_all), axis=1), 2)
    cx_stsd_with_gr_all = np.split(np.expand_dims(s_scaler.fit_transform(cx_with_gr_all), axis=1), 2)
    cy_stsd_with_gr_all = np.split(np.expand_dims(s_scaler.fit_transform(cy_with_gr_all), axis=1), 2)

    fx_stsd_with_gr_sho = np.split(np.expand_dims(s_scaler.fit_transform(fx_with_gr_sho), axis=1), 2)
    fy_stsd_with_gr_sho = np.split(np.expand_dims(s_scaler.fit_transform(fy_with_gr_sho), axis=1), 2)
    fz_stsd_with_gr_sho = np.split(np.expand_dims(s_scaler.fit_transform(fz_with_gr_sho), axis=1), 2)
    cx_stsd_with_gr_sho = np.split(np.expand_dims(s_scaler.fit_transform(cx_with_gr_sho), axis=1), 2)
    cy_stsd_with_gr_sho = np.split(np.expand_dims(s_scaler.fit_transform(cy_with_gr_sho), axis=1), 2)

    # Re-use variable names:
    fx_with_gr_all = np.concatenate((fx_stsd_with_gr_all[0], fx_stsd_with_gr_all[1]), axis=2)
    fy_with_gr_all = np.concatenate((fy_stsd_with_gr_all[0], fy_stsd_with_gr_all[1]), axis=2)
    fz_with_gr_all = np.concatenate((fz_stsd_with_gr_all[0], fz_stsd_with_gr_all[1]), axis=2)
    cx_with_gr_all = np.concatenate((cx_stsd_with_gr_all[0], cx_stsd_with_gr_all[1]), axis=2)
    cy_with_gr_all = np.concatenate((cy_stsd_with_gr_all[0], cy_stsd_with_gr_all[1]), axis=2)

    fx_with_gr_sho = np.concatenate((fx_stsd_with_gr_sho[0], fx_stsd_with_gr_sho[1]), axis=2)
    fy_with_gr_sho = np.concatenate((fy_stsd_with_gr_sho[0], fy_stsd_with_gr_sho[1]), axis=2)
    fz_with_gr_sho = np.concatenate((fz_stsd_with_gr_sho[0], fz_stsd_with_gr_sho[1]), axis=2)
    cx_with_gr_sho = np.concatenate((cx_stsd_with_gr_sho[0], cx_stsd_with_gr_sho[1]), axis=2)
    cy_with_gr_sho = np.concatenate((cy_stsd_with_gr_sho[0], cy_stsd_with_gr_sho[1]), axis=2)

    sigs_bal_with_gr_all = np.concatenate((fx_with_gr_all,
                                           fy_with_gr_all,
                                           fz_with_gr_all,
                                           cx_with_gr_all,
                                           cy_with_gr_all), axis=1)
    sigs_bal_with_gr_sho = np.concatenate((fx_with_gr_sho,
                                           fy_with_gr_sho,
                                           fz_with_gr_sho,
                                           cx_with_gr_sho,
                                           cy_with_gr_sho), axis=1)

    sigs_bal_with_gr_all = torch.tensor(sigs_bal_with_gr_all, dtype=torch.float32)
    sigs_bal_with_gr_sho = torch.tensor(sigs_bal_with_gr_sho, dtype=torch.float32)

    mod_with_gr_all = TwoF(in_features=int(sigs_bal_with_gr_all.size(1) * sigs_bal_with_gr_all.size(2)),
                           fc1_out=800,
                           out_features=600)
    mod_with_gr_sho = TwoF(in_features=int(sigs_bal_with_gr_sho.size(1) * sigs_bal_with_gr_sho.size(2)),
                           fc1_out=800,
                           out_features=600)

    mod_check_with_gr_all = torch.load(os.path.join(results_path,
                                                    'mod_checks_fou-umap-_%s-%s-%s-%s_%s_%s.pth' %
                                                    (datasets[0],
                                                     datasets[1],
                                                     datasets[3],
                                                     datasets[4],
                                                     arch,
                                                     bs)))[cv_fold]
    mod_check_with_gr_sho = torch.load(os.path.join(results_path,
                                                    'mod_checks_fou-umap-_%s-%s-%s-%s_%s_%s.pth' %
                                                    (datasets[0],
                                                     datasets[2],
                                                     datasets[3],
                                                     datasets[4],
                                                     arch,
                                                     bs)))[cv_fold]

    mod_with_gr_all.load_state_dict(mod_check_with_gr_all[0])
    mod_with_gr_sho.load_state_dict(mod_check_with_gr_sho[0])

    mod_with_gr_all.eval()
    mod_with_gr_sho.eval()

    embs_with_gr_all = mod_with_gr_all(sigs_bal_with_gr_all, cuda).detach().numpy()
    embs_with_gr_sho = mod_with_gr_sho(sigs_bal_with_gr_sho, cuda).detach().numpy()

    mapper_with_gr_all = umap.UMAP(n_neighbors=n_neighbors,
                                   min_dist=min_dist,
                                   n_components=n_components,
                                   metric=metric).fit(embs_with_gr_all)
    mapper_with_gr_sho = umap.UMAP(n_neighbors=n_neighbors,
                                   min_dist=min_dist,
                                   n_components=n_components,
                                   metric=metric).fit(embs_with_gr_sho)

    # Note that the umap points function requires a mapper object. The process above shows how the mapper objects were
    # originally created and then input into the points function. However, UMAP is a stochastic process, so we saved the
    # coordinates for reproducibility. We then loaded the coordinates and set the points function to use them instead of
    # the mapper object when generating UMAP plots:
    coords_with_gr_all = np.load('./Results/umap_coords/coords_umap_tr_set_embs_%s_%s_%s_%s_%s.npy' %
                                 (arch, bs, 'gr-all', cv_fold, n_neighbors), allow_pickle=True)
    coords_with_gr_sho = np.load('./Results/umap_coords/coords_umap_tr_set_embs_%s_%s_%s_%s_%s.npy' %
                                 (arch, bs, 'gr-sho', cv_fold, n_neighbors), allow_pickle=True)

    if attribute == 'Datasets':

        coords_with_gr_all = coords_with_gr_all[::-1]
        coords_with_gr_sho = coords_with_gr_sho[::-1]

    color_key_with_gr_all = {lab: colors[i] for i, lab in enumerate(categories_bal_with_gr_all)}
    color_key_with_gr_sho = {lab: colors[i] for i, lab in enumerate(categories_bal_with_gr_sho)}

    ax_with_gr_all = points(mapper_with_gr_all,
                            figure=None,
                            points=coords_with_gr_all,
                            labels=labels_bal_with_gr_all,
                            show_legend=False,
                            color_key=color_key_with_gr_all,
                            ax=fig.add_subplot(gs[idx, 1]),
                            attr=attribute)
    ax_with_gr_sho = points(mapper_with_gr_sho,
                            figure=None,
                            points=coords_with_gr_sho,
                            labels=labels_bal_with_gr_sho,
                            show_legend=True,
                            color_key=color_key_with_gr_sho,
                            ax=fig.add_subplot(gs[idx, 0]),
                            attr=attribute)

    ax_with_gr_sho.set_ylabel('%s' % attribute, size=11)

    axs_with_gr_all.append(ax_with_gr_all)
    axs_with_gr_sho.append(ax_with_gr_sho)

if save_fig:

    plt.savefig('./Figures/fig_umap_tr_set_embs_%s_%s_%s.pdf' % (arch, bs, n_neighbors), dpi=1200)
    plt.savefig('./Figures/fig_umap_tr_set_embs_%s_%s_%s.png' % (arch, bs, n_neighbors), dpi=1200)
    plt.savefig('./Figures/fig_umap_tr_set_embs_%s_%s_%s.svg' % (arch, bs, n_neighbors), dpi=1200)

if show_fig:

    plt.show()


