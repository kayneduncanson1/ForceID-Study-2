import numpy as np
import os
import torch
import umap.plot
import matplotlib.pyplot as plt
from Utils import set_seed, get_full_metadata, load_list, get_unique_unsorted, points
from PrePro import get_directional_components
from Models import TwoF
from sklearn.preprocessing import StandardScaler

"""This script was used to create the UMAP figure in the Supplementary Information. It follows largely the same process
as the umap_main.py script, though generates subplots for individual CV folds with different n_neighbors settings."""

g = set_seed()

show_fig = False
save_fig = True

settings_n_neighbors = [5, 15, 30, 50, 75, 100]
s_scaler = StandardScaler()
arch = '-----2F'
bs = 512
attribute = 'Datasets'
cv_folds = np.arange(5)
min_dist = 0.0
n_components = 2
metric = 'euclidean'
datasets = ['fi-all', 'gr-sho', 'gb-all', 'ai-all']
cuda = False
results_path = './Results/main'

trial_names_fi_all, labels_fi_all, ids_fi_all, counts_samples_fi_all, sexes_fi_all, ages_fi_all, heights_fi_all,\
masses_fi_all, footwear_fi_all, speeds_fi_all = get_full_metadata(datasets[0])
trial_names_gr_sho, labels_gr_sho, ids_gr_sho, counts_samples_gr_sho, sexes_gr_sho, ages_gr_sho, heights_gr_sho,\
masses_gr_sho, footwear_gr_sho, speeds_gr_sho = get_full_metadata(datasets[1])
trial_names_gb_all, labels_gb_all, ids_gb_all, counts_samples_gb_all, sexes_gb_all, ages_gb_all, heights_gb_all,\
masses_gb_all, footwear_gb_all, speeds_gb_all = get_full_metadata(datasets[2])
trial_names_ai_all, labels_ai_all, ids_ai_all, counts_samples_ai_all, sexes_ai_all, ages_ai_all, heights_ai_all,\
masses_ai_all, footwear_ai_all, speeds_ai_all = get_full_metadata(datasets[3])

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
sigs_r_gr_sho = sigs_all_r[1]
sigs_r_gb_all = sigs_all_r[2]
sigs_r_ai_all = sigs_all_r[3]

sigs_l_fi_all = sigs_all_l[0]
sigs_l_gr_sho = sigs_all_l[1]
sigs_l_gb_all = sigs_all_l[2]
sigs_l_ai_all = sigs_all_l[3]

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

plt.rcParams["font.family"] = "Times New Roman"

fig = plt.figure(tight_layout=True, figsize=(7.0, 9.5))
colors = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1']

gs = fig.add_gridspec(6, 5)

axs_with_gr_sho = []

trial_names_fou_umap_with_gr_sho = load_list(os.path.join(results_path,
                                                          'trial_names_tr_by_id_fou-umap-_%s-%s-%s-%s.txt' %
                                                          (datasets[0],
                                                           datasets[1],
                                                           datasets[2],
                                                           datasets[3])))

for cv_fold in cv_folds:

    trial_names_fold_with_gr_sho = np.concatenate(trial_names_fou_umap_with_gr_sho[cv_fold])
    indices_trial_names_bal_gr_sho = np.concatenate([np.asarray(trial_names_with_gr_sho == name).nonzero()[0]
                                                     for name in trial_names_fold_with_gr_sho])

    if attribute == 'Datasets':

        attribute_binned_gr_sho = ['AIST', 'Gutenberg', 'GaitRec-S', 'ForceID-A']

    else:

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

    attribute_binned_gr_sho = np.array(attribute_binned_gr_sho)

    sigs_r_bal_with_gr_sho = sigs_r_with_gr_sho[indices_trial_names_bal_gr_sho]
    sigs_l_bal_with_gr_sho = sigs_l_with_gr_sho[indices_trial_names_bal_gr_sho]

    if attribute == 'Datasets':

        labels_bal_with_gr_sho = attribute_binned_gr_sho.repeat(trial_names_fold_with_gr_sho.shape[0] / 4)
        categories_bal_with_gr_sho = get_unique_unsorted(labels_bal_with_gr_sho)

    else:

        labels_bal_with_gr_sho = attribute_binned_gr_sho.copy()
        categories_bal_with_gr_sho = np.unique(labels_bal_with_gr_sho)

    sigs_bal_with_gr_sho = np.concatenate((sigs_r_bal_with_gr_sho, sigs_l_bal_with_gr_sho), axis=0)

    fx_with_gr_sho, fy_with_gr_sho, fz_with_gr_sho, cx_with_gr_sho, cy_with_gr_sho =\
        get_directional_components(sigs_bal_with_gr_sho)

    fx_stsd_with_gr_sho = np.split(np.expand_dims(s_scaler.fit_transform(fx_with_gr_sho), axis=1), 2)
    fy_stsd_with_gr_sho = np.split(np.expand_dims(s_scaler.fit_transform(fy_with_gr_sho), axis=1), 2)
    fz_stsd_with_gr_sho = np.split(np.expand_dims(s_scaler.fit_transform(fz_with_gr_sho), axis=1), 2)
    cx_stsd_with_gr_sho = np.split(np.expand_dims(s_scaler.fit_transform(cx_with_gr_sho), axis=1), 2)
    cy_stsd_with_gr_sho = np.split(np.expand_dims(s_scaler.fit_transform(cy_with_gr_sho), axis=1), 2)

    fx_with_gr_sho = np.concatenate((fx_stsd_with_gr_sho[0], fx_stsd_with_gr_sho[1]), axis=2)
    fy_with_gr_sho = np.concatenate((fy_stsd_with_gr_sho[0], fy_stsd_with_gr_sho[1]), axis=2)
    fz_with_gr_sho = np.concatenate((fz_stsd_with_gr_sho[0], fz_stsd_with_gr_sho[1]), axis=2)
    cx_with_gr_sho = np.concatenate((cx_stsd_with_gr_sho[0], cx_stsd_with_gr_sho[1]), axis=2)
    cy_with_gr_sho = np.concatenate((cy_stsd_with_gr_sho[0], cy_stsd_with_gr_sho[1]), axis=2)

    sigs_bal_with_gr_sho = np.concatenate((fx_with_gr_sho,
                                           fy_with_gr_sho,
                                           fz_with_gr_sho,
                                           cx_with_gr_sho,
                                           cy_with_gr_sho), axis=1)

    sigs_bal_with_gr_sho = torch.tensor(sigs_bal_with_gr_sho, dtype=torch.float32)

    mod_with_gr_sho = TwoF(in_features=int(sigs_bal_with_gr_sho.size(1) * sigs_bal_with_gr_sho.size(2)),
                           fc1_out=800,
                           out_features=600)

    mod_check_with_gr_sho = torch.load(os.path.join(results_path,
                                                    'mod_checks_fou-umap-_%s-%s-%s-%s_%s_%s.pth' %
                                                    (datasets[0],
                                                     datasets[1],
                                                     datasets[2],
                                                     datasets[3],
                                                     arch,
                                                     bs)))[cv_fold]

    mod_with_gr_sho.load_state_dict(mod_check_with_gr_sho[0])
    mod_with_gr_sho.eval()
    embs_with_gr_sho = mod_with_gr_sho(sigs_bal_with_gr_sho, cuda).detach().numpy()

    for idx, n_neighbors in enumerate(settings_n_neighbors):

        mapper_with_gr_sho = umap.UMAP(n_neighbors=n_neighbors,
                                       min_dist=min_dist,
                                       n_components=n_components,
                                       metric=metric).fit(embs_with_gr_sho)

        coords_with_gr_sho = np.load('./Results/umap_coords/coords_umap_tr_set_embs_%s_%s_%s_%s_%s.npy' %
                                     (arch, bs, 'gr-sho', cv_fold, n_neighbors), allow_pickle=True)

        if attribute == 'Datasets':

            coords_with_gr_sho = coords_with_gr_sho[::-1]

        color_key_with_gr_sho = {lab: colors[i] for i, lab in enumerate(categories_bal_with_gr_sho)}
        ax_with_gr_sho = points(mapper_with_gr_sho,
                                figure=None,
                                points=coords_with_gr_sho,
                                labels=labels_bal_with_gr_sho,
                                show_legend=False,
                                color_key=color_key_with_gr_sho,
                                ax=fig.add_subplot(gs[idx, cv_fold]),
                                attr=attribute)

        if n_neighbors == 100:

            ax_with_gr_sho.set_xlabel('%s' % (cv_fold + 1), size=11)

        if cv_fold == 0:

            ax_with_gr_sho.set_ylabel('%s' % n_neighbors, size=11)

        axs_with_gr_sho.append(ax_with_gr_sho)

if save_fig:

    plt.savefig('./Figures/fig_umap_tr_set_embs_%s_%s_supp.pdf' % (arch, bs), dpi=1200)
    plt.savefig('./Figures/fig_umap_tr_set_embs_%s_%s_supp.png' % (arch, bs), dpi=1200)
    plt.savefig('./Figures/fig_umap_tr_set_embs_%s_%s_supp.svg' % (arch, bs), dpi=1200)

if show_fig:

    plt.show()
