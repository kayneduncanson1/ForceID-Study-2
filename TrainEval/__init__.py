import time
import os
import numpy as np
from sklearn import preprocessing
import torch
from torch.utils.data import WeightedRandomSampler
import torch.optim as optim
from PrePro import get_directional_components, standardise_then_split
from DataLoaders import DataLoaderLabels, DataLoaderNoLabels, init_data_loaders_no_labs_va_te
from Models import OneF, TwoF, OneCOneF, OneCTwoF, ThrCOneF, ThrCTwoF, ThrCOneLUOneF, ThrCOneLUTwoF, ThrCOneLBOneF,\
    ThrCOneLBTwoF, OneCOneTOneF, OneCOneTTwoF, ThrCOneTOneF, ThrCOneTTwoF, TOneF, TTwoF
from Losses import get_dists


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience=20, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            # print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                # print('INFO: Early stopping')
                self.early_stop = True


def calc_acc_train(dist_ap, dist_an):

    dist_ap_max = torch.max(dist_ap, dim=1)[0]
    indices_dist_ap_max_valid = torch.where(dist_ap_max != float('-inf'))[0]

    if len(indices_dist_ap_max_valid) == 0:

        acc = None
        print('No valid positives in batch')

    else:

        dist_ap_max = dist_ap_max[indices_dist_ap_max_valid].unsqueeze(1)
        dist_an_min = torch.max(dist_an, dim=1)[0][indices_dist_ap_max_valid] * -1
        dist_an_min = dist_an_min.unsqueeze(1)

        with torch.no_grad():

            acc = (dist_an_min > dist_ap_max).float().mean().item()

    return acc


def train_val(mod, loader_tr, loader_va, labels_va, criterion_tr, criterion_eval, opt, epochs, ptm=True,
              miner=None, criterion_opt=None, cuda_tr=True, cuda_va=True, single_input=True, count_samples_min=None,
              indices_s1=None, indices_s2=None, n_similarity_search=None):

    if not ptm and (miner is not None or criterion_opt is not None):

        raise Exception('Currently only support PTM miners and criterion_opts. Check whether ptm should be True.')

    elif count_samples_min is not None and indices_s1 is not None:

        raise Exception('count_samples_min and session indices are not required simultaneously.')

    elif count_samples_min is None and indices_s1 is None:

        raise Exception('Either count_samples_min or session indices must be specified.')

    elif count_samples_min is None and n_similarity_search is not None:

        raise Exception('count_samples_min is required if n_similarity_search is specified.')

    early_stopping = EarlyStopping()
    t_start = time.time()
    hist_acc_tr = []
    hist_loss_tr = []
    hist_acc_va = []
    hist_loss_va = []
    acc_tr_best = 0
    acc_va_best = 0

    for epoch in range(1, epochs + 1):

        for phase in ['train', 'val']:

            if phase == 'train':

                losses = AverageMeter()
                accs = AverageMeter()

                if cuda_tr:

                    mod.cuda().train()

                else:

                    mod.train()

                if single_input:

                    for batch_idx, (inputs, labs) in enumerate(loader_tr):

                        if cuda_tr:

                            inputs, labs = inputs.cuda(), labs.cuda()

                        opt.zero_grad()

                        with torch.set_grad_enabled(True):

                            # compute output
                            embs = mod(inputs, cuda_tr)

                            dist_ap, dist_an = get_dists(embs, labs, 'train')

                            if ptm and miner is None:

                                loss = criterion_tr(embs, labs)
                                acc = calc_acc_train(dist_ap, dist_an)

                            elif ptm and miner is not None:

                                miner_output = miner(embs, labs)
                                loss = criterion_tr(embs, labs, miner_output)
                                acc = calc_acc_train(dist_ap, dist_an)

                            elif not ptm:

                                loss, acc = criterion_tr(dist_ap, dist_an)

                            loss.backward()

                            if criterion_opt is not None:

                                criterion_opt.step()

                            opt.step()
                            losses.update(loss.detach().cpu().numpy().item())

                            if acc is not None:

                                accs.update(acc)

                    hist_loss_tr.append(losses.avg)
                    hist_acc_tr.append(accs.avg)

                    if accs.avg > acc_tr_best:

                        acc_tr_best = accs.avg

                else: # Ignore this section. Not used in this study.

                    for batch_idx, (inputs1, inputs2, labs) in enumerate(loader_tr):

                        if cuda_tr:

                            inputs1, inputs2, labs = inputs1.cuda(), inputs2.cuda(), labs.cuda()

                        opt.zero_grad()

                        with torch.set_grad_enabled(True):

                            # compute output
                            embs = mod(inputs1, inputs2, cuda_tr)

                            dist_ap, dist_an = get_dists(embs, labs, 'train')

                            if ptm and miner is None:

                                loss = criterion_tr(embs, labs)
                                acc = calc_acc_train(dist_ap, dist_an)

                            elif ptm and miner is not None:

                                miner_output = miner(embs, labs)
                                loss = criterion_tr(embs, labs, miner_output)
                                acc = calc_acc_train(dist_ap, dist_an)

                            elif not ptm:

                                loss, acc = criterion_tr(dist_ap, dist_an)

                            loss.backward()

                            if criterion_opt is not None:

                                criterion_opt.step()

                            opt.step()
                            losses.update(loss.detach().cpu().numpy().item())

                            if acc is not None:

                                accs.update(acc)

                    hist_loss_tr.append(losses.avg)
                    hist_acc_tr.append(accs.avg)

                    if accs.avg > acc_tr_best:

                        acc_tr_best = accs.avg

            else:

                if cuda_va:

                    mod.cuda().eval()

                else:

                    mod.cpu().eval()

                embs = None

                if single_input:

                    for batch_idx, inputs in enumerate(loader_va):

                        if cuda_va:

                            inputs = inputs.cuda()

                        opt.zero_grad()

                        with torch.set_grad_enabled(False):

                            outputs = mod(inputs, cuda_va)
                            embs = torch.cat((embs, outputs), dim=0) if embs is not None else outputs

                else: # Ignore this section. Not used in this study.

                    for batch_idx, (inputs1, inputs2) in enumerate(loader_va):

                        if cuda_va:

                            inputs1, inputs2 = inputs1.cuda(), inputs2.cuda()

                        opt.zero_grad()

                        with torch.set_grad_enabled(False):

                            outputs = mod(inputs1, inputs2, cuda_va)
                            embs = torch.cat((embs, outputs), dim=0) if embs is not None else outputs

                outputs = None

                if cuda_va:

                    dist_ap, dist_an = get_dists(embs, labels_va.cuda(), 'val')

                else:

                    dist_ap, dist_an = get_dists(embs, labels_va, 'val')

                if count_samples_min is not None and n_similarity_search is None:

                    loss, acc = criterion_eval(dist_ap, dist_an, count_samples_min)

                elif count_samples_min is not None and n_similarity_search is not None:

                    loss, acc = criterion_eval(dist_ap, dist_an, count_samples_min, n_similarity_search)

                elif count_samples_min is None and indices_s1 is not None:

                    loss, acc = criterion_eval(dist_ap, dist_an, indices_s1, indices_s2)

                dist_ap = None
                dist_an = None

                hist_loss_va.append(loss.detach().cpu().numpy().item())
                hist_acc_va.append(acc)

                if acc > acc_va_best:

                    acc_va_best = acc
                    checkpoint = {'model_state_dict': mod.state_dict(), 'optimizer_state_dict': opt.state_dict()}

        early_stopping(loss)

        if early_stopping.early_stop:

            break

    time_elapsed = time.time() - t_start

    return time_elapsed, hist_acc_tr, hist_loss_tr, hist_acc_va, hist_loss_va, checkpoint['model_state_dict'],\
        checkpoint['optimizer_state_dict'], embs


def test(mod, loader_te, labels_te, criterion_eval, opt, cuda=True, single_input=True,
         count_samples_min=None, indices_s1=None, indices_s2=None, n_similarity_search=None):

    if count_samples_min is not None and indices_s1 is not None:

        raise Exception('count_samples_min and session indices are not required simultaneously.')

    elif count_samples_min is None and indices_s1 is None:

        raise Exception('Either count_samples_min or session indices must be specified.')

    elif count_samples_min is None and n_similarity_search is not None:

        raise Exception('count_samples_min is required if n_similarity_search is specified.')

    if cuda:

        mod.cuda().eval()

    else:

        mod.cpu().eval()

    embs = None

    if single_input:

        for batch_idx, inputs in enumerate(loader_te):

            if cuda:

                inputs = inputs.cuda()

            opt.zero_grad()

            with torch.set_grad_enabled(False):

                outputs = mod(inputs, cuda)
                embs = torch.cat((embs, outputs), dim=0) if embs is not None else outputs

    else:

        for batch_idx, (inputs1, inputs2) in enumerate(loader_te):

            if cuda:

                inputs1, inputs2 = inputs1.cuda(), inputs2.cuda()

            opt.zero_grad()

            with torch.set_grad_enabled(False):

                outputs = mod(inputs1, inputs2, cuda)
                embs = torch.cat((embs, outputs), dim=0) if embs is not None else outputs

    outputs = None

    if cuda:

        dist_ap, dist_an = get_dists(embs, labels_te.cuda(), 'trial_names')

    else:

        dist_ap, dist_an = get_dists(embs, labels_te, 'trial_names')

    if count_samples_min is not None and n_similarity_search is None:

        loss, acc = criterion_eval(dist_ap, dist_an, count_samples_min)

    elif count_samples_min is not None and n_similarity_search is not None:

        loss, acc = criterion_eval(dist_ap, dist_an, count_samples_min, n_similarity_search)

    elif count_samples_min is None and indices_s1 is not None:

        loss, acc = criterion_eval(dist_ap, dist_an, indices_s1, indices_s2)

    dist_ap = None
    dist_an = None

    return loss.cpu().numpy().item(), acc, embs


le = preprocessing.LabelEncoder()


# Currently only supports EvalHard as criterion_eval:
def run_expt(trial_names, sigs_all_r, sigs_all_l, trial_names_tr_by_id, trial_names_va_by_id, trial_names_te_by_id,
             n_samples_per_id_bal, bs, arch, epochs, criterion_tr, criterion_eval, ptm, miner, criterion_opt, cuda_tr,
             cuda_va, cuda_te, single_input):

    n_folds = 5

    times = []
    mod_checks = []
    losses_tr_all = []
    losses_va_all = []
    losses_te_all = []
    accs_tr_all = []
    accs_va_all = []
    accs_te_all = []
    embs_va_all = []
    embs_te_all = []

    for fold in range(n_folds):

        print('Fold %s:' % (fold + 1))

        count_va_min = n_samples_per_id_bal
        count_te_min = n_samples_per_id_bal

        trial_names_tr = np.concatenate(trial_names_tr_by_id[fold])
        trial_names_va = np.concatenate(trial_names_va_by_id[fold])
        trial_names_te = np.concatenate(trial_names_te_by_id[fold])

        labels_tr = np.array([int(name[3:7]) for name in trial_names_tr])
        labels_va = np.array([int(name[3:7]) for name in trial_names_va])
        labels_te = np.array([int(name[3:7]) for name in trial_names_te])

        le.fit(labels_tr)
        # Transform labels to start from 0:
        labels_tr = torch.tensor(le.transform(labels_tr)).long()

        le.fit(labels_va)
        labels_va = torch.tensor(le.transform(labels_va)).long()

        le.fit(labels_te)
        labels_te = torch.tensor(le.transform(labels_te)).long()

        indices_tr = np.concatenate([np.asarray(trial_names == name).nonzero()[0] for name in trial_names_tr])
        indices_va = np.concatenate([np.asarray(trial_names == name).nonzero()[0] for name in trial_names_va])
        indices_te = np.concatenate([np.asarray(trial_names == name).nonzero()[0] for name in trial_names_te])

        trial_names_tr = None
        trial_names_va = None
        trial_names_te = None

        sigs_all_r_tr = sigs_all_r[indices_tr]
        sigs_all_r_va = sigs_all_r[indices_va]
        sigs_all_r_te = sigs_all_r[indices_te]

        sigs_all_l_tr = sigs_all_l[indices_tr]
        sigs_all_l_va = sigs_all_l[indices_va]
        sigs_all_l_te = sigs_all_l[indices_te]

        indices_tr = None
        indices_va = None
        indices_te = None

        sigs_tr = np.concatenate((sigs_all_r_tr, sigs_all_l_tr), axis=0)
        sigs_va = np.concatenate((sigs_all_r_va, sigs_all_l_va), axis=0)
        sigs_te = np.concatenate((sigs_all_r_te, sigs_all_l_te), axis=0)

        sigs_all_r_tr, sigs_all_l_tr = None, None
        sigs_all_r_va, sigs_all_l_va = None, None
        sigs_all_r_te, sigs_all_l_te = None, None

        fx_tr, fy_tr, fz_tr, cx_tr, cy_tr = get_directional_components(sigs_tr)
        fx_va, fy_va, fz_va, cx_va, cy_va = get_directional_components(sigs_va)
        fx_te, fy_te, fz_te, cx_te, cy_te = get_directional_components(sigs_te)

        fx_tr_stsd, fx_va_stsd, fx_te_stsd = standardise_then_split(fx_tr, fx_va, fx_te)
        fy_tr_stsd, fy_va_stsd, fy_te_stsd = standardise_then_split(fy_tr, fy_va, fy_te)
        fz_tr_stsd, fz_va_stsd, fz_te_stsd = standardise_then_split(fz_tr, fz_va, fz_te)
        cx_tr_stsd, cx_va_stsd, cx_te_stsd = standardise_then_split(cx_tr, cx_va, cx_te)
        cy_tr_stsd, cy_va_stsd, cy_te_stsd = standardise_then_split(cy_tr, cy_va, cy_te)

        fx_tr, fy_tr, fz_tr, cx_tr, cy_tr = None, None, None, None, None
        fx_va, fy_va, fz_va, cx_va, cy_va = None, None, None, None, None
        fx_te, fy_te, fz_te, cx_te, cy_te = None, None, None, None, None

        # Combine r (at index 0) and l (at index 1) sigs along sequence dim:
        fx_tr_stsd = torch.cat((fx_tr_stsd[0], fx_tr_stsd[1]), dim=2)
        fy_tr_stsd = torch.cat((fy_tr_stsd[0], fy_tr_stsd[1]), dim=2)
        fz_tr_stsd = torch.cat((fz_tr_stsd[0], fz_tr_stsd[1]), dim=2)
        cx_tr_stsd = torch.cat((cx_tr_stsd[0], cx_tr_stsd[1]), dim=2)
        cy_tr_stsd = torch.cat((cy_tr_stsd[0], cy_tr_stsd[1]), dim=2)

        fx_va_stsd = torch.cat((fx_va_stsd[0], fx_va_stsd[1]), dim=2)
        fy_va_stsd = torch.cat((fy_va_stsd[0], fy_va_stsd[1]), dim=2)
        fz_va_stsd = torch.cat((fz_va_stsd[0], fz_va_stsd[1]), dim=2)
        cx_va_stsd = torch.cat((cx_va_stsd[0], cx_va_stsd[1]), dim=2)
        cy_va_stsd = torch.cat((cy_va_stsd[0], cy_va_stsd[1]), dim=2)

        fx_te_stsd = torch.cat((fx_te_stsd[0], fx_te_stsd[1]), dim=2)
        fy_te_stsd = torch.cat((fy_te_stsd[0], fy_te_stsd[1]), dim=2)
        fz_te_stsd = torch.cat((fz_te_stsd[0], fz_te_stsd[1]), dim=2)
        cx_te_stsd = torch.cat((cx_te_stsd[0], cx_te_stsd[1]), dim=2)
        cy_te_stsd = torch.cat((cy_te_stsd[0], cy_te_stsd[1]), dim=2)

        if arch in ['--1T-1F', '--1T-2F', '--2T-1F', '--2T-2F', '--3T-1F', '--3T-2F']:

            sigs_tr = torch.cat((fx_tr_stsd, fy_tr_stsd, fz_tr_stsd, cx_tr_stsd, cy_tr_stsd,
                                 torch.zeros_like(fx_tr_stsd)), dim=1)
            sigs_va = torch.cat((fx_va_stsd, fy_va_stsd, fz_va_stsd, cx_va_stsd, cy_va_stsd,
                                 torch.zeros_like(fx_va_stsd)), dim=1)
            sigs_te = torch.cat((fx_te_stsd, fy_te_stsd, fz_te_stsd, cx_te_stsd, cy_te_stsd,
                                 torch.zeros_like(fx_te_stsd)), dim=1)

        else:

            # Re-combine channels. Re-declaring sigs_tr, sigs_va, and sigs_te to save memory:
            sigs_tr = torch.cat((fx_tr_stsd, fy_tr_stsd, fz_tr_stsd, cx_tr_stsd, cy_tr_stsd), dim=1)
            sigs_va = torch.cat((fx_va_stsd, fy_va_stsd, fz_va_stsd, cx_va_stsd, cy_va_stsd), dim=1)
            sigs_te = torch.cat((fx_te_stsd, fy_te_stsd, fz_te_stsd, cx_te_stsd, cy_te_stsd), dim=1)

        fx_tr_stsd, fy_tr_stsd, fz_tr_stsd, cx_tr_stsd, cy_tr_stsd = None, None, None, None, None
        fx_va_stsd, fy_va_stsd, fz_va_stsd, cx_va_stsd, cy_va_stsd = None, None, None, None, None
        fx_te_stsd, fy_te_stsd, fz_te_stsd, cx_te_stsd, cy_te_stsd = None, None, None, None, None

        loader_tr, loader_va, loader_te = init_data_loaders_no_labs_va_te(DataLoaderLabels,
                                                                          DataLoaderNoLabels,
                                                                          sigs_tr, labels_tr,
                                                                          sigs_va, sigs_te,
                                                                          batch_size=bs)

        if arch == '-----1F':

            mod = OneF(in_features=int(sigs_tr.size(1) * sigs_tr.size(2)), out_features=600)

        elif arch == '-----2F':

            mod = TwoF(in_features=int(sigs_tr.size(1) * sigs_tr.size(2)), fc1_out=800, out_features=600)

        elif arch == '1C---1F':

            mod = OneCOneF(nc0=sigs_tr.size(1), nc1=32, out_features=800)

        elif arch == '1C---2F':

            mod = OneCTwoF(nc0=sigs_tr.size(1), nc1=32, fc1_out=1600, out_features=800)

        elif arch == '3C---1F':

            mod = ThrCOneF(nc0=sigs_tr.size(1), nc1=32, nc2=64, nc3=128, out_features=800)

        elif arch == '3C---2F':

            mod = ThrCTwoF(nc0=sigs_tr.size(1), nc1=32, nc2=64, nc3=128, fc1_out=1600, out_features=800)

        elif arch == '3C1LU1F':

            mod = ThrCOneLUOneF(nc0=sigs_tr.size(1), nc1=32, nc2=64, nc3=128, out_features=800)

        elif arch == '3C1LU2F':

            mod = ThrCOneLUTwoF(nc0=sigs_tr.size(1), nc1=32, nc2=64, nc3=128, fc1_out=1600, out_features=800)

        elif arch == '3C1LB1F':

            mod = ThrCOneLBOneF(nc0=sigs_tr.size(1), nc1=32, nc2=64, nc3=128, out_features=800)

        elif arch == '3C1LB2F':

            mod = ThrCOneLBTwoF(nc0=sigs_tr.size(1), nc1=32, nc2=64, nc3=128, fc1_out=1600, out_features=800)

        elif arch == '1C1T-1F':

            mod = OneCOneTOneF(nc0=sigs_tr.size(1), nc1=32, out_features=800)

        elif arch == '1C1T-2F':

            mod = OneCOneTTwoF(nc0=sigs_tr.size(1), nc1=32, fc1_out=1600, out_features=800)

        elif arch == '3C1T-1F':

            mod = ThrCOneTOneF(nc0=sigs_tr.size(1), nc1=32, nc2=64, nc3=128, out_features=800)

        elif arch == '3C1T-2F':

            mod = ThrCOneTTwoF(nc0=sigs_tr.size(1), nc1=32, nc2=64, nc3=128, fc1_out=1600, out_features=800)

        elif arch == '--1T-1F':

            mod = TOneF(nc0=sigs_tr.size(1), num_t_layers=1, out_features=800)

        elif arch == '--1T-2F':

            mod = TTwoF(nc0=sigs_tr.size(1), num_t_layers=1, fc1_out=1000, out_features=800)

        elif arch == '--2T-1F':

            mod = TOneF(nc0=sigs_tr.size(1), num_t_layers=2, out_features=800)

        elif arch == '--2T-2F':

            mod = TTwoF(nc0=sigs_tr.size(1), num_t_layers=2, fc1_out=1000, out_features=800)

        elif arch == '--3T-1F':

            mod = TOneF(nc0=sigs_tr.size(1), num_t_layers=3, out_features=800)

        elif arch == '--3T-2F':

            mod = TTwoF(nc0=sigs_tr.size(1), num_t_layers=3, fc1_out=1000, out_features=800)

        else:

            raise Exception('The specified arch is currently not supported.')

        # To get the total number of parameters for each model architecture, we uncommented this and set a breakpoint...
        # at the next line:
        # mod_n_params = sum(p.numel() for p in mod.parameters() if p.requires_grad)

        opt = optim.Adam(mod.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)

        tr_va_time, accs_tr, losses_tr, accs_va, losses_va, mod_state_dict, opt_state_dict, embs_va = \
            train_val(mod, loader_tr, loader_va, labels_va, criterion_tr, criterion_eval, opt, epochs, ptm, miner,
                      criterion_opt, cuda_tr, cuda_va, single_input, count_va_min, indices_s1=None, indices_s2=None,
                      n_similarity_search=None)

        times.append(tr_va_time)
        accs_tr_all.append(accs_tr)
        losses_tr_all.append(losses_tr)
        accs_va_all.append(accs_va)
        losses_va_all.append(losses_va)
        mod_checks.append((mod_state_dict, opt_state_dict))
        embs_va_all.append(embs_va.cpu())

        mod.load_state_dict(mod_state_dict)

        labels_tr, labels_va = None, None
        sampler_tr = None
        sigs_tr, sigs_va = None, None
        loader_tr, loader_va = None, None
        accs_tr, accs_va = None, None
        losses_tr, losses_va = None, None
        mod_state_dict = None
        opt_state_dict = None
        embs_va = None

        loss_te, acc_te, embs_te = test(mod, loader_te, labels_te, criterion_eval, opt, cuda_te, single_input,
                                        count_te_min, indices_s1=None, indices_s2=None, n_similarity_search=None)

        losses_te_all.append(loss_te)
        accs_te_all.append(acc_te)
        embs_te_all.append(embs_te.cpu())
        embs_te = None

    return times, mod_checks, losses_tr_all, losses_va_all, losses_te_all, accs_tr_all, accs_va_all, accs_te_all, \
        embs_va_all, embs_te_all


def run_expt_osa(results_path, config, trial_names, sigs_all_r, sigs_all_l, channel, window, trial_names_tr_by_id,
                 trial_names_va_by_id, trial_names_te_by_id, n_samples_per_id_bal, bs, arch, criterion_eval, cuda_te,
                 single_input):

    n_folds = 5

    losses_te_all = []
    accs_te_all = []
    embs_te_all = []

    for fold in range(n_folds):

        print('Fold %s:' % (fold + 1))

        count_va_min = n_samples_per_id_bal
        count_te_min = n_samples_per_id_bal

        trial_names_tr = np.concatenate(trial_names_tr_by_id[fold])
        trial_names_va = np.concatenate(trial_names_va_by_id[fold])
        trial_names_te = np.concatenate(trial_names_te_by_id[fold])

        labels_tr = np.array([int(name[3:7]) for name in trial_names_tr])
        labels_va = np.array([int(name[3:7]) for name in trial_names_va])
        labels_te = np.array([int(name[3:7]) for name in trial_names_te])

        le.fit(labels_tr)
        # Transform labels to start from 0:
        labels_tr = torch.tensor(le.transform(labels_tr)).long()

        le.fit(labels_va)
        labels_va = torch.tensor(le.transform(labels_va)).long()

        le.fit(labels_te)
        labels_te = torch.tensor(le.transform(labels_te)).long()

        indices_tr = np.concatenate([np.asarray(trial_names == name).nonzero()[0] for name in trial_names_tr])
        indices_va = np.concatenate([np.asarray(trial_names == name).nonzero()[0] for name in trial_names_va])
        indices_te = np.concatenate([np.asarray(trial_names == name).nonzero()[0] for name in trial_names_te])

        trial_names_tr = None
        trial_names_va = None
        trial_names_te = None

        sigs_all_r_tr = sigs_all_r[indices_tr]
        sigs_all_r_va = sigs_all_r[indices_va]
        sigs_all_r_te = sigs_all_r[indices_te]

        sigs_all_l_tr = sigs_all_l[indices_tr]
        sigs_all_l_va = sigs_all_l[indices_va]
        sigs_all_l_te = sigs_all_l[indices_te]

        indices_tr = None
        indices_va = None
        indices_te = None

        sigs_tr = np.concatenate((sigs_all_r_tr, sigs_all_l_tr), axis=0)
        sigs_va = np.concatenate((sigs_all_r_va, sigs_all_l_va), axis=0)
        sigs_te = np.concatenate((sigs_all_r_te, sigs_all_l_te), axis=0)

        sigs_all_r_tr, sigs_all_l_tr = None, None
        sigs_all_r_va, sigs_all_l_va = None, None
        sigs_all_r_te, sigs_all_l_te = None, None

        fx_tr, fy_tr, fz_tr, cx_tr, cy_tr = get_directional_components(sigs_tr)
        fx_va, fy_va, fz_va, cx_va, cy_va = get_directional_components(sigs_va)
        fx_te, fy_te, fz_te, cx_te, cy_te = get_directional_components(sigs_te)

        fx_tr_stsd, fx_va_stsd, fx_te_stsd = standardise_then_split(fx_tr, fx_va, fx_te)
        fy_tr_stsd, fy_va_stsd, fy_te_stsd = standardise_then_split(fy_tr, fy_va, fy_te)
        fz_tr_stsd, fz_va_stsd, fz_te_stsd = standardise_then_split(fz_tr, fz_va, fz_te)
        cx_tr_stsd, cx_va_stsd, cx_te_stsd = standardise_then_split(cx_tr, cx_va, cx_te)
        cy_tr_stsd, cy_va_stsd, cy_te_stsd = standardise_then_split(cy_tr, cy_va, cy_te)

        fx_tr, fy_tr, fz_tr, cx_tr, cy_tr = None, None, None, None, None
        fx_va, fy_va, fz_va, cx_va, cy_va = None, None, None, None, None
        fx_te, fy_te, fz_te, cx_te, cy_te = None, None, None, None, None

        # Combine r (at index 0) and l (at index 1) sigs along sequence dim:
        fx_tr_stsd = torch.cat((fx_tr_stsd[0], fx_tr_stsd[1]), dim=2)
        fy_tr_stsd = torch.cat((fy_tr_stsd[0], fy_tr_stsd[1]), dim=2)
        fz_tr_stsd = torch.cat((fz_tr_stsd[0], fz_tr_stsd[1]), dim=2)
        cx_tr_stsd = torch.cat((cx_tr_stsd[0], cx_tr_stsd[1]), dim=2)
        cy_tr_stsd = torch.cat((cy_tr_stsd[0], cy_tr_stsd[1]), dim=2)

        fx_va_stsd = torch.cat((fx_va_stsd[0], fx_va_stsd[1]), dim=2)
        fy_va_stsd = torch.cat((fy_va_stsd[0], fy_va_stsd[1]), dim=2)
        fz_va_stsd = torch.cat((fz_va_stsd[0], fz_va_stsd[1]), dim=2)
        cx_va_stsd = torch.cat((cx_va_stsd[0], cx_va_stsd[1]), dim=2)
        cy_va_stsd = torch.cat((cy_va_stsd[0], cy_va_stsd[1]), dim=2)

        fx_te_stsd = torch.cat((fx_te_stsd[0], fx_te_stsd[1]), dim=2)
        fy_te_stsd = torch.cat((fy_te_stsd[0], fy_te_stsd[1]), dim=2)
        fz_te_stsd = torch.cat((fz_te_stsd[0], fz_te_stsd[1]), dim=2)
        cx_te_stsd = torch.cat((cx_te_stsd[0], cx_te_stsd[1]), dim=2)
        cy_te_stsd = torch.cat((cy_te_stsd[0], cy_te_stsd[1]), dim=2)

        if arch in ['--1T-1F', '--1T-2F', '--2T-1F', '--2T-2F', '--3T-1F', '--3T-2F']:

            sigs_tr = torch.cat((fx_tr_stsd, fy_tr_stsd, fz_tr_stsd, cx_tr_stsd, cy_tr_stsd,
                                 torch.zeros_like(fx_tr_stsd)), dim=1)
            sigs_va = torch.cat((fx_va_stsd, fy_va_stsd, fz_va_stsd, cx_va_stsd, cy_va_stsd,
                                 torch.zeros_like(fx_va_stsd)), dim=1)
            sigs_te = torch.cat((fx_te_stsd, fy_te_stsd, fz_te_stsd, cx_te_stsd, cy_te_stsd,
                                 torch.zeros_like(fx_te_stsd)), dim=1)

        else:

            # Re-combine channels. Re-declaring sigs_tr, sigs_va, and sigs_te to save memory:
            sigs_tr = torch.cat((fx_tr_stsd, fy_tr_stsd, fz_tr_stsd, cx_tr_stsd, cy_tr_stsd), dim=1)
            sigs_va = torch.cat((fx_va_stsd, fy_va_stsd, fz_va_stsd, cx_va_stsd, cy_va_stsd), dim=1)
            sigs_te = torch.cat((fx_te_stsd, fy_te_stsd, fz_te_stsd, cx_te_stsd, cy_te_stsd), dim=1)

        fx_tr_stsd, fy_tr_stsd, fz_tr_stsd, cx_tr_stsd, cy_tr_stsd = None, None, None, None, None
        fx_va_stsd, fy_va_stsd, fz_va_stsd, cx_va_stsd, cy_va_stsd = None, None, None, None, None
        fx_te_stsd, fy_te_stsd, fz_te_stsd, cx_te_stsd, cy_te_stsd = None, None, None, None, None

        sigs_tr[:, channel, window[0]:window[1]] = 0
        sigs_va[:, channel, window[0]:window[1]] = 0
        sigs_te[:, channel, window[0]:window[1]] = 0

        loader_tr, loader_va, loader_te = init_data_loaders_no_labs_va_te(DataLoaderLabels,
                                                                          DataLoaderNoLabels,
                                                                          sigs_tr, labels_tr,
                                                                          sigs_va, sigs_te,
                                                                          batch_size=bs)

        if arch == '-----1F':

            mod = OneF(in_features=int(sigs_tr.size(1) * sigs_tr.size(2)), out_features=600)

        elif arch == '-----2F':

            mod = TwoF(in_features=int(sigs_tr.size(1) * sigs_tr.size(2)), fc1_out=800, out_features=600)

        elif arch == '1C---1F':

            mod = OneCOneF(nc0=sigs_tr.size(1), nc1=32, out_features=800)

        elif arch == '1C---2F':

            mod = OneCTwoF(nc0=sigs_tr.size(1), nc1=32, fc1_out=1600, out_features=800)

        elif arch == '3C---1F':

            mod = ThrCOneF(nc0=sigs_tr.size(1), nc1=32, nc2=64, nc3=128, out_features=800)

        elif arch == '3C---2F':

            mod = ThrCTwoF(nc0=sigs_tr.size(1), nc1=32, nc2=64, nc3=128, fc1_out=1600, out_features=800)

        elif arch == '3C1LU1F':

            mod = ThrCOneLUOneF(nc0=sigs_tr.size(1), nc1=32, nc2=64, nc3=128, out_features=800)

        elif arch == '3C1LU2F':

            mod = ThrCOneLUTwoF(nc0=sigs_tr.size(1), nc1=32, nc2=64, nc3=128, fc1_out=1600, out_features=800)

        elif arch == '3C1LB1F':

            mod = ThrCOneLBOneF(nc0=sigs_tr.size(1), nc1=32, nc2=64, nc3=128, out_features=800)

        elif arch == '3C1LB2F':

            mod = ThrCOneLBTwoF(nc0=sigs_tr.size(1), nc1=32, nc2=64, nc3=128, fc1_out=1600, out_features=800)

        elif arch == '1C1T-1F':

            mod = OneCOneTOneF(nc0=sigs_tr.size(1), nc1=32, out_features=800)

        elif arch == '1C1T-2F':

            mod = OneCOneTTwoF(nc0=sigs_tr.size(1), nc1=32, fc1_out=1600, out_features=800)

        elif arch == '3C1T-1F':

            mod = ThrCOneTOneF(nc0=sigs_tr.size(1), nc1=32, nc2=64, nc3=128, out_features=800)

        elif arch == '3C1T-2F':

            mod = ThrCOneTTwoF(nc0=sigs_tr.size(1), nc1=32, nc2=64, nc3=128, fc1_out=1600, out_features=800)

        elif arch == '--1T-1F':

            mod = TOneF(nc0=sigs_tr.size(1), num_t_layers=1, out_features=800)

        elif arch == '--1T-2F':

            mod = TTwoF(nc0=sigs_tr.size(1), num_t_layers=1, fc1_out=1000, out_features=800)

        elif arch == '--2T-1F':

            mod = TOneF(nc0=sigs_tr.size(1), num_t_layers=2, out_features=800)

        elif arch == '--2T-2F':

            mod = TTwoF(nc0=sigs_tr.size(1), num_t_layers=2, fc1_out=1000, out_features=800)

        elif arch == '--3T-1F':

            mod = TOneF(nc0=sigs_tr.size(1), num_t_layers=3, out_features=800)

        elif arch == '--3T-2F':

            mod = TTwoF(nc0=sigs_tr.size(1), num_t_layers=3, fc1_out=1000, out_features=800)

        else:

            raise Exception('The specified arch is currently not supported.')

        mod_check = torch.load(os.path.join(results_path, 'mod_checks_one------_%s_%s_%s.pth' %
                                            (config[0], arch, bs)))[fold]
        mod.load_state_dict(mod_check[0])

        opt = optim.Adam(mod.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)

        loss_te, acc_te, embs_te = test(mod, loader_te, labels_te, criterion_eval, opt, cuda_te, single_input,
                                        count_te_min, indices_s1=None, indices_s2=None, n_similarity_search=None)

        losses_te_all.append(loss_te)
        accs_te_all.append(acc_te)
        embs_te_all.append(embs_te.cpu())
        embs_te = None

    return losses_te_all, accs_te_all, embs_te_all
