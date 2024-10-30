import numpy as np
from scipy.signal import butter, filtfilt, decimate
from scipy.interpolate import interp1d
import torch
from sklearn.preprocessing import StandardScaler


# sigs must be of shape (N, C, L). ForceID-A is handled slightly different to the other datasets and this is indicated
# via the 'ds' param:
# Options
# --------
# 'fi-all' = ForceID-A (all measures)
# 'not_fi-all' = AIST, Gutenberg or GaitRec
# --------
def trim_filt_norm(sigs, fz_thresh, cutoff_freq, sampling_rate, interp_len, ds='fi-all'):

    sigs_pro = []
    start_1_margins = [] # List of margins between the measurement start and the first frame at which Fz is above 50N
    end_1_margins = [] # List of margins between the measurement end and the last frame at which Fz is above 50N
    lens_cop = []
    count_invalid_start_cop = 0
    count_invalid_end_cop = 0

    # Loop through individual footsteps/stance phases:
    for i in range(sigs.shape[0]):

        # Get first and last index where fz is above the 50N threshold:
        start_1 = np.asarray(sigs[i, 2, :] > fz_thresh).nonzero()[0][0]
        end_1 = np.asarray(sigs[i, 2, :] > fz_thresh).nonzero()[0][-1]

        start_1_margins.append(start_1)
        end_1_margins.append(sigs[i, 2, :].shape[0] - end_1)

        # Get the no. frames that is 5% of the total no. frames above the 50N threshold. This is used to further...
        # trim COP measures:
        cop_margin = int(np.round(0.05 * (end_1 - start_1)))

        if ds == 'fi-all':

            start_cop = start_1 + cop_margin
            end_cop = end_1 - cop_margin

            len_cop = end_cop - start_cop
            lens_cop.append(len_cop)

            # Down sample measures to 250 Hz (originally sampled at 2000Hz). Retain an additional 20 frames at each...
            # end for GRFs:
            grf_dns = decimate(sigs[i, :3, start_1 - 20:end_1 + 20], q=int(sampling_rate / 250), axis=1)
            cop_dns = decimate(sigs[i, 3:, start_cop:end_cop], q=int(sampling_rate / 250), axis=1)

            grf_filt = butterworth_lowpass(grf_dns, order=4, normal_cutoff=cutoff_freq / (0.5 * 250), pad_len=40)
            cop_filt = butterworth_lowpass(cop_dns, order=4, normal_cutoff=cutoff_freq / (0.5 * 250), pad_len=40)

        else: # 'not_fi-all'

            # Get the indices that would start and end the cop if using the method for ForceID-A:
            start_cop_fi = start_1 + cop_margin
            end_cop_fi = end_1 - cop_margin

            # Get the indices that would start and end the cop if using the method used to process the other datasets...
            # (e.g., Gutenberg original dataset description: "Furthermore, we cropped the filtered COP signals with...
            # a vertical GRF threshold of 80 N to avoid artifacts in COP calculation at small GRF signal values."):
            start_cop_other_ds = np.asarray(sigs[i, 2, :] > 80).nonzero()[0][0]
            end_cop_other_ds = np.asarray(sigs[i, 2, :] > 80).nonzero()[0][-1]

            start_cop = start_cop_fi - start_cop_other_ds
            end_cop = end_cop_fi - end_cop_other_ds
            len_cop = end_cop_fi - start_cop_fi
            # lens_cop.append(len_cop)

            # Validate the start and end indices for cop:
            if start_cop < 0:

                count_invalid_start_cop += 1

            if end_cop >= 0:

                count_invalid_end_cop += 1

            grf_filt = butterworth_lowpass(sigs[i, :3, :end_1 + 10], order=4,
                                           normal_cutoff=cutoff_freq / (0.5 * sampling_rate), pad_len=50)
            cop_filt = butterworth_lowpass(sigs[i, 3:, start_cop:start_cop + len_cop], order=4,
                                           normal_cutoff=cutoff_freq / (0.5 * sampling_rate), pad_len=50)

        # Get first and last index where filtered fz is above the 50N threshold:
        start_2 = np.asarray(grf_filt[2, :] > fz_thresh).nonzero()[0][0]
        end_2 = np.asarray(grf_filt[2, :] > fz_thresh).nonzero()[0][-1]

        # Interpolate between the times before and after filtered Fz crosses the threshold to determine the...
        # exact time points where Fz = 50N:
        interp_start = interp1d([grf_filt[2, start_2 - 1], grf_filt[2, start_2]], [start_2 - 1, start_2], axis=0)
        t_start = interp_start(fz_thresh)
        interp_end = interp1d([grf_filt[2, end_2], grf_filt[2, end_2 + 1]], [end_2, end_2 + 1], axis=0)
        t_end = interp_end(fz_thresh)

        # Interpolate GRFs to interp_len evenly spaced points between the start and end times defined above:
        t_grf = np.arange(0, grf_filt.shape[1], 1)
        interp_func = interp1d(t_grf, grf_filt, axis=1)
        t_new_grf = np.linspace(t_start, t_end, interp_len)
        grf_interp = interp_func(t_new_grf)

        # Interpolate COP coordinates to interp_len evenly spaced points between 0 and 100:
        t_cop = np.linspace(0, interp_len, cop_filt.shape[1])
        interp_func = interp1d(t_cop, cop_filt, axis=1)
        t_new_cop = np.linspace(0, interp_len, interp_len)
        cop_interp = interp_func(t_new_cop)

        # Set COP coordinates to start at (0, 0):
        cop_zeroed = cop_interp - np.expand_dims(cop_interp[:, 0], axis=1)

        # Recombine GRF and COP then append to processed sigs object:
        sigs_pro.append(np.concatenate((grf_interp, cop_zeroed), axis=0))

    return np.array(sigs_pro)


# Butterworth low-pass filter:
def butterworth_lowpass(sig, order, normal_cutoff, pad_len):
    b, a = butter(order, normal_cutoff)
    sigs_filtered = filtfilt(b, a, sig, axis=1, padlen=pad_len)
    return sigs_filtered


# Get individual directional components of the GRF and COP (respectively) from a data object shape (N, C, L),
# where N = no. samples, C = no. channels (5), and L = sequence length:
def get_directional_components(sigs):

    fx = sigs[:, 0, :]
    fy = sigs[:, 1, :]
    fz = sigs[:, 2, :]
    cx = sigs[:, 3, :]
    cy = sigs[:, 4, :]

    return fx, fy, fz, cx, cy


# Standardise (z-score normalise) measures of a given directional component in training, validation, and test sets...
# based on mean and std of training set. Inputs are arrays of shape (2N, L) and outputs are tuples of shape...
# (2,) with each tuple being an (N, 1, L) tensor, where N = no. samples and L = length. Inputs axis 0 is 2N because...
# measures of each stance side from each sample are concatenated along the sample axis for normalisation. Each element
# of the output tuple is a stance side. Within each element, dim 1 is there so that the directional components can...
# subsequently be concatenated along the first dimension outside this function:
def standardise_then_split(component_tr, component_va, component_te):

    s_scaler = StandardScaler()
    s_scaler.fit(component_tr)
    component_tr_stsd = torch.split(torch.tensor(np.expand_dims(s_scaler.transform(component_tr),
                                                                axis=1), dtype=torch.float32),
                                    int(component_tr.shape[0] / 2))
    component_va_stsd = torch.split(torch.tensor(np.expand_dims(s_scaler.transform(component_va),
                                                                axis=1), dtype=torch.float32),
                                    int(component_va.shape[0] / 2))
    component_te_stsd = torch.split(torch.tensor(np.expand_dims(s_scaler.transform(component_te),
                                                                axis=1), dtype=torch.float32),
                                    int(component_te.shape[0] / 2))

    return component_tr_stsd, component_va_stsd, component_te_stsd
