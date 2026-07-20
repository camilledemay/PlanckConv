#!/usr/bin/env python
"""Core functions for Smarties map-making and QuickPol beam matrices."""

import os
import time

import healpy as hp
import numpy as np
from smarties.hn import Spin_maps
from smarties.mapmaking import FrameworkSystematics
from smarties.systematics.beam_convolution import convert_alm_spin_to_plusminus

from PlanckConv.external_qp_planck import (
    get_angles,
    get_blms_fits,
)
# ----------------------------------------------------------------------
# Load Planck hit‑map moments and build spin maps


def load_hmap_planck_1_det(
    path_to_moments, det_name, smax, spin_ref, RIMO, dtype=np.complex128
):
    """Load the Planck h-maps of one detector and rotate it so that they follow the same conventiona as smarties."""
    hitfile = os.path.join(path_to_moments, f"polmoments_{det_name}_hits.fits")
    momfile = os.path.join(path_to_moments, f"polmoments_{det_name}.fits")

    if spin_ref == "Pxx":
        myangle = -get_angles(RIMO, [det_name], ref="Dxx")[0]
    else:
        myangle = 0

    t1 = time.time()
    hit = hp.read_map(hitfile)
    spins = hp.read_map(momfile, None)
    h_maps = np.zeros((smax + 1, hit.shape[0]), dtype=dtype)

    print(f"Loaded hits+spins in {time.time() - t1:.2f}s", flush=True)

    for s in range(smax + 1):
        if s == 0:
            buf = hit.astype(dtype)
        else:
            buf = (spins[2 * s - 2] + 1j * spins[2 * s - 1]) / hit
            if myangle != 0:
                buf *= np.cos(s * myangle) + 1j * np.sin(s * myangle)
        h_maps[s] = buf
    return h_maps


def build_Planck_h_maps_dictionnary(
    det_names, moments_dir, smax, spin_ref, RIMO, dtype, detector_weights
):
    """Load all detectors and build h_n_spin_dict up to a spin smax."""
    h_maps_list = []
    hits_list = []
    for det in det_names:
        h_maps = load_hmap_planck_1_det(moments_dir, det, smax, spin_ref, RIMO, dtype)
        h_maps_list.append(h_maps)
        hits_list.append(h_maps[0].real)
        assert np.all(h_maps[0].real > 0), "minute papillion"
    hits_arr = np.array(hits_list)

    for idet, det in enumerate(det_names):
        print(f"Weight of det {det}: {detector_weights[det[:-1]]}")
        hits_arr[idet] *= detector_weights[det[:-1]]  # 100-1a -> 100-1
    total_hits = hits_arr.sum(axis=0)
    mask_hits = (total_hits > 0).astype(np.int8)

    list_hn_spins = np.arange(0, smax + 1)  # up to smax
    h_n_dict = {
        s: np.zeros((len(det_names), total_hits.size), dtype=dtype)
        for s in list_hn_spins
    }
    for idet, (hits, h_map) in enumerate(zip(hits_arr, h_maps_list)):
        h_n_dict[0][idet] = hits / (total_hits)
        for s in list_hn_spins:
            if s == 0:
                continue
            h_n_dict[s][idet] = h_map[s] * hits / total_hits
    # add negative spins
    for s in list_hn_spins:
        if s != 0:
            h_n_dict[-s] = np.conj(h_n_dict[s])

    assert np.all(h_n_dict[0].imag < 1e-7), "h_n_dict[0] has non-zero imaginary part"
    return Spin_maps.from_dictionary(h_n_dict), mask_hits


# ----------------------------------------------------------------------
# CMB generation
def generate_cmb_alms(
    det_names,
    seed_cmb,
    path_to_cl,
    polarized,
    nside,
    lmax,
    apply_pixel_window=False,
):
    """Generate CMB alms from a Cl"""
    np.random.seed(seed_cmb)

    cls = hp.read_cl(path_to_cl)
    if apply_pixel_window:
        match cls.shape:
            case (1,):
                cls *= hp.pixwin(nside, lmax=lmax, pol=False) ** 2
            case (3,):
                cls *= hp.pixwin(nside, lmax=lmax, pol=True) ** 2
    alms = hp.synalm(cls=cls, lmax=lmax, new=True)
    if alms.shape[0] == 1:
        print("Alms only contain temperature, padding polarization with zeros")
        alms = np.atleast_2d(alms)
        alms = np.pad(alms, ((0, 2), (0, 0)), mode="constant", constant_values=0)
    if not polarized:
        alms[1] *= 0
        alms[2] *= 0
    if apply_pixel_window:
        apply_pixwin(alms, nside, lmax)
    alms_dict = {}
    for det in det_names:
        alms_dict[det] = alms
    return alms_dict


def apply_pixwin(alms, nside, lmax):
    Twindow, Pwindow = hp.pixwin(nside, lmax=lmax, pol=True)
    hp.almxfl(alms[0], Twindow, inplace=True)
    hp.almxfl(alms[1], Pwindow, inplace=True)
    hp.almxfl(alms[2], Pwindow, inplace=True)


# ----------------------------------------------------------------------
# Map‑making with Smarties
def run_smarties_mapmaking(
    h_n_spin_dict,
    mask_hits,
    spin_sky_maps,
    spin_systematics_maps,
    lmax,
    pol_ang_rad,
    pol_efficiency,
    inverse_mapmaking_matrix,
    return_inverse_mapmaking_matrix,
    condition_number_mask,
    condition_number_threshold=10,
):
    """Compute final T, Q, U maps using FrameworkSystematics."""
    syst = FrameworkSystematics(
        map_shape=(1, mask_hits.size), nstokes=3, lmax=lmax, list_spin_output=[0, -2, 2]
    )
    out = syst.compute_total_maps(
        mask_hits,
        h_n_spin_dict,
        spin_sky_maps,
        spin_systematics_maps,
        return_Q_U=False,
        inverse_mapmaking_matrix=inverse_mapmaking_matrix,
        return_inverse_mapmaking_matrix=return_inverse_mapmaking_matrix
        or condition_number_mask,
        mask_input=False,
        polar_angle=pol_ang_rad,
        polar_efficiency_coeff=pol_efficiency,
    )
    if return_inverse_mapmaking_matrix or condition_number_mask:
        final_spin_maps, inverse_mapmaking_matrix = out
    else:
        final_spin_maps = out

    final_I = final_spin_maps[0].real
    final_Q = ((final_spin_maps[-2] + final_spin_maps[2]) / 2).real
    final_U = (1j * (final_spin_maps[-2] - final_spin_maps[2]) / 2).real

    tqu = np.ones((3, mask_hits.size), dtype=float) * hp.UNSEEN

    if condition_number_mask:
        cond_number = np.linalg.cond(inverse_mapmaking_matrix)
        cond_mask = cond_number < condition_number_threshold
        print(cond_mask)
        print(mask_hits)
        print(cond_mask.shape)
        print(final_I.shape)
        full_mask = cond_mask & mask_hits.astype(bool)
        print(full_mask.shape)

        print(
            f"Maximum value of the condition number: {np.max(cond_number)}", flush=True
        )
    else:
        full_mask = mask_hits.astype(bool)
    tqu[0, full_mask] = final_I[full_mask]
    tqu[1, full_mask] = final_Q[full_mask]
    tqu[2, full_mask] = final_U[full_mask]

    if return_inverse_mapmaking_matrix:
        return tqu, inverse_mapmaking_matrix
    else:
        return tqu


def get_Planck_det_blms(
    det_names,
    path_to_beams,
    lmax,
    mmax_beam,
    pol_ang_rad,
    polarisation_efficiencies,
):
    blms_dict = {}
    for idet, det in enumerate(det_names):
        polarisation_efficiency = polarisation_efficiencies[idet]
        beam_path = os.path.join(path_to_beams, f"blm_{det}.fits")
        blms = load_Planck_blms_copolar(
            beam_path,
            lmax=lmax,
            mmax=mmax_beam,
            isbalm=False,
            renorm=True,
            polang=pol_ang_rad[idet],
            poleff=polarisation_efficiency,
        )
        blms *= 1 / np.sqrt(4 * np.pi)  # renormalize to match smarties convention
        blms_dict[det] = blms

    return blms_dict


def convert_Planck_blms_to_hp_format(blms, lmax, mmax):
    blms_output = np.zeros((3, hp.Alm.getsize(lmax, mmax)), dtype=np.complex128)
    for l in range(lmax + 1):
        for m in range(min(l, mmax) + 1):
            idx_m = hp.Alm.getidx(lmax, l, m)
            blms_output[0, idx_m] = blms[l, m, 0]
            blms_output[1, idx_m] = blms[l, m, 1]
            blms_output[2, idx_m] = blms[l, m, 2]
    blms_output[1], blms_output[2] = convert_alm_spin_to_plusminus(
        blms_output[1].copy(), blms_output[2].copy(), spin=2
    )
    return blms_output


def load_Planck_blms_copolar(
    fitsfile, lmax, mmax, polang=0, poleff=1, isbalm=False, renorm=True
):
    """Load the beam harmonic coefficients from a FITS file and convert them to the healpy format, if they do not contain polarization assumes copolarity."""
    blms_grasp = get_blms_fits(
        fitsfile, lmax=lmax, mmax=mmax, isbalm=isbalm, renorm=renorm
    )
    if blms_grasp.shape[2] == 3:
        print(f"Blms in {fitsfile} contains polarization.")
        blms_grasp = convert_Planck_blms_to_hp_format(
            blms_grasp, lmax, mmax
        )  # do not apply poleff to already polarized blms
        return blms_grasp
    else:
        print(f"Blms in {fitsfile} do not contain polarization, assuming copolarity.")

        blms_grasp_temp = blms_grasp
    blms_grasp = np.zeros((3, hp.Alm.getsize(lmax, mmax)), dtype=np.complex128)

    def get_blm_lm(l: int, m: int):
        # Return b_lm
        if abs(m) > l or abs(m) > mmax:
            return 0.0j
        if m >= 0:
            return blms_grasp_temp[l, m, 0]
        else:
            mp = -m
            return ((-1) ** mp) * np.conjugate(blms_grasp_temp[l, mp, 0])

    phase_p2 = np.exp(2j * polang)
    phase_m2 = np.exp(-2j * polang)

    for l in range(lmax + 1):
        for m in range(min(l, mmax) + 1):
            idx_m = hp.Alm.getidx(lmax, l, m)
            blms_grasp[0, idx_m] = blms_grasp_temp[l, m, 0]

            b_m_plus_2 = phase_p2 * get_blm_lm(l, m + 2)
            b_m_minus_2 = phase_m2 * get_blm_lm(l, m - 2)

            blms_grasp[1, idx_m] = -0.5 * (b_m_plus_2 + b_m_minus_2) * poleff  # blm E
            blms_grasp[2, idx_m] = 0.5j * (b_m_plus_2 - b_m_minus_2) * poleff  # blm B

    return blms_grasp
