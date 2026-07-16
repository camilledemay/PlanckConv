import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    from PlanckConv.core_functions import (
        SkyData,
        PlanckDetectorsData,
        compute_convolved_planck_map,
    )
    import matplotlib.pyplot as plt
    import healpy as hp
    import marimo as mo
    import numpy as np

    return (
        PlanckDetectorsData,
        SkyData,
        compute_convolved_planck_map,
        hp,
        mo,
        np,
        plt,
    )


@app.cell
def _():
    nside = 128
    lmax = 2 * nside
    return lmax, nside


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Load the detector informations: blms and h-maps
    """)
    return


@app.cell
def _(PlanckDetectorsData, lmax):
    mmax = 4
    det_planck_data = PlanckDetectorsData(
        detector_set="100A",  # you can refer to list_planck() to check the different set of detectors
        path_to_blms="/home/camille/Documents/PhD/litebird_thingy/h-maps_beam_conv_paper/GRASP beam/beams/",
        path_to_pol_moments="/home/camille/Documents/PhD/litebird_thingy/h-maps_beam_conv_paper/polmoments_ns0128_ideal",  # planck h-maps are called polmoments
        path_to_rimo="inputs/RIMOs/RIMO_HFI_npipe5v16_symmetrized.fits",
        mmax_beam=mmax,
        lmax=lmax,
        blm_polar_efficiency="IMO", # Set this to ideal if youre blms already contains the polarisation efficiency
        mapmaking_polar_efficiency="IMO", # Polarisation efficiency assumed by the mapmaking
        ref_frame_beams="Dxx",
        ref_frame_polmoments="Pxx",
    )

    # det_planck_data.rho_blm = (
    #     np.ones_like(det_planck_data.detector_names, dtype=int) * 0.1
    # )
    # det_planck_data.rho_mapmaking = (
    #     np.ones_like(det_planck_data.detector_names, dtype=int) * 0.1
    # )
    det_planck_data.fill_blms_dict()

    det_planck_data.fill_h_maps_dict()
    return det_planck_data, mmax


@app.cell
def _():
    return


@app.cell
def _(det_planck_data):
    det_planck_data.rho_mapmaking
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Set the sky alms
    """)
    return


@app.cell
def _(SkyData, det_planck_data, lmax, nside):
    sky = SkyData(
        nside=nside,
        lmax=lmax,
        apply_pixel_window=False,
        temperature_only=False,
    )
    sky.fill_cmb_alms(
        det_planck_data.detector_names,
        path_to_cl="inputs/Cls_Planck2018_for_PTEP_2020_r0.fits",
        seed_cmb=1,
    )
    # Or you can provide whatever set of alms you want with SkyData.set_alms_dict(alms_dict), where alms_dict is a dictionnary with a set of alms for each detector, the keys being the detector_names
    return (sky,)


@app.cell
def _():
    # import pickle

    # lbs_alms = pickle.load(
    #     open(
    #         "/home/camille/Documents/PhD/litebird_thingy/h-maps_beam_conv_paper/smarties_with_planck_polmoments/cmb_alms_dict.pkl",
    #         mode="rb",
    #     )
    # )
    # sky.set_alms_dict(lbs_alms)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Generate the spin maps and run the Smarties mapmaking
    """)
    return


@app.cell
def _(compute_convolved_planck_map, det_planck_data, sky):
    TQU_conv, inverse_mm = compute_convolved_planck_map(
        sky_data=sky,
        detector_data=det_planck_data,
        inverse_mapmaking_matrix=None,  # You can reuse the mapmaking matrix of a detector to avoid recomputing at each mapmaking
        return_inverse_mapmaking_matrix=True,
        output_directory=None,
        condition_number_threshold=None,
    )
    return TQU_conv, inverse_mm


@app.cell
def _(det_planck_data, hp, nside, plt, sky):
    input_map = hp.alm2map(
        sky.alms_dict[det_planck_data.detector_names[0]], nside=nside, pol=True
    )
    plt.figure(figsize=(12, 3), dpi=200)
    for i, label in enumerate(["T", "Q", "U"]):
        hp.mollview(
            input_map[i],
            sub=(1, 3, i + 1),
            title=f"{label}",
            unit=r"$\mu$K",
            cmap="viridis",
            format="%.1e",
            cbar=True,
            fontsize={"": 00},
        )
    plt.suptitle("Input", fontsize=15)
    plt.show()
    return


@app.cell
def _(TQU_conv, hp, plt):
    plt.figure(figsize=(12, 3), dpi=200)
    for j, label2 in enumerate(["T", "Q", "U"]):
        hp.mollview(
            TQU_conv[j],
            sub=(1, 3, j + 1),
            title=f"{label2}",
            unit=r"$\mu$K",
            cmap="seismic",
            format="%.1e",
            cbar=True,
            fontsize={"": 00},
        )
    plt.suptitle("Convolved maps", fontsize=15)
    plt.show()
    return


@app.cell
def _(TQU_conv, det_planck_data, hp, lmax, sky):
    spect_conv = hp.anafast(TQU_conv, lmax=lmax)
    spect_input = hp.alm2cl(sky.alms_dict[det_planck_data.detector_names[0]])
    return spect_conv, spect_input


@app.cell
def _(lmax, np, plt, spect_conv, spect_input):
    # ---------- Plot mean spectra comparison ----------
    spec_order = [0, 1, 2, 3, 4, 5]
    spec_labels = ["TT", "EE", "BB", "TE", "EB", "TB"]
    ells = np.arange(lmax + 1)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharex=True)
    axes = axes.ravel()

    for ax, idx, cur_label in zip(axes, spec_order, spec_labels):
        ax.plot(ells, spect_conv[idx], label="Smarties -> anafast", alpha=0.8)
        if cur_label in ["TB", "EB"]:
            ax.plot(
                ells,
                np.zeros_like(ells),
                "--",
                color="green",
                label="Input (0)",
            )
        else:
            ax.plot(
                ells,
                spect_input[idx][: lmax + 1],
                "--",
                color="green",
                label="Input",
            )
        qp_label = "TT" if cur_label in ["TT", "TE", "TB"] else cur_label
        # Shaded error band for Smarties
        ax.set_title(cur_label)
        if cur_label in ["TE", "TB", "EB"]:
            ax.set_yscale("symlog", linthresh=1e-7)
        else:
            ax.set_yscale("log")
        # ax.set_xscale("log")
        ax.set_xlim(2, lmax)
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    plt.show()
    return


@app.cell
def _(det_planck_data):
    det_planck_data.pol_angles_rad
    return


@app.cell
def _(det_planck_data, hp, mmax, np, plt):
    plt.figure(figsize=(15, 10), dpi=130)
    det_index = 0
    det_names = det_planck_data.detector_names
    h_n_spin_dict = det_planck_data.h_maps_dict
    for s in np.arange(mmax + 3):
        print(
            f"mean of h_{s} for detector {det_names[det_index]}: {np.mean(h_n_spin_dict[s][det_index].real)}"
        )
        hp.mollview(
            h_n_spin_dict[s][det_index].real,
            title=f"h_{s} for detector {det_names[det_index]}, mean: {np.mean(h_n_spin_dict[s][det_index].real):.1e}",
            sub=(3, 3, s + 1),
        )
    plt.show()
    return det_names, h_n_spin_dict


@app.cell
def _(det_names, h_n_spin_dict, hp, mmax, np, plt):
    def _():
        fig = plt.figure(figsize=(15, 10), dpi=130)
        det_index = 0
        for s in np.arange(mmax + 3):
            hp.mollview(
                h_n_spin_dict[s][det_index].real
                - h_n_spin_dict[s][det_index + 1].real,
                title=f"h_{s} for detector {det_names[det_index]}, max: {np.max(h_n_spin_dict[s][det_index].real - h_n_spin_dict[s][det_index + 1].real):.1e}",
                sub=(3, 3, s + 1),
            )

    _()
    plt.show()
    return


@app.cell
def _(hp, inverse_mm, np, nside, plt):
    def _():
        nstokes = 3
        total_mask = np.ones(hp.nside2npix(nside))
        mapmaking_matrix = np.linalg.pinv(inverse_mm)

        extended_mapmaking_matrix = (
            np.ones(
                (
                    nstokes,
                    nstokes,
                )
                + total_mask.shape
            ).squeeze()
            * hp.UNSEEN
        )

        # value = 10
        plt.figure(figsize=(10, 8))
        extended_mapmaking_matrix[:, :, total_mask != 0] = (
            mapmaking_matrix.T.real
        )
        for i in range(nstokes**2):
            row_ = i // nstokes
            col_ = i % nstokes

            hp.mollview(
                extended_mapmaking_matrix[row_, col_, :],
                sub=(nstokes, nstokes, i + 1),
                cmap="seismic",
                title=f"M element ({row_},{col_}) real \n <mean> = {mapmaking_matrix[:, row_, col_].real.mean():.2e}",
            )  # , min=-value, max=value)

        plt.figure(figsize=(10, 8))
        extended_mapmaking_matrix[:, :, total_mask != 0] = (
            mapmaking_matrix.T.imag
        )
        for i in range(nstokes**2):
            row_ = i // nstokes
            col_ = i % nstokes
        return hp.mollview(
            extended_mapmaking_matrix[row_, col_, :],
            sub=(nstokes, nstokes, i + 1),
            cmap="seismic",
            title=f"M element ({row_},{col_}) imag \n <mean> = {mapmaking_matrix[:, row_, col_].imag.mean():.2e}",
        )  # , min=-value, max=value)

    _()
    plt.show()
    return


@app.cell
def _(det_planck_data, np, sky):
    import smarties.systematics.beam_convolution as sm_beam_conv

    spin_syst = sm_beam_conv.get_systematic_maps_from_alms_blms(
        sky.alms_dict,
        det_planck_data.blms_dict,
        np.ones(len(det_planck_data.detector_names)),
        det_planck_data.detector_names,
        sky.lmax,
        det_planck_data.mmax_beam,
        sky.nside,
        det_planck_data.pol_angles_rad,
        substract_gaussian_beam=False,
    )
    return (spin_syst,)


@app.cell
def _(det_names, hp, mmax, np, plt, spin_syst):
    def _():
        fig = plt.figure(figsize=(15, 10), dpi=130)
        det_index = 0
        for s in np.arange(0, mmax + 2, 2):
            hp.mollview(
                spin_syst[s][det_index].real,
                title=f"Spin map {s} for detector {det_names[det_index]}, mean: {np.mean(spin_syst[s][det_index].real):.1e}",
                sub=(3, 3, s + 1),
            )
        return plt.show()

    _()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
