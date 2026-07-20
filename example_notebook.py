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
        path_to_blms="inputs/gaussian_elliptical_beams/",
        path_to_pol_moments="inputs/polmoments_ns0128/",  # planck h-maps are called polmoments
        path_to_rimo="inputs/RIMOs/RIMO_HFI_npipe5v16_symmetrized.fits",
        mmax_beam=mmax,
        lmax=lmax,
        blm_polar_efficiency="Ideal",  # If the blms contains polarization this does nothing, if they don't it applies the polarization efficiency when assuming copolar beams
        mapmaking_polar_efficiency="Ideal",  # Polarisation efficiency assumed by the mapmaking
        ref_frame_beams="Dxx",
        ref_frame_polmoments="Pxx",
    )
    det_planck_data.fill_blms_dict()

    det_planck_data.fill_h_maps_dict()
    return (det_planck_data,)


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
        temperature_only=False,
    )
    sky.fill_cmb_alms(
        det_planck_data.detector_names,
        path_to_cl="inputs/Cls_Planck2018_for_PTEP_2020_r0.fits",
        seed_cmb=1,
        apply_pixel_window=False,
    )
    # Or you can provide whatever set of alms you want with SkyData.set_alms_dict(alms_dict), where alms_dict is a dictionnary with a set of alms for each detector, the keys being the detector_names


    return (sky,)


@app.cell
def _(SkyData, det_planck_data, lmax, nside):
    # If some of your input, e.g foreground map, is convolved with a circular gaussian beam
    # the circular gaussian beam can be deconvolved
    # example
    sky2 = SkyData(
        nside=nside,
        lmax=lmax,
        temperature_only=False,
    )
    sky2.fill_cmb_alms(
        det_planck_data.detector_names,
        path_to_cl="inputs/Cls_Planck2018_for_PTEP_2020_r0.fits",
        seed_cmb=1,
        apply_pixel_window=False,
    )
    sky2.convolve_circular_gaussian(fwhm_arcmim=100)
    sky2.deconvolve_circular_gaussian(fwhm_arcmim=100)
    return


@app.cell
def _():
    return


@app.cell
def _():
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
    return (TQU_conv,)


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
            cmap="viridis",
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
    spec_order = [0, 1, 2, 3, 4, 5]
    spec_labels = ["TT", "EE", "BB", "TE", "EB", "TB"]
    ells = np.arange(lmax + 1)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharex=True)
    axes = axes.ravel()

    for ax, idx, cur_label in zip(axes, spec_order, spec_labels):
        ax.plot(ells, spect_conv[idx], label="Smarties -> anafast", alpha=0.8)

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
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
