from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import healpy as hp
import numpy as np
import smarties.systematics.beam_convolution as sm_beam_conv
from smarties.hn import Spin_maps
from smarties.tools import transform_array_maps_into_spin_maps

from PlanckConv.core_functions import (
    build_Planck_h_maps_dictionnary,
    generate_cmb_alms,
    get_Planck_det_blms,
    run_smarties_mapmaking
)
from PlanckConv.external_qp_planck import (
    detector_weights,
    get_angles,
    list_planck,
    load_RIMO,
)


@dataclass(slots=True)
class PlanckDetectorsData:
    """Detector-specific informations."""

    detector_set: str
    path_to_blms: str
    path_to_pol_moments: str
    path_to_rimo: str
    mmax_beam: int
    lmax: int
    ref_frame_beams: str
    ref_frame_polmoments: str
    blm_polar_efficiency: str
    mapmaking_polar_efficiency: str

    detector_subset: int | None = None

    # Dynamic attributes initialized later
    rimo: Any = field(init=False)
    detector_names: Any = field(init=False)
    rho_mapmaking: Any = field(init=False)
    rho_blm: Any = field(init=False)

    pol_angles_rad: Any = field(init=False)
    blms_dict: Any = field(init=False)
    h_maps_dict: Any = field(init=False)

    def __post_init__(self):
        self.rimo = load_RIMO(self.path_to_rimo)
        self._set_detector_names()
        if self.detector_names == -1:
            raise ValueError("Invalid detector subset")
        self._set_polarisation_efficiencies()
        self._set_pol_angles_rad()

    def _set_detector_names(self):
        self.detector_names = list_planck(
            self.detector_set, subset=self.detector_subset
        )

    def _set_polarisation_efficiencies(self):
        if self.blm_polar_efficiency == "IMO":
            rho_blm = [
                (1 - self.rimo[det].epsilon) / (1 + self.rimo[det].epsilon)
                for det in self.detector_names
            ]
        elif self.blm_polar_efficiency == "Ideal":
            rho_blm = [1 for det in self.detector_names]
        else:
            raise ValueError(
                f"Unknown polarisation efficiency model: {self.blm_polar_efficiency}"
            )

        if self.mapmaking_polar_efficiency == "IMO":
            rho_mapmaking = [
                (1 - self.rimo[det].epsilon) / (1 + self.rimo[det].epsilon)
                for det in self.detector_names
            ]
        elif self.mapmaking_polar_efficiency == "Ideal":
            rho_mapmaking = [1 for det in self.detector_names]
        else:
            raise ValueError(
                f"Unknown polarisation efficiency model: {self.mapmaking_polar_efficiency}"
            )

        self.rho_blm = np.array(rho_blm)
        self.rho_mapmaking = np.array(rho_mapmaking)

    def _set_pol_angles_rad(self):
        pol_angles_rad = get_angles(
            RIMO=self.rimo, shorts=self.detector_names, ref=self.ref_frame_beams
        )
        self.pol_angles_rad = pol_angles_rad

    def fill_blms_dict(self):

        blms_dict = get_Planck_det_blms(
            det_names=self.detector_names,
            path_to_beams=self.path_to_blms,
            lmax=self.lmax,
            mmax_beam=self.mmax_beam,
            pol_ang_rad=self.pol_angles_rad,
            polarisation_efficiencies=self.rho_blm,
        )
        self.blms_dict = blms_dict

    def set_h_maps_dict(self, h_maps_dict):
        self.h_maps_dict = h_maps_dict

    def fill_h_maps_dict(self, dtype: type = np.complex128):
        if not hasattr(self, "detector_names"):
            self._set_detector_names()
        h_maps_dict, _ = build_Planck_h_maps_dictionnary(
            det_names=self.detector_names,
            moments_dir=self.path_to_pol_moments,
            smax=self.mmax_beam + 2,
            spin_ref=self.ref_frame_polmoments,
            RIMO=self.rimo,
            dtype=dtype,
            detector_weights=detector_weights,
        )
        self.h_maps_dict = h_maps_dict


@dataclass(slots=True)
class SkyData:
    """Sky-specific inputs that can be reused across detector sets."""

    nside: int
    lmax: int
    temperature_only: bool = False

    alms_dict: Any = field(default_factory=dict)

    def fill_cmb_alms(
        self,
        detector_names,
        path_to_cl: str,
        seed_cmb: int | None = None,
        apply_pixel_window: bool = False,
    ):
        alms_dict = generate_cmb_alms(
            det_names=detector_names,
            path_to_cl=path_to_cl,
            lmax=self.lmax,
            nside=self.nside,
            seed_cmb=seed_cmb,
            apply_pixel_window=apply_pixel_window,
            polarized=not self.temperature_only,
        )
        self.alms_dict = alms_dict

    def set_alms_dict(self, alms_dict):
        """Set the alms_dict attribute."""
        self.alms_dict = alms_dict

    def deconvolve_circular_gaussian(self, fwhm_arcmim: float):
        """Deconvolve the circular Gaussian."""
        Bl = hp.sphtfunc.gauss_beam(
            np.deg2rad(fwhm_arcmim / 60), lmax=self.lmax, pol=True
        )
        for key in self.alms_dict.keys():
            hp.sphtfunc.almxfl(self.alms_dict[key][0], 1 / Bl[:, 0], inplace=True)
            hp.sphtfunc.almxfl(self.alms_dict[key][1], 1 / Bl[:, 1], inplace=True)
            hp.sphtfunc.almxfl(self.alms_dict[key][2], 1 / Bl[:, 2], inplace=True)

    def convolve_circular_gaussian(self, fwhm_arcmim: float):
        """Convolve the alms with a circular Gaussian beam."""
        Bl = hp.sphtfunc.gauss_beam(
            np.deg2rad(fwhm_arcmim / 60), lmax=self.lmax, pol=True
        )
        for key in self.alms_dict.keys():
            hp.sphtfunc.almxfl(self.alms_dict[key][0], Bl[:, 0], inplace=True)
            hp.sphtfunc.almxfl(self.alms_dict[key][1], Bl[:, 1], inplace=True)
            hp.sphtfunc.almxfl(self.alms_dict[key][2], Bl[:, 2], inplace=True)

    def __add__(self, other: "SkyData") -> "SkyData":
        assert self.lmax == other.lmax, (
            "The lmax of the two SkyData objects do not match"
        )
        assert self.nside == other.nside, (
            "The nside of the two SkyData objects do not match"
        )
        return SkyData(
            nside=self.nside,
            lmax=self.lmax,
            temperature_only=self.temperature_only,
            alms_dict={
                key: self.alms_dict[key] + other.alms_dict[key]
                for key in self.alms_dict.keys()
            },
        )

    def __sub__(self, other: "SkyData") -> "SkyData":
        assert self.lmax == other.lmax, (
            "The lmax of the two SkyData objects do not match"
        )
        assert self.nside == other.nside, (
            "The nside of the two SkyData objects do not match"
        )
        return SkyData(
            nside=self.nside,
            lmax=self.lmax,
            temperature_only=self.temperature_only,
            alms_dict={
                key: self.alms_dict[key] - other.alms_dict[key]
                for key in self.alms_dict.keys()
            },
        )


def compute_convolved_planck_map(
    sky_data: SkyData,
    detector_data: PlanckDetectorsData,
    output_directory: Path | str | None = None,
    inverse_mapmaking_matrix: np.ndarray | None = None,
    return_inverse_mapmaking_matrix: bool = False,
    condition_number_threshold: float | None = None,
):

    assert sky_data.lmax == detector_data.lmax, "The blms and alms lmax do not match"
    assert list(sky_data.alms_dict.keys()) == detector_data.detector_names, (
        "The alms_dict keys do not match the detector names"
    )
    spin_syst_dict = sm_beam_conv.get_systematic_maps_from_alms_blms(
        sky_data.alms_dict,
        detector_data.blms_dict,
        np.ones(len(detector_data.detector_names)),
        detector_data.detector_names,
        sky_data.lmax,
        detector_data.mmax_beam,
        sky_data.nside,
        detector_data.pol_angles_rad,
        substract_gaussian_beam=False,
    )

    spin_syst = Spin_maps.from_dictionary(spin_syst_dict)
    print(np.max(np.abs(spin_syst[4])))
    print(np.max(np.abs(spin_syst[3])))
    print(np.max(np.abs(spin_syst[2])))
    print(np.max(np.abs(spin_syst[4])))
    empty_sky = transform_array_maps_into_spin_maps(
        np.zeros((3, hp.nside2npix(sky_data.nside))), n_stokes_output=3
    )
    output = run_smarties_mapmaking(
        h_n_spin_dict=detector_data.h_maps_dict,
        mask_hits=np.ones(hp.nside2npix(sky_data.nside)),
        spin_sky_maps=empty_sky,
        spin_systematics_maps=spin_syst,
        lmax=sky_data.lmax,
        pol_ang_rad=detector_data.pol_angles_rad,
        pol_efficiency=detector_data.rho_mapmaking,
        inverse_mapmaking_matrix=inverse_mapmaking_matrix,
        return_inverse_mapmaking_matrix=return_inverse_mapmaking_matrix,
        condition_number_mask=condition_number_threshold is not None,
        condition_number_threshold=condition_number_threshold,
    )
    if return_inverse_mapmaking_matrix:
        inverse_mapmaking_matrix = output[1]
        TQU_convolved_map = output[0]
    else:
        TQU_convolved_map = output
    if output_directory is not None:
        if isinstance(output_directory, str):
            output_directory = Path(output_directory)
        file_name = Path(
            f"TQU_map_nside{sky_data.nside}_{detector_data.detector_set}_mmax_{detector_data.mmax_beam}.npy"
        )
        np.save(output_directory / file_name, TQU_convolved_map)
    if return_inverse_mapmaking_matrix:
        return TQU_convolved_map, inverse_mapmaking_matrix
    else:
        return TQU_convolved_map
