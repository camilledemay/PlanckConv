from dataclasses import dataclass
import numpy as np
from astropy.io import fits
from typing import Sequence, Optional,Any,Union,Dict,List
# The following function are taken from the qp_planck repo: https://github.com/paganol/qp_planck/tree/main

#: Conversion factor from degrees to radians.
degree: float = np.pi / 180.0

def get_angles(RIMO, shorts, ref="Dxx"):
    """Return detector angles (in radians) for a list of short names."""
    angles = np.zeros(len(shorts))
    for idet, det in enumerate(shorts):
        if ref == "Dxx":
            angles[idet] = RIMO[det].psi_uv + RIMO[det].psi_pol
        elif ref == "Pxx":
            if "LFI" in det:
                angles[idet] = RIMO[det].psi_pol
            else:
                angles[idet] = 0
        else:
            raise RuntimeError(f"Unknown referential: {ref}")
    angles = np.radians(angles)
    return angles

@dataclass
class DetectorData:
    """
    Container for a single Planck detector's RIMO / focal plane information.

    Parameters
    ----------
    name : str
        Detector identifier (e.g. 'LFI27M', '143-1a', etc.).
    phi_uv : float
        Azimuth angle of the detector direction in degrees (UV frame).
    theta_uv : float
        Polar angle of the detector direction in degrees (UV frame).
    psi_uv : float
        Additional rotation angle in degrees (UV frame).
    psi_pol : float
        Polarization angle in degrees.
    epsilon : float
        Polarization efficiency (0–1).
    fsample : float
        Sample frequency in Hz.
    fknee : float
        1/f noise knee frequency in Hz.
    alpha : float
        1/f noise spectral index.
    net : float
        Noise Equivalent Temperature (NET).
    fwhm : float
        Beam full width at half maximum, typically in arcminutes.
    """

    name: str
    phi_uv: float
    theta_uv: float
    psi_uv: float
    psi_pol: float
    epsilon: float
    fsample: float
    fknee: float
    alpha: float
    net: float
    fwhm: float




#: Namespace object to mimic 'qa.mult' used in legacy code.

#: Bore-sight rotation quaternion. Set to identity by default (no extra rotation).
SPINROT: np.ndarray = np.array([0.0, 0.0, 0.0, 1.0])
def load_RIMO(path: str, comm: Optional[Any] = None) -> dict[str, DetectorData]:
    """
    Load and (optionally) broadcast the reduced instrument model (RIMO).

    The RIMO is the "focal plane database" for Planck: it describes
    detector positions, orientations, beam properties and noise parameters.

    Parameters
    ----------
    path : str
        Path to the FITS RIMO file.
    comm : MPI communicator, optional
        MPI communicator with attributes `rank`, and methods `Barrier()` and
        `bcast(obj, root=0)`. If provided, the RIMO dictionary is only
        loaded on rank 0 and then broadcast to all ranks.

    Returns
    -------
    dict
        Mapping from detector name (str) to a :class:`DetectorData` instance.
    """
    if comm is not None:
        comm.Barrier()



    RIMO: Dict[str, DetectorData] = {}
    is_root = comm is None or getattr(comm, "rank", 0) == 0

    if is_root:
        print(f"Loading RIMO from {path}", flush=True)
        hdulist = fits.open(path, "readonly")

        data = hdulist[1].data
        detectors = data.field("detector").ravel()
        phi_uvs = data.field("phi_uv").ravel()
        theta_uvs = data.field("theta_uv").ravel()
        psi_uvs = data.field("psi_uv").ravel()
        psi_pols = data.field("psi_pol").ravel()
        epsilons = data.field("epsilon").ravel()
        fsamples = data.field("f_samp").ravel()
        fknees = data.field("f_knee").ravel()
        alphas = data.field("alpha").ravel()
        nets = data.field("net").ravel()
        fwhms = data.field("fwhm").ravel()

        for i in range(len(detectors)):
            phi = phi_uvs[i] * degree
            theta = theta_uvs[i] * degree
            # Make sure we don't double-count psi rotation already included in phi
            psi = (psi_uvs[i] + psi_pols[i]) * degree - phi


            RIMO[detectors[i]] = DetectorData(
                name=detectors[i],
                phi_uv=phi_uvs[i],
                theta_uv=theta_uvs[i],
                psi_uv=psi_uvs[i],
                psi_pol=psi_pols[i],
                epsilon=epsilons[i],
                fsample=fsamples[i],
                fknee=fknees[i],
                alpha=alphas[i],
                net=nets[i],
                fwhm=fwhms[i],
            )

        hdulist.close()

    if comm is not None:
        # Broadcast to all ranks
        RIMO = comm.bcast(RIMO, root=0)


    return RIMO


def list_planck(
    detset: Union[int, str],
    good: bool = True,
    subset: Optional[int] = None,
    extend_857: bool = True,
    extend_545: bool = False,
) -> Union[List[str], int]:
    """
    Return lists of Planck detectors / horns for common detector sets.

    The logic follows the usual Planck conventions for LFI and HFI:

    Examples
    --------
    >>> list_planck(100)              # all 100 GHz detectors
    >>> list_planck("143A")           # subset 'A' at 143 GHz
    >>> list_planck("LFI")            # all LFI detectors
    >>> list_planck("PLANCK")         # all Planck detectors

    Special values
    --------------
    detset = "ROWS"
        Returns a list of lists that group detectors into rows, useful for
        plots or tables.
    detset = "LFI", "HFI", "PLANCK"
        Returns lists of all detectors in the respective instrument(s).

    Parameters
    ----------
    detset : int or str
        Detector set identifier, e.g. 30, "100", "143A", "LFI", "HFI", "PLANCK".
    good : bool, optional
        If True, only include "good" detectors where that distinction exists.
    subset : int, optional
        For some frequencies, can select specific subsets (1, 2, or 3)
        following Planck HFI conventions.
    extend_857 : bool, optional
        If True, at 857 GHz include extended set when `good` is False.
    extend_545 : bool, optional
        If True, at 545 GHz include extended set even if `good` is True.

    Returns
    -------
    list[str] or int
        A list of detector names, or `-1` if the detector set is unknown.
    """
    detectors: List[str] = []
    if subset is None:
        subset = 0

    # --- LFI channel selections ------------------------------------------------
    if detset in (30, "30", "030", "30GHz", "030GHz", "30A", "030A", "30B", "030B"):
        horns = range(27, 29)
        instrument = "LFI"
    elif detset in (44, "44", "044", "44GHz", "044GHz", "44A", "044A", "44B", "044B"):
        horns = range(24, 27)
        instrument = "LFI"
    elif detset in (70, "70", "070", "70GHz", "070GHz"):
        horns = range(18, 24)
        # subset=1
        if subset == 1:
            horns = [18, 23]
        elif subset == 2:
            horns = [19, 22]
        elif subset == 3:
            horns = [20, 21]
        elif subset == 4:
            horns = [20]
        instrument = "LFI"
    elif detset in ["70A", "070A"]:
        horns = [18, 20, 23]
        instrument = "LFI"
    elif detset in ["70B", "070B"]:
        horns = [19, 21, 22]
        instrument = "LFI"
    elif isinstance(detset, str) and detset.upper() == "LFI":
        detectors.extend(list_planck(30, good=good))
        detectors.extend(list_planck(44, good=good))
        detectors.extend(list_planck(70, good=good))
        return detectors

    # --- HFI channel selections ------------------------------------------------
    elif detset in (100, "100", "100GHz"):
        psb_horns = range(1, 5)
        swb_horns: Sequence[int] = []
        if subset == 1:
            psb_horns = [1, 4]
        elif subset == 2:
            psb_horns = [2, 3]
        instrument = "HFI"
        freq = "100-"
    elif detset == "100A":
        psb_horns = [1, 4]
        swb_horns: Sequence[int] = []
        instrument = "HFI"
        freq = "100-"

    elif detset == "100B":
        psb_horns = [2, 3]
        swb_horns = []
        instrument = "HFI"
        freq = "100-"
    elif detset == "100C":
        psb_horns = [1, 2]
        swb_horns = []
        instrument = "HFI"
        freq = "100-"
    elif detset == "100D":
        psb_horns = [1]
        swb_horns = []
        instrument = "HFI"
        freq = "100-"
    elif detset in (143, "143", "143GHz"):
        psb_horns = np.arange(1, 5)
        if good:
            swb_horns = range(5, 8)
        else:
            swb_horns = range(5, 9)
        if subset == 1:
            psb_horns, swb_horns = [1, 3], []
        elif subset == 2:
            psb_horns, swb_horns = [2, 4], []
        elif subset == 3:
            psb_horns, swb_horns = [], [5, 6, 7]
        instrument = "HFI"
        freq = "143-"
    elif detset == "143A":
        psb_horns = [1, 3]
        swb_horns = [5, 7]
        instrument = "HFI"
        freq = "143-"
    elif detset == "143B":
        psb_horns = [2, 4]
        swb_horns = [6]
        instrument = "HFI"
        freq = "143-"
    elif detset in (217, "217", "217GHz"):
        psb_horns = np.arange(5, 9)
        swb_horns = np.arange(1, 5)
        if subset == 1:
            psb_horns, swb_horns = [5, 7], []
        elif subset == 2:
            psb_horns, swb_horns = [6, 8], []
        elif subset == 3:
            psb_horns, swb_horns = [], [1, 2, 3, 4]
        instrument = "HFI"
        freq = "217-"
    elif detset == "217A":
        psb_horns = [5, 7]
        swb_horns = [1, 3]
        instrument = "HFI"
        freq = "217-"
    elif detset == "217B":
        psb_horns = [6, 8]
        swb_horns = [2, 4]
        instrument = "HFI"
        freq = "217-"
    elif detset in (353, "353", "353GHz"):
        psb_horns = np.arange(3, 7)
        swb_horns = [1, 2, 7, 8]
        if subset == 1:
            psb_horns, swb_horns = [3, 5], []
        elif subset == 2:
            psb_horns, swb_horns = [4, 6], []
        elif subset == 3:
            psb_horns, swb_horns = [], [1, 2, 7, 8]
        instrument = "HFI"
        freq = "353-"
    elif detset == "353A":
        psb_horns = [3, 5]
        swb_horns = [1, 7]
        instrument = "HFI"
        freq = "353-"
    elif detset == "353B":
        psb_horns = [4, 6]
        swb_horns = [2, 8]
        instrument = "HFI"
        freq = "353-"
    elif detset in (545, "545", "545GHz"):
        psb_horns = []
        if good and not extend_545:
            swb_horns = [1, 2, 4]
        else:
            swb_horns = np.arange(1, 5)
        instrument = "HFI"
        freq = "545-"
    elif detset == "545A":
        psb_horns = []
        swb_horns = [1]
        instrument = "HFI"
        freq = "545-"
    elif detset == "545B":
        psb_horns = []
        swb_horns = [2, 4]
        instrument = "HFI"
        freq = "545-"
    elif detset in (857, "857", "857GHz"):
        psb_horns = []
        if good and not extend_857:
            swb_horns = [1, 2, 3]
        else:
            swb_horns = np.arange(1, 5)
        instrument = "HFI"
        freq = "857-"
    elif detset == "857A":
        psb_horns = []
        swb_horns = [1, 3]
        instrument = "HFI"
        freq = "857-"
    elif detset == "857B":
        psb_horns = []
        swb_horns = [2, 4]
        instrument = "HFI"
        freq = "857-"
    elif isinstance(detset, str) and detset.upper() == "HFI":
        detectors.extend(list_planck(100, good=good, extend_857=extend_857))
        detectors.extend(list_planck(143, good=good, extend_857=extend_857))
        detectors.extend(list_planck(217, good=good, extend_857=extend_857))
        detectors.extend(list_planck(353, good=good, extend_857=extend_857))
        detectors.extend(list_planck(545, good=good, extend_857=extend_857))
        detectors.extend(list_planck(857, good=good, extend_857=extend_857))
        return detectors
    elif isinstance(detset, str) and detset.upper() == "PLANCK":
        detectors.extend(list_planck("LFI", good=good, extend_857=extend_857))
        detectors.extend(list_planck("HFI", good=good, extend_857=extend_857))
        return detectors
    elif isinstance(detset, str) and detset.upper() == "ROWS":
        # Row-grouping of detectors; kept as in the original code
        return [
            ["LFI27M", "LFI27S", "LFI28M", "LFI28S"],
            ["LFI24M", "LFI24S"],
            ["LFI25M", "LFI25S", "LFI26M", "LFI26S"],
            ["LFI18M", "LFI23S"],
            ["LFI19M", "LFI22S"],
            ["LFI20M", "LFI21S"],
            ["100-1a", "100-1b", "100-4a", "100-4b"],
            ["100-2a", "100-2b", "100-3a", "100-3b"],
            ["143-1a", "143-1b", "143-3a", "143-3b"],
            ["143-2a", "143-2b", "143-4a", "143-4b"],
            ["143-5", "143-7"],
            ["143-6"],
            ["217-1", "217-3"],
            ["217-2", "217-4"],
            ["217-5a", "217-5b", "217-7a", "217-7b"],
            ["217-6a", "217-6b", "217-8a", "217-8b"],
            ["353-1", "353-7"],
            ["353-3a", "353-3b", "353-5a", "353-5b"],
            ["353-4a", "353-4b", "353-6a", "353-6b"],
            ["353-2", "353-8"],
            ["545-1"],
            ["545-2", "545-4"],
            ["857-1", "857-3"],
            ["857-2"],
        ]
    else:
        # Single detectors and horns
        lfidets = list_planck("LFI")
        hfidets = list_planck("HFI")
        if detset in lfidets or detset in hfidets:
            return [detset]  # type: ignore[list-item]
        if isinstance(detset, str) and detset + "M" in lfidets:
            return [detset + "M", detset + "S"]  # type: ignore[return-value]
        if isinstance(detset, str) and detset + "a" in hfidets:
            return [detset + "a", detset + "b"]  # type: ignore[return-value]
        # All other cases
        print("ERROR: unknown detector set: ", detset)
        return -1

    # Build detector name list for LFI / HFI cases above
    if instrument == "LFI":
        for horn in horns:
            for arm in ["S", "M"]:
                detectors.append("LFI" + str(horn) + arm)
    elif instrument == "HFI":
        for horn in psb_horns:
            for arm in ["a", "b"]:
                detectors.append(freq + str(horn) + arm)
        for horn in swb_horns:
            detectors.append(freq + str(horn))

    return detectors


def get_blms_fits(fitsfile, lmax=None, mmax=None, isbalm=True, renorm=True):
    """Read beam multipoles B_lm from a FITS file."""
    data = fits.getdata(fitsfile)
    Tix = data.field(0)
    Tre = data.field(1)
    Tim = data.field(2)
    polbeam_in = False
    ndb = 1
    try:
        dataG = fits.getdata(fitsfile, 2)
        Gre = dataG.field(1)
        Gim = dataG.field(2)
        dataC = fits.getdata(fitsfile, 3)
        Cre = dataC.field(1)
        Cim = dataC.field(2)
        polbeam_in = True
        ndb = 3
    except Exception:
        print(
            "#ff0000  ",
            "WARNING: Polarized Blm not found in %s" % (fitsfile),
            "\x1b[0m",
            flush=True,
        )
    ls = np.array(np.floor(np.sqrt(Tix - 1)), dtype=np.int64)
    ms = Tix - ls * ls - ls - 1
    print(
        f"maximum l in file: {np.max(ls)}, maximum m in file: {np.max(ms)}", flush=True
    )
    if lmax is None:
        lmax = np.max(ls)
    if mmax is None:
        mmax = np.max(ms)
    idxs = (ls <= lmax) * (ms <= mmax)
    ret = np.zeros((lmax + 1, mmax + 1, ndb), dtype=np.complex128)
    ret[ls[idxs], ms[idxs], 0] = Tre[idxs] + 1j * Tim[idxs]
    if polbeam_in:
        ret[ls[idxs], ms[idxs], 1] = -(
            (Gre[idxs] - Cim[idxs]) + 1j * (Gim[idxs] + Cre[idxs])
        )
        ret[ls[idxs], ms[idxs], 2] = -(
            (Gre[idxs] + Cim[idxs]) + 1j * (Gim[idxs] - Cre[idxs])
        )
    if renorm:
        print("Renormalizing the beam")
        ret /= ret[0, 0, 0]
    if isbalm:
        print("Converting from balm")
        # file is balm, so renormalize and scale.
        for l in range(lmax + 1):
            for kb in range(ndb):
                ret[l, :, kb] *= 1 / (1 * np.sqrt((2 * l + 1)))
    return ret
# ----------------------------------------------------------------------
# Detector weights
# ----------------------------------------------------------------------
#: Map-level detector weights (for example from white-noise levels).
detector_weights: Dict[str, float] = {
    # 30 GHz
    "LFI27": 0.40164e06,
    "LFI28": 0.36900e06,
    # 44 GHz
    "LFI24": 0.12372e06,
    "LFI25": 0.14049e06,
    "LFI26": 0.11233e06,
    # 70 GHz
    "LFI18": 53650.0,
    "LFI19": 42141.0,
    "LFI20": 36579.0,
    "LFI21": 50355.0,
    "LFI22": 49363.0,
    "LFI23": 47966.0,
    # 100 GHz
    "100-1": 0.76343e06,
    "100-2": 0.12661e07,
    "100-3": 0.10631e07,
    "100-4": 0.10532e07,
    # 143 GHz
    "143-1": 0.16407e07,
    "143-2": 0.18577e07,
    "143-3": 0.16439e07,
    "143-4": 0.14458e07,
    "143-5": 0.27630e07,
    "143-6": 0.26942e07,
    "143-7": 0.28599e07,
    # 217 GHz
    "217-1": 0.11058e07,
    "217-2": 0.10261e07,
    "217-3": 0.10958e07,
    "217-4": 0.10593e07,
    "217-5": 0.67318e06,
    "217-6": 0.71092e06,
    "217-7": 0.76576e06,
    "217-8": 0.71226e06,
    # 353 GHz
    "353-1": 0.12829e06,
    "353-2": 0.13475e06,
    "353-3": 48067.0,
    "353-4": 42187.0,
    "353-5": 56914.0,
    "353-6": 25293.0,
    "353-7": 87730.0,
    "353-8": 74453.0,
    # 545 GHz
    "545-1": 4475.5,
    "545-2": 5540.3,
    "545-4": 4321.0,
    # 857 GHz
    "857-1": 6.8895,
    "857-2": 6.3108,
    "857-3": 6.5964,
    "857-4": 3.6785,
}
