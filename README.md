# Smarties4Planck

Efficient simulation of beam-convolved Planck maps from arbitrary input skies.

Smarties4Planck is a high-level wrapper around the
[Smarties](https://github.com/simonsobs/smarties) package. It reads Planck beams
and polmoments (aka h-maps), converts them into the format and conventions used
by Smarties, and calls Smarties to perform the beam convolution.

The underlying formalism is described in [Demay et al., 2026]() and
[Morshed et al., in prep.](). For practical usage you can refer to the
[example notebook](example_notebook.ipynb).

## Inputs

The code takes the following inputs:

- `alms` — spherical harmonic coefficients of the input sky.
- `blms` — Planck beam harmonic coefficients.
- Planck polmoments.

The polmoments and `blms` are available at NERSC:

- `blms`: `/global/cfs/cdirs/cmb/data/planck2020/npipe/aux/beams`
- polmoments: `/global/cfs/cdirs/cmb/data/planck2020/npipe/aux/polmoments`

## Installation

To install this package, clone this repository and run 
```bash
pip install path\to\repo\dir
``` 
Alternatively, if you use uv you can run 
```bash
uv add "smarties4planck @ git+https://github.com/camilledemay/smarties4planck"
```


This code relies on [DUCC](https://gitlab.mpcdf.mpg.de/mtr/ducc) to compute harmonic transforms. As detailed in DUCC documentation, for best performance, it is recommended to compile it from source.
