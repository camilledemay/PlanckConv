# Smarties4Planck

Efficient simulation of beam-convolved Planck maps from arbitrary input skies.

Smarties4Planck is a high-level wrapper around the
[Smarties](https://github.com/simonsobs/smarties) package. It reads Planck beams
and polmoments (aka h-maps), converts them into the format
and conventions used by Smarties, and calls Smarties to perform the beam
convolution.

The underlying formalism is described in [Paper I]() and [Paper II]()
(links TBD). Practical usage is demonstrated in the
[example notebook](example_notebook.ipynb.ipynb).

## Inputs

The code takes the following inputs:

- `alms`, spherical harmonic coefficients of the input sky.
- `blms`, Planck beam harmonic coefficients.
- Planck polmoments.

The polmoments and `blms` are available in NERSC: /global/cfs/cdirs/cmb/data/planck2020/npipe/aux/beams and /aux/polmoments respectively.

# Installation

To install this package, clone this repository and run 
```
pip install path\to\repo\dir
``` 
Alternatively, if you use uv you can run 
```
uv add "smarties4planck @ git+https://github.com/camilledemay/smarties4planck"
```

You may want to compile ducc from source to speed-up the map2alm transforms, see  https://gitlab.mpcdf.mpg.de/mtr/ducc
