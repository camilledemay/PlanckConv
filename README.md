This code enables efficient simulation of beam-convolved Planck maps from arbitrary input skies,it takes the following inputs:

- harmonic coefficients (`alms`)
- Planck beam harmonic coefficients (`blms`) 
- Planck polmoments

The required data files can be downloaded here: 

This is a wrapper around the Smarties package ([https://github.com/simonsobs/smarties](https://github.com/simonsobs/smarties)), the underlying formalism is described here [] and here [].

For practical usage, you can refer to the Jupyter notebook.

You may want to compile ducc from source to speed-up the map2alm transforms, see  https://gitlab.mpcdf.mpg.de/mtr/ducc
