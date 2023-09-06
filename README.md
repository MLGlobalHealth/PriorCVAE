# PriorCVAE in JAX

This repository is based on the following two papers:

1. Semenova, Elizaveta, et al. ["PriorVAE: encoding spatial priors with variational autoencoders for small-area estimation."](https://royalsocietypublishing.org/doi/full/10.1098/rsif.2022.0094) Journal of the Royal Society Interface 19.191 (2022): 20220094. Original code is avilable [here](https://github.com/elizavetasemenova/PriorVAE). 
2. Semenova, Elizaveta, Max Cairney-Leeming, and Seth Flaxman. ["PriorCVAE: scalable MCMC parameter inference with Bayesian deep generative modelling."](https://arxiv.org/abs/2304.04307) arXiv preprint arXiv:2304.04307 (2023). Original code is avilable [here](https://github.com/elizavetasemenova/PriorcVAE).

## Environment

We recommend setting up a [conda](https://docs.conda.io/projects/conda/en/latest/index.html) environment.
```shell
conda create -n prior_cvae -c conda-forge python==3.10.1
conda activate prior_cvae
```

Within the virtual environment, install the dependencies by running
```shell
pip install -r requirements.txt
```

**Note:** The code has been tested with `Python 3.10.1`. There is a known issue with `Python 3.10.0` related to loading a saved model  because of the [bug](https://bugs.python.org/issue45416) which is resolved in `Python 3.10.1`. 

## Install the package

```shell
python setup.py install
```

To install in the develop mode:
```shell
python setup.py develop
```


## Examples

Example notebooks can be found in the `examples\` directory. Remember to install the priorCVAE package before running the notebooks.

Sample command:
```shell
cd examples/
jupyter notebook GP-PriorCVAE.ipynb
```

**Note:** For experiments it is recommended to use float64 precision to avoid numerical instability:
```python
import jax.config as config
config.update("jax_enable_x64", True)
```

## To runs tests

First install the test-requirements by running the following command from within the conda environment:
```shell
pip install -r requirements-test.txt
```
Then, run the following command:
```shell
pytest -v tests/
```

### Projects using PriorVAE or PriorCVAE


| Project | Description | Publication | Uses current library |
| --- | --- | --- | --- |
| [aggVAE](https://github.com/MLGlobalHealth/aggVAE) | "Deep learning and MCMC with aggVAE for shifting administrative boundaries: mapping malaria prevalence in Kenya", Elizaveta Semenova, Swapnil Mishra, Samir Bhatt, Seth Flaxman, Juliette Unwin | [arvix](https://arxiv.org/pdf/2305.19779.pdf) Accepted to the "Epistemic Uncertainty in Artificial Intelligence" workshop of the "Uncertainty in Artificial Intelligence (UAI 2023)" conference.| no

### Contributing

For all correspondence, please contact [elizaveta.semenova@cs.ox.ac.uk](mailto:elizaveta.semenova@cs.ox.ac.uk).

### License

This software is provided under the [MIT license](LICENSE).
