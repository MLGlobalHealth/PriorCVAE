import pytest

from jax.config import config
config.update("jax_enable_x64", True)


@pytest.fixture(name="dimension", params=[1, 3])
def _dimension_fixture(request):
    return request.param


@pytest.fixture(name="num_data", params=[2, 5, 10])
def _num_data_fixture(request):
    return request.param


@pytest.fixture(name="boolean_variable", params=[True, False])
def _boolean_variable_fixture(request):
    return request.param
