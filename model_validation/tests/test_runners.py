"""
File contains classes used to run the correct tests and visualizations for a kernel.
"""

from abc import ABC, abstractmethod
import wandb

from .tests import (
    bootstrap_mean_test,
    norm_of_kernel_diff,
    mmd_two_sample_test,
    bootstrap_covariance_test,
)
from .visualizations import (
    plot_correlation,
    plot_vae_realizations,
    plot_kernel,
    plot_covariance,
)


class AbstractTestRunner(ABC):
    def __init__(self, kernel, grid):
        self.kernel = kernel
        self.kernel_matrix = kernel(grid, grid)
        self.kernel_name = kernel.__class__.__name__
        self.grid = grid

    @abstractmethod
    def run_tests(self, samples):
        pass

    @abstractmethod
    def run_visualizations(self, samples):
        pass


class SquaredExponentialTestRunner(AbstractTestRunner):
    def __init__(self, kernel, grid):
        super().__init__(kernel, grid)
        self.tests = [
            bootstrap_mean_test,
            norm_of_kernel_diff,
            mmd_two_sample_test,
            bootstrap_covariance_test,
        ]

        self.visualizations = [
            plot_covariance,
            plot_kernel,
            plot_vae_realizations,
            plot_correlation,
        ]

    def run_tests(self, samples, test_set):
        test_samples = test_set[1]
        for test in self.tests:
            result = test(
                samples=samples,
                kernel=self.kernel,
                target_samples=test_samples,
                grid=self.grid,
            )
            if wandb.run is not None:
                wandb.run.summary[test.__name__] = result
            else:
                print(f"{test.__name__}: {result}")

    def run_visualizations(self, samples, test_set):
        test_samples = test_set[1]
        for visualization in self.visualizations:
            visualization(
                samples=samples,
                kernel=self.kernel,
                grid=self.grid,
                kernel_name=self.kernel_name,
                gp_samples=test_samples,
            )


class MaternTestRunner(AbstractTestRunner):
    def __init__(self, kernel, grid):
        super().__init__(kernel, grid)

    def run_tests(self, samples):
        print("run matern tests")
        pass

    def run_visualizations(self, samples):
        pass
