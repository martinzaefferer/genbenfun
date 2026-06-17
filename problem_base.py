import ioh
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF
from typing import Callable
from scipy.stats import qmc

class Problem():
    def __init__(self, problem_id: int, instance_id: int, dimension: int,
                 amplitude: float = 1.0, frequency: float = 0.0, mode: str = "sin_f"):
        self.problem_id = problem_id
        self.instance_id = instance_id
        self.dimension = dimension
        self.amplitude = amplitude
        self.frequency = frequency
        self.mode = mode
        self._y_internal = np.nan
        self.problem = ioh.get_problem(problem_id, instance_id, dimension)

    def apply_modification(self, x, y_internal):
        if self.mode == "sin_f":
            arg = 2*np.pi*(10**self.frequency)*(y_internal)
            return y_internal + (10**self.amplitude) * (np.sin(arg)**2)
        elif self.mode == "sin_x":
            r =  np.linalg.norm(x - self.problem.optimum.x)
            arg = 2*np.pi*(10**self.frequency)*r
            return y_internal + (10**self.amplitude) * (1-np.cos(arg))
        return y_internal

    def evaluate_internal(self, x):
        y_internal = self.problem(x) - self.problem.optimum.y
        self._y_internal = y_internal
        return self.apply_modification(x, y_internal)

    def get_problem(self):
        return ioh.wrap_problem(self.evaluate_internal,
                                f"F{self.problem_id}_instance{self.instance_id}_dim{self.dimension}_{self.mode}_amp{self.amplitude}_freq{self.frequency}",
                                ioh.ProblemClass.REAL, self.dimension, 1, lb=-5, ub=5)

    @property
    def y_internal(self):
        return self._y_internal

class SurrogateProblem(Problem):
    def __init__(self, problem_id: int, instance_id: int, dimension: int,
                 amplitude: float = 1.0, frequency: float = 0.0, mode: str = "sin_f"):
        super().__init__(problem_id, instance_id, dimension, amplitude, frequency, mode)
        self.surrogate_model: GaussianProcessRegressor | None = None
        self._y_modified_internal : float = np.nan
        self.evaluate_surrogate : Callable | None = None

    def get_samples(self, n_samples: int, method: str = "random"):
        if method == "random":
            samples = np.random.uniform(-5, 5, (n_samples, self.dimension))
        elif method == "halton":
            sampler = qmc.Halton(d=self.dimension, scramble=True)
            samples = sampler.random(n_samples) * 10 - 5
        else:
            raise ValueError(f"Unknown sampling method: {method}")

        y_samples = np.array([self.apply_modification(x, self.problem(x) - self.problem.optimum.y) for x in samples])
        return samples, y_samples
    
    def create_surrogate(self, kernel_type: str = "matern52", n_samples: int = 100,
                         method: str = "halton", normalize_y: bool = True,
                         noise_level: float = 1e-6):
        x_train, y_train_norm = self.get_samples(n_samples=n_samples, method=method)

        if kernel_type == "rbf":
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
                length_scale=np.ones(self.dimension),
                length_scale_bounds=(1e-2, 1e2),
            )
        elif kernel_type == "matern52":
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
                length_scale=np.ones(self.dimension),
                length_scale_bounds=(1e-2, 1e2),
                nu=2.5,
            )
        else:
            raise ValueError(f"Unknown kernel: {kernel_type}")

        surrogate_model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=max(float(noise_level), 1e-12),
            normalize_y=normalize_y,
            n_restarts_optimizer=2,
            random_state=0,
        )
        surrogate_model.fit(x_train, y_train_norm)
        self.surrogate_model = surrogate_model

        def predict_mean(x):
            x_arr = np.asarray(x, dtype=float).reshape(1, -1)
            pred_mean = float(surrogate_model.predict(x_arr, return_std=False)[0])
            return pred_mean

        self.evaluate_surrogate = predict_mean

    def evaluate_internal(self, x):
        y_internal = self.problem(x) - self.problem.optimum.y
        self._y_internal = y_internal
        self._y_modified_internal =  self.apply_modification(x, y_internal)
        return self.evaluate_surrogate(x)

    def get_problem(self, version = "surrogate"):
        if self.evaluate_surrogate is None:
            raise ValueError("Surrogate model has not been created. Call create_surrogate() first.")
        if version == "surrogate":
            return ioh.wrap_problem(self.evaluate_internal, 
                                f"Surrogate_F{self.problem_id}_instance{self.instance_id}_dim{self.dimension}_{self.mode}_amp{self.amplitude}_freq{self.frequency}",
                                ioh.ProblemClass.REAL, self.dimension, 1, lb=-5, ub=5)
        elif version == "modified":
            return ioh.wrap_problem(self.evaluate_internal, 
                                f"Modified_F{self.problem_id}_instance{self.instance_id}_dim{self.dimension}_{self.mode}_amp{self.amplitude}_freq{self.frequency}",
                                ioh.ProblemClass.REAL, self.dimension, 1, lb=-5, ub=5)
        elif version == "base": 
            return ioh.get_problem(self.problem_id, self.instance_id, self.dimension)
        else:
            raise ValueError("Unknown option for 'version'. Supported: 'surrogate', 'modified', 'base'.")
            
    @property
    def y_modified_internal(self):
        return self._y_modified_internal
    
    def evaluate_model_quality(self, n_samples: int = 100, method: str = "halton"):
        if self.evaluate_surrogate is None:
            raise ValueError("Surrogate model has not been created. Call create_surrogate() first.")

        x_test, y_test_true = self.get_samples(n_samples=n_samples, method=method)
        y_test_pred = np.array([self.evaluate_surrogate(x) for x in x_test])
        y_test_disturbed = np.array([self.apply_modification(x, y) for x, y in zip(x_test, y_test_true)])

        rmse_true = float(np.sqrt(np.mean((y_test_pred - y_test_true) ** 2)))
        rmse_disturbed = float(np.sqrt(np.mean((y_test_pred - y_test_disturbed) ** 2)))
        return rmse_true, rmse_disturbed