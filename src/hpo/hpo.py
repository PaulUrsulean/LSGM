import skopt
import torch
from skopt.space import Real, Integer, Categorical
from skopt.optimizer import Optimizer


class HPO:

    def __init__(self,
                 objective_function,
                 search_space,
                 n_iters=10,
                 gpu_ids=[],
                 n_initial_points=10):
        """

        :param objective_function: Function that takes in device (gpu) and parameters and computes objective
        :param search_space:
        :param n_iters:
        :param gpu_ids:
        :param n_initial_points:
        """

        self.objective_function = objective_function
        self.space = self._parameters_to_space(search_space)
        self.n_iters = n_iters

        if len(gpu_ids) == 0:  # No GPU selected manually TODO: Choose randomly then
            self.n_gpus = 1
            self.devices = [None]
        else:
            self.n_gpus = len(gpu_ids)
            self.devices = gpu_ids

        self.optimizer = Optimizer(
            dimensions=self.space,
            n_initial_points=n_initial_points,
            random_state=42
        )

    def _parameters_to_space(self, parameters):
        return parameters

    def _eval_config(self, parameter_sets):
        assert(len(parameter_sets) == self.n_gpus)
        # TODO: Make use of parallelization if not already happening
        return [self.objective_function(self.devices[i], *params) for i, params in enumerate(parameter_sets)]

    def tune(self, return_best_n=1):

        for curr_iter in range(self.n_iters):
            new_points = self.optimizer.ask(self.n_gpus)
            res = self._eval_config(new_points)
            self.optimizer.tell(new_points, res)

        # Return best configuration
        return sorted(list(zip(self.optimizer.Xi, self.optimizer.yi)), key=lambda x:x[1])[:return_best_n]
