import math
from torch.optim.lr_scheduler import LambdaLR
from typing import List


def burn_in(current_it, max_iters, power=4):
    if max_iters > 0:
        return math.pow(current_it / max_iters, power)
    else:
        return 1.0


class steps(LambdaLR):
    def __init__(self, optimizer, steps: List[int], scales: List[float],
                 burnin_iters=0, burnin_power=4, last_epoch=-1, verbose=False):

        # Scheduler implementation
        class _steps_impl(object):
            def __init__(self, steps, scales, burnin_iters, burnin_power):

                self.steps = steps
                self.scales = scales
                assert len(self.steps) == len(self.scales), \
                    'Length of steps and scales must be same.'

                self.burninIters = burnin_iters
                self.burninPower = burnin_power

            def __call__(self, it):
                it += 1
                if it <= self.burninIters:
                    return burn_in(it, self.burninIters, self.burninPower)

                factor = 1.0
                for step, scale in zip(self.steps, self.scales):
                    if it > step:
                        factor *= scale

                return factor

        # Construct parent class
        super().__init__(optimizer,
                         lr_lambda=_steps_impl(
                             steps, scales, burnin_iters, burnin_power),
                         last_epoch=last_epoch, verbose=verbose)
