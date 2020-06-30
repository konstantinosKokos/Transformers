from math import cos, radians
from typing import *

from torch.optim import Optimizer


def make_cosine_schedule(max_lr: float, warmup_steps: int, decay_over: int) -> Callable[[int], float]:
    """
        Makes a schedule that increases the lr from 0 to max_lr over warmup_steps,
        then reduces it to 0 over decay_over steps.
    """
    linear_factor = max_lr / warmup_steps
    cos_window = _cosine_window(max_lr, warmup_steps, decay_over)

    def cosine_schedule(step: int) -> float:
        if step < warmup_steps:
            return linear_factor * step
        else:
            return cos_window(step)

    return cosine_schedule


def _cosine_window(max_lr: float, offset: int, decay_over: int) -> Callable[[int], float]:
    f = 90 / decay_over
    b = - f * offset

    def schedule(step: int) -> float:
        angle = f * step + b
        return cos(radians(angle)) * max_lr

    return schedule


def make_cosine_schedule_with_restarts(max_lr: float, warmup_steps: int, restart_every: int,
                                       decay_over: int, restart_type: Union[Literal['linear'], Literal['cosine']])  \
        -> Callable[[int], float]:
    linear_factor = max_lr / warmup_steps
    if restart_type not in {'linear', 'cosine'}:
        raise TypeError(f'Restart type can be either linear or cosine.')
    envelope_fn = _cosine_window if restart_type == 'cosine' else _linear_envelope
    envelope = envelope_fn(max_lr=max_lr, offset=warmup_steps, decay_over=decay_over)

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return linear_factor * step
        outer_factor = envelope(step)
        current_restart = (step - warmup_steps) // restart_every
        offset = warmup_steps + current_restart * restart_every
        inner_cos_fn = _cosine_window(max_lr=outer_factor, offset=offset, decay_over=restart_every)
        return inner_cos_fn(step)

    return schedule


def _linear_envelope(max_lr: float, offset: int, decay_over: int) -> Callable[[int], float]:
    def schedule(step: int) -> float:
        return (decay_over - step) * max_lr / (decay_over - offset)
    return schedule


def make_noam_scheme(d_model: int, warmup_steps: int, factor: float) -> Callable[[int], float]:
    def noam_scheme(step: int) -> float:
        step += 1
        return d_model**-0.5 * min(step**-0.5, step*warmup_steps**-1.5) * factor
    return noam_scheme


class Scheduler(object):
    def __init__(self, opt: Optimizer, schedule: Callable[[int], float], param_group_scales: Sequence[float] = (1,)):
        self.opt = opt
        self.schedule = schedule
        self.step_num = 0
        self.lr = 0.
        self.param_group_scales = param_group_scales

    def step(self) -> None:
        self.step_num += 1
        self.lr = self.schedule(self.step_num)
        for i, p in enumerate(self.opt.param_groups):
            p['lr'] = self.lr * self.param_group_scales[i]
        self.opt.step()

    def zero_grad(self) -> None:
        self.opt.zero_grad()
