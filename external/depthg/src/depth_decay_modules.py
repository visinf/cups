from typing import Optional, Union


class Decay:
    def __init__(self, init_value: Union[int, float], decay_rate: float, update_every: int,
                 min_value: Union[int, float]):
        self.init_value = init_value
        self.decay_rate = decay_rate
        self.update_every = update_every
        self.min_value = min_value
        self.return_type = type(init_value)

        assert self.decay_rate > 0, "Decay rate must be positive"
        assert type(self.init_value) == type(self.min_value), "Init value and min value must be of the same type"

    def calculate(self, step: int):
        raise NotImplementedError

    def return_update(self, step: int):
        step = step // self.update_every

        if step == 0:
            return self.init_value

        value = self.calculate(step)

        if type(value) != self.return_type:
            value = self.return_type(value)

        return value


class ExponentialDecay(Decay):
    def __init__(self, init_value: Union[int, float], decay_rate: float, update_every: int,
                 min_value: Union[int, float]):
        super().__init__(init_value, decay_rate, update_every, min_value)

    def calculate(self, step: int):
        assert type(step) == int, "Step must be an integer"

        return max(self.init_value * self.decay_rate ** step, self.min_value)


class LinearDecay(Decay):
    def __init__(self, init_value: Union[int, float], decay_rate: float, update_every: int,
                 min_value: Union[int, float]):
        super().__init__(init_value, decay_rate, update_every, min_value)

    def calculate(self, step: int):
        assert type(step) == int, "Step must be an integer"

        return max(self.init_value - step * self.decay_rate, self.min_value)


class StepDecay(Decay):
    pass


def get_depth_scheduler(version: str):
    if version == "exp":
        return ExponentialDecay
    elif version == "lin":
        return LinearDecay
    else :
        raise NotImplementedError

