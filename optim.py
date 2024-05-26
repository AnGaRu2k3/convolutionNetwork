import math
from torch.optim import Optimizer

class CosineSchedule:
    def __init__(self, optimizer: Optimizer, base_lr: float, target_lr: float, max_steps: int, warmup_steps: int = 0):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.target_lr = target_lr
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self):
        if self.step_num < self.warmup_steps:
            return self.base_lr * (self.step_num / self.warmup_steps)
        else:
            progress = (self.step_num - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            return self.target_lr + 0.5 * (self.base_lr - self.target_lr) * (1 + math.cos(math.pi * progress))
