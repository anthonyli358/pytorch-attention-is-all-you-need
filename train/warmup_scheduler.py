class WarmupScheduler:
    """
    Adam optimizer, learning rate scales with d_model.
    Overrites the learning rate.
    """

    def __init__(self, optimizer, d_model=512, warmup_steps=4000):
        self.optimizer = optimizer  # Adam
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self.d_model ** (-0.5) * min(
            self.step_num ** (-0.5), self.step_num * self.warmup_steps ** (-1.5)
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def state_dict(self):
        return {k: v for k, v in self.__dict__.items() if k != "optimizer"}

    def load_state_dict(self, state):
        self.__dict__.update(state)