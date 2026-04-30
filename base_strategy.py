class BaseExecutionStrategy:
    def __init__(self, side):
        self.side = side

    def on_tick(self, current_row, current_time) -> bool:
        raise NotImplementedError
