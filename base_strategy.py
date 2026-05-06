class BaseStrategy:
    def __init__(self, side):
        self.side = side

    def on_tick(self, row, current_time) -> bool:
        raise NotImplementedError
