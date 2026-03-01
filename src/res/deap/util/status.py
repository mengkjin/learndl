class gpStatus:
    def __init__(self , n_iter : int , n_gen : int , start_iter : int = 0 , start_gen : int = 0 , **kwargs) -> None:
        self.n_iter      = n_iter
        self.n_gen       = n_gen

        self.start_iter  = start_iter
        self.start_gen   = start_gen

        self.forbidden  : list[str] = []

    def iter_iteration(self):
        for self._i_iter in range(self.start_iter, self.n_iter):
            yield self._i_iter

    def iter_generation(self):
        if self._i_iter == self.start_iter:
            for self._i_gen in range(self.start_gen, self.n_gen):
                yield self._i_gen
        else:
            for self._i_gen in range(0, self.n_gen):
                yield self._i_gen

    @property
    def i_iter(self) -> int:
        if not hasattr(self , '_i_iter'):
            self._i_iter = self.start_iter
        return self._i_iter
    @property
    def i_gen(self) -> int:
        if not hasattr(self , '_i_gen'):
            self._i_gen = self.iter_start_gen
        return self._i_gen
    @property
    def iter_start_gen(self) -> int:
        return self.start_gen if self._i_iter == self.start_iter else 0

    