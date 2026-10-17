class EarlyStopping:
    """Early stopping class that stops training when a specified number of epochs have passed without improvement.

    Attributes:
        best_fitness (float): Best fitness value observed.
        best_epoch (int): Epoch where best fitness was observed.
        patience (int): Number of epochs to wait after fitness stops improving before stopping.
        possible_stop (bool): Flag indicating if stopping may occur next epoch.
    """

    def __init__(self, patience=50):
        """Initialize early stopping object.

        Args:
            patience (int, optional): Number of epochs to wait after fitness stops improving before stopping. 0 or None
                disables early stopping. The trainer always passes the cfg `patience` value (100 by default).
        """
        self.best_fitness = 0.0  # i.e. mAP
        self.best_epoch = 0
        self.patience = patience or float("inf")  # epochs to wait after fitness stops improving to stop
        self.possible_stop = False  # possible stop may occur next epoch

    def __call__(self, epoch, fitness):
        """Check whether to stop training.

        Args:
            epoch (int): Current epoch of training.
            fitness (float): Fitness value of current epoch.

        Returns:
            (bool): True if training should stop, False otherwise.
        """
        if fitness is None:  # check if fitness=None (happens when val=False)
            return False

        if fitness > self.best_fitness or self.best_fitness == 0:  # allow for early zero-fitness stage of training
            self.best_epoch = epoch
            self.best_fitness = fitness
        delta = epoch - self.best_epoch  # epochs without improvement
        self.possible_stop = delta >= (self.patience - 1)  # possible stop may occur next epoch
        stop = delta >= self.patience  # stop training if patience exceeded
        if stop:
            prefix = colorstr("EarlyStopping: ")
            LOGGER.info(
                f"{prefix}Training stopped early as no improvement observed in last {self.patience} epochs. "
                f"Best results observed at epoch {self.best_epoch}, best model saved as best.pt.\n"
                f"To update EarlyStopping(patience={self.patience}) pass a new patience value, "
                f"i.e. `patience=300` or use `patience=0` to disable EarlyStopping."
            )
        return stop
