from abc import ABC, abstractmethod


class AbstractMeasureStorage(ABC):
    """
    Abstract interface for handling the storage of computed measures. They can be written to file or simply logged.
    The subclasses implementing this interface should be used in combination with MeasureCallback.
    """

    @abstractmethod
    def store_on_epoch_end(self):
        """Called with single measured value for each measure at the end of each epoch."""
        ...

    @abstractmethod
    def store_on_train_end(self):
        """Called with single measured value for each measure at the end of training."""
        ...

    def write(self):
        """
        Write stored measures to file or log. This function can be empty for storage methods writing after
        each call, for example logging or updating storage file after each epoch to save memory for large measures.
        """
        ...
