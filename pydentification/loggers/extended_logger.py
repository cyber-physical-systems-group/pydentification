import logging
from typing import Literal, Optional

SUCCESS = 21  # logging.INFO + 1
STOP = 31  # logging.WARNING + 1
FAIL = 41  # logging.ERROR + 1
NOTIFY = 19  # logging.INFO - 1


class ExtendedLogger(logging.getLoggerClass()):
    """
    Custom logger supporting additional logging levels:
        * SUCCESS
        * STOP
        * FAIL
        * NOTIFY

    For most use cases this is an overkill, but logging metrics during long training runs can be useful, especially
    when running multiple trainings in parallel, it makes catching the eye on the most important information easier.
    """

    COLOR_FORMATTING = {
        "red": "\033[91m{message}\033[0m",
        "green": "\033[92m{message}\033[0m",
        "yellow": "\033[93m{message}\033[0m",
        "blue": "\033[94m{message}\033[0m",
        "purple": "\033[95m{message}\033[0m",
        "cyan": "\033[96m{message}\033[0m",
    }

    def __init__(self, name, level: int = logging.NOTSET):
        super().__init__(name, level)

        logging.addLevelName(SUCCESS, "SUCCESS")
        logging.addLevelName(STOP, "STOP")
        logging.addLevelName(FAIL, "FAIL")
        logging.addLevelName(NOTIFY, "NOTIFY")

    def _format_header(
        self, header: Optional[str], color: Literal["red", "green", "yellow", "blue", "purple", "cyan"]
    ) -> str:
        """Create header with color formatting for given logging level"""
        if header is not None:
            header = self.COLOR_FORMATTING[color].format(message=header)
        else:
            header = ""
        return header

    def success(self, msg: str, header: Optional[str] = None, *args, **kwargs) -> None:
        if self.isEnabledFor(SUCCESS):
            header = self._format_header(header, "green")
            self._log(SUCCESS, header + " " + msg, args, **kwargs)

    def stop(self, msg: str, header: Optional[str] = None, *args, **kwargs) -> None:
        if self.isEnabledFor(STOP):
            header = self._format_header(header, "yellow")
            self._log(SUCCESS, header + " " + msg, args, **kwargs)

    def fail(self, msg: str, header: Optional[str] = None, *args, **kwargs) -> None:
        if self.isEnabledFor(FAIL):
            header = self._format_header(header, "red")
            self._log(SUCCESS, header + " " + msg, args, **kwargs)

    def notify(self, msg: str, header: Optional[str] = None, *args, **kwargs) -> None:
        if self.isEnabledFor(NOTIFY):
            header = self._format_header(header, "purple")
            self._log(NOTIFY, header + " " + msg, args, **kwargs)
