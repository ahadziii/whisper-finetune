import sys
import os
from google.cloud import logging
from google.logging.type import log_severity_pb2 as severity
from src import config, credentials

class Log:
    def __init__(self, logger = None):
        self._logger = logger

    def info(self, message: str, **parameters) -> None:
        """Send informational message to Google Cloud Log.

        Example:

            from src import log

            # Send string to Google Cloud
            log.info('Pull message {id}',
                id=id,
            )

        """
        self._log(
            message=message.format(**parameters),
            severity=severity.INFO,
        );

    def error(self, message: str, **parameters) -> None:
        """Send error message to Google Cloud Log."""
        self._log(
            message=message.format(**parameters),
            severity=severity.ERROR,
        );

    def debug(self, message: str, **parameters) -> None:
        """Print debug message to console."""
        if sys.stdin.isatty():
            print(message.format(**parameters))

    def _log(self, message, severity) -> None:
        # Send formatted message to Google API
        self.logger.log_text(
            message,
            severity=severity,
        )

        # Also print message on console for local debugging
        if sys.stdin.isatty():
            print(message)

    @property
    def logger(self):
        # Initialize Google logger if we dont have one
        if not self._logger:
            client = logging.Client(
                project=config.get('project'),
                credentials=credentials.get(),
            )
            self._logger = client.logger(config.get('log'))

        # Return reference to a previously created logger instance
        return self._logger

# Export single instance
log = Log()
