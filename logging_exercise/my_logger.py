import logging 
from logging.config import dictConfig
import sys 
from pathlib import Path
from rich.logging import RichHandler

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)


logging_config = {
    "version": 1,
    "formatters": { # 
        "minimal": {"format": "%(message)s"},
        "detailed": {
            "format": "%(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
        },
    },
    "handlers": { # 
        "console": {
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "formatter": "minimal",
            "level": logging.DEBUG,
        },
        "info": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "info.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.INFO,
        },
        "error": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "error.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.ERROR,
        },
    },
    "root": {
        "handlers": ["console", "info", "error"],
        "level": logging.DEBUG,
        "propagate": True,
    },
}


# Create super basic logger
# basicConfig is a method to configure the logging system
# stream specifies where the log messages should be output, here: standard output stream (console)
# level: only level DEBUG or higher will be logged
# LEVELS: OFF -> CRITICAL -> ERROR -> WARNING -> INFO -> DEBUG 
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

dictConfig(logging_config)

# The built-in variable __name__ always contains the record of the script or module that is currently being run. 
# Therefore if we initialize our logger base using this variable, it will always be unique 
# to our application and not conflict with logger setup by any third-party package.
logger = logging.getLogger(__name__)

logger.root.handlers[0] = RichHandler(markup=True)

# Logging levels (from lowest to highest priority)
logger.debug("Used for debugging your code.")
logger.info("Informative messages from your code.")
logger.warning("Everything works but there is something to be aware of.")
logger.error("There's been a mistake with the process.")
logger.critical("There is something terribly wrong and process may terminate.")