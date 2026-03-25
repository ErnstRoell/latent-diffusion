"""
Example of structlog that sends nice formatted logs to the console and json to a file.
IMPORTANT: Save for later.
"""

import os
import logging.config
import structlog
import pathlib

timestamper = structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S")


class TypeFilter:
    def __init__(self, type_name: str):
        self.type_name = type_name

    def __call__(self, record):
        if isinstance(record.msg, str):
            return False
        if record.msg.get("type", None) != self.type_name:
            return False
        else:
            return True


pre_chain = [
    # Add the log level and a timestamp to the event_dict if the log entry
    # is not from structlog.
    structlog.stdlib.add_log_level,
    # Add extra attributes of LogRecord objects to the event dictionary
    # so that values passed in the extra parameter of log methods pass
    # through to log output.
    structlog.stdlib.ExtraAdder(),
    timestamper,
]


def extract_from_record(_, __, event_dict):
    """
    Extract thread and process names and add them to the event dict.
    """
    record = event_dict["_record"]
    event_dict["thread_name"] = record.threadName
    event_dict["process_name"] = record.processName
    return event_dict


def setup_loggers(result_folder: str, name: str, remove_logs: bool = False):
    metrics_path = pathlib.Path(f"{result_folder}/metrics_{name}.log")
    config_path = pathlib.Path(f"{result_folder}/configs_{name}.log")

    if remove_logs:
        if metrics_path.exists():
            os.remove(metrics_path)
        if config_path.exists():
            os.remove(config_path)

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "filters": {
                "config_filter": {
                    "()": "loggers.TypeFilter",
                    "type_name": "config",
                },
                "metric_filter": {
                    "()": "loggers.TypeFilter",
                    "type_name": "metric",
                },
            },
            "formatters": {
                "plain": {
                    "()": structlog.stdlib.ProcessorFormatter,
                    "processors": [
                        structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                        structlog.dev.ConsoleRenderer(colors=False),
                    ],
                    "foreign_pre_chain": pre_chain,
                },
                "colored": {
                    "()": structlog.stdlib.ProcessorFormatter,
                    "processors": [
                        extract_from_record,
                        structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                        structlog.dev.ConsoleRenderer(colors=True),
                    ],
                    "foreign_pre_chain": pre_chain,
                },
                "metrics": {
                    "()": structlog.stdlib.ProcessorFormatter,
                    "processors": [
                        extract_from_record,
                        structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                        structlog.processors.JSONRenderer(),
                    ],
                    "foreign_pre_chain": pre_chain,
                },
                "configs": {
                    "()": structlog.stdlib.ProcessorFormatter,
                    "processors": [
                        extract_from_record,
                        structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                        structlog.processors.JSONRenderer(),
                    ],
                    "foreign_pre_chain": pre_chain,
                },
            },
            "handlers": {
                "default": {
                    "level": "INFO",
                    "class": "logging.StreamHandler",
                    "formatter": "colored",
                },
                "metrics": {
                    "level": "DEBUG",
                    "class": "logging.handlers.WatchedFileHandler",
                    "filename": f"{metrics_path}",
                    "formatter": "metrics",
                    "filters": ["metric_filter"],
                },
                "configs": {
                    "level": "DEBUG",
                    "class": "logging.handlers.WatchedFileHandler",
                    "filename": f"{config_path}",
                    "filters": ["config_filter"],
                    "formatter": "configs",
                },
            },
            "loggers": {
                "": {
                    "handlers": ["default", "metrics", "configs"],
                    "level": "DEBUG",
                    "propagate": True,
                },
            },
        }
    )
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.PositionalArgumentsFormatter(),
            timestamper,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
