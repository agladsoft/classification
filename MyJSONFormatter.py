import json_log_formatter
from json_log_formatter import JSONFormatter


class MyJSONFormatter(JSONFormatter):
    def __init__(self):
        super().__init__()

    def to_json(self, record):
        try:
            return self.json_lib.dumps(record, default=json_log_formatter._json_serializable, ensure_ascii=False)
        except (TypeError, ValueError):
            try:
                return self.json_lib.dumps(record, ensure_ascii=False)
            except (TypeError, ValueError, OverflowError):
                return '{}'