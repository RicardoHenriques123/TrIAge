"""Telemetry logging utilities."""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


@dataclass
class TelemetryEvent:
    """Structured telemetry event.

    Args:
        event: Event name.
        timestamp: ISO8601 timestamp.
        data: Event payload.
    """

    event: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    data: Dict[str, Any] = field(default_factory=dict)


class TelemetryLogger:
    """JSONL telemetry logger for operational metrics.

    Args:
        log_path: Path to the JSONL file.
    """

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_event(self, event: str, data: Dict[str, Any]) -> None:
        """Write an event to the JSONL log.

        Args:
            event: Event name.
            data: Event payload.
        """

        payload = TelemetryEvent(event=event, data=data)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(payload), ensure_ascii=True) + "\n")

    def log_decision(self, data: Dict[str, Any]) -> None:
        """Convenience wrapper for routing decisions.

        Args:
            data: Routing decision payload.
        """

        self.log_event("routing_decision", data)
