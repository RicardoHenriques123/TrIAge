"""Tests for telemetry logging."""

import json
import pytest
from pathlib import Path
from datetime import datetime, timezone

from triage.telemetry.logger import TelemetryLogger, TelemetryEvent


class TestTelemetryEvent:
    """Tests for TelemetryEvent dataclass."""

    def test_creation_with_timestamp(self):
        """Test creating a TelemetryEvent with explicit timestamp."""
        timestamp = "2024-01-01T12:00:00+00:00"
        event = TelemetryEvent(
            event="test_event",
            timestamp=timestamp,
            data={"key": "value"}
        )
        
        assert event.event == "test_event"
        assert event.timestamp == timestamp
        assert event.data == {"key": "value"}

    def test_creation_auto_timestamp(self):
        """Test that timestamp is auto-generated if not provided."""
        event = TelemetryEvent(event="test_event", data={})
        
        # Verify it's a valid ISO8601 timestamp
        assert event.timestamp is not None
        assert "T" in event.timestamp
        assert event.timestamp.endswith("+00:00") or event.timestamp.endswith("Z")

    def test_default_empty_data(self):
        """Test that data defaults to empty dict."""
        event = TelemetryEvent(event="test_event")
        
        assert event.data == {}


class TestTelemetryLogger:
    """Tests for TelemetryLogger."""

    def test_creates_parent_directory(self, temp_dir):
        """Test that logger creates parent directories."""
        log_path = temp_dir / "subdir" / "logs" / "triage.jsonl"
        
        logger = TelemetryLogger(log_path)
        
        assert log_path.parent.exists()

    def test_log_event_writes_jsonl(self, temp_log_path):
        """Test that log_event writes valid JSONL."""
        logger = TelemetryLogger(temp_log_path)
        
        logger.log_event("test_event", {"key": "value", "number": 42})
        
        # Read and verify
        assert temp_log_path.exists()
        lines = temp_log_path.read_text().strip().split("\n")
        assert len(lines) == 1
        
        record = json.loads(lines[0])
        assert record["event"] == "test_event"
        assert record["data"]["key"] == "value"
        assert record["data"]["number"] == 42
        assert "timestamp" in record

    def test_log_event_appends(self, temp_log_path):
        """Test that multiple log_event calls append to file."""
        logger = TelemetryLogger(temp_log_path)
        
        logger.log_event("event1", {"id": 1})
        logger.log_event("event2", {"id": 2})
        logger.log_event("event3", {"id": 3})
        
        lines = temp_log_path.read_text().strip().split("\n")
        assert len(lines) == 3
        
        record1 = json.loads(lines[0])
        record2 = json.loads(lines[1])
        record3 = json.loads(lines[2])
        
        assert record1["data"]["id"] == 1
        assert record2["data"]["id"] == 2
        assert record3["data"]["id"] == 3

    def test_log_decision_convenience_method(self, temp_log_path):
        """Test log_decision convenience method."""
        logger = TelemetryLogger(temp_log_path)
        
        decision_data = {
            "issue_number": 42,
            "label": "bug",
            "confidence": 0.95,
            "source": "local"
        }
        logger.log_decision(decision_data)
        
        lines = temp_log_path.read_text().strip().split("\n")
        record = json.loads(lines[0])
        
        assert record["event"] == "routing_decision"
        assert record["data"]["issue_number"] == 42
        assert record["data"]["label"] == "bug"

    def test_log_with_complex_data(self, temp_log_path):
        """Test logging complex nested data structures."""
        logger = TelemetryLogger(temp_log_path)
        
        complex_data = {
            "string": "value",
            "number": 123,
            "float": 0.95,
            "boolean": True,
            "null": None,
            "list": [1, 2, 3],
            "nested": {
                "key1": "val1",
                "key2": ["a", "b", "c"]
            }
        }
        
        logger.log_event("complex_event", complex_data)
        
        lines = temp_log_path.read_text().strip().split("\n")
        record = json.loads(lines[0])
        
        assert record["data"]["string"] == "value"
        assert record["data"]["number"] == 123
        assert record["data"]["float"] == 0.95
        assert record["data"]["boolean"] is True
        assert record["data"]["null"] is None
        assert record["data"]["list"] == [1, 2, 3]
        assert record["data"]["nested"]["key1"] == "val1"

    def test_jsonl_format_valid(self, temp_log_path):
        """Test that output is valid JSONL (each line is valid JSON)."""
        logger = TelemetryLogger(temp_log_path)
        
        logger.log_event("event1", {"data": "value1"})
        logger.log_event("event2", {"data": "value2"})
        
        with temp_log_path.open("r") as f:
            for line_num, line in enumerate(f, 1):
                # Each line should be valid JSON
                try:
                    record = json.loads(line)
                    assert "event" in record
                    assert "timestamp" in record
                    assert "data" in record
                except json.JSONDecodeError as e:
                    pytest.fail(f"Line {line_num} is not valid JSON: {e}")

    def test_special_characters_escaped(self, temp_log_path):
        """Test that special characters are properly escaped."""
        logger = TelemetryLogger(temp_log_path)
        
        special_data = {
            "quotes": 'He said "hello"',
            "backslash": "path\\to\\file",
            "newline": "line1\nline2",
            "unicode": "🚀 emoji"
        }
        
        logger.log_event("special_event", special_data)
        
        lines = temp_log_path.read_text().strip().split("\n")
        record = json.loads(lines[0])
        
        assert 'He said "hello"' in record["data"]["quotes"]
        assert "emoji" in record["data"]["unicode"]

    def test_ensure_ascii_true(self, temp_log_path):
        """Test that ensure_ascii=True produces ASCII output."""
        logger = TelemetryLogger(temp_log_path)
        
        logger.log_event("unicode_event", {"text": "🚀 Rocket"})
        
        content = temp_log_path.read_bytes()
        # Should be ASCII-safe (no raw unicode bytes in the 128-255 range)
        try:
            content.decode("ascii")
        except UnicodeDecodeError:
            pytest.fail("Output contains non-ASCII bytes")

    def test_multiple_loggers_same_file(self, temp_log_path):
        """Test that multiple logger instances can write to the same file."""
        logger1 = TelemetryLogger(temp_log_path)
        logger2 = TelemetryLogger(temp_log_path)
        
        logger1.log_event("from_logger1", {"id": 1})
        logger2.log_event("from_logger2", {"id": 2})
        logger1.log_event("from_logger1_again", {"id": 3})
        
        lines = temp_log_path.read_text().strip().split("\n")
        assert len(lines) == 3
        
        records = [json.loads(line) for line in lines]
        assert records[0]["event"] == "from_logger1"
        assert records[1]["event"] == "from_logger2"
        assert records[2]["event"] == "from_logger1_again"

    def test_timestamp_format(self, temp_log_path):
        """Test that timestamps are in valid ISO8601 format."""
        logger = TelemetryLogger(temp_log_path)
        logger.log_event("test", {})
        
        lines = temp_log_path.read_text().strip().split("\n")
        record = json.loads(lines[0])
        
        timestamp = record["timestamp"]
        # Should be parseable as ISO8601
        try:
            # Try both formats commonly used
            if timestamp.endswith("Z"):
                datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            else:
                datetime.fromisoformat(timestamp)
        except ValueError:
            pytest.fail(f"Invalid ISO8601 timestamp: {timestamp}")

    def test_empty_event_handling(self, temp_log_path):
        """Test logging with minimal data."""
        logger = TelemetryLogger(temp_log_path)
        logger.log_event("", {})  # Empty event name
        logger.log_event("event", {})  # Empty data
        
        lines = temp_log_path.read_text().strip().split("\n")
        assert len(lines) == 2
