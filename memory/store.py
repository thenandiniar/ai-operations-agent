
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_MEMORY_FILE = Path("memory/execution_history.jsonl")


class MemoryError(Exception):
    """Raised when execution history cannot be stored or read."""


def make_json_safe(value: Any) -> Any:
    """Convert common Python objects into JSON-compatible values."""

    if hasattr(value, "model_dump"):
        return make_json_safe(value.model_dump())

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]

    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def save_execution(
    goal: str,
    execution_report: Any,
    memory_file: str | Path = DEFAULT_MEMORY_FILE,
) -> dict:
    """Append one execution record to persistent JSONL history."""

    if not isinstance(goal, str) or not goal.strip():
        raise MemoryError("Execution goal cannot be empty.")

    if execution_report is None:
        raise MemoryError("Execution report cannot be empty.")

    path = Path(memory_file)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "goal": goal.strip(),
        "execution": make_json_safe(execution_report),
    }

    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    except (OSError, TypeError, ValueError) as error:
        raise MemoryError(
            f"Could not save execution history: {error}"
        ) from error

    return record


def load_history(
    memory_file: str | Path = DEFAULT_MEMORY_FILE,
) -> list[dict]:
    """Read all valid execution records from JSONL history."""

    path = Path(memory_file)

    if not path.exists():
        return []

    records = []

    try:
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise MemoryError(
                        f"Invalid JSON at line {line_number}: {error}"
                    ) from error

                if not isinstance(record, dict):
                    raise MemoryError(
                        f"Invalid record at line {line_number}."
                    )

                records.append(record)

    except OSError as error:
        raise MemoryError(
            f"Could not read execution history: {error}"
        ) from error

    return records


def get_recent_history(
    limit: int = 5,
    memory_file: str | Path = DEFAULT_MEMORY_FILE,
) -> list[dict]:
    """Return the most recent execution records."""

    if limit < 1:
        raise ValueError("Limit must be at least 1.")

    records = load_history(memory_file)

    return records[-limit:]
