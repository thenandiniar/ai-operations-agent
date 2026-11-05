
import pytest

from memory.store import (
    save_execution,
    load_history,
    get_recent_history,
    MemoryError,
)


def test_save_execution(tmp_path):
    memory_file = tmp_path / "history.jsonl"

    record = save_execution(
        "Analyze sales data",
        {"status": "success", "total_revenue": 81300},
        memory_file,
    )

    assert record["goal"] == "Analyze sales data"
    assert record["execution"]["status"] == "success"
    assert memory_file.exists()


def test_load_execution_history(tmp_path):
    memory_file = tmp_path / "history.jsonl"

    save_execution("First request", {"status": "success"}, memory_file)
    save_execution("Second request", {"status": "partial"}, memory_file)

    history = load_history(memory_file)

    assert len(history) == 2
    assert history[0]["goal"] == "First request"
    assert history[1]["goal"] == "Second request"


def test_get_recent_history(tmp_path):
    memory_file = tmp_path / "history.jsonl"

    for number in range(5):
        save_execution(
            f"Request {number}",
            {"status": "success"},
            memory_file,
        )

    recent = get_recent_history(2, memory_file)

    assert len(recent) == 2
    assert recent[0]["goal"] == "Request 3"
    assert recent[1]["goal"] == "Request 4"


def test_empty_history(tmp_path):
    memory_file = tmp_path / "nonexistent.jsonl"

    assert load_history(memory_file) == []


def test_reject_empty_goal(tmp_path):
    with pytest.raises(MemoryError):
        save_execution(
            "",
            {"status": "success"},
            tmp_path / "history.jsonl",
        )


def test_reject_invalid_history(tmp_path):
    memory_file = tmp_path / "broken.jsonl"
    memory_file.write_text("{invalid json}\n", encoding="utf-8")

    with pytest.raises(MemoryError):
        load_history(memory_file)
