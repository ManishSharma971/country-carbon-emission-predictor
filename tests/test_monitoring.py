from pathlib import Path
from src.monitoring import record_request, get_metrics, reset_metrics, configure_logger

def test_request_metrics():
    reset_metrics()
    record_request(12.5); record_request(20, error=True)
    m = get_metrics()
    assert m["requests"] == 2 and m["errors"] == 1
    assert m["average_latency_ms"] == 16.25
    reset_metrics()

def test_logger_uses_test_directory(isolated_test_logs):
    configure_logger().info("test log entry")
    assert (Path(isolated_test_logs) / "app.log").exists()
