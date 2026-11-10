import logging, os
from pathlib import Path
from threading import Lock

_lock = Lock()
_metrics = {"requests": 0, "errors": 0, "total_latency_ms": 0.0, "last_latency_ms": 0.0}


def configure_logger():
    log_dir = Path(
        os.getenv(
            "APP_LOG_DIR",
            Path(__file__).resolve().parents[1] / "logs"
        )
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("country_co2")
    logger.setLevel(logging.INFO)

    # Close existing handlers before replacing them.
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    handler = logging.FileHandler(
        log_dir / "app.log",
        encoding="utf-8"
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    logger.addHandler(handler)
    logger.propagate = False

    return logger


def record_request(latency_ms, error=False):
    with _lock:
        _metrics["requests"] += 1
        _metrics["errors"] += int(error)
        _metrics["total_latency_ms"] += float(latency_ms)
        _metrics["last_latency_ms"] = float(latency_ms)

def get_metrics():
    with _lock:
        result = dict(_metrics)
    result["average_latency_ms"] = result["total_latency_ms"] / result["requests"] if result["requests"] else 0.0
    return result

def reset_metrics():
    with _lock:
        _metrics.update(requests=0, errors=0, total_latency_ms=0.0, last_latency_ms=0.0)
