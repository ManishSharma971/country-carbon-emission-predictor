
import logging
import os
import tempfile

import pytest


@pytest.fixture(scope="function")
def isolated_test_logs(monkeypatch):
    tmp = tempfile.mkdtemp(prefix="co2_test_logs_")
    monkeypatch.setenv("APP_LOG_DIR", tmp)

    logger = logging.getLogger("country_co2")

    # Close existing handlers so they cannot lock the test log file.
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    yield tmp

    # Close test handlers before the temporary directory is deleted.
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    # Remove the temporary directory after releasing the file.
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
