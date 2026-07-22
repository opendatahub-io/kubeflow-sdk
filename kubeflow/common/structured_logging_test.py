# Copyright 2026 The Kubeflow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import warnings

import pytest
import structlog

import kubeflow.common.structured_logging as kf_logging
from kubeflow.common.structured_logging import configure_logging, get_logger


@pytest.fixture(autouse=True)
def _reset_logging_state():
    """Reset module-level state, stdlib handlers, and structlog globals between tests."""
    kf_logging._configured = False
    structlog.reset_defaults()
    ns_logger = logging.getLogger("kubeflow")
    ns_logger.handlers.clear()
    ns_logger.propagate = True
    ns_logger.setLevel(logging.NOTSET)
    yield
    kf_logging._configured = False
    structlog.reset_defaults()
    ns_logger = logging.getLogger("kubeflow")
    ns_logger.handlers.clear()
    ns_logger.propagate = True
    ns_logger.setLevel(logging.NOTSET)


def test_get_logger_returns_named_logger():
    logger = get_logger("kubeflow.test.logging")
    assert logger is not None
    assert hasattr(logger, "info")
    assert hasattr(logger, "warning")
    assert hasattr(logger, "error")


def test_get_logger_none():
    logger = get_logger(None)
    assert logger is not None
    assert hasattr(logger, "info")


def test_get_logger_routes_through_stdlib(capsys):
    ns_logger = logging.getLogger("kubeflow.test.stdlib")
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    ns_logger.addHandler(handler)
    ns_logger.setLevel(logging.DEBUG)

    logger = get_logger("kubeflow.test.stdlib")
    logger.info("hello_stdlib")

    captured = capsys.readouterr()
    assert "hello_stdlib" in (captured.out + captured.err)
    ns_logger.handlers.clear()


def test_configure_logging_json_output(capsys, monkeypatch):
    monkeypatch.delenv("CI", raising=False)
    configure_logging(level="INFO", json_output=True)

    logger = get_logger("kubeflow.test.json")
    logger.info("json_event", job_name="demo")

    captured = capsys.readouterr()
    output = (captured.out + captured.err).strip().splitlines()
    assert output, "expected at least one log line"
    payload = json.loads(output[-1])
    assert payload["event"] == "json_event"
    assert payload["job_name"] == "demo"
    assert payload["level"] == "info"


def test_configure_logging_console_output(capsys, monkeypatch):
    monkeypatch.delenv("CI", raising=False)
    configure_logging(level="INFO", json_output=False)

    logger = get_logger("kubeflow.test.console")
    logger.info("console_event")

    captured = capsys.readouterr()
    output = (captured.out + captured.err).strip()
    assert "console_event" in output
    # Console output should NOT be valid JSON
    with pytest.raises(json.JSONDecodeError):
        json.loads(output.splitlines()[-1])


def test_configure_logging_idempotent(monkeypatch):
    monkeypatch.delenv("CI", raising=False)
    configure_logging(level="DEBUG", json_output=False)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        configure_logging(level="ERROR", json_output=True)
        assert len(w) == 1
        assert "already been called" in str(w[0].message)

    ns_logger = logging.getLogger("kubeflow")
    assert ns_logger.level == logging.DEBUG


def test_configure_logging_invalid_level_raises():
    with pytest.raises(ValueError, match="Invalid log level"):
        configure_logging(level="DEUBG")


def test_ci_true_enables_json(capsys, monkeypatch):
    monkeypatch.setenv("CI", "true")
    configure_logging(level="INFO")

    logger = get_logger("kubeflow.test.ci_true")
    logger.info("ci_json_check")

    captured = capsys.readouterr()
    output = (captured.out + captured.err).strip().splitlines()
    assert output
    payload = json.loads(output[-1])
    assert payload["event"] == "ci_json_check"


def test_ci_false_does_not_enable_json(capsys, monkeypatch):
    monkeypatch.setenv("CI", "false")
    configure_logging(level="INFO")

    logger = get_logger("kubeflow.test.ci_false")
    logger.info("ci_console_check")

    captured = capsys.readouterr()
    output = (captured.out + captured.err).strip()
    assert "ci_console_check" in output
    with pytest.raises(json.JSONDecodeError):
        json.loads(output.splitlines()[-1])


def test_ci_zero_does_not_enable_json(capsys, monkeypatch):
    monkeypatch.setenv("CI", "0")
    configure_logging(level="INFO")

    logger = get_logger("kubeflow.test.ci_zero")
    logger.info("ci_zero_check")

    captured = capsys.readouterr()
    output = (captured.out + captured.err).strip()
    assert "ci_zero_check" in output
    with pytest.raises(json.JSONDecodeError):
        json.loads(output.splitlines()[-1])


def test_existing_handlers_warns(monkeypatch):
    monkeypatch.delenv("CI", raising=False)
    ns_logger = logging.getLogger("kubeflow")
    ns_logger.addHandler(logging.StreamHandler())

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        configure_logging(level="INFO", json_output=False)
        assert any("already has handlers" in str(x.message) for x in w)

    # Handler is appended, not replaced
    assert len(ns_logger.handlers) == 2
