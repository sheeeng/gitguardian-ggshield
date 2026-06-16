"""Tests for truststore setup in __main__.py."""

import sys
from unittest import mock

import pytest


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="truststore requires Python 3.10+"
)
def test_setup_truststore_swallows_errors(monkeypatch) -> None:
    """A failure while injecting truststore must not crash the CLI (see #1265)."""
    import ggshield.__main__ as main_module

    fake_truststore = mock.MagicMock()
    fake_truststore.inject_into_ssl.side_effect = ValueError(
        "invalid literal for int() with base 10: ''"
    )
    monkeypatch.setitem(sys.modules, "truststore", fake_truststore)

    # Should not raise.
    main_module.setup_truststore()

    fake_truststore.inject_into_ssl.assert_called_once()
