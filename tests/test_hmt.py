"""Tests for the main module."""


def test_version():
    """Check that the version is acceptable."""
    from hmt import __version__

    assert __version__ == "0.1.0"
