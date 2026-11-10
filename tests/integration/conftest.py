"""Shares the live gate + mock upstream fixture with every integration test module."""

from tests.integration.test_http import env

__all__ = ["env"]
