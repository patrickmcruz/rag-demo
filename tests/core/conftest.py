"""Shared fixtures for core (non-API) tests."""

import pytest

from tests.helpers import DummyChain


@pytest.fixture
def dummy_chain():
    """A fresh DummyChain for each test."""
    return DummyChain()
