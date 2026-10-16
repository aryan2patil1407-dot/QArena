"""Test-local configuration.

`core.contracts.TestCase` is a domain model (a *test case* under test), not a
pytest test class — tell pytest not to try to collect it.
"""
from core.contracts import TestCase

TestCase.__test__ = False
