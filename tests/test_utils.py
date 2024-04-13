#!/usr/bin/env python3

import unittest
import pytorch_influence_functions as ptif
from typing import Any

class TestSaveJson(unittest.TestCase):
    def test_placeholder(self) -> None:
        self.skipTest("This test is not implemented!")

class TestDisplayProgress(unittest.TestCase):
    def test_placeholder(self) -> None:
        self.skipTest("This test is not implemented!")

class TestInitLogging(unittest.TestCase):
    def test_placeholder(self) -> None:
        self.skipTest("This test is not implemented!")

class TestGetDefaultConfig(unittest.TestCase):
    def test_get_default_config(self) -> None:
        config: dict[str, Any] = ptif.get_default_config()

        self.assertIsNotNone(config)
        self.assertIsInstance(config, dict)