from __future__ import annotations

import unittest

from stablemimic.sim import close_simulation_app


class FakeApplication:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class SimulationShutdownTests(unittest.TestCase):
    def test_normal_close_returns_without_forcing_exit(self) -> None:
        application = FakeApplication()
        close_simulation_app(application, timeout_seconds=0.1)
        self.assertTrue(application.closed)

    def test_timeout_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be positive"):
            close_simulation_app(FakeApplication(), timeout_seconds=0.0)


if __name__ == "__main__":
    unittest.main()
