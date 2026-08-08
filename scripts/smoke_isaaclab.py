#!/usr/bin/env python3
"""Minimal headless Isaac Lab startup/reset/step smoke test."""

from __future__ import annotations

import argparse
import faulthandler
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

faulthandler.enable()
faulthandler.dump_traceback_later(60.0, repeat=False)
print("[STAGE] Launching Isaac Sim application...", flush=True)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("[STAGE] Isaac Sim application launched.", flush=True)

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext

from stablemimic.sim import close_simulation_app


def main() -> None:
    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args_cli.device))
    print("[STAGE] SimulationContext created; entering reset.", flush=True)
    sim.reset()
    print("[STAGE] Reset complete; entering one step.", flush=True)
    sim.step()
    print("[PASS] Isaac Lab headless reset and step completed.", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        faulthandler.cancel_dump_traceback_later()
        print("[STAGE] Closing Isaac Sim application...", flush=True)
        close_simulation_app(
            simulation_app,
            timeout_seconds=15.0,
            forced_exit_code=1 if sys.exc_info()[0] is not None else 0,
        )
