
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Under-voltage cutoff finder for the PoE shield (DUT) using ONLY the programmable power supply and electronic load.

It applies a small, safe constant load and uses a bracket + binary search on Vin to locate the cutoff edge
where the DUT stops producing a valid 5 V output.

No scope is used in this test.
"""

import argparse
import csv
import os
import time
from datetime import datetime

import numpy as np
from lxml import etree

import tntapi
import yangrpc
from yangcli import yangcli  # noqa: F401 (kept for parity with your stack)
from storageBucket import storageBucket

from powersupply import Power
from load import Load

namespaces = {
    "nc": "urn:ietf:params:xml:ns:netconf:base:1.0",
    "nd": "urn:ietf:params:xml:ns:yang:ietf-network",
    "nt": "urn:ietf:params:xml:ns:yang:ietf-network-topology",
}

# --------------------------
# CLI
# --------------------------
parser = argparse.ArgumentParser(
    description="Find UV cutoff on the PoE shield by binary-searching Vin with a small constant load (no scope)."
)

# Infra
parser.add_argument("--config", required=True, help="Path to the NETCONF config XML (ietf-networks + topology).")
parser.add_argument("--power-node", default="raspberrypi", help="Power supply node name")
parser.add_argument("--load-node", default="raspberrypi", help="Electronic load node name")

# Load & detection
parser.add_argument("--load-current", type=float, default=0.2, help="Constant current to draw (A) via resistance emulation")
parser.add_argument("--vout-alive-th", type=float, default=4.5, help="Minimum Vout considered 'alive' (V)")
parser.add_argument("--settle", type=float, default=0.25, help="Seconds to wait after setting Vin before reading Vout")

# Search bounds & behavior
parser.add_argument("--vin-min", type=float, default=40.0, help="Minimum Vin to consider during search (V)")
parser.add_argument("--vin-max", type=float, default=60.0, help="Maximum Vin to consider during search (V)")
parser.add_argument("--start-high", type=float, default=48.0, help="Initial Vin to probe (should be inside valid window)")
parser.add_argument("--bracket-step", type=float, default=1.0, help="Coarse step for auto-bracketing (V)")
parser.add_argument("--resolution", type=float, default=0.05, help="Binary-search stop width (V)")
parser.add_argument("--max-iters", type=int, default=20, help="Max iterations for binary search")

# Latch-off handling (re-enable when increasing Vin)
parser.add_argument("--rearm-on-rise", action="store_true", default=True,
                    help="If coming from a dead state and raising Vin, power-cycle to rearm the DUT")
parser.add_argument("--rearm-off-delay", type=float, default=0.20, help="Seconds to wait after PSU off before reapplying")
parser.add_argument("--rearm-on-delay", type=float, default=0.25, help="Seconds to wait after PSU on before measuring")

# Retry
parser.add_argument("--retries", type=int, default=2, help="Retries for a single measurement point on comm errors")

# Output
parser.add_argument("--out-dir", default="results", help="Directory for CSV & artifacts")
parser.add_argument("--tag", default="", help="Optional tag in filenames")

args = parser.parse_args()

# --------------------------
# NETCONF infra
# --------------------------

tree = etree.parse(args.config)
network = tree.xpath("/nc:config/nd:networks/nd:network", namespaces=namespaces)[0]
conns = tntapi.network_connect(network)
yconns = tntapi.network_connect_yangrpc(network)

POWER = Power(yconns, conns, node_name=args.power_node)
LOAD = Load(yconns, conns, network, node_name=args.load_node)

# --------------------------
# Helpers
# --------------------------

def current_to_resistance(i_a: float, v_nom: float = 5.0) -> float:
    """Approximate constant-current via a fixed resistance using nominal Vout."""
    if i_a <= 0:
        return 1e6
    return v_nom / i_a


def psu_off():
    try:
        POWER.delete_powersupply()
    except Exception:
        pass


def apply_vin(vin: float):
    psu_off()
    POWER.set_voltage_dual(float(vin))


def read_vout_imax_retry() -> tuple[float, float]:
    exc = None
    for _ in range(max(1, args.retries)):
        try:
            vout, i_meas = LOAD.get_load_data()
            return float(vout), float(i_meas)
        except Exception as e:
            exc = e
            time.sleep(0.05)
    if exc:
        raise exc
    return 0.0, 0.0


def measure_alive(vin: float, from_dead: bool = False) -> tuple[bool, float, float]:
    """Set Vin, wait, read Vout -> return (alive?, vout, imeas). Handles optional rearm when raising Vin."""
    if from_dead and args.rearm_on_raise:
        # full power-cycle before setting a higher Vin
        psu_off()
        time.sleep(args.rearm_off_delay)
        POWER.set_voltage_dual(vin)
        time.sleep(args.rearm_on_delay)
    else:
        apply_vin(vin)
        time.sleep(args.settle)

    vout, imeas = read_vout_imax_retry()
    return (vout >= args.vout_alive_th), vout, imeas


# --------------------------
# Main
# --------------------------

def main():
    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    csv_path = os.path.join(args.out_dir, f"uv_cutoff_binary{tag}_{ts}.csv")

    # Prepare constant load via resistance approximation
    R = current_to_resistance(args.load_current)
    try:
        LOAD.delete_load()
    except Exception:
        pass
    LOAD.set_resistance(R)

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "vin_cmd_v", "vout_v", "i_meas_a", "alive", "phase"
        ])

    def log_point(vin, vout, imeas, alive, phase):
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([f"{vin:.6f}", f"{vout:.6f}", f"{imeas:.6f}", int(alive), phase])

    # --- Bracket: find (hi_alive, lo_dead) around the edge ---
    hi = None  # alive vin
    lo = None  # dead vin

    # Start high
    vin = float(np.clip(args.start_high, args.vin_min, args.vin_max))
    alive, vout, imeas = measure_alive(vin, from_dead=False)
    log_point(vin, vout, imeas, alive, phase="probe_start")

    if alive:
        hi = vin
        # step downward until dead
        step = abs(args.bracket_step)
        while vin - step >= args.vin_min and lo is None:
            vin = vin - step
            alive2, vout2, imeas2 = measure_alive(vin, from_dead=False)
            log_point(vin, vout2, imeas2, alive2, phase="bracket_down")
            if not alive2:
                lo = vin
            else:
                hi = vin
    else:
        lo = vin
        # step upward until alive
        step = abs(args.bracket_step)
        from_dead = True
        while vin + step <= args.vin_max and hi is None:
            vin = vin + step
            alive2, vout2, imeas2 = measure_alive(vin, from_dead=from_dead)
            from_dead = not alive2  # continue rearming while still dead
            log_point(vin, vout2, imeas2, alive2, phase="bracket_up")
            if alive2:
                hi = vin
            else:
                lo = vin

    if hi is None or lo is None:
        print("Could not bracket the UV cutoff within vin-min/max.")
        psu_off()
        return

    # Ensure ordering: hi is alive, lo is dead
    if hi < lo:
        hi, lo = lo, hi

    print(f"Bracketed edge: hi(alive)={hi:.3f} V, lo(dead)={lo:.3f} V")

    # --- Binary search ---
    it = 0
    last_state_dead = True  # we came from lo(dead)
    while (hi - lo) > args.resolution and it < args.max_iters:
        it += 1
        mid = (hi + lo) / 2.0
        alive, vout, imeas = measure_alive(mid, from_dead=last_state_dead)
        log_point(mid, vout, imeas, alive, phase=f"bisect_{it}")
        if alive:
            hi = mid
            last_state_dead = False
        else:
            lo = mid
            last_state_dead = True

    cutoff_v = hi  # lowest Vin that is still alive (approx. UV enable)

    print(f"\nEstimated UV cutoff (low-side edge) ≈ {cutoff_v:.3f} V @ {args.load_current:.3f} A load")

    psu_off()


if __name__ == "__main__":
    main()
