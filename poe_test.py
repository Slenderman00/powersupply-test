
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import base64
import csv
import io
import json
import os
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from lxml import etree
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)
import tntapi
import yangrpc
from yangcli import yangcli
from storageBucket import storageBucket

from powersupply import Power
from load import Load
from scope import Scope
import wave

namespaces = {
    "nc": "urn:ietf:params:xml:ns:netconf:base:1.0",
    "nd": "urn:ietf:params:xml:ns:yang:ietf-network",
    "nt": "urn:ietf:params:xml:ns:yang:ietf-network-topology",
}

# --------------------------
# CLI
# --------------------------
parser = argparse.ArgumentParser(description="PoE shield PSU test with live 3D plot and CSV output.")

# Required infra
parser.add_argument(
    "--config",
    required=True,
    help="Path to the NETCONF config XML (ietf-networks + topology). E.g. ../networks.xml",
)

# Node names (default to your 'raspberrypi' nodes)
parser.add_argument("--power-node", default="raspberrypi", help="Power supply node name")
parser.add_argument("--load-node", default="raspberrypi", help="Electronic load node name")
parser.add_argument("--scope-node", default="raspberrypi", help="Scope node name")

# PoE defaults (44–57 V typical for 802.3af/at)
parser.add_argument("--min-v", type=float, default=44.0, help="Minimum input voltage (V)")
parser.add_argument("--max-v", type=float, default=57.0, help="Maximum input voltage (V) (inclusive)")
parser.add_argument("--v-step", type=float, default=0.5, help="Voltage step (V). Supports fractional steps.")

# Output regulation test params
parser.add_argument("--start-current", type=float, default=5.0, help="Starting draw (A) at each input voltage")
parser.add_argument("--min-current", type=float, default=0.2, help="Lowest current to try before giving up (A)")
parser.add_argument("--current-step", type=float, default=-0.25, help="Current step (A); negative to step down")
parser.add_argument("--vout-pass-threshold", type=float, default=4.75,
                    help="Minimum acceptable output voltage to count as passing (V)")

# Scope acquisition / retries
parser.add_argument("--scope-threshold", type=float, default=1.0,
                    help="Scope load threshold trigger (arbitrary, forwarded to scope)")
parser.add_argument("--scope-samples", type=int, default=1000, help="Scope samples")
parser.add_argument("--scope-rate", type=int, default=20000, help="Scope sample rate (Hz)")
parser.add_argument("--scope-channel", default="ch1", help="Scope channel")
parser.add_argument("--scope-retries", type=int, default=3, help="Retries on scope errors per (Vin, I) point")
parser.add_argument("--scope-retry-delay", type=float, default=0.5, help="Delay between scope retries (s)")

# Plot
parser.add_argument("--no-plot", action="store_true", help="Disable live plotting (useful for headless runs)")

# CSV / artifacts
parser.add_argument("--out-dir", default="results", help="Directory to write CSV & artifacts")
parser.add_argument("--tag", default="", help="Optional tag to include in output filename")

args = parser.parse_args()

# --------------------------
# NETCONF infra
# --------------------------
tree = etree.parse(args.config)
network = tree.xpath("/nc:config/nd:networks/nd:network", namespaces=namespaces)[0]

conns = tntapi.network_connect(network)
yconns = tntapi.network_connect_yangrpc(network)

# GLOBAL instrument instances (node names fixed for this run)
POWER = Power(yconns, conns, node_name=args.power_node)
LOAD = Load(yconns, conns, network, node_name=args.load_node)
SCOPE = Scope(yconns, conns, node_name=args.scope_node)

# --------------------------
# Helpers
# --------------------------
def data_to_signal(data, rng=16):
    f = io.BytesIO(data)
    spf = wave.open(f, "r")
    signal = spf.readframes(-1)
    signal = np.frombuffer(signal, np.uint8)
    # Map 0..255 -> approx ±rng/?
    def _map(i):
        return ((i - 128) / 100.0) * rng
    return np.vectorize(_map)(signal)

def get_average_voltage(data):
    signal = data_to_signal(data)
    half = signal[800:]
    voltage = float(np.average(half))
    return voltage

def inclusive_arange(start, stop, step):
    # numpy.arange may miss the last step due to FP; include last point if close
    n = int(np.floor((stop - start) / step + 0.5))
    xs = np.array([start + i * step for i in range(n + 1)], dtype=float)
    xs[-1] = stop
    return xs

def current_to_resistance(target_current_a, v_nominal=5.0):
    # R = V / I  (avoid division by ~0)
    if target_current_a <= 0:
        return 9999.0
    return v_nominal / target_current_a

# --------------------------
# Live plots (3D + quick 2D)
# --------------------------
fig3d = None
ax3d = None
fig2d = None
ax2d_v = None
ax2d_i = None

def init_plots():
    global fig3d, ax3d, fig2d, ax2d_v, ax2d_i
    if args.no-plot:
        return
    plt.ion()
    # 3D: X=Vin, Y=Requested I (A), Z=Vout
    fig3d = plt.figure(figsize=(9, 8))
    ax3d = fig3d.add_subplot(111, projection='3d')
    ax3d.set_xlabel("Input Voltage (V)")
    ax3d.set_ylabel("Requested Current (A)")
    ax3d.set_zlabel("Output Voltage (V)")
    ax3d.set_title("Live PoE PSU Sweep (3D)")

    # 2D helper: Vout vs requested I at latest Vin
    fig2d, (ax2d_v, ax2d_i) = plt.subplots(2, 1, figsize=(8, 8))
    ax2d_v.set_title("Vout vs Requested Current (latest Vin)")
    ax2d_v.set_xlabel("Requested Current (A)")
    ax2d_v.set_ylabel("Vout (V)")
    ax2d_v.axhline(y=args.vout_pass_threshold, linestyle='--')

    ax2d_i.set_title("Measured Current vs Requested Current (latest Vin)")
    ax2d_i.set_xlabel("Requested Current (A)")
    ax2d_i.set_ylabel("Measured Current (A)")
    fig3d.canvas.draw(); fig3d.canvas.flush_events()
    fig2d.canvas.draw(); fig2d.canvas.flush_events()

def update_plots(all_points, current_vin):
    if args.no-plot:
        return
    # all_points: list of dicts with keys: vin, req_i, meas_i, vout, pass
    # 3D scatter (incremental)
    xs = [p["vin"] for p in all_points]
    ys = [p["req_i"] for p in all_points]
    zs = [p["vout"] for p in all_points]
    colors = ['g' if p["pass"] else 'r' for p in all_points]

    ax3d.clear()
    ax3d.set_xlabel("Input Voltage (V)")
    ax3d.set_ylabel("Requested Current (A)")
    ax3d.set_zlabel("Output Voltage (V)")
    ax3d.set_title("Live PoE PSU Sweep (3D)")
    ax3d.scatter(xs, ys, zs, c=colors, depthshade=True)
    ax3d.axhline(y=args.vout_pass_threshold)  # projected line

    # 2D: filter for latest Vin
    curr = [p for p in all_points if abs(p["vin"] - current_vin) < 1e-6]
    curr_sorted = sorted(curr, key=lambda p: p["req_i"])
    ax2d_v.clear(); ax2d_i.clear()
    if curr_sorted:
        ax2d_v.plot([p["req_i"] for p in curr_sorted], [p["vout"] for p in curr_sorted])
        ax2d_v.axhline(y=args.vout_pass_threshold, linestyle='--')
        ax2d_v.set_title(f"Vout vs Requested Current @ Vin={current_vin:.2f} V")
        ax2d_v.set_xlabel("Requested Current (A)")
        ax2d_v.set_ylabel("Vout (V)")

        ax2d_i.plot([p["req_i"] for p in curr_sorted], [p["meas_i"] for p in curr_sorted])
        ax2d_i.set_title(f"Measured I vs Requested I @ Vin={current_vin:.2f} V")
        ax2d_i.set_xlabel("Requested Current (A)")
        ax2d_i.set_ylabel("Measured Current (A)")
    fig3d.canvas.draw(); fig3d.canvas.flush_events()
    fig2d.canvas.draw(); fig2d.canvas.flush_events()
    plt.pause(0.001)

# --------------------------
# Instrument ops
# --------------------------
def capture_startup(resistance, vin):
    """
    Arm scope, set load, then apply input voltage; return raw scope data (wav bytes).
    """
    # Reset / set state
    POWER.delete_voltage_dual()
    LOAD.set_resistance(resistance)
    SCOPE.start_acquisition(
        args.scope_samples, args.scope_rate,
        args.scope_channel, "positive",
        args.scope_threshold, args.scope_channel,
        16, "-"
    )
    POWER.set_voltage_dual(vin)
    data = SCOPE.recieve()
    return data

def get_load_snapshot():
    vout, i_meas = LOAD.get_load_data()
    return float(vout), float(i_meas)

# --------------------------
# Main sweep
# --------------------------
def main():
    # Ensure output dir
    os.makedirs(args.out_dir, exist_ok=True)

    # Build filename stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    fname_stem = (
        f"poe_sweep_min{args.min_v}_max{args.max_v}_step{args.v_step}"
        f"_Istart{args.start_current}_Istep{args.current_step}"
        f"_Vth{args.vout_pass_threshold}{tag}_{timestamp}"
    ).replace(".", "p")

    csv_path = os.path.join(args.out_dir, f"{fname_stem}.csv")

    # Init CSV
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "input_voltage_v", "requested_current_a", "measured_current_a",
            "output_voltage_v", "computed_resistance_ohm", "pass", "attempt"
        ])

    # Storage bucket per Vin (kept from your code)
    storage = storageBucket()

    # Initialize instruments cleanly
    try:
        LOAD.delete_load()
    except Exception:
        pass
    try:
        POWER.delete_powersupply()
    except Exception:
        pass

    init_plots()
    all_points = []

    # Iterate Vin with inclusive fractional steps
    for vin in inclusive_arange(args.min_v, args.max_v, args.v_step):
        print(f"\n### Input voltage {vin:.2f} V")
        write, get_path = storage.create_folder(vin)

        # Start at 5 A (default) -> resistance ~ 1 Ω, step current downward (increase R)
        req_i = args.start_current
        last_status_pass = None

        vin_points = []

        while req_i >= args.min_current - 1e-9:
            R = current_to_resistance(req_i)

            # Scope-acquisition with retry
            data = None
            attempt = 0
            while attempt < args.scope_retries:
                attempt += 1
                try:
                    data = capture_startup(R, vin)
                    break
                except Exception as e:
                    print(f"[Scope error] {e}; retry {attempt}/{args.scope_retries} ...")
                    time.sleep(args.scope_retry_delay)
                    # Power off before re-arm to avoid latched states
                    try:
                        POWER.delete_voltage_dual()
                    except Exception:
                        pass

            if data is None:
                # Could not acquire this point—record as fail and move on
                print(f"[WARN] Skipping point Vin={vin:.2f}V, Ireq={req_i:.2f}A after retries.")
                vout, i_meas = (0.0, 0.0)
                passed = False
                attempt_idx = args.scope_retries
            else:
                # Persist raw waveform
                write(data, f"reqI_{req_i:.2f}_startup.wav")

                # Read instantaneous load data
                vout, i_meas = get_load_snapshot()
                passed = (vout >= args.vout_pass_threshold)
                attempt_idx = 1

            # Log / plot
            point = {
                "vin": float(vin),
                "req_i": float(req_i),
                "meas_i": float(i_meas),
                "vout": float(vout),
                "R": float(R),
                "pass": bool(passed),
                "attempt": attempt_idx
            }
            vin_points.append(point)
            all_points.append(point)

            print(f"Vin={vin:.2f} V | Ireq={req_i:.2f} A (R={R:.3f} Ω) "
                  f"=> Vout={vout:.3f} V, Imeas={i_meas:.3f} A | {'PASS' if passed else 'FAIL'}")

            update_plots(all_points, vin)

            # Write incremental CSV row
            with open(csv_path, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    f"{vin:.4f}", f"{req_i:.4f}", f"{i_meas:.4f}",
                    f"{vout:.4f}", f"{R:.6f}", int(passed), attempt_idx
                ])

            # Stop condition:
            # "the draw shall then be lowered until it stops failing"
            # -> Start at 5A and step DOWN until we reach the last PASS (i.e., not failing).
            # In other words: keep reducing current while it's FAIL; once it becomes PASS, stop for this Vin.
            if passed:
                print("→ PASS reached; stopping current reduction for this Vin.")
                break

            # Otherwise, reduce current
            req_i = req_i + args.current_step  # note: default step is negative to lower current

        # Persist per-Vin JSON summary and a snapshot of plots
        try:
            out_json = {
                "vin": vin,
                "points": vin_points,
                "vout_pass_threshold": args.vout_pass_threshold
            }
            write(json.dumps(out_json), "data.json", type="ascii")
        except Exception:
            pass

        if not args.no-plot:
            try:
                plt.savefig(os.path.join(get_path(), f"plot_{vin:.2f}V.png"))
            except Exception:
                pass

        # Reset load for next Vin
        try:
            LOAD.delete_load()
        except Exception:
            pass
        try:
            POWER.delete_powersupply()
        except Exception:
            pass

    print(f"\nDone. CSV written to: {csv_path}")
    if not args.no-plot:
        plt.ioff()
        plt.show(block=False)

if __name__ == "__main__":
    main()
