#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import csv
import io
import json
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from lxml import etree
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
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
parser = argparse.ArgumentParser(description="PoE shield PSU test with robust scope averaging, live 3D plot, CSV, current refinement, and hold-time test.")

# Required infra
parser.add_argument("--config", required=True, help="Path to the NETCONF config XML (ietf-networks + topology).")

# Node names
parser.add_argument("--power-node", default="raspberrypi", help="Power supply node name")
parser.add_argument("--load-node", default="raspberrypi", help="Electronic load node name")
parser.add_argument("--scope-node", default="raspberrypi", help="Scope node name")

# PoE defaults (44–57 V)
parser.add_argument("--min-v", type=float, default=44.0, help="Minimum input voltage (V)")
parser.add_argument("--max-v", type=float, default=57.0, help="Maximum input voltage (V) (inclusive)")
parser.add_argument("--v-step", type=float, default=0.5, help="Voltage step (V)")

# Output regulation test params (start at 5 A, then lower)
parser.add_argument("--start-current", type=float, default=5.0, help="Starting draw (A) at each input voltage")
parser.add_argument("--min-current", type=float, default=0.2, help="Lowest current to try before giving up (A)")
parser.add_argument("--current-step", type=float, default=-0.25, help="Current step (A); negative lowers draw")
parser.add_argument("--vout-pass-threshold", type=float, default=4.75, help="PASS threshold for 5 V rail, based on scope average (V)")

# Refinement (feature 1)
parser.add_argument("--refine", action="store_true", help="After first PASS per Vin, refine to find the highest passing current (binary search)")
parser.add_argument("--refine-min-step", type=float, default=0.02, help="Stop refining when bracket width < this (A)")
parser.add_argument("--refine-max-iter", type=int, default=10, help="Safety cap on refinement iterations")

# Hold-time test (feature 2)
parser.add_argument("--hold-test", action="store_true", help="Measure how long the PSU can hold the highest passing current until Vout dips below threshold on a downward slope")
parser.add_argument("--hold-threshold", type=float, default=4.5, help="Vout threshold for hold test (V)")
parser.add_argument("--hold-timeout", type=float, default=120.0, help="Max seconds to wait during hold test before timing out")
parser.add_argument("--hold-poll-period", type=float, default=0.02, help="Polling period (s) when sampling Vout for hold-time detection")

# Scope acquisition / retries
parser.add_argument("--scope-threshold", type=float, default=1.0, help="Scope trigger level")
parser.add_argument("--scope-samples", type=int, default=1000, help="Scope samples")
parser.add_argument("--scope-rate", type=int, default=20000, help="Scope sample rate (Hz)")
parser.add_argument("--scope-channel", default="ch1", help="Scope channel")
parser.add_argument("--scope-retries", type=int, default=3, help="Retries on scope errors per point")
parser.add_argument("--scope-retry-delay", type=float, default=0.5, help="Delay between scope retries (s)")

# Robust averaging controls
parser.add_argument("--scope-range", type=float, default=16.0, help="Scope voltage range used for mapping")
parser.add_argument("--smooth-win", type=int, default=25, help="Moving-average window (samples) for smoothing")
parser.add_argument("--trim-percent", type=float, default=0.10, help="Winsorize fraction (0.10 = clip 10% tails)")

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

POWER = Power(yconns, conns, node_name=args.power_node)
LOAD = Load(yconns, conns, network, node_name=args.load_node)
SCOPE = Scope(yconns, conns, node_name=args.scope_node)

# --------------------------
# Helpers
# --------------------------
def data_to_signal(data, rng=16.0):
    f = io.BytesIO(data)
    spf = wave.open(f, "r")
    raw = spf.readframes(-1)
    sig = np.frombuffer(raw, np.uint8).astype(np.float32)
    # Map 0..255 → approx ±rng
    return ((sig - 128.0) / 100.0) * float(rng)


def get_scope_metrics(data, rng=16.0, smooth_win=25, trim_percent=0.10):
    """
    Robust voltage metrics from the LAST QUARTER of the scope trace:
      - moving-average smoothing
      - winsorized (clipped) mean to resist spikes
    Returns dict: avg_v, v_min, v_max, v_std, ripple_pp, n
    """
    sig = data_to_signal(data, rng=float(rng))
    n = len(sig)
    if n == 0:
        return dict(avg_v=0.0, v_min=0.0, v_max=0.0, v_std=0.0, ripple_pp=0.0, n=0)

    start = max(0, int(0.75 * n))
    tail = sig[start:] if start < n else sig[-1:]
    if tail.size == 0:
        tail = sig[-1:]

    # Moving-average smoothing
    win = max(1, int(smooth_win))
    if tail.size >= win and win > 1:
        kernel = np.ones(win, dtype=np.float32) / float(win)
        smooth = np.convolve(tail, kernel, mode="same")
    else:
        smooth = tail

    # Winsorize extremes
    p = float(trim_percent)
    if smooth.size > 1 and 0.0 < p < 0.5:
        lo = np.percentile(smooth, p * 100.0)
        hi = np.percentile(smooth, (1.0 - p) * 100.0)
        clipped = np.clip(smooth, lo, hi)
    else:
        clipped = smooth

    avg_v = float(np.mean(clipped))
    v_min = float(np.min(smooth))
    v_max = float(np.max(smooth))
    v_std = float(np.std(smooth))
    ripple_pp = v_max - v_min

    return dict(avg_v=avg_v, v_min=v_min, v_max=v_max, v_std=v_std, ripple_pp=ripple_pp, n=int(tail.size))


def inclusive_arange(start, stop, step):
    n = int(np.floor((stop - start) / step + 0.5))
    xs = np.array([start + i * step for i in range(n + 1)], dtype=float)
    xs[-1] = stop
    return xs


def current_to_resistance(target_current_a, v_nominal=5.0):
    if target_current_a <= 0:
        return 9999.0
    return v_nominal / target_current_a

# --------------------------
# Live plots (3D + helper 2D)
# --------------------------
fig3d = None
ax3d = None
fig2d = None
ax2d_v = None
ax2d_i = None


def init_plots():
    global fig3d, ax3d, fig2d, ax2d_v, ax2d_i
    if args.no_plot:
        return
    plt.ion()
    fig3d = plt.figure(figsize=(9, 8))
    ax3d = fig3d.add_subplot(111, projection='3d')
    ax3d.set_xlabel("Input Voltage (V)")
    ax3d.set_ylabel("Requested Current (A)")
    ax3d.set_zlabel("Scope Avg Vout (V)")
    ax3d.set_title("Live PoE PSU Sweep (3D)")

    fig2d, (ax2d_v, ax2d_i) = plt.subplots(2, 1, figsize=(8, 8))
    ax2d_v.set_title("Scope Avg Vout vs Ireq (latest Vin)")
    ax2d_v.set_xlabel("Requested Current (A)")
    ax2d_v.set_ylabel("Scope Avg Vout (V)")
    ax2d_v.axhline(y=args.vout_pass_threshold, linestyle="--")

    ax2d_i.set_title("Measured Current vs Ireq (latest Vin)")
    ax2d_i.set_xlabel("Requested Current (A)")
    ax2d_i.set_ylabel("Measured Current (A)")
    fig3d.canvas.draw(); fig3d.canvas.flush_events()
    fig2d.canvas.draw(); fig2d.canvas.flush_events()


def update_plots(all_points, current_vin):
    if args.no_plot:
        return
    xs = [p["vin"] for p in all_points]
    ys = [p["req_i"] for p in all_points]
    zs = [p["vout_scope"] for p in all_points]
    colors = ["g" if p["pass"] else "r" for p in all_points]

    ax3d.clear()
    ax3d.set_xlabel("Input Voltage (V)")
    ax3d.set_ylabel("Requested Current (A)")
    ax3d.set_zlabel("Scope Avg Vout (V)")
    ax3d.set_title("Live PoE PSU Sweep (3D)")
    ax3d.scatter(xs, ys, zs, c=colors, depthshade=True)

    curr = [p for p in all_points if abs(p["vin"] - current_vin) < 1e-6]
    curr_sorted = sorted(curr, key=lambda p: p["req_i"])
    ax2d_v.clear(); ax2d_i.clear()
    if curr_sorted:
        ax2d_v.plot([p["req_i"] for p in curr_sorted], [p["vout_scope"] for p in curr_sorted])
        ax2d_v.axhline(y=args.vout_pass_threshold, linestyle="--")
        ax2d_v.set_title(f"Scope Avg Vout vs Ireq @ Vin={current_vin:.2f} V")
        ax2d_v.set_xlabel("Requested Current (A)")
        ax2d_v.set_ylabel("Scope Avg Vout (V)")

        ax2d_i.plot([p["req_i"] for p in curr_sorted], [p["meas_i"] for p in curr_sorted])
        ax2d_i.set_title(f"Measured I vs Ireq @ Vin={current_vin:.2f} V")
        ax2d_i.set_xlabel("Requested Current (A)")
        ax2d_i.set_ylabel("Measured Current (A)")
    fig3d.canvas.draw(); fig3d.canvas.flush_events()
    fig2d.canvas.draw(); fig2d.canvas.flush_events()
    plt.pause(0.001)

# --------------------------
# Instrument ops
# --------------------------

def capture_startup(resistance, vin, trigger_slope="positive", trigger_thresh=None):
    POWER.delete_voltage_dual()
    LOAD.set_resistance(resistance)
    SCOPE.start_acquisition(
        args.scope_samples,
        args.scope_rate,
        args.scope_channel,
        trigger_slope,
        (args.scope_threshold if trigger_thresh is None else trigger_thresh),
        args.scope_channel,
        16,
        "-",
    )
    POWER.set_voltage_dual(vin)
    return SCOPE.recieve()


def get_load_snapshot():
    vout, i_meas = LOAD.get_load_data()
    return float(vout), float(i_meas)

# --------------------------
# Core point runner + refinement + hold-time
# --------------------------

def log_point(csv_path, point):
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            f"{point['vin']:.6f}",
            f"{point['req_i']:.6f}",
            f"{point['meas_i']:.6f}",
            f"{point['vout_scope']:.6f}",
            f"{point['vmin']:.6f}",
            f"{point['vmax']:.6f}",
            f"{point['vstd']:.6f}",
            f"{point['ripple']:.6f}",
            f"{point['vout_load']:.6f}",
            f"{point['R']:.6f}",
            int(point['pass']),
            point['attempt'],
            int(point.get('refined', 0)),
            ("" if point.get('hold_time_s') is None else f"{point['hold_time_s']:.6f}"),
        ])


def run_point(vin, req_i, storage_write, label_prefix, all_points, csv_path, trigger_slope="positive"):
    R = current_to_resistance(req_i)

    data = None
    attempt = 0
    while attempt < args.scope_retries:
        attempt += 1
        try:
            data = capture_startup(R, vin, trigger_slope=trigger_slope)
            break
        except Exception as e:
            print(f"[Scope error] {e}; retry {attempt}/{args.scope_retries} ...")
            time.sleep(args.scope_retry_delay)
            try:
                POWER.delete_voltage_dual()
            except Exception:
                pass

    vout_scope = vmin = vmax = vstd = ripple = 0.0
    vout_load = 0.0
    i_meas = 0.0
    passed = False

    if data is not None:
        storage_write(data, f"{label_prefix}_I_{req_i:.2f}_startup.wav")
        m = get_scope_metrics(
            data,
            rng=args.scope_range,
            smooth_win=args.smooth_win,
            trim_percent=args.trim_percent,
        )
        vout_scope = m["avg_v"]
        vmin = m["v_min"]
        vmax = m["v_max"]
        vstd = m["v_std"]
        ripple = m["ripple_pp"]
        vout_load, i_meas = get_load_snapshot()
        passed = (vout_scope >= args.vout_pass_threshold)
        attempt_idx = 1
    else:
        print(f"[WARN] Skipping point Vin={vin:.2f}V, Ireq={req_i:.2f}A after retries.")
        attempt_idx = args.scope_retries

    point = {
        "vin": float(vin),
        "req_i": float(req_i),
        "meas_i": float(i_meas),
        "vout_scope": float(vout_scope),
        "vout_load": float(vout_load),
        "vmin": float(vmin),
        "vmax": float(vmax),
        "vstd": float(vstd),
        "ripple": float(ripple),
        "R": float(R),
        "pass": bool(passed),
        "attempt": attempt_idx,
    }
    all_points.append(point)

    print(
        "Vin={:.2f} V | Ireq={:.2f} A (R={:.3f} Ω) => ScopeAvg={:.3f} V  [min={:.3f}, max={:.3f}, std={:.3f}, pp={:.3f}]  "
        "LoadVout={:.3f} V  Imeas={:.3f} A | {}".format(
            vin, req_i, R, vout_scope, vmin, vmax, vstd, ripple, vout_load, i_meas, "PASS" if passed else "FAIL"
        )
    )

    update_plots(all_points, vin)
    log_point(csv_path, {**point, "refined": 0, "hold_time_s": None})

    return point


def refine_max_current(vin, last_fail_i, last_pass_i, storage_write, all_points, csv_path):
    """Binary-search the boundary between FAIL (too high I) and PASS.
    Returns (best_pass_i, best_point).
    """
    if last_pass_i is None:
        return None, None
    if last_fail_i is None:
        # Nothing to bracket; can't refine
        return last_pass_i, None

    lo = last_pass_i  # known PASS
    hi = last_fail_i  # known FAIL (higher current)

    best_i = lo
    best_point = None

    it = 0
    while (abs(hi - lo) > args.refine_min_step) and (it < args.refine_max_iter):
        it += 1
        mid = (hi + lo) / 2.0
        p = run_point(vin, mid, storage_write, label_prefix=f"refine{it}", all_points=all_points, csv_path=csv_path)
        if p["pass"]:
            lo = mid
            best_i = mid
            best_point = p
        else:
            hi = mid
    return best_i, best_point


def measure_hold_time(vin, hold_current_a, storage_write):
    """Measure time at a fixed current until Vout dips below hold-threshold on a DOWNWARD slope.
    We configure the scope with a negative slope & the threshold, and also poll the load voltage at a
    small interval to detect the first crossing on a falling edge. Returns time (s) or None on timeout.
    """
    R = current_to_resistance(hold_current_a)

    # Try to have the scope armed with the intended trigger
    try:
        POWER.delete_voltage_dual()
        LOAD.set_resistance(R)
        SCOPE.start_acquisition(
            args.scope_samples,
            args.scope_rate,
            args.scope_channel,
            "negative",  # falling-edge trigger
            args.hold_threshold,
            args.scope_channel,
            16,
            "-",
        )
        POWER.set_voltage_dual(vin)
    except Exception as e:
        print(f"[Hold] Failed to arm scope for hold-time: {e}")
        # Still proceed with polling approach

    t0 = time.monotonic()
    last_v = None
    while True:
        now = time.monotonic()
        if (now - t0) >= args.hold_timeout:
            return None
        try:
            vout, _ = get_load_snapshot()
        except Exception:
            vout = None
        if vout is not None:
            if last_v is not None:
                if (vout < args.hold_threshold) and (vout < last_v):
                    return now - t0
            last_v = vout
        time.sleep(args.hold_poll_period)


# --------------------------
# Main sweep
# --------------------------

def main():
    os.makedirs(args.out_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    fname_stem = (
        f"poe_sweep_min{args.min_v}_max{args.max_v}_step{args.v_step}"
        f"_Istart{args.start_current}_Istep{args.current_step}"
        f"_Vth{args.vout_pass_threshold}_sw{args.smooth_win}_tp{args.trim_percent}"
        f"{tag}_{timestamp}"
    ).replace(".", "p")
    csv_path = os.path.join(args.out_dir, f"{fname_stem}.csv")

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "input_voltage_v",
            "requested_current_a",
            "measured_current_a",
            "scope_avg_v",
            "scope_min_v",
            "scope_max_v",
            "scope_std_v",
            "scope_ripple_pp_v",
            "load_vout_v",
            "computed_resistance_ohm",
            "pass",
            "attempt",
            "refined",          # 0 normal sweep, 1 refinement point
            "hold_time_s",      # only filled for the final max-pass current measurement
        ])

    storage = storageBucket()

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

    for vin in inclusive_arange(args.min_v, args.max_v, args.v_step):
        print(f"\n### Input voltage {vin:.2f} V")
        write, get_path = storage.create_folder(vin)

        req_i = args.start_current
        last_fail_i = None
        last_pass_i = None

        # Sweep downward until the first PASS (or min current)
        while req_i >= args.min_current - 1e-9:
            p = run_point(vin, req_i, write, label_prefix="sweep", all_points=all_points, csv_path=csv_path)
            if p["pass"]:
                last_pass_i = req_i
                break
            else:
                last_fail_i = req_i
            req_i = req_i + args.current_step  # negative by default

        # If we never passed, record and move on
        if last_pass_i is None:
            print("→ No PASS reached for this Vin.")
        else:
            print(f"→ PASS reached at ~{last_pass_i:.3f} A.")
            # Optional refinement to pinpoint the exact maximum passing current
            best_i = last_pass_i
            best_point = None
            if args.refine:
                bi, bp = refine_max_current(vin, last_fail_i, last_pass_i, write, all_points, csv_path)
                if bi is not None:
                    best_i = bi
                    best_point = bp
                print(f"→ Refined max PASS current ≈ {best_i:.3f} A")

            # Optional hold-time test at the refined maximum current
            if args.hold_test:
                print("→ Measuring hold time at refined current ...")
                hold_s = measure_hold_time(vin, best_i, write)
                if hold_s is None:
                    print(f"   Hold test timed out after {args.hold_timeout:.1f} s (no dip < {args.hold_threshold:.2f} V detected).")
                else:
                    print(f"   Hold time until Vout < {args.hold_threshold:.2f} V (falling) = {hold_s:.3f} s")

                # Log a synthetic row capturing hold time at the best current
                vout, imeas = get_load_snapshot()
                point = {
                    "vin": float(vin),
                    "req_i": float(best_i),
                    "meas_i": float(imeas),
                    "vout_scope": float(vout),  # store DC snapshot here for convenience
                    "vout_load": float(vout),
                    "vmin": 0.0,
                    "vmax": 0.0,
                    "vstd": 0.0,
                    "ripple": 0.0,
                    "R": float(current_to_resistance(best_i)),
                    "pass": 1,
                    "attempt": 0,
                }
                all_points.append(point)
                update_plots(all_points, vin)
                log_point(csv_path, {**point, "refined": 1, "hold_time_s": hold_s})

        # Save per-Vin summary & plot snapshot
        try:
            out_json = {
                "vin": vin,
                "points": [p for p in all_points if abs(p["vin"] - vin) < 1e-6],
                "vout_pass_threshold": args.vout_pass_threshold,
            }
            write(json.dumps(out_json), "data.json", type="ascii")
        except Exception:
            pass

        if not args.no_plot:
            try:
                plt.savefig(os.path.join(get_path(), f"plot_{vin:.2f}V.png"))
            except Exception:
                pass

        try:
            LOAD.delete_load()
        except Exception:
            pass
        try:
            POWER.delete_powersupply()
        except Exception:
            pass

    print(f"\nDone. CSV written to: {csv_path}")
    if not args.no_plot:
        plt.ioff()
        plt.show(block=False)


if __name__ == "__main__":
    main()
