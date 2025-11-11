#!/usr/bin/python3

from lxml import etree
import time
import sys, os
import argparse
import tntapi
import yangrpc
from yangcli import yangcli
import base64
import matplotlib.pyplot as plt
import numpy as np
import io
import wave
from storageBucket import storageBucket
import json

from powersupply import Power
from load import Load
from scope import Scope

from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

namespaces = {
    "nc": "urn:ietf:params:xml:ns:netconf:base:1.0",
    "nd": "urn:ietf:params:xml:ns:yang:ietf-network",
    "nt": "urn:ietf:params:xml:ns:yang:ietf-network-topology",
}

# -----------------------------
# CLI
# -----------------------------
parser = argparse.ArgumentParser(description="PoE shield PSU test (original flow + CLI, 3D live & CSV).")
parser.add_argument(
    "--config", required=True,
    help="Path to the netconf configuration *.xml file defining the configuration according to ietf-networks, ietf-networks-topology and netconf-node models e.g. ../networks.xml",
)
parser.add_argument("--power-node", default="raspberrypi", help="Power supply node name")
parser.add_argument("--load-node", default="raspberrypi", help="Load node name")
parser.add_argument("--scope-node", default="raspberrypi", help="Scope node name")

# PoE sweep
parser.add_argument("--min-v", type=float, default=44.0, help="Minimum PoE input voltage (V)")
parser.add_argument("--max-v", type=float, default=57.0, help="Maximum PoE input voltage (V), inclusive")
parser.add_argument("--v-step", type=float, default=1.0, help="Voltage step (V). Supports floats")

# Current / resistance behavior
parser.add_argument("--start-current", type=float, default=5.0, help="Start draw (A) at each Vin (default 5A)")
parser.add_argument("--resistance-step", type=float, default=0.5, help="How much to increase resistance per step (Ω)")
parser.add_argument("--min-current", type=float, default=0.2, help="Stop if requested current would go below this (A)")
parser.add_argument("--vout-threshold", type=float, default=4.75, help="Pass threshold for DUT 5V rail (V)")

# Scope robustness
parser.add_argument("--scope-retries", type=int, default=3, help="Retries when scope recieve() errors")
parser.add_argument("--scope-retry-delay", type=float, default=0.3, help="Delay between scope retries (s)")
parser.add_argument("--samples", type=int, default=1000, help="Scope samples")
parser.add_argument("--sample-rate", type=float, default=20000.0, help="Scope sample rate (Hz)")
parser.add_argument("--trigger-source", default="ch1", help="Scope trigger/source channel")
parser.add_argument("--trigger-level", type=float, default=1.0, help="Scope trigger level")
parser.add_argument("--trigger-slope", default="positive", help="Scope trigger slope")
parser.add_argument("--scope-range", type=float, default=16.0, help="Scope channel range")

# Output artifacts
parser.add_argument("--out-dir", default="results", help="Directory for CSV and plots")
parser.add_argument("--tag", default="", help="Optional tag for filenames")

# Live plotting controls
parser.add_argument("--no-3d", action="store_true", help="Disable the extra live 3D plot")

args = parser.parse_args()

# -----------------------------
# NETCONF init
# -----------------------------
tree = etree.parse(args.config)
network = tree.xpath("/nc:config/nd:networks/nd:network", namespaces=namespaces)[0]

conns = tntapi.network_connect(network)
yconns = tntapi.network_connect_yangrpc(network)

# GLOBAL instrument instances (node names from CLI)
POWER = Power(yconns, conns, node_name=args.power_node)
LOAD = Load(yconns, conns, network, node_name=args.load_node)
SCOPE = Scope(yconns, conns, node_name=args.scope_node)

# Initialize figure and axes globally (your 2D)
fig, axs = None, None

# Optional 3D
fig3d, ax3d = None, None
xs3d, ys3d, zs3d = [], [], []

def data_to_signal(data, range=16):
    f = io.BytesIO(data)
    spf = wave.open(f, "r")
    signal = spf.readframes(-1)
    signal = np.frombuffer(signal, np.uint8)
    def map(i):
        return ((i - 128) / 100) * range
    signal = np.vectorize(map)(signal)
    return signal

def get_average_voltage(data):
    signal = data_to_signal(data)
    half = signal[800:]
    voltage = np.average(half)
    return voltage

def plot(data, range, currents, volts, loads):
    global fig, axs

    plt.ion()

    if fig is None or not plt.fignum_exists(fig.number if fig else 0):
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        fig.show()
        axs[1].invert_xaxis()
        axs[2].invert_xaxis()

    signal = data_to_signal(data, range)

    if axs[0].lines:
        axs[0].lines[0].set_ydata(signal)
    else:
        axs[0].plot(signal)
    axs[0].set_title("Boot")
    axs[0].set_ylabel("Voltage")
    axs[0].axhline(y=5, color="r", linestyle="-")
    axs[0].relim()
    axs[0].autoscale_view()

    if axs[1].lines:
        axs[1].lines[0].set_data(loads, currents)
    else:
        axs[1].plot(loads, currents)
    axs[1].set_title("Current over Load")
    axs[1].set_xlabel("Load")
    axs[1].set_ylabel("Current")
    axs[1].set_ylim(ymin=0, ymax=6)
    axs[1].relim()
    axs[1].autoscale_view()

    if axs[2].lines:
        axs[2].lines[0].set_data(loads, volts)
    else:
        axs[2].plot(loads, volts)
    axs[2].set_title("Voltage over Load")
    axs[2].set_xlabel("Load")
    axs[2].set_ylabel("Voltage")
    axs[2].axhline(y=5, color="r", linestyle="-")
    axs[2].set_ylim(ymin=0, ymax=6)
    axs[2].relim()
    axs[2].autoscale_view()

    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.001)

def plot3d_update(vin, req_current, vout):
    """Extra live 3D scatter (Vin, Ireq, Vout) without touching your 2D plots."""
    global fig3d, ax3d, xs3d, ys3d, zs3d
    if args.no_3d:
        return
    plt.ion()
    if fig3d is None or not plt.fignum_exists(fig3d.number if fig3d else 0):
        fig3d = plt.figure(figsize=(7,6))
        ax3d = fig3d.add_subplot(111, projection="3d")
        ax3d.set_xlabel("Vin (V)")
        ax3d.set_ylabel("Requested I (A)")
        ax3d.set_zlabel("Vout (V)")
        ax3d.set_title("Live PoE sweep (3D)")
    xs3d.append(vin); ys3d.append(req_current); zs3d.append(vout)
    ax3d.cla()
    ax3d.set_xlabel("Vin (V)")
    ax3d.set_ylabel("Requested I (A)")
    ax3d.set_zlabel("Vout (V)")
    ax3d.set_title("Live PoE sweep (3D)")
    ax3d.scatter(xs3d, ys3d, zs3d)
    fig3d.canvas.draw(); fig3d.canvas.flush_events()
    plt.pause(0.001)

def capture_startup(
    load_threshold=1, load=5, voltage=12,
    load_node_name="load0", scope_node_name="scope0", power_node_name='power0'
):
    """
    Keep your original ordering:
      1) POWER.delete_voltage_dual()
      2) LOAD.set_resistance(load)
      3) SCOPE.start_acquisition(...)
      4) POWER.set_voltage_dual(voltage)
      5) SCOPE.recieve()  (with small retry loop only)
    """
    POWER.delete_voltage_dual()
    LOAD.set_resistance(load)
    SCOPE.start_acquisition(args.samples, args.sample_rate,
                            args.trigger_source, args.trigger_slope,
                            load_threshold, args.trigger_source, args.scope_range, "-")
    POWER.set_voltage_dual(voltage)

    last_err = None
    for attempt in range(1, args.scope_retries + 1):
        try:
            data = SCOPE.recieve()
            return data
        except Exception as e:
            last_err = e
            print(f"[scope recieve() error] attempt {attempt}/{args.scope_retries}: {e}")
            time.sleep(args.scope_retry_delay)
    # If all retries failed, re-raise to be handled by caller (repeat current test point)
    raise last_err if last_err else RuntimeError("scope recieve() failed")

def load_sweep(resistance=5, step=-1, threshold=4.6, node_name="load0"):
    LOAD.set_resistance(resistance)
    voltage = 9999
    while voltage > threshold:
        LOAD.set_resistance(resistance)
        voltage, current = LOAD.get_load_data()
        print(f"# {resistance} Ohm, {voltage} v, {current} l")
        resistance = resistance + step
    return (resistance, voltage)

def callback(voltage):
    resistance, _voltage = load_sweep(8, -0.5, 4.5)
    print(f"# powersupply died at {resistance} ohm producing {_voltage} volts with {voltage} volts as input")

def _inclusive_voltages(vmin, vmax, vstep):
    vals = []
    v = vmin
    # guard floating point drift
    eps = abs(vstep) * 1e-6 + 1e-9
    while v <= vmax + eps:
        vals.append(round(v, 6))
        v += vstep
    # ensure last is exactly vmax
    if abs(vals[-1] - vmax) > eps:
        vals.append(vmax)
    return vals

def sweep_sweep(
    min_voltage=44,
    max_voltage=57,
    voltage_step=1.0,
    voltage_node_name="power0",
    # start at 5A -> ~1.0 Ω
    start_current=5.0,
    resistance_step=0.5,   # increase R -> lower current
    voltage_threshold=4.75,
    resistance_pass=None,  # unused in current algo
    run_until_failure=False,
    run_until_failure_step=-0.05,
    load_node_name="load0",
    scope_node_name="scope0",
):
    LOAD.delete_load()
    POWER.delete_powersupply()

    storage = storageBucket()

    # CSV setup
    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    csv_name = (
        f"poe_sweep_min{min_voltage}_max{max_voltage}_step{voltage_step}"
        f"_Istart{start_current}_Rstep{resistance_step}_Vth{voltage_threshold}{tag}_{ts}.csv"
    ).replace(".", "p")
    csv_path = os.path.join(args.out_dir, csv_name)
    with open(csv_path, "w") as f:
        f.write("vin_v,requested_current_a,measured_current_a,vout_v,resistance_ohm,pass\n")

    # iterate voltages (float-inclusive)
    for voltage in _inclusive_voltages(min_voltage, max_voltage, voltage_step):
        write, get_path = storage.create_folder(voltage)

        # Start at 5A draw => ~1.0 Ω
        if start_current <= 0:
            start_resistance = 9999.0
        else:
            start_resistance = 5.0 / start_current

        resistance = start_resistance
        print(f"# input voltage {voltage} volts")
        POWER.set_voltage_dual(voltage)

        currents = []
        voltages = []
        loads = []

        # We'll lower draw (increase resistance) until it stops failing.
        _voltage = 9999.0
        passed = False

        while True:
            # Acquire boot waveform; if scope burps, REPEAT this resistance point
            try:
                data = capture_startup(load=resistance, voltage=voltage,
                                       load_node_name=load_node_name,
                                       scope_node_name=scope_node_name,
                                       power_node_name=voltage_node_name)
            except Exception as e:
                print(f"[retry same point] Vin={voltage}, R={resistance}Ω due to scope error: {e}")
                # Repeat current test (do not change resistance)
                continue

            write(data, f"resistance_{resistance}_startup.wav")
            _voltage, current = LOAD.get_load_data()
            currents.append(current)
            loads.append(resistance)
            voltages.append(_voltage)
            plot(data, args.scope_range, currents, voltages, loads)
            plot3d_update(voltage, 5.0 / resistance if resistance > 0 else 0.0, _voltage)

            req_current = 5.0 / resistance if resistance > 0 else 0.0
            print(f"# R={resistance:.3f} Ω  =>  Vout={_voltage:.3f} V, I={current:.3f} A (req≈{req_current:.3f} A)")

            # Log CSV
            with open(csv_path, "a") as f:
                f.write(f"{voltage:.4f},{req_current:.4f},{current:.4f},{_voltage:.4f},{resistance:.6f},{int(_voltage>=voltage_threshold)}\n")

            # Stop condition: "lower draw until it stops failing"
            if _voltage >= voltage_threshold:
                print("PASSED! stopping current reduction for this Vin.")
                passed = True
                break

            # Otherwise, lower draw: increase resistance
            # Also stop if requested current would go below minimum
            next_res = resistance + resistance_step
            next_req_i = 5.0 / next_res if next_res > 0 else 0.0
            if next_req_i < args.min_current:
                print(f"Stopping: requested current would drop below {args.min_current} A.")
                break
            resistance = next_res

        print(f"# stopping at {resistance} ohm producing {_voltage} volts {current} amps with {voltage} volts as input")
        write(json.dumps({"currents": currents, "voltages": voltages, "loads": loads}), "data.json", type="ascii")
        plt.savefig(f"{get_path()}/plot.png")
        LOAD.delete_load()

    LOAD.delete_load()
    POWER.delete_powersupply()
    print(f"\nCSV written: {csv_path}")

plt.show(block=False)

# Use CLI params to drive sweep
sweep_sweep(
    min_voltage=args.min_v,
    max_voltage=args.max_v,
    voltage_step=args.v_step,
    voltage_node_name=args.power_node,
    start_current=args.start_current,
    resistance_step=args.resistance_step,
    voltage_threshold=args.vout_threshold,
    load_node_name=args.load_node,
    scope_node_name=args.scope_node,
)
