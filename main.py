#!/usr/bin/python

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

from scope import Scope
from powersupply import Power
from load import Load

namespaces = {
    "nc": "urn:ietf:params:xml:ns:netconf:base:1.0",
    "nd": "urn:ietf:params:xml:ns:yang:ietf-network",
    "nt": "urn:ietf:params:xml:ns:yang:ietf-network-topology",
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    help="Path to the netconf configuration *.xml file defining the configuration according to ietf-networks, ietf-networks-topology and netconf-node models e.g. ../networks.xml",
    required=True,
)
args = parser.parse_args()

tree = etree.parse(args.config)
network = tree.xpath("/nc:config/nd:networks/nd:network", namespaces=namespaces)[0]

# Connect once and share with class instances
conns = tntapi.network_connect(network)
yconns = tntapi.network_connect_yangrpc(network)

fig, axs = None, None

def data_to_signal(data, range=16):
    f = io.BytesIO(data)
    spf = wave.open(f, "r")

    # Extract Raw Audio from Wav File
    signal = spf.readframes(-1)
    signal = np.frombuffer(signal, np.uint8)

    def _map(i):
        # map 8-bit unsigned to approx ±range/?
        return ((i - 128) / 100) * range

    applyall = np.vectorize(_map)
    signal = applyall(signal)
    return signal

def get_average_voltage(data):
    signal = data_to_signal(data)
    half = signal[800:]
    voltage = np.average(half)
    return voltage

def plot(data, range, currents, volts, loads):
    global fig, axs

    plt.ion()

    # Create figure and axes if they don't exist
    if fig is None or not plt.fignum_exists(fig.number if fig else 0):
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        fig.show()
        axs[1].invert_xaxis()
        axs[2].invert_xaxis()

    signal = data_to_signal(data, range)

    # Boot signal plot
    if axs[0].lines:
        axs[0].lines[0].set_ydata(signal)
    else:
        axs[0].plot(signal)
    axs[0].set_title("Boot")
    axs[0].set_ylabel("Voltage")
    axs[0].axhline(y=5, color="r", linestyle="-")
    axs[0].relim()
    axs[0].autoscale_view()

    # Current vs Load
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

    # Voltage vs Load
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

# ---------------------------------
# Class-based measurement primitives
# ---------------------------------
def capture_startup(
    pwr: Power,
    load: Load,
    scp: Scope,
    load_threshold=1,
    load_resistance=5,
    voltage=12,
):
    """
    Arms scope, sets load, then enables supply to capture startup waveform.
    """
    # Ensure outputs off then set load
    pwr.delete_voltage_dual()
    load.set_resistance(load_resistance)

    # Arm scope then power on
    scp.start_acquisition(
        samples=1000,
        sample_rate=20000,
        scope_trigger_source="ch1",
        scope_trigger_slope="positive",
        scope_trigger_level=load_threshold,
        scope_channel_name="ch1",
        scope_channel_range=16,
        scope_channel_parameters="-",
    )
    pwr.set_voltage_dual(voltage)
    data = scp.recieve()
    return data

def load_sweep(load: Load, start_resistance=5, step=-1, threshold=4.6):
    """
    Decrease resistance until output voltage drops below threshold.
    """
    resistance = start_resistance
    voltage = 9999.0

    load.set_resistance(resistance)
    while voltage > threshold:
        load.set_resistance(resistance)
        voltage, current = load.get_load_data()
        print(f"# {resistance} Ohm, {voltage} v, {current} A")
        resistance = resistance + step

    return resistance, voltage

# --------------
# Main sweep run
# --------------
def sweep_sweep(
    min_voltage=44,
    max_voltage=57,
    voltage_step=1,
    voltage_node_name="power0",
    start_resistance=5,
    resistance_step=-0.5,
    voltage_threshold=4,
    resistance_pass=0.5,
    run_until_failure=False,
    run_until_failure_step=-0.05,
    load_node_name="load0",
    scope_node_name="scope0",
):
    # Instantiate instruments for the requested nodes
    pwr = Power(conns, yconns, node_name=voltage_node_name)
    load = Load(conns, yconns, network, node_name=load_node_name)
    scp = Scope(yconns, conns, node_name=scope_node_name)

    # Fresh state
    load.delete_load()
    pwr.delete_powersupply()

    _voltage = 9999.0
    storage = storageBucket()

    for vin in range(min_voltage, max_voltage, voltage_step):
        write, get_path = storage.create_folder(vin)

        resistance = start_resistance
        print(f"# input voltage {vin} volts")
        pwr.set_voltage_dual(vin)

        currents = []
        voltages = []
        loads = []

        while _voltage > voltage_threshold:
            # Capture startup at this resistance
            data = capture_startup(
                pwr=pwr,
                load=load,
                scp=scp,
                load_threshold=1,
                load_resistance=resistance,
                voltage=vin,
            )

            # Save waveform
            write(data, f"resistance_{resistance}_startup.wav")

            # Read steady-state from load
            _voltage, current = load.get_load_data()

            currents.append(current)
            loads.append(resistance)
            voltages.append(_voltage)
            plot(data, 16, currents, voltages, loads)

            print(f"# {resistance} Ohm, {_voltage} v, {current} A")
            if resistance <= resistance_pass:
                if run_until_failure:
                    resistance = resistance + run_until_failure_step
                else:
                    print("PASSED!")
                    break
            else:
                resistance = resistance + resistance_step

        print(
            f"# stopping at {resistance} ohm producing {_voltage} volts {current} amps with {vin} volts as input"
        )
        write(
            json.dumps({"currents": currents, "voltages": voltages, "loads": loads}),
            "data.json",
            type="ascii",
        )
        plt.savefig(f"{get_path()}/plot.png")

        # Reset for next VIN
        _voltage = 9999.0
        load.delete_load()

    # Cleanup
    load.delete_load()
    pwr.delete_powersupply()

plt.show(block=False)
sweep_sweep(voltage_node_name='raspberrypi', load_node_name='raspberrypi', scope_node_name='raspberrypi')
