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

from powersupply import Power
from load import Load
from scope import Scope

namespaces = {
    "nc": "urn:ietf:params:xml:ns:netconf:base:1.0",
    "nd": "urn:ietf:params:xml:ns:yang:ietf-network",
    "nt": "urn:ietf:params:xml:ns:yang:ietf-network-topology",
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    help="Path to the netconf configuration *.xml file defining the configuration according to ietf-networks, ietf-networks-topology and netconf-node models e.g. ../networks.xml",
)
args = parser.parse_args()

tree = etree.parse(args.config)
network = tree.xpath("/nc:config/nd:networks/nd:network", namespaces=namespaces)[0]

conns = tntapi.network_connect(network)
yconns = tntapi.network_connect_yangrpc(network)

# GLOBAL instrument instances (node names fixed for this run)
POWER = Power(yconns, conns, node_name="raspberrypi")
LOAD = Load(yconns, conns, network, node_name="raspberrypi")
SCOPE = Scope(yconns, conns, node_name="raspberrypi")

# Initialize figure and axes globally
fig, axs = None, None

def data_to_signal(data, range=16):
    f = io.BytesIO(data)
    spf = wave.open(f, "r")
    signal = spf.readframes(-1)
    signal = np.frombuffer(signal, np.uint8)
    def map(i):
        return ((i - 128) / 100) * 16
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

def capture_startup(
    load_threshold=1, load=5, voltage=12,
    load_node_name="load0", scope_node_name="scope0", power_node_name='power0'
):
    POWER.delete_voltage_dual()
    LOAD.set_resistance(load)
    SCOPE.start_acquisition(1000, 20000, "ch1", "positive", load_threshold, "ch1", 16, "-")
    POWER.set_voltage_dual(voltage)
    data = SCOPE.recieve()
    return data

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
    LOAD.delete_load()
    POWER.delete_powersupply()

    _voltage = 9999
    storage = storageBucket()

    for voltage in range(min_voltage, max_voltage, voltage_step):
        write, get_path = storage.create_folder(voltage)

        resistance = start_resistance
        print(f"# input voltage {voltage} volts")
        POWER.set_voltage_dual(voltage)

        currents = []
        voltages = []
        loads = []

        while _voltage > voltage_threshold:
            data = capture_startup(load=resistance, voltage=voltage,
                                   load_node_name=load_node_name,
                                   scope_node_name=scope_node_name,
                                   power_node_name=voltage_node_name)

            write(data, f"resistance_{resistance}_startup.wav")
            _voltage, current = LOAD.get_load_data()
            currents.append(current)
            loads.append(resistance)
            voltages.append(_voltage)
            plot(data, 16, currents, voltages, loads)
            print(f"# {resistance} Ohm, {_voltage} v, {current} l")
            if resistance <= resistance_pass:
                if run_until_failure:
                    resistance = resistance + run_until_failure_step
                else:
                    print("PASSED!")
                    break
            else:
                resistance = resistance + resistance_step

        print(f"# stopping at {resistance} ohm producing {_voltage} volts {current} amps with {voltage} volts as input")
        write(json.dumps({"currents": currents, "voltages": voltages, "loads": loads}), "data.json", type="ascii")
        plt.savefig(f"{get_path()}/plot.png")
        _voltage = 9999
        LOAD.delete_load()

    LOAD.delete_load()
    POWER.delete_powersupply()

plt.show(block=False)
sweep_sweep(voltage_node_name='raspberrypi', load_node_name='raspberrypi', scope_node_name='raspberrypi')
