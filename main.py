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

namespaces = {
    "nc": "urn:ietf:params:xml:ns:netconf:base:1.0",
    "nd": "urn:ietf:params:xml:ns:yang:ietf-network",
    "nt": "urn:ietf:params:xml:ns:yang:ietf-network-topology",
}

global args
args = None

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    help="Path to the netconf configuration *.xml file defining the configuration according to ietf-networks, ietf-networks-topology and netconf-node models e.g. ../networks.xml",
)

args = parser.parse_args()

tree = etree.parse(args.config)
network = tree.xpath("/nc:config/nd:networks/nd:network", namespaces=namespaces)[0]

#conns_notification = tntapi.network_connect(network)
conns = tntapi.network_connect(network)
yconns = tntapi.network_connect_yangrpc(network)


def delete_powersupply(node_name="power0"):
    yangcli(yconns[node_name], """delete /outputs""").xpath("./ok")
    tntapi.network_commit(conns)


def delete_load(node_name="load0"):
    yangcli(yconns[node_name], """delete /lsi-ivi-load:load""").xpath("./ok")
    tntapi.network_commit(conns)


def set_voltage(voltage, node_name="power0"):
    yangcli(
        yconns[node_name],
        """replace /outputs/output[name='out1'] -- voltage-level=%.9f current-limit=%.9f"""
        % (voltage, 0.5),
    ).xpath("./ok")
    tntapi.network_commit(conns)


def delete_voltage_dual(node_name="power0"):
    yangcli(yconns[node_name], "delete /outputs/output[name='out1']").xpath("./ok")
    yangcli(yconns[node_name], "delete /outputs/output[name='out2']").xpath("./ok")
    tntapi.network_commit(conns)


def set_voltage_dual(voltage, node_name="power0"):
    yangcli(
        yconns[node_name],
        """replace /outputs/output[name='out1'] -- voltage-level=%.9f current-limit=%.9f"""
        % (voltage, 0.5),
    ).xpath("./ok")
    yangcli(
        yconns[node_name],
        """replace /outputs/output[name='out2'] -- voltage-level=%.9f current-limit=%.9f"""
        % (voltage, 0.5),
    ).xpath("./ok")
    tntapi.network_commit(conns)


def voltage_sweep(callback, min=10, max=48, step=1, node_name="power0"):
    for voltage in range(min, max, step):
        print(f"# input voltage {voltage} volts")
        set_voltage(voltage)
        callback(voltage)


def set_resistance(resistance, node_name="load0"):
    yangcli(
        yconns[node_name],
        f"replace /lsi-ivi-load:load/channel[name='out1'] -- resistance={format(resistance, '.9f')}",
    ).xpath("./ok")
    tntapi.network_commit(conns)


def get_load_data(node_name="load0"):
    state = tntapi.network_get_state(
        network,
        conns,
        filter="""<filter type="xpath" select="/*[local-name()='load' or local-name()='load-state']/channel"/>""",
    )
    state_wo_ns = tntapi.strip_namespaces(state)
    voltage = state_wo_ns.xpath(
        "node[node-id='%s']/data/load-state/channel[name='%s']/measurement/voltage"
        % (node_name, "out2")
    )[0].text
    voltage = float(voltage)
    current = state_wo_ns.xpath(
        "node[node-id='%s']/data/load-state/channel[name='%s']/measurement/current"
        % (node_name, "out2")
    )[0].text
    current = float(current)

    return voltage, current


class scope:
    def __init__(self, yconns, conns, node_name):
        self.yconns = yconns
        self.conns = conns
        self.node_name = node_name
        self.scope_channel_name = "ch1"

        self.scope_subscribe()

    def scope_subscribe(self):
        filter = """
        <filter xmlns="urn:ietf:params:xml:ns:netconf:notification:1.0" xmlns:netconf="urn:ietf:params:xml:ns:netconf:base:1.0" netconf:type="subtree">
        <acquisition-complete xmlns="urn:lsi:params:xml:ns:yang:ivi-scope"/>
        </filter>"""

        rpc_xml_str = """
        <create-subscription xmlns="urn:ietf:params:xml:ns:netconf:notification:1.0">
        %(filter)s
        </create-subscription>
        """

        print(rpc_xml_str % {"filter": filter})

        result = conns[self.node_name].rpc(rpc_xml_str % {"filter": filter})
        rpc_error = result.xpath("rpc-error")
        assert len(rpc_error) == 0

    def start_acquisition(
        self,
        samples=1000,
        sample_rate=100000,
        scope_trigger_source=None,
        scope_trigger_slope=None,
        scope_trigger_level=0,
        scope_channel_name="ch1",
        scope_channel_range=16,
        scope_channel_parameters="-",
    ):
        yangcli(self.yconns[self.node_name], """delete /acquisition""")
        tntapi.network_commit(self.conns)

        ok = yangcli(
            self.yconns[self.node_name],
            """merge /acquisition -- samples=%d sample-rate=%d"""
            % (float(samples), float(sample_rate)),
        ).xpath("./ok")
        assert len(ok) == 1

        if scope_trigger_source is not None and scope_trigger_slope is not None:
            ok = yangcli(
                self.yconns[self.node_name],
                """merge /acquisition/trigger -- source=%s level=%f slope=%s"""
                % (
                    scope_trigger_source,
                    float(scope_trigger_level),
                    scope_trigger_slope,
                ),
            ).xpath("./ok")
            assert len(ok) == 1

        ok = yangcli(
            self.yconns[self.node_name],
            """merge /acquisition/channels/channel[name='%s'] -- range=%f parameters='%s'"""
            % (
                scope_channel_name,
                float(scope_channel_range),
                scope_channel_parameters,
            ),
        ).xpath("./ok")
        assert len(ok) == 1

        self.scope_channel_name = scope_channel_name

        tntapi.network_commit(conns)

    def recieve(self):
        while 1:
            (notification_xml, ret) = conns[self.node_name].receive()
            # print(f"# {notification_xml}")
            if ret != 1:  # timeout
                break
        if notification_xml is None:
            print("[FAILED] Receiving <acquisition-complete> notification")
            sys.exit(-1)

        result = yangcli(
            self.yconns[self.node_name],
            """xget /acquisition/channels/channel[name='%s']"""
            % (self.scope_channel_name),
        )

        print(etree.tostring(result))
        data = result.xpath("./data/acquisition/channels/channel/data")

        data = data[0].text
        data = base64.b64decode(data)

        return data

    def run_plot(
        self,
        samples=1000,
        sample_rate=100000,
        scope_trigger_source=None,
        scope_trigger_slope=None,
        scope_trigger_level=0,
        scope_channel_name="ch1",
        scope_channel_range=16,
        scope_channel_parameters="-",
    ):

        self.start_acquisition(
            samples,
            sample_rate,
            scope_trigger_source,
            scope_trigger_slope,
            scope_trigger_level,
            scope_channel_name,
            scope_channel_range,
            scope_channel_parameters,
        )

        data = self.recieve()

        f2 = open("signal.wav", "wb")
        f2.write(data)

        f = io.BytesIO(data)

        spf = wave.open(f, "r")

        # Extract Raw Audio from Wav File
        signal = spf.readframes(-1)
        signal = np.frombuffer(signal, np.int8)

        # Signal from 50 to - 50

        # ((Signal + 50) / 100) * range

        # print(f"# {etree.tostring(data[0])}")
        def map(i):
            return ((i) / 100) * scope_channel_range

        applyall = np.vectorize(map)
        signal = applyall(signal)

        plt.plot(signal)
        plt.ylabel("some numbers")

        # Redraw the plot
        plt.draw()

        # Pause for a short duration to allow visualization
        plt.pause(0.001)

        return data


# Initialize figure and axes globally
fig, axs = None, None


def data_to_signal(data, range=16):
    f = io.BytesIO(data)
    spf = wave.open(f, "r")

    # Extract Raw Audio from Wav File
    signal = spf.readframes(-1)
    signal = np.frombuffer(signal, np.int8)

    def map(i):
        return ((i) / 100) * range

    applyall = np.vectorize(map)
    signal = applyall(signal)

    return signal


def get_average_voltage(data):
    signal = data_to_signal(data)
    # mid = len(signal) // 2
    half = signal[800:]
    # print(f'# {half}')
    voltage = np.average(half)

    return voltage


def plot(data, range, currents, volts, loads):
    global fig, axs

    plt.ion()

    # Create figure and axes if they don't exist
    if fig is None or not plt.fignum_exists(fig.number):
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        fig.show()

        axs[1].invert_xaxis()
        axs[2].invert_xaxis()

    signal = data_to_signal(data, range)

    # Update the boot signal plot
    if axs[0].lines:
        axs[0].lines[0].set_ydata(signal)
    else:
        axs[0].plot(signal)
    axs[0].set_title("Boot")
    axs[0].set_ylabel("Voltage")
    axs[0].axhline(y=5, color="r", linestyle="-")
    axs[0].set_ylim(ymin=0, ymax=6)
    axs[0].relim()
    axs[0].autoscale_view()

    # Update the current over load plot
    if axs[1].lines:
        axs[1].lines[0].set_data(loads, currents)
    else:
        axs[1].plot(loads, currents)
    axs[1].set_title("Current over Load")
    axs[1].set_xlabel("Load")
    axs[1].set_ylabel("Current")
    axs[1].set_ylim(ymin=0, ymax=3)
    axs[1].relim()
    axs[1].autoscale_view()

    # Update the voltage over load plot
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

    # Redraw the plots
    fig.canvas.draw()
    fig.canvas.flush_events()

    # plt.ioff()
    plt.pause(0.001)


def capture_startup(
    load_threshold=1, load=5, voltage=12, load_node_name="load0", scope_node_name="scope0", power_node_name='power0'
):
    s1 = scope(yconns, conns, scope_node_name)
    # delete_load(load_node_name)
    delete_voltage_dual(power_node_name)
    set_resistance(load, load_node_name)
    #time.sleep(5)
    s1.start_acquisition(1000, 20000, "ch1", "positive", load_threshold, "ch1", 16, "-")
    set_voltage_dual(voltage, power_node_name)
    data = s1.recieve()
    return data


def load_sweep(resistance=5, step=-1, threshold=4.6, node_name="load0"):
    set_resistance(resistance, node_name=node_name)

    voltage = 9999
    while voltage > threshold:
        set_resistance(resistance, node_name=node_name)

        voltage, current = get_load_data(node_name)

        print(f"# {resistance} Ohm, {voltage} v, {current} l")

        resistance = resistance + step

    return (resistance, voltage)


def callback(voltage):
    resistance, _voltage = load_sweep(8, -0.5, 4.5)
    print(
        f"# powersupply died at {resistance} ohm producing {_voltage} volts with {voltage} volts as input"
    )


# def transient_load(voltage_node_name="power0", load_node_name="load0", scope_node_name="scope0"):


def sweep_sweep(
    min_voltage=48,
    max_voltage=50,
    voltage_step=2,
    voltage_node_name="power0",
    start_resistance=2,
    resistance_step=-0.5,
    voltage_threshold=4.6,
    resistance_pass=2,
    run_until_failure=False,
    run_until_failure_step=-0.05,
    load_node_name="load0",
    scope_node_name="scope0",

):
    delete_load(load_node_name)
    delete_powersupply(node_name=voltage_node_name)

    _voltage = 9999
    storage = storageBucket()

    for voltage in range(min_voltage, max_voltage, voltage_step):
        write, get_path = storage.create_folder(voltage)

        resistance = start_resistance
        print(f"# input voltage {voltage} volts")
        # set_voltage_dual(voltage, node_name=voltage_node_name)

        currents = []
        voltages = []
        loads = []

        while _voltage > voltage_threshold:
            # set_resistance(retsistance, node_name=resistance_node_name)
            data = capture_startup(load=resistance, voltage=voltage, load_node_name=load_node_name, scope_node_name=scope_node_name, power_node_name=voltage_node_name)

            write(data, f"resistance_{resistance}_startup.wav")
            # _voltage, _ = get_load_data(load_node_name)
            _voltage = get_average_voltage(data)
            current = _voltage / resistance
            currents.append(current)
            loads.append(resistance)
            voltages.append(_voltage)
            plot(data, 16, currents, voltages, loads)
            # plt.savefig('test.png')
            # s1.run_plot()
            print(f"# {resistance} Ohm, {_voltage} v, {current} l")
            if resistance <= resistance_pass:
                if run_until_failure:
                    resistance = resistance + run_until_failure_step
                else:
                    print("PASSED!")
                    break
            else:
                resistance = resistance + resistance_step
        print(
            f"# stopping at {resistance} ohm producing {_voltage} volts {current} amps with {voltage} volts as input"
        )
        write(
            json.dumps({"currents": currents, "voltages": voltages, "loads": loads}),
            "data.json",
            type="ascii",
        )
        plt.savefig(f"{get_path()}/plot.png")
        _voltage = 9999
        delete_load(load_node_name)

    delete_load(load_node_name)
    delete_powersupply(voltage_node_name)


plt.show(block=False)

# scope()
sweep_sweep(voltage_node_name='raspberrypi', load_node_name='raspberrypi', scope_node_name='raspberrypi')
