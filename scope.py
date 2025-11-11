from lxml import etree
import sys
import tntapi
from yangcli import yangcli
import base64
import matplotlib.pyplot as plt
import numpy as np
import io
import wave


class Scope:
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

        result = self.conns[self.node_name].rpc(rpc_xml_str % {"filter": filter})
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
        signal = np.frombuffer(signal, np.uint8)

        # Signal from 50 to - 50

        # ((Signal + 50) / 100) * range

        # print(f"# {etree.tostring(data[0])}")
        def map(i):
            return ((i) / 100)
 
        applyall = np.vectorize(map)
        signal = applyall(signal)

        plt.plot(signal)
        plt.ylabel("some numbers")

        # Redraw the plot
        plt.draw()

        # Pause for a short duration to allow visualization
        plt.pause(0.001)

        return data
