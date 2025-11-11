import tntapi
from yangcli import yangcli

class Load:
    def __init__(self, conns, yconns, network, node_name="load0"):
        self.yconns = yconns
        self.conns = conns
        self.network = network
        self.node_name = node_name

def delete_load(node_name="load0"):
    yangcli(self.yconns[node_name], """delete /lsi-ivi-load:load""").xpath("./ok")
    tntapi.network_commit(self.conns)


def set_resistance(self, resistance):
    yangcli(
        self.conns[self.node_name],
        f"replace /lsi-ivi-load:load/channel[name='out1'] -- resistance={format(resistance, '.9f')} transient-frequency=0",
    ).xpath("./ok")
    tntapi.network_commit(self.conns)


def get_load_data(self):
    state = tntapi.network_get_state(
        self.network,
        self.conns,
        filter="""<filter type="xpath" select="/*[local-name()='load' or local-name()='load-state']/channel"/>""",
    )
    state_wo_ns = tntapi.strip_namespaces(state)
    voltage = state_wo_ns.xpath(
        "node[node-id='%s']/data/load-state/channel[name='%s']/measurement/voltage"
        % (self.node_name, "out1")
    )[0].text
    voltage = float(voltage)
    current = state_wo_ns.xpath(
        "node[node-id='%s']/data/load-state/channel[name='%s']/measurement/current"
        % (node_name, "out1")
    )[0].text
    current = float(current)

    return voltage, current
