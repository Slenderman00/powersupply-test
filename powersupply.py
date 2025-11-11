import tntapi
from yangcli import yangcli


class Power:
    def __init__(self, yconns, conns, node_name="power0"):
        self.yconns = yconns
        self.conns = conns
        self.node_name = node_name

    def delete_powersupply(self):
        yangcli(self.yconns[self.node_name], """delete /outputs""").xpath("./ok")
        tntapi.network_commit(self.conns)


    def set_voltage(self, voltage):
        yangcli(
            self.yconns[self.node_name],
            """replace /outputs/output[name='out1'] -- voltage-level=%.9f current-limit=%.9f"""
            % (voltage, 0.5),
        ).xpath("./ok")
        tntapi.network_commit(self.conns)


    def delete_voltage_dual(self):
        yangcli(self.yconns[self.node_name], "delete /outputs/output[name='out1']").xpath("./ok")
        yangcli(self.yconns[self.node_name], "delete /outputs/output[name='out2']").xpath("./ok")
        tntapi.network_commit(self.conns)


    def set_voltage_dual(self, voltage):
        yangcli(
            self.yconns[self.node_name],
            """replace /outputs/output[name='out1'] -- voltage-level=%.9f current-limit=%.9f"""
            % (voltage, 0.5),
        ).xpath("./ok")
        yangcli(
            self.yconns[self.node_name],
            """replace /outputs/output[name='out2'] -- voltage-level=%.9f current-limit=%.9f"""
            % (voltage, 0.5),
        ).xpath("./ok")
        tntapi.network_commit(self.conns)


    def voltage_sweep(self, callback, min=10, max=48, step=1, node_name="power0"):
        for voltage in range(min, max, step):
            print(f"# input voltage {voltage} volts")
            set_voltage(voltage)
            callback(voltage)
