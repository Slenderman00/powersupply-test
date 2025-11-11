import tntapi
from yangcli import yangcli

class Power:
    def __init__(self, conns, yconns, node_name="power0"):
        self.conns = conns
        self.yconns = yconns
        self.node_name = node_name

    def delete_powersupply(self):
        yangcli(self.yconns[self.node_name], "delete /outputs").xpath("./ok")
        tntapi.network_commit(self.conns)

    def set_voltage(self, voltage, current_limit=0.5, output="out1"):
        yangcli(
            self.yconns[self.node_name],
            "replace /outputs/output[name='%s'] -- voltage-level=%.9f current-limit=%.9f"
            % (output, float(voltage), float(current_limit)),
        ).xpath("./ok")
        tntapi.network_commit(self.conns)

    def delete_voltage_dual(self):
        yangcli(self.yconns[self.node_name], "delete /outputs/output[name='out1']").xpath("./ok")
        yangcli(self.yconns[self.node_name], "delete /outputs/output[name='out2']").xpath("./ok")
        tntapi.network_commit(self.conns)

    def set_voltage_dual(self, voltage, current_limit=0.5):
        for out in ("out1", "out2"):
            yangcli(
                self.yconns[self.node_name],
                "replace /outputs/output[name='%s'] -- voltage-level=%.9f current-limit=%.9f"
                % (out, float(voltage), float(current_limit)),
            ).xpath("./ok")
        tntapi.network_commit(self.conns)

    def voltage_sweep(self, callback, vmin=10, vmax=48, step=1):
        for v in range(int(vmin), int(vmax), int(step)):
            print(f"# input voltage {v} volts")
            self.set_voltage(v)
            callback(v)
