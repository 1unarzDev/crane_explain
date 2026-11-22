"""Scenario format whose load() ignores the file: one stub scenario per seed (the task config is the experiment)."""
import json
import re
from pathlib import Path

from fastsim.scenario import JsonScenarioFormat


class StubFormat(JsonScenarioFormat):
    def load(self, path):
        m = re.search(r"s(\d+)", Path(path).stem)
        s = int(m.group(1)) if m else 0
        stub = {"name": "ep-%d" % s, "start": [0, 0, 0], "elements": [],
                "identity": {"pair_key": "ep-%d" % s, "truth_sha256": "stub"}}
        return self.parse(json.dumps(stub).encode())
