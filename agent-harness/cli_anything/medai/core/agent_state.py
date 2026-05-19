from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from .json_utils import write_json


class AgentState:
    def __init__(self, patient_id: str, output_folder: str | Path, config: dict | None = None):
        self.patient_id = patient_id
        self.output_folder = Path(output_folder).resolve()
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.state_path = self.output_folder / "agent_state.json"
        self.data = {"patient_id": patient_id, "created_at": datetime.now(timezone.utc).isoformat(), "config": config or {}, "events": [], "summary": None}
        self.save()

    def add_event(self, step: str, status: str, details: dict | None = None, scan_id: str | None = None):
        ev = {"time": datetime.now(timezone.utc).isoformat(), "step": step, "status": status, "scan_id": scan_id, "details": details or {}}
        self.data["events"].append(ev)
        self.save()
        return ev

    def set_summary(self, summary: dict, status: str | None = None):
        self.data["summary"] = summary
        if status: self.data["status"] = status
        self.save()

    def save(self):
        write_json(self.state_path, self.data)
        return str(self.state_path)
