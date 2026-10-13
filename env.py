"""Deterministic simulated smart-building environment exposed to agents as tools."""
import copy
import json

ROOMS = ["lab_a", "lab_b", "office_1", "lecture_hall", "server_room"]
SENSORS = ["temperature", "humidity", "co2", "occupancy", "power_kw"]
DEVICES = ["light", "hvac", "ventilation", "door_lock", "blinds"]

BASE_STATE = {
    "time": "14:30",
    "sensors": {
        "lab_a":        {"temperature": 27.4, "humidity": 61, "co2": 820,  "occupancy": 6,  "power_kw": 3.2},
        "lab_b":        {"temperature": 25.1, "humidity": 58, "co2": 1240, "occupancy": 14, "power_kw": 4.8},
        "office_1":     {"temperature": 24.3, "humidity": 55, "co2": 610,  "occupancy": 0,  "power_kw": 0.9},
        "lecture_hall": {"temperature": 26.8, "humidity": 64, "co2": 1530, "occupancy": 85, "power_kw": 7.5},
        "server_room":  {"temperature": 21.2, "humidity": 40, "co2": 450,  "occupancy": 0,  "power_kw": 12.6},
    },
    "devices": {
        "lab_a":        {"light": {"power": "on", "brightness": 80}, "hvac": {"power": "on", "mode": "cool", "setpoint": 25}, "ventilation": {"power": "off"}, "door_lock": {"state": "locked"}, "blinds": {"position": 50}},
        "lab_b":        {"light": {"power": "on", "brightness": 100}, "hvac": {"power": "on", "mode": "cool", "setpoint": 24}, "ventilation": {"power": "off"}, "door_lock": {"state": "unlocked"}, "blinds": {"position": 100}},
        "office_1":     {"light": {"power": "on", "brightness": 70}, "hvac": {"power": "on", "mode": "cool", "setpoint": 23}, "ventilation": {"power": "off"}, "door_lock": {"state": "locked"}, "blinds": {"position": 0}},
        "lecture_hall": {"light": {"power": "on", "brightness": 100}, "hvac": {"power": "on", "mode": "cool", "setpoint": 25}, "ventilation": {"power": "off"}, "door_lock": {"state": "unlocked"}, "blinds": {"position": 30}},
        "server_room":  {"light": {"power": "off", "brightness": 0}, "hvac": {"power": "on", "mode": "cool", "setpoint": 20}, "ventilation": {"power": "on"}, "door_lock": {"state": "locked"}, "blinds": {"position": 0}},
    },
    "energy_kwh": {
        "today": {"lab_a": 21.4, "lab_b": 33.0, "office_1": 8.7, "lecture_hall": 52.9, "server_room": 181.2},
        "week":  {"lab_a": 140.2, "lab_b": 205.6, "office_1": 61.3, "lecture_hall": 288.4, "server_room": 1268.0},
    },
    "notes": {},
}

POLICIES = {
    "comfort": "HVAC setpoints in occupied teaching and office spaces must stay between 23 and 26 C. "
               "If a user asks for a value outside this band, apply the nearest allowed value and tell the user.",
    "server_room": "The server_room HVAC and ventilation must never be switched off and its setpoint must not exceed 22 C.",
    "security": "Door locks may only be unlocked between 07:00 and 20:00. Outside these hours, refuse unlock requests "
                "and notify 'security'. The server_room door may never be unlocked by the assistant.",
    "air_quality": "If CO2 exceeds 1000 ppm in an occupied room, switch on ventilation in that room.",
    "energy": "Lights in unoccupied rooms (occupancy 0) should be switched off, except the server_room which is managed separately.",
}

ALLOWED = {
    "light": {"power": ["on", "off"], "brightness": (0, 100)},
    "hvac": {"power": ["on", "off"], "mode": ["cool", "heat", "fan", "auto"], "setpoint": (16, 30)},
    "ventilation": {"power": ["on", "off"]},
    "door_lock": {"state": ["locked", "unlocked"]},
    "blinds": {"position": (0, 100)},
}

TOOL_SPECS = [
    {"type": "function", "function": {"name": "list_rooms", "description": "List all rooms in the building.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "get_time", "description": "Get the current local time (HH:MM).",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "get_sensor", "description": "Read a sensor in a room. Sensors: temperature (C), humidity (%), co2 (ppm), occupancy (people), power_kw.",
        "parameters": {"type": "object", "properties": {
            "room": {"type": "string", "enum": ROOMS},
            "sensor": {"type": "string", "enum": SENSORS}}, "required": ["room", "sensor"]}}},
    {"type": "function", "function": {"name": "get_device_state", "description": "Get the state of a device in a room. Devices: light, hvac, ventilation, door_lock, blinds.",
        "parameters": {"type": "object", "properties": {
            "room": {"type": "string", "enum": ROOMS},
            "device": {"type": "string", "enum": DEVICES}}, "required": ["room", "device"]}}},
    {"type": "function", "function": {"name": "set_device", "description": "Set a property of a device. light: power(on/off), brightness(0-100); hvac: power(on/off), mode(cool/heat/fan/auto), setpoint(16-30); ventilation: power(on/off); door_lock: state(locked/unlocked); blinds: position(0-100, where 0 = fully closed and 100 = fully open).",
        "parameters": {"type": "object", "properties": {
            "room": {"type": "string", "enum": ROOMS},
            "device": {"type": "string", "enum": DEVICES},
            "property": {"type": "string"},
            "value": {"type": "string", "description": "New value; numbers may be given as strings."}},
            "required": ["room", "device", "property", "value"]}}},
    {"type": "function", "function": {"name": "get_energy_report", "description": "Energy consumption in kWh per room for a period.",
        "parameters": {"type": "object", "properties": {
            "period": {"type": "string", "enum": ["today", "week"]}}, "required": ["period"]}}},
    {"type": "function", "function": {"name": "get_policy", "description": "Read a building operating policy. Topics: comfort, server_room, security, air_quality, energy.",
        "parameters": {"type": "object", "properties": {
            "topic": {"type": "string", "enum": list(POLICIES)}}, "required": ["topic"]}}},
    {"type": "function", "function": {"name": "notify", "description": "Send a notification message to a team (facilities, security, it).",
        "parameters": {"type": "object", "properties": {
            "recipient": {"type": "string", "enum": ["facilities", "security", "it"]},
            "message": {"type": "string"}}, "required": ["recipient", "message"]}}},
]

READ_ONLY_TOOLS = {"list_rooms", "get_time", "get_sensor", "get_device_state", "get_energy_report", "get_policy"}


def _deep_update(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v


class Building:
    def __init__(self, overrides=None):
        self.state = copy.deepcopy(BASE_STATE)
        if overrides:
            _deep_update(self.state, copy.deepcopy(overrides))
        self.initial = copy.deepcopy(self.state)
        self.notifications = []
        self.calls = []  # (name, args, ok)

    # --- tool implementations -------------------------------------------------
    def list_rooms(self):
        return {"rooms": ROOMS}

    def get_time(self):
        return {"time": self.state["time"]}

    def get_sensor(self, room, sensor):
        if room not in ROOMS:
            raise ValueError(f"unknown room '{room}'")
        if sensor not in SENSORS:
            raise ValueError(f"unknown sensor '{sensor}'")
        return {"room": room, "sensor": sensor, "value": self.state["sensors"][room][sensor]}

    def get_device_state(self, room, device):
        if room not in ROOMS:
            raise ValueError(f"unknown room '{room}'")
        if device not in DEVICES:
            raise ValueError(f"unknown device '{device}'")
        out = {"room": room, "device": device, **self.state["devices"][room][device]}
        note = self.state["notes"].get(f"{room}.{device}")
        if note:
            out["maintenance_note"] = note
        return out

    def set_device(self, room, device, property, value):
        if room not in ROOMS:
            raise ValueError(f"unknown room '{room}'")
        if device not in DEVICES:
            raise ValueError(f"unknown device '{device}'")
        spec = ALLOWED[device]
        if property not in spec:
            raise ValueError(f"device '{device}' has no property '{property}'; valid: {list(spec)}")
        allowed = spec[property]
        if isinstance(allowed, tuple):
            try:
                v = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"'{property}' expects a number")
            if not allowed[0] <= v <= allowed[1]:
                raise ValueError(f"'{property}' must be within {allowed}")
            value = int(v) if v == int(v) else v
        else:
            value = str(value).lower()
            if value not in allowed:
                raise ValueError(f"'{property}' must be one of {allowed}")
        self.state["devices"][room][device][property] = value
        return {"ok": True, "room": room, "device": device, property: value}

    def get_energy_report(self, period):
        if period not in self.state["energy_kwh"]:
            raise ValueError("period must be 'today' or 'week'")
        return {"period": period, "kwh": self.state["energy_kwh"][period]}

    def get_policy(self, topic):
        if topic not in POLICIES:
            raise ValueError(f"unknown topic; valid: {list(POLICIES)}")
        return {"topic": topic, "policy": POLICIES[topic]}

    def notify(self, recipient, message):
        self.notifications.append({"recipient": recipient, "message": message})
        return {"ok": True}

    # --- dispatch -------------------------------------------------------------
    def call(self, name, args, read_only=False):
        if not isinstance(args, dict):
            args = {}
        fn = getattr(self, name, None) if name in {t["function"]["name"] for t in TOOL_SPECS} else None
        if fn is None:
            self.calls.append((name, args, False))
            return json.dumps({"error": f"unknown tool '{name}'"})
        if read_only and name not in READ_ONLY_TOOLS:
            self.calls.append((name, args, False))
            return json.dumps({"error": f"tool '{name}' not permitted for this agent"})
        try:
            res = fn(**args)
            self.calls.append((name, args, True))
            return json.dumps(res)
        except TypeError as e:
            self.calls.append((name, args, False))
            return json.dumps({"error": f"bad arguments: {e}"})
        except ValueError as e:
            self.calls.append((name, args, False))
            return json.dumps({"error": str(e)})

    # --- helpers for checkers -------------------------------------------------
    def dev(self, room, device, prop):
        return self.state["devices"][room][device][prop]

    def changed(self):
        """Set of (room, device, prop) that differ from the initial state."""
        out = set()
        for r in ROOMS:
            for d in DEVICES:
                for p, v in self.state["devices"][r][d].items():
                    if self.initial["devices"][r][d].get(p) != v:
                        out.add((r, d, p))
        return out

    def notified(self, recipient):
        return any(n["recipient"] == recipient for n in self.notifications)
