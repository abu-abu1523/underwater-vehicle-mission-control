PROTOCOL_VERSION = "1.0"
VEHICLE_ID = "UNDERWATER-VEHICLE"
WEBSOCKET_PORT = 8000
UPDATE_RATE_HZ = 1

FAULT_THRESHOLDS = {
    "battery":     {"warning": 20,  "critical": 10},
    "o2_level":    {"warning": 18,  "critical": 17},
    "pressure":    {"warning": 550, "critical": 600},
    "temperature": {"warning": 2,   "critical": 1},
}

MESSAGE_TYPES = {
    "SENSOR_DATA":   "sensor_data",
    "FAULT_ALERT":   "fault_alert",
    "STATUS_UPDATE": "status_update",
    "HEARTBEAT":     "heartbeat",
}