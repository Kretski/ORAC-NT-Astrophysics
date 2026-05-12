import os
import json
import time
import shutil
from pathlib import Path
from datetime import datetime, timezone

WATCH_FILE = "latest_event.json"
ARCHIVE_DIR = "archive"
REPLAY_DIR = "replay"

os.makedirs(ARCHIVE_DIR, exist_ok=True)
os.makedirs(REPLAY_DIR, exist_ok=True)

last_mtime = 0
last_event_uid = None


# ============================================================
# TIME
# ============================================================

def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# ============================================================
# LOGGING
# ============================================================

def log(msg):
    print(f"[{utc_now()}] {msg}")


# ============================================================
# SAFE JSON LOAD
# ============================================================

def load_json_safe(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log(f"ERROR | JSON parse failed: {e}")
        return None


# ============================================================
# FLEXIBLE FIELD PARSER
# ============================================================

def parse_event(data):

    # ---- event id ----
    event_id = (
        data.get("superevent_id")
        or data.get("event_id")
        or data.get("id")
        or data.get("uid")
        or "UNKNOWN"
    )

    # ---- state/type ----
    state = (
        data.get("alert_type")
        or data.get("state")
        or data.get("notice_type")
        or data.get("event")
        or "UNKNOWN"
    )

    # ---- timestamp ----
    ts = (
        data.get("time_created")
        or data.get("created")
        or data.get("timestamp")
        or utc_now()
    )

    return {
        "event_id": str(event_id),
        "state": str(state),
        "timestamp": str(ts),
        "raw": data
    }


# ============================================================
# ARCHIVE
# ============================================================

def archive_event(event):

    eid = event["event_id"]
    state = event["state"]

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    fname = f"{ts}_{eid}_{state}.json"
    out = os.path.join(ARCHIVE_DIR, fname)

    with open(out, "w", encoding="utf-8") as f:
        json.dump(event, f, indent=2)

    log(f"ARCHIVE | {eid} -> {state}")


# ============================================================
# EXPORT REPLAY COPY
# ============================================================

def export_replay(event):

    eid = event["event_id"]

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    out = os.path.join(REPLAY_DIR, f"{ts}_{eid}.json")

    with open(out, "w", encoding="utf-8") as f:
        json.dump(event, f, indent=2)

    log("EXPORT | Replay exported")


# ============================================================
# MAIN LOOP
# ============================================================

print("================================================")
print("ORAC Archive Layer v1.2")
print(f"Watching {WATCH_FILE}")
print("================================================")

log("CONNECT | Wrapper started")

try:

    while True:

        if not os.path.exists(WATCH_FILE):
            time.sleep(1)
            continue

        mtime = os.path.getmtime(WATCH_FILE)

        if mtime != last_mtime:

            last_mtime = mtime

            data = load_json_safe(WATCH_FILE)

            if data is None:
                time.sleep(1)
                continue

            event = parse_event(data)

            uid = f"{event['event_id']}::{event['state']}"

            # avoid duplicates
            if uid == last_event_uid:
                time.sleep(1)
                continue

            last_event_uid = uid

            archive_event(event)

            print()
            print(f"📡 EVENT | {event['event_id']} | {event['state']}")

            export_replay(event)

        time.sleep(1)

except KeyboardInterrupt:

    print()
    log("DISCONNECT | Wrapper stopped")