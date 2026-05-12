"""
orac_archive.py
========================================

ORAC Event Archiver

Създава structured archive за всички GCN/LVK events.

Структура:
archive/
    2026-05-11/
        MS260511d/
            PRELIMINARY_2026-05-11T14-39-46.json
            INITIAL_2026-05-11T14-39-52.json
            metadata.json

УПОТРЕБА:
    from orac_archive import archive_event
    archive_event(event_dict)

Author: Dimitar Kretski
"""

import os
import json
from datetime import datetime, timezone


# ============================================================
# HELPERS
# ============================================================

def utc_now():

    return datetime.now(
        timezone.utc
    )


def safe_timestamp(dt=None):

    if dt is None:
        dt = utc_now()

    return dt.strftime(
        "%Y-%m-%dT%H-%M-%S"
    )


def ensure_dir(path):

    os.makedirs(
        path,
        exist_ok=True
    )


# ============================================================
# MAIN ARCHIVE FUNCTION
# ============================================================

def archive_event(
    event,
    archive_root="archive"
):

    """
    event = {
        "event_id": "MS260511d",
        "state": "INITIAL",
        "topic": "igwn.gwalert",
        "received_at": "...",
        ...
    }
    """

    try:

        # ----------------------------------------------------
        # Extract metadata
        # ----------------------------------------------------

        event_id = str(
            event.get(
                "event_id",
                "UNKNOWN"
            )
        )

        state = str(
            event.get(
                "state",
                "UNKNOWN"
            )
        )

        now = utc_now()

        date_dir = now.strftime(
            "%Y-%m-%d"
        )

        timestamp = safe_timestamp(now)

        # ----------------------------------------------------
        # Build paths
        # ----------------------------------------------------

        event_dir = os.path.join(
            archive_root,
            date_dir,
            event_id
        )

        ensure_dir(event_dir)

        # ----------------------------------------------------
        # Save raw event
        # ----------------------------------------------------

        filename = (
            f"{state}_{timestamp}.json"
        )

        filepath = os.path.join(
            event_dir,
            filename
        )

        with open(
            filepath,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                event,
                f,
                indent=2,
                ensure_ascii=False
            )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        metadata_path = os.path.join(
            event_dir,
            "metadata.json"
        )

        metadata = {
            "event_id": event_id,
            "latest_state": state,
            "last_update_utc": now.isoformat(),
            "archive_path": event_dir
        }

        # preserve existing metadata fields
        if os.path.exists(metadata_path):

            try:

                with open(
                    metadata_path,
                    "r",
                    encoding="utf-8"
                ) as f:

                    old_meta = json.load(f)

                metadata.update(old_meta)

                metadata["latest_state"] = state
                metadata["last_update_utc"] = now.isoformat()

            except:
                pass

        with open(
            metadata_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                metadata,
                f,
                indent=2,
                ensure_ascii=False
            )

        # ----------------------------------------------------
        # Console log
        # ----------------------------------------------------

        print(
            f"[ARCHIVE] "
            f"{event_id} "
            f"{state}"
        )

        print(
            f"          → {filepath}"
        )

        return filepath

    except Exception as e:

        print(
            f"[ARCHIVE ERR] {e}"
        )

        return None


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_event = {

        "event_id": "MS260511d",

        "state": "INITIAL",

        "topic": "igwn.gwalert",

        "received_at": utc_now().isoformat(),

        "far": 1.2e-7,

        "group": "CBC",

        "pipeline": "gstlal"
    }

    archive_event(test_event)