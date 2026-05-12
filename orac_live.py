"""
ORAC-NT GCN Live Listener — ORAC-Live Edition
===============================================
NASA GCN Kafka listener за реални astrophysical alerts

СТАРТИРАНЕ:
  pip install gcn-kafka python-dotenv

  python orac_live.py              # слуша непрекъснато
  python orac_live.py --keepalive  # keepalive mode

ВАЖНО:
  Сложи credentials в .env файл:
  
  GCN_CLIENT_ID=...
  GCN_CLIENT_SECRET=...
"""

import json
import time
import os
from datetime import datetime, timezone
from dotenv import load_dotenv

try:
    from plyer import notification as _plyer_notify
    NOTIFY_OK = True
except ImportError:
    NOTIFY_OK = False

# ─────────────────────────────────────────────────────────────
# Serious alert filter
# ─────────────────────────────────────────────────────────────

def is_serious(event_id, alert_type, bns_prob, bbh_prob, nsbh_prob, far):
    """
    Реален сериозен GW alert:
    - Не е MDC (MS prefix)
    - Не е RETRACTION
    - BNS или NSBH > 50% ИЛИ BBH > 80%
    - FAR < 1e-6 /yr (ако е наличен)
    """
    if str(event_id).startswith('MS'):
        return False
    if alert_type == 'RETRACTION':
        return False
    prob_ok = (bns_prob > 0.5 or nsbh_prob > 0.5 or bbh_prob > 0.8)
    if not prob_ok:
        return False
    if far is not None and far > 1e-6:
        return False
    return True


def desktop_notify(event_id, bns_prob, bbh_prob, nsbh_prob, far):
    """Windows desktop notification при сериозен alert."""
    far_str = f"{far:.2e} /yr" if far else "N/A"
    msg = (
        f"BNS:{bns_prob*100:.0f}%  BBH:{bbh_prob*100:.0f}%  NSBH:{nsbh_prob*100:.0f}%\n"
        f"FAR: {far_str}"
    )
    if NOTIFY_OK:
        try:
            _plyer_notify.notify(
                title=f"ORAC ALERT: {event_id}",
                message=msg,
                app_name="ORAC-NT",
                timeout=30
            )
        except Exception as e:
            print(f"  [NOTIFY ERR] {e}")
    else:
        # Fallback: Windows PowerShell toast (без plyer)
        import subprocess
        ps_cmd = (
            f'Add-Type -AssemblyName System.Windows.Forms; '
            f'$n = New-Object System.Windows.Forms.NotifyIcon; '
            f'$n.Icon = [System.Drawing.SystemIcons]::Information; '
            f'$n.Visible = $true; '
            f'$n.ShowBalloonTip(30000, "ORAC ALERT: {event_id}", "{msg.replace(chr(10), " ")}", '
            f'[System.Windows.Forms.ToolTipIcon]::Warning); '
            f'Start-Sleep 2; $n.Dispose()'
        )
        try:
            subprocess.Popen(
                ["powershell", "-WindowStyle", "Hidden", "-Command", ps_cmd],
                creationflags=0x08000000  # CREATE_NO_WINDOW
            )
        except Exception as e:
            print(f"  [PS NOTIFY ERR] {e}")

try:
    from gcn_kafka import Consumer
    GCN_OK = True
except ImportError:
    GCN_OK = False
    print("[ERR] pip install gcn-kafka")
    exit(1)

# ─────────────────────────────────────────────────────────────
# Зареждане на .env
# ─────────────────────────────────────────────────────────────

load_dotenv()

CLIENT_ID = os.getenv("GCN_CLIENT_ID")
CLIENT_SECRET = os.getenv("GCN_CLIENT_SECRET")

if not CLIENT_ID or not CLIENT_SECRET:
    print("[ERR] Липсват GCN credentials в .env")
    exit(1)

# ─────────────────────────────────────────────────────────────
# NASA GCN Topics
# ─────────────────────────────────────────────────────────────

TOPICS = [

    # ── Gravitational Waves ────────────────────────────────
    'igwn.gwalert',

    # ── Einstein Probe X-ray ──────────────────────────────
    'gcn.notices.einstein_probe.wxt.alert',

    # ── IceCube neutrinos ─────────────────────────────────
    'gcn.notices.icecube.lvk_nu_track_search',
    'gcn.notices.icecube.gold_bronze_track_alerts',

    # ── Swift GRB ─────────────────────────────────────────
    'gcn.notices.swift.bat.guano',

    # ── Heartbeat / keepalive ─────────────────────────────
    'gcn.heartbeat',

    # ── Legacy classic notices (plain text format) ────────
    'gcn.classic.text.FERMI_GBM_FIN_POS',
    'gcn.classic.text.LVC_INITIAL',
]

# Classic text topics — не са JSON
CLASSIC_TEXT_TOPICS = {
    'gcn.classic.text.FERMI_GBM_FIN_POS',
    'gcn.classic.text.LVC_INITIAL',
}

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_FILE    = os.path.join(BASE_DIR, 'latest_event.json')
CONNECTION_LOG = os.path.join(BASE_DIR, 'history', 'connection.log')

# ─────────────────────────────────────────────────────────────
# Connection logger
# ─────────────────────────────────────────────────────────────

def log_connection(event_type, detail=""):
    """
    Записва connection events в history/connection.log
    event_type: CONNECT | DISCONNECT | ERROR | RECONNECT
    """
    os.makedirs(os.path.dirname(CONNECTION_LOG), exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{ts}] {event_type}"
    if detail:
        line += f" | {detail}"
    with open(CONNECTION_LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')
    if event_type != "CONNECT":
        print(f"\n  📋 Logged → connection.log: {event_type}")

# ─────────────────────────────────────────────────────────────
# Topic icons
# ─────────────────────────────────────────────────────────────

TOPIC_ICONS = {
    'igwn.gwalert': '🌊 GW',
    'gcn.notices.boom': '💥 BOOM',
    'gcn.notices.einstein': '☢️ X-ray',
    'gcn.notices.icecube': '🧊 Nu',
    'gcn.notices.superK': '💫 SN',
    'gcn.notices.swift': '⚡ GRB',
    'gcn.classic.text.FERMI': '🛸 FERMI',
    'gcn.classic.text.LVC': '🌊 LVC',
}

def topic_icon(topic):
    for key, icon in TOPIC_ICONS.items():
        if key in topic:
            return icon
    return '📡'

# ─────────────────────────────────────────────────────────────
# Plain text parser за classic GCN notices
# ─────────────────────────────────────────────────────────────

def parse_classic_text(raw_text, topic):
    fields = {}
    for line in raw_text.splitlines():
        if ':' in line:
            key, _, val = line.partition(':')
            fields[key.strip()] = val.strip()

    event_id = (
        fields.get('TRIGGER_NUM') or
        fields.get('NOTICE_TYPE') or
        fields.get('GRB_NAME') or
        'UNKNOWN'
    )

    notice_type = fields.get('NOTICE_TYPE', 'CLASSIC')

    return {
        "id": event_id,
        "alert_type": notice_type,
        "topic": topic,
        "gps_time": 0.0,
        "bbh_prob": 0.0,
        "bns_prob": 0.0,
        "nsbh_prob": 0.0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "icon": topic_icon(topic),
        "raw_payload": fields
    }

# ─────────────────────────────────────────────────────────────
# Save event for ORAC pipeline
# ─────────────────────────────────────────────────────────────

def save_event(event_data):
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(event_data, f, indent=2, ensure_ascii=False)

# ─────────────────────────────────────────────────────────────
# Main alert parser
# ─────────────────────────────────────────────────────────────

def process_alert(msg):

    topic = msg.topic()

    if topic == "gcn.heartbeat":
        ts = datetime.now().strftime('%H:%M:%S')
        print(f"  💓 [{ts}] Heartbeat OK (няма нови alerts)", end='\r')
        return

    raw_bytes = msg.value()

    # ── Classic text topics ───────────────────────────────
    if topic in CLASSIC_TEXT_TOPICS:
        try:
            raw_text = raw_bytes.decode('utf-8', errors='replace')
        except Exception as e:
            print(f"\n[ERR] Decode error on classic notice: {e}")
            return

        print(f"\n{'=' * 58}")
        print(f"  {topic_icon(topic)}  CLASSIC NOTICE  [{topic}]")
        print(f"  Time: {datetime.now(timezone.utc).isoformat()}")
        print(f"  --- Raw ---")
        for line in raw_text.splitlines()[:10]:
            print(f"  {line}")
        print(f"  ...")

        event_data = parse_classic_text(raw_text, topic)
        save_event(event_data)

        print(f"  💾 → {OUTPUT_FILE}")
        print(f"  → orac_spinqit_wrapper.py ще реагира автоматично")
        return

    # ── JSON topics ───────────────────────────────────────
    try:
        payload = json.loads(raw_bytes)
    except Exception as e:
        try:
            raw_text = raw_bytes.decode('utf-8', errors='replace')
            print(f"\n[WARN] Non-JSON payload on {topic}, treating as text.")
            print(f"  First 200 chars: {raw_text[:200]}")
        except Exception:
            print(f"\n[ERR] Cannot decode message on {topic}: {e}")
        return

    alert_type = payload.get('alert_type', payload.get('type', 'NOTICE'))
    event_id   = payload.get('superevent_id', payload.get('id', payload.get('trigger_id', 'UNKNOWN')))
    now        = datetime.now(timezone.utc).isoformat()
    icon       = topic_icon(topic)

    print(f"\n{'=' * 58}")
    print(f"  {icon}  EVENT: {event_id}  [{alert_type}]")
    print(f"  Topic:  {topic}")
    print(f"  Time:   {now}")

    # RETRACTION алертите нямат 'event' поле — само съобщение за оттегляне
    if alert_type == 'RETRACTION':
        print(f"  ⚠️  RETRACTED — събитието е оттеглено като false alarm.")
        log_connection("RETRACTION", f"event_id={event_id}")
        return

    event     = payload.get('event') or {}
    gps_time  = event.get('time', 0.0)
    classif   = event.get('classification', {})
    bns_prob  = classif.get('BNS', 0.0)
    bbh_prob  = classif.get('BBH', 0.0)
    nsbh_prob = classif.get('NSBH', 0.0)
    far       = event.get('far', None)

    if (bns_prob + bbh_prob + nsbh_prob) > 0:
        print(
            f"  BBH:{bbh_prob * 100:.0f}%  "
            f"BNS:{bns_prob * 100:.0f}%  "
            f"NSBH:{nsbh_prob * 100:.0f}%"
        )

    if far:
        print(f"  FAR: {far:.2e} /yr")

    event_data = {
        "id":          event_id,
        "alert_type":  alert_type,
        "topic":       topic,
        "gps_time":    gps_time,
        "bbh_prob":    bbh_prob,
        "bns_prob":    bns_prob,
        "nsbh_prob":   nsbh_prob,
        "timestamp":   now,
        "icon":        icon,
        "raw_payload": payload
    }

    # ── Desktop notification при сериозен alert ──────────
    if is_serious(event_id, alert_type, bns_prob, bbh_prob, nsbh_prob, far):
        print(f"\n  🚨 СЕРИОЗЕН ALERT — изпращане на desktop notification...")
        desktop_notify(event_id, bns_prob, bbh_prob, nsbh_prob, far)

    save_event(event_data)

    print(f"  💾 → {OUTPUT_FILE}")
    print(f"  → orac_spinqit_wrapper.py ще реагира автоматично")

# ─────────────────────────────────────────────────────────────
# Live listener
# ─────────────────────────────────────────────────────────────

def listen():

    print("=" * 58)
    print("  ORAC-NT GCN Live Listener")
    print(f"  Client: {CLIENT_ID[:8]}...")
    print(f"  Топици: {len(TOPICS)}")

    for t in TOPICS:
        print(f"    {topic_icon(t)}  {t}")

    print("  Ctrl+C за спиране")
    print("=" * 58 + "\n")

    consumer = Consumer(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET
    )

    consumer.subscribe(TOPICS)

    log_connection("CONNECT", f"client={CLIENT_ID[:8]}... topics={len(TOPICS)}")
    print("[OK] Свързан към NASA GCN Kafka\n")

    last_error_str = None   # за да не дублираме едни и същи грешки

    try:

        while True:

            msgs = consumer.consume(
                num_messages=10,
                timeout=2.0
            )

            if not msgs:
                ts = datetime.now().strftime('%H:%M:%S')
                print(
                    f"  📡 [{ts}] Слушам... (няма нови alerts)",
                    end='\r'
                )
                continue

            for m in msgs:

                if m.error():
                    err_str = str(m.error())

                    # ── Логва само нови/различни грешки ──
                    if err_str != last_error_str:
                        last_error_str = err_str
                        print(f"\n  ⚠️  Kafka error: {err_str}")
                        log_connection("ERROR", err_str)

                    continue

                # Успешно съобщение — нулира error tracker
                if last_error_str is not None:
                    log_connection("RECONNECT", "Stream resumed after error")
                    print(f"\n  ✅ Връзката е възстановена.")
                    last_error_str = None

                process_alert(m)

    except KeyboardInterrupt:

        log_connection("DISCONNECT", "Stopped by user (Ctrl+C)")
        print("\n\n[ORAC] Спрян.")

    finally:

        consumer.close()

# ─────────────────────────────────────────────────────────────
# Keepalive mode
# ─────────────────────────────────────────────────────────────

def keepalive(duration=300):

    print(f"[KEEPALIVE] Свързване за {duration}s...")

    consumer = Consumer(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET
    )

    consumer.subscribe(TOPICS)

    log_connection("CONNECT", f"keepalive mode duration={duration}s")

    start = time.time()
    count = 0

    try:

        while time.time() - start < duration:

            msgs = consumer.consume(
                num_messages=5,
                timeout=5.0
            )

            elapsed = int(time.time() - start)

            if msgs:

                count += len(msgs)

                for m in msgs:

                    if not m.error():

                        topic = m.topic()

                        if topic == "gcn.heartbeat":
                            print(
                                f"  💓 [{elapsed:>3}s] heartbeat",
                                end='\r'
                            )
                            continue

                        print(f"\n  📡 [{elapsed:>3}s] Alert: {topic}")

            else:

                print(
                    f"  ⏳ [{elapsed:>3}s/{duration}s] Чакам...",
                    end='\r'
                )

        log_connection("DISCONNECT", f"keepalive completed. alerts={count}")
        print(f"\n\n[KEEPALIVE] ✅ Credential обновен!")
        print(f"[KEEPALIVE] Получени alerts: {count}")
        print(f"[KEEPALIVE] Следващо обновяване: до 30 дни")

    except KeyboardInterrupt:

        log_connection("DISCONNECT", "keepalive stopped by user")
        print("\n[KEEPALIVE] Прекъснато.")

    finally:

        consumer.close()

# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    import sys

    if "--keepalive" in sys.argv:
        keepalive()
    else:
        listen()