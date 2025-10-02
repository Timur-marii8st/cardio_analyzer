from __future__ import annotations
import argparse
import pandas as pd
import time, requests
from datetime import datetime, timezone, timedelta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with ts,channel,value")
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--session-id", default="replay-session")
    ap.add_argument("--device-id", default="feeder-001")
    ap.add_argument("--speed", type=float, default=1.0, help="Playback speed (1.0 = realtime)")
    ap.add_argument("--batch-size", type=int, default=200)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if "ts" not in df.columns:
        raise ValueError("CSV must contain ts")
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    # переигрываем относительно текущего времени
    start_real = datetime.now(timezone.utc)
    start_src  = df["ts"].min()
    df["ts"] = df["ts"].apply(lambda t: start_real + (t - start_src) / args.speed)

    rows = df.to_dict(orient="records")
    url = f"{args.api}/v1/ingest/batch"
    i = 0
    while i < len(rows):
        batch = rows[i:i+args.batch_size]
        # подождём до времени первой точки батча
        now = datetime.now(timezone.utc)
        wait_sec = (batch[0]["ts"] - now).total_seconds()
        if wait_sec > 0:
            time.sleep(min(wait_sec, 2.0))
        payload = {
            "device_id": args.device_id,
            "session_id": args.session_id,
            "samples": [
                {"ts": r["ts"].isoformat(), "channel": str(r["channel"]), "value": float(r["value"])} for r in batch
            ]
        }
        r = requests.post(url, json=payload, timeout=30)
        if not r.ok:
            print("POST failed:", r.status_code, r.text)
        i += args.batch_size

if __name__ == "__main__":
    main()