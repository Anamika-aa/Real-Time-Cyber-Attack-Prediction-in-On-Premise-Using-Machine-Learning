#!/usr/bin/env python3

import pandas as pd
import numpy as np

# ---------------- FEATURE ENGINEERING ---------------- #

HIGH_RISK_PORTS = {22, 23, 3389, 445, 4444, 135, 139}

def ip_features(ip_str):
    try:
        parts = str(ip_str).split(".")
        first = int(parts[0])
        private = (
            first == 10 or
            (first == 172 and 16 <= int(parts[1]) <= 31) or
            (first == 192 and int(parts[1]) == 168)
        )
        return first, int(private)
    except:
        return 0, 0


def build_features(df_raw, le_label=None, le_band=None, training=False):
    d = df_raw.copy()

    # Timestamp features
    d["_ts"] = pd.to_datetime(d.get("timestamp"), errors="coerce", utc=True)
    d["hour"] = d["_ts"].dt.hour.fillna(12).astype(int)
    d["dow"] = d["_ts"].dt.dayofweek.fillna(0).astype(int)
    d["is_night"] = ((d["hour"] < 6) | (d["hour"] > 22)).astype(int)
    d["is_weekend"] = (d["dow"] >= 5).astype(int)

    # Rule features
    d["rule_level"] = pd.to_numeric(d.get("rule_level", 0), errors="coerce").fillna(0)
    d["rule_high_severity"] = (d["rule_level"] >= 10).astype(int)
    d["rule_medium_severity"] = ((d["rule_level"] >= 5) & (d["rule_level"] < 10)).astype(int)

    # Network features
    d["src_port"] = pd.to_numeric(d.get("src_port", 0), errors="coerce").fillna(0)
    d["dst_port"] = pd.to_numeric(d.get("dst_port", 0), errors="coerce").fillna(0)

    d["dst_high_risk_port"] = d["dst_port"].apply(lambda p: int(p in HIGH_RISK_PORTS))

    d[["src_first_octet", "src_private"]] = d.get("src_ip", "0.0.0.0").apply(
        lambda x: pd.Series(ip_features(x))
    )
    d[["dst_first_octet", "dst_private"]] = d.get("dst_ip", "0.0.0.0").apply(
        lambda x: pd.Series(ip_features(x))
    )

    # Protocol
    d["protocol"] = d.get("protocol", "tcp").astype(str).str.lower()
    d["is_tcp"] = (d["protocol"] == "tcp").astype(int)
    d["is_udp"] = (d["protocol"] == "udp").astype(int)

    FEATURES = [
        "rule_level","rule_high_severity","rule_medium_severity",
        "hour","dow","is_night","is_weekend",
        "src_port","dst_port","dst_high_risk_port",
        "src_first_octet","dst_first_octet","src_private","dst_private",
        "is_tcp","is_udp"
    ]

    return d[FEATURES].fillna(0)


# ---------------- HELPER: FLATTEN ALERT ---------------- #

def flatten_alert(alert):
    return {
        "timestamp": alert.get("@timestamp"),
        "rule_level": alert.get("rule", {}).get("level"),
        "src_ip": alert.get("data", {}).get("srcip"),
        "dst_ip": alert.get("data", {}).get("dstip"),
        "src_port": alert.get("data", {}).get("srcport"),
        "dst_port": alert.get("data", {}).get("dstport"),
        "protocol": alert.get("data", {}).get("protocol"),
    }