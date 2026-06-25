#!/usr/bin/env python3

import json
import pandas as pd
import numpy as np
import joblib

# 🔹 CHANGE THIS PATH (VERY IMPORTANT)
MODEL_PATH = "wazuh_outputs/wazuh_rf_model.pkl"

# 🔹 Load model
loaded = joblib.load(MODEL_PATH)

rf_model = loaded["model"]
le_label = loaded["label_encoder"]
le_band = loaded["band_encoder"]
FEATURE_COLS = loaded["feature_names"]

# ---------------- HELPERS ---------------- #

HIGH_RISK_PORTS = {22, 23, 3389, 445, 4444, 135, 139}

def ip_features(ip_str):
    try:
        parts = str(ip_str).split(".")
        first = int(parts[0])
        private = (first==10 or
                   (first==172 and 16<=int(parts[1])<=31) or
                   (first==192 and int(parts[1])==168))
        return first, int(private)
    except:
        return 0, 0

def rule_band(rule_id):
    try:
        rid = int(rule_id)
    except:
        return "other"

    if 5700 <= rid < 5800: return "ssh"
    if 5500 <= rid < 5600: return "auth"
    if 18100 <= rid < 18200: return "windows"
    if 31000 <= rid < 32000: return "web"
    if 40100 <= rid < 40200: return "firewall"
    return "other"

def mitre_risk(tactic):
    if not isinstance(tactic, str):
        return 0
    t = tactic.lower()
    if "lateral" in t or "privilege" in t: return 8
    if "exfiltration" in t or "impact" in t: return 9
    if "credential" in t: return 7
    return 1

# ---------------- FEATURE ENGINEERING ---------------- #

def build_features(df_raw):
    d = df_raw.copy()

    # Timestamp
    d["_ts"] = pd.to_datetime(d["timestamp"], errors="coerce", utc=True)
    d["hour"] = d["_ts"].dt.hour.fillna(12)
    d["dow"] = d["_ts"].dt.dayofweek.fillna(0)
    d["is_night"] = ((d["hour"]<6)|(d["hour"]>22)).astype(int)
    d["is_weekend"] = (d["dow"]>=5).astype(int)

    # Rule
    d["rule_level"] = pd.to_numeric(d["rule_level"], errors="coerce").fillna(0)
    d["rule_high_severity"] = (d["rule_level"]>=10).astype(int)
    d["rule_medium_severity"] = ((d["rule_level"]>=5)&(d["rule_level"]<10)).astype(int)

    d["rule_band_str"] = d["rule_id"].apply(rule_band)
    d["rule_band_enc"] = le_band.transform(
        d["rule_band_str"].apply(lambda x: x if x in le_band.classes_ else "other")
    )

    d["mitre_risk"] = d.get("rule_mitre_tactic", "").apply(mitre_risk)

    # Network
    d["src_port"] = pd.to_numeric(d.get("src_port",0), errors="coerce").fillna(0)
    d["dst_port"] = pd.to_numeric(d.get("dst_port",0), errors="coerce").fillna(0)
    d["event_count"] = pd.to_numeric(d.get("event_count",1), errors="coerce").fillna(1)

    d["dst_port_privileged"] = (d["dst_port"]<1024).astype(int)
    d["src_port_privileged"] = (d["src_port"]<1024).astype(int)
    d["dst_high_risk_port"] = d["dst_port"].apply(lambda p: int(p in HIGH_RISK_PORTS))
    d["same_src_dst_port"] = (d["src_port"]==d["dst_port"]).astype(int)

    d["log_event_count"] = np.log1p(d["event_count"])
    d["high_event_count"] = (d["event_count"]>10).astype(int)

    d[["src_first_octet","src_private"]] = d["src_ip"].fillna("0.0.0.0").apply(
        lambda x: pd.Series(ip_features(x)))
    d[["dst_first_octet","dst_private"]] = d["dst_ip"].fillna("0.0.0.0").apply(
        lambda x: pd.Series(ip_features(x)))

    d["cross_subnet"] = (d["src_first_octet"]!=d["dst_first_octet"]).astype(int)
    d["external_src"] = (1-d["src_private"]).astype(int)

    # Protocol / action / status
    for col in ["protocol","action","status"]:
        if col not in d.columns:
            d[col] = "unknown"
        d[col] = d[col].astype(str).str.lower()

    d["is_deny"] = d["action"].str.contains("den|drop|block", na=False).astype(int)
    d["is_failure"] = d["status"].str.contains("fail|error", na=False).astype(int)
    d["is_tcp"] = (d["protocol"]=="tcp").astype(int)
    d["is_udp"] = (d["protocol"]=="udp").astype(int)
    d["is_icmp"] = (d["protocol"]=="icmp").astype(int)

    # Compound
    d["fail_high_severity"] = d["is_failure"] * d["rule_high_severity"]
    d["deny_high_event"] = d["is_deny"] * d["high_event_count"]
    d["external_high_severity"] = d["external_src"] * d["rule_high_severity"]
    d["night_high_severity"] = d["is_night"] * d["rule_high_severity"]
    d["high_risk_fail"] = d["dst_high_risk_port"] * d["is_failure"]
    d["mitre_event_product"] = d["mitre_risk"] * d["log_event_count"]

    return d[FEATURE_COLS].fillna(0)

# ---------------- PREDICT ---------------- #

def predict_alert(alert):
    try:
        df = pd.DataFrame([alert])
        X = build_features(df)

        pred = rf_model.predict(X)[0]
        label = le_label.inverse_transform([pred])[0]

        return label
    except Exception as e:
        return f"error: {e}"

# ---------------- FLATTEN WAZUH ---------------- #

def flatten_alert(alert):
    return {
        "timestamp": alert.get("@timestamp"),
        "rule_id": alert.get("rule", {}).get("id"),
        "rule_level": alert.get("rule", {}).get("level"),
        "rule_mitre_tactic": alert.get("rule", {}).get("mitre", {}).get("tactic"),
        "src_ip": alert.get("data", {}).get("srcip"),
        "dst_ip": alert.get("data", {}).get("dstip"),
        "src_port": alert.get("data", {}).get("srcport"),
        "dst_port": alert.get("data", {}).get("dstport"),
        "protocol": alert.get("data", {}).get("protocol"),
        "action": alert.get("data", {}).get("action"),
        "status": alert.get("data", {}).get("status"),
        "event_count": alert.get("data", {}).get("extra_data", {}).get("count"),
    }

# ---------------- MAIN ---------------- #

import time

if __name__ == "__main__":

    print("🚀 ML Prediction Started...")

    with open("alerts.json") as f:   # 👉 use local test file OR wazuh path later
        for line in f:
            try:
                alert = json.loads(line)
                flat = flatten_alert(alert)

                result = predict_alert(flat)

                if result != "Normal":
                    print(f"🚨 ML ALERT: {result}")

            except Exception as e:
                print("Error:", e)