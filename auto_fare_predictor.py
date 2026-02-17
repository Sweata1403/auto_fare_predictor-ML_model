"""
=====================================================
  AUTO FARE PREDICTOR — OLA vs UBER vs RAPIDO
  Route: Blue Ridge Tower 20, Phase 1 Hinjewadi
       → Capgemini Office, Phase 3 Hinjewadi
=====================================================
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib joblib
"""

# ── 1. IMPORTS ────────────────────────────────────────────────
import os, csv, joblib, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")


# ── 2. CONFIG ─────────────────────────────────────────────────
CONFIG = {
    "source":       "Blue Ridge Tower 20 Phase 1 Hinjewadi",
    "destination":  "Capgemini Office Phase 3 Hinjewadi",
    "distance_km":  3.8,
    "apps":         ["ola", "uber", "rapido"],
    "dataset_path": "hinjewadi_fare_data.csv",
    "model_dir":    "models/",
}

# Column definitions for the wide-format CSV
FARE_COLUMNS = [
    "date", "hour", "day_of_week", "is_weekend", "is_holiday",
    "is_raining", "weather_condition", "time_of_day",
    "ola_fare",    "ola_surge_active",    "ola_surge_multiplier",
    "uber_fare",   "uber_surge_active",   "uber_surge_multiplier",
    "rapido_fare", "rapido_surge_active", "rapido_surge_multiplier",
    "cheapest_app",
    "source", "destination", "distance_km"
]


# ── 3. HELPER ─────────────────────────────────────────────────
def get_time_of_day(hour: int) -> str:
    if   7  <= hour <= 10: return "Morning Peak"
    elif 11 <= hour <= 16: return "Off Peak"
    elif 17 <= hour <= 21: return "Evening Peak"
    else:                  return "Late Night"


# ── 4. CREATE BLANK CSV TEMPLATE ─────────────────────────────
def create_dataset_template():
    if not os.path.exists(CONFIG["dataset_path"]):
        pd.DataFrame(columns=FARE_COLUMNS).to_csv(CONFIG["dataset_path"], index=False)
        print(f"✅ Blank dataset created → {CONFIG['dataset_path']}")
    else:
        print(f"📂 Dataset exists → {CONFIG['dataset_path']}")

    os.makedirs(CONFIG["model_dir"], exist_ok=True)


# ── 5. MANUAL LOGGING ─────────────────────────────────────────
def log_fare_entry(
    ola_fare:    float,
    uber_fare:   float,
    rapido_fare: float,
    is_raining:  int   = 0,
    weather_condition:  str   = "Clear",
    ola_surge_active:   int   = 0,   ola_surge_multiplier:    float = 1.0,
    uber_surge_active:  int   = 0,   uber_surge_multiplier:   float = 1.0,
    rapido_surge_active:int   = 0,   rapido_surge_multiplier: float = 1.0,
    is_holiday:  int   = 0,
):
    """
    Call this ONCE each time you check all 3 apps simultaneously.

    Example (weekday evening, raining, Uber has surge):
        log_fare_entry(
            ola_fare=78.0,   ola_surge_active=0,
            uber_fare=112.0, uber_surge_active=1, uber_surge_multiplier=1.8,
            rapido_fare=58.0,
            is_raining=1, weather_condition="Heavy Rain"
        )
    """
    now = datetime.now()
    fares = {"ola": ola_fare, "uber": uber_fare, "rapido": rapido_fare}
    cheapest = min(fares, key=fares.get)

    row = {
        "date":              now.strftime("%Y-%m-%d"),
        "hour":              now.hour,
        "day_of_week":       now.strftime("%A"),
        "is_weekend":        1 if now.weekday() >= 5 else 0,
        "is_holiday":        is_holiday,
        "is_raining":        is_raining,
        "weather_condition": weather_condition,
        "time_of_day":       get_time_of_day(now.hour),
        # Ola
        "ola_fare":             ola_fare,
        "ola_surge_active":     ola_surge_active,
        "ola_surge_multiplier": ola_surge_multiplier,
        # Uber
        "uber_fare":             uber_fare,
        "uber_surge_active":     uber_surge_active,
        "uber_surge_multiplier": uber_surge_multiplier,
        # Rapido
        "rapido_fare":             rapido_fare,
        "rapido_surge_active":     rapido_surge_active,
        "rapido_surge_multiplier": rapido_surge_multiplier,
        # Meta
        "cheapest_app": cheapest,
        "source":       CONFIG["source"],
        "destination":  CONFIG["destination"],
        "distance_km":  CONFIG["distance_km"],
    }

    file_exists = os.path.exists(CONFIG["dataset_path"])
    with open(CONFIG["dataset_path"], "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FARE_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"✅ Logged at {now.strftime('%H:%M on %A %d-%b-%Y')}")
    print(f"   Ola:₹{ola_fare}  Uber:₹{uber_fare}  Rapido:₹{rapido_fare}")
    print(f"   🏆 Cheapest → {cheapest.title()} (₹{fares[cheapest]})")


# ── 6. SYNTHETIC DATA (demo — replace with real data) ────────
def generate_synthetic_data(n: int = 2000):
    """
    Mimics realistic Hinjewadi pricing patterns:
    - Morning (7-10am) and Evening (5-9pm) peaks have surge
    - Rapido is usually cheapest, Uber most expensive
    - Rain pushes all fares up
    - Weekends are cheaper (less demand from offices)
    """
    np.random.seed(42)
    rows = []
    days_list = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

    # Base fares for this route (short ~3.8km IT corridor route)
    base = {"ola": 65, "uber": 78, "rapido": 50}

    for _ in range(n):
        hour    = np.random.randint(0, 24)
        day_idx = np.random.randint(0, 7)
        day     = days_list[day_idx]
        wknd    = 1 if day_idx >= 5 else 0
        rain    = np.random.choice([0,1], p=[0.70, 0.30])   # Pune gets good rain!
        hday    = np.random.choice([0,1], p=[0.93, 0.07])
        tod     = get_time_of_day(hour)

        # Surge is more common in peaks and rain
        is_peak = 1 if tod in ("Morning Peak", "Evening Peak") else 0

        fares, surges, smuls = {}, {}, {}
        for app in ["ola", "uber", "rapido"]:
            # Rapido surges less aggressively
            surge_prob = 0.7 if (is_peak and not wknd) else 0.2
            if app == "rapido": surge_prob *= 0.6
            if rain:            surge_prob = min(surge_prob + 0.3, 0.95)

            surge  = np.random.choice([0,1], p=[1-surge_prob, surge_prob])
            smul   = round(np.random.uniform(1.2, 2.2), 1) if surge else 1.0

            fare   = base[app] * smul
            fare  += 8  if wknd else 0          # slight weekend premium (less autos)
            fare  += 12 if rain else 0
            fare  += np.random.normal(0, 4)
            fare   = max(fare, 35)

            fares[app]  = round(fare, 2)
            surges[app] = surge
            smuls[app]  = smul

        cheapest = min(fares, key=fares.get)
        rows.append({
            "date":              f"2025-{np.random.randint(1,12):02d}-{np.random.randint(1,28):02d}",
            "hour":              hour,
            "day_of_week":       day,
            "is_weekend":        wknd,
            "is_holiday":        hday,
            "is_raining":        rain,
            "weather_condition": "Light Rain" if (rain and np.random.rand()>0.4) else (
                                 "Heavy Rain" if rain else "Clear"),
            "time_of_day":       tod,
            "ola_fare":             fares["ola"],
            "ola_surge_active":     surges["ola"],
            "ola_surge_multiplier": smuls["ola"],
            "uber_fare":             fares["uber"],
            "uber_surge_active":     surges["uber"],
            "uber_surge_multiplier": smuls["uber"],
            "rapido_fare":             fares["rapido"],
            "rapido_surge_active":     surges["rapido"],
            "rapido_surge_multiplier": smuls["rapido"],
            "cheapest_app":   cheapest,
            "source":         CONFIG["source"],
            "destination":    CONFIG["destination"],
            "distance_km":    CONFIG["distance_km"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(CONFIG["dataset_path"], index=False)
    print(f"✅ Synthetic data ({n} rows) → {CONFIG['dataset_path']}")
    return df


# ── 7. FEATURE ENGINEERING ────────────────────────────────────
SHARED_FEATURES = [
    "hour", "hour_sin", "hour_cos",
    "is_weekend", "is_holiday", "is_raining",
    "weather_encoded", "time_of_day_encoded", "day_encoded",
    "distance_km"
]

def engineer_features(df: pd.DataFrame):
    df = df.copy()
    encoders = {}

    for col, key in [("weather_condition","weather"),
                     ("time_of_day","time_of_day"),
                     ("day_of_week","day")]:
        le = LabelEncoder()
        df[f"{key}_encoded"] = le.fit_transform(df[col])
        encoders[key] = le

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df, encoders


# ── 8. TRAIN ONE MODEL PER APP ────────────────────────────────
def get_app_features(app: str) -> list:
    """Each app model also gets its own surge columns as features."""
    return SHARED_FEATURES + [
        f"{app}_surge_active",
        f"{app}_surge_multiplier",
    ]


def train_models(df: pd.DataFrame):
    print("\n📊 Training one model per app...\n")
    df, encoders = engineer_features(df)
    trained = {}

    for app in CONFIG["apps"]:
        features   = get_app_features(app)
        target_col = f"{app}_fare"
        X = df[features]; y = df[target_col]
        Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.2, random_state=42)

        candidates = {
            "XGBoost":  xgb.XGBRegressor(n_estimators=200, learning_rate=0.05,
                                          max_depth=5, random_state=42, verbosity=0),
            "LightGBM": lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05,
                                           max_depth=5, random_state=42, verbose=-1),
            "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        }

        best_m, best_mae, best_name = None, float("inf"), ""
        for name, m in candidates.items():
            m.fit(Xtr, ytr)
            mae = mean_absolute_error(yte, m.predict(Xte))
            if mae < best_mae:
                best_mae, best_m, best_name = mae, m, name

        trained[app] = best_m
        preds = best_m.predict(Xte)
        print(f"  {app.upper():8s} → Best: {best_name:15s} | "
              f"MAE: ₹{best_mae:.2f} | "
              f"RMSE: ₹{np.sqrt(mean_squared_error(yte,preds)):.2f} | "
              f"R²: {r2_score(yte,preds):.3f}")

    model_path = CONFIG["model_dir"] + "fare_models.pkl"
    joblib.dump({"models": trained, "encoders": encoders}, model_path)
    print(f"\n💾 All models saved → {model_path}")
    return trained, encoders


# ── 9. PREDICT — Which app is cheapest at a given hour? ───────
def predict_all_apps(
    hour=None, is_weekend=None, is_holiday=0,
    is_raining=0, weather_condition="Clear",
    ola_surge_active=0,    ola_surge_multiplier=1.0,
    uber_surge_active=0,   uber_surge_multiplier=1.0,
    rapido_surge_active=0, rapido_surge_multiplier=1.0,
):
    model_path = CONFIG["model_dir"] + "fare_models.pkl"
    saved    = joblib.load(model_path)
    models   = saved["models"]; enc = saved["encoders"]

    now      = datetime.now()
    hour     = hour if hour is not None else now.hour
    is_wknd  = is_weekend if is_weekend is not None else (1 if now.weekday()>=5 else 0)
    day_name = now.strftime("%A")
    tod      = get_time_of_day(hour)

    surge_info = {
        "ola":    (ola_surge_active,    ola_surge_multiplier),
        "uber":   (uber_surge_active,   uber_surge_multiplier),
        "rapido": (rapido_surge_active, rapido_surge_multiplier),
    }

    shared_base = {
        "hour": hour,
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "is_weekend":  is_wknd,
        "is_holiday":  is_holiday,
        "is_raining":  is_raining,
        "weather_encoded":     enc["weather"].transform([weather_condition])[0],
        "time_of_day_encoded": enc["time_of_day"].transform([tod])[0],
        "day_encoded":         enc["day"].transform([day_name])[0],
        "distance_km": CONFIG["distance_km"],
    }

    predictions = {}
    for app in CONFIG["apps"]:
        surge_a, surge_m = surge_info[app]
        row = {**shared_base,
               f"{app}_surge_active":     surge_a,
               f"{app}_surge_multiplier": surge_m}
        features = get_app_features(app)
        predictions[app] = round(models[app].predict(pd.DataFrame([row])[features])[0], 2)

    cheapest = min(predictions, key=predictions.get)

    print(f"\n{'='*52}")
    print(f"  Route  : {CONFIG['source']}")
    print(f"           → {CONFIG['destination']}")
    print(f"  Time   : {hour:02d}:00 ({tod})")
    print(f"  Day    : {day_name} | Weekend: {'Yes' if is_wknd else 'No'}")
    print(f"  Rain   : {'Yes 🌧️' if is_raining else 'No ☀️'}")
    print(f"{'='*52}")
    print(f"  {'App':10s} {'Fare':>8s}  {'Surge':>6s}")
    print(f"  {'-'*30}")
    for app, fare in sorted(predictions.items(), key=lambda x: x[1]):
        sa, sm = surge_info[app]
        surge_tag = f"x{sm}" if sa else "—"
        tag = "  ← 🏆 CHEAPEST" if app == cheapest else ""
        print(f"  {app.title():10s} ₹{fare:>7.2f}  {surge_tag:>6s}{tag}")
    print(f"{'='*52}\n")

    return predictions, cheapest


# ── 10. CHART 1: Hourly Fare Line Chart ───────────────────────
def plot_hourly_fares(models, enc,
                      is_raining=0, is_weekend=0,
                      day_name="Monday"):
    colors = {"ola":"#28a745", "uber":"#222222", "rapido":"#ff6b00"}
    labels = {"ola":"Ola", "uber":"Uber", "rapido":"Rapido"}
    results= {app: [] for app in CONFIG["apps"]}
    tod_enc= enc["time_of_day"]
    w_enc  = enc["weather"]
    d_enc  = enc["day"]

    for h in range(24):
        tod = get_time_of_day(h)
        surge_h = 1 if (7<=h<=10 or 17<=h<=21) else 0
        weather_label = "Light Rain" if is_raining else "Clear"

        shared = {
            "hour":h, "hour_sin":np.sin(2*np.pi*h/24), "hour_cos":np.cos(2*np.pi*h/24),
            "is_weekend":is_weekend, "is_holiday":0, "is_raining":is_raining,
            "weather_encoded":     w_enc.transform([weather_label])[0],
            "time_of_day_encoded": tod_enc.transform([tod])[0],
            "day_encoded":         d_enc.transform([day_name])[0],
            "distance_km": CONFIG["distance_km"],
        }
        for app in CONFIG["apps"]:
            smul = 1.4 if surge_h else 1.0
            row  = {**shared, f"{app}_surge_active": surge_h,
                               f"{app}_surge_multiplier": smul}
            features = get_app_features(app)
            results[app].append(round(models[app].predict(pd.DataFrame([row])[features])[0], 2))

    fig, ax = plt.subplots(figsize=(15, 6))
    for app in CONFIG["apps"]:
        ax.plot(range(24), results[app], label=labels[app],
                color=colors[app], linewidth=2.5, marker="o", markersize=5)

    # Shade peak hours
    ax.axvspan(7, 10, alpha=0.08, color="red",  label="Morning Peak")
    ax.axvspan(17,21, alpha=0.08, color="blue", label="Evening Peak")

    ax.set_xlabel("Hour of Day", fontsize=12)
    ax.set_ylabel("Predicted Fare (₹)", fontsize=12)
    title = (f"Auto Fare by Hour — Hinjewadi Phase 1 → Phase 3\n"
             f"[{day_name}{'  |  Raining' if is_raining else ''}{'  |  Weekend' if is_weekend else ''}]")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h}:00" for h in range(24)], rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=11); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("hourly_fares.png", dpi=150)
    plt.show()
    print("📊 Saved → hourly_fares.png")


# ── 11. CHART 2: Cheapest App Heatmap (Day × Hour) ───────────
def plot_cheapest_heatmap(models, enc):
    days    = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    app_num = {"ola":0, "uber":1, "rapido":2}
    hmap    = np.zeros((7, 24))
    fmap    = np.zeros((7, 24))

    for di, day in enumerate(days):
        wknd = 1 if di >= 5 else 0
        for h in range(24):
            surge_h = 1 if (7<=h<=10 or 17<=h<=21) and not wknd else 0
            tod     = get_time_of_day(h)
            shared  = {
                "hour":h, "hour_sin":np.sin(2*np.pi*h/24), "hour_cos":np.cos(2*np.pi*h/24),
                "is_weekend":wknd, "is_holiday":0, "is_raining":0,
                "weather_encoded":     enc["weather"].transform(["Clear"])[0],
                "time_of_day_encoded": enc["time_of_day"].transform([tod])[0],
                "day_encoded":         enc["day"].transform([day])[0],
                "distance_km":         CONFIG["distance_km"],
            }
            fares = {}
            for app in CONFIG["apps"]:
                smul = 1.4 if surge_h else 1.0
                row  = {**shared, f"{app}_surge_active": surge_h,
                                  f"{app}_surge_multiplier": smul}
                features = get_app_features(app)
                fares[app] = models[app].predict(pd.DataFrame([row])[features])[0]

            cheapest        = min(fares, key=fares.get)
            hmap[di][h]     = app_num[cheapest]
            fmap[di][h]     = round(fares[cheapest], 1)

    cmap = ListedColormap(["#28a745", "#222222", "#ff6b00"])
    fig, ax = plt.subplots(figsize=(22, 6))
    ax.imshow(hmap, cmap=cmap, aspect="auto", vmin=-0.5, vmax=2.5)

    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h}:00" for h in range(24)],
                       rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(7))
    ax.set_yticklabels(days, fontsize=10)
    ax.set_title(
        "Cheapest App by Hour & Day — Blue Ridge Phase 1 → Capgemini Phase 3\n"
        "(Cell shows cheapest fare ₹ | Color shows which app)",
        fontsize=13, fontweight="bold", pad=12
    )

    for d in range(7):
        for h in range(24):
            ax.text(h, d, f"₹{fmap[d][h]:.0f}",
                    ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold")

    legend_handles = [
        mpatches.Patch(facecolor="#28a745", label="Ola"),
        mpatches.Patch(facecolor="#222222", label="Uber"),
        mpatches.Patch(facecolor="#ff6b00", label="Rapido"),
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              bbox_to_anchor=(1.08, 1), fontsize=11, title="Cheapest App")
    plt.tight_layout()
    plt.savefig("cheapest_heatmap.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("🗺️  Saved → cheapest_heatmap.png")


# ── 12. CHART 3: Average Fare Per App Per Hour (bar chart) ────
def plot_avg_fare_bar(df: pd.DataFrame):
    """
    Uses your actual logged data to show the real average fare
    per app at each hour. Only useful once you have real data.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    colors = {"ola":"#28a745", "uber":"#222222", "rapido":"#ff6b00"}
    labels = {"ola":"Ola", "uber":"Uber", "rapido":"Rapido"}

    for ax, app in zip(axes, CONFIG["apps"]):
        avg = df.groupby("hour")[f"{app}_fare"].mean()
        ax.bar(avg.index, avg.values, color=colors[app], alpha=0.85, width=0.7)
        ax.set_title(f"{labels[app]} — Avg Fare by Hour", fontsize=12, fontweight="bold")
        ax.set_xlabel("Hour"); ax.set_ylabel("Avg Fare (₹)")
        ax.set_xticks(range(24))
        ax.set_xticklabels([str(h) for h in range(24)], rotation=45, fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Average Auto Fare by Hour — Hinjewadi Phase 1 → Phase 3",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("avg_fare_by_hour.png", dpi=150)
    plt.show()
    print("📊 Saved → avg_fare_by_hour.png")


# ── 13. SUMMARY: How often is each app cheapest? ─────────────
def print_cheapest_summary(df: pd.DataFrame):
    counts = df["cheapest_app"].value_counts()
    total  = len(df)
    print("\n📊 Cheapest App Frequency (across all observations):")
    print(f"  {'App':10s} {'Count':>8s}  {'%':>6s}")
    print(f"  {'-'*28}")
    for app, count in counts.items():
        bar = "█" * int((count/total)*30)
        print(f"  {app.title():10s} {count:>8d}  {count/total*100:>5.1f}%  {bar}")


# ── 14. MAIN ──────────────────────────────────────────────────
if __name__ == "__main__":

    print("="*55)
    print("  AUTO FARE PREDICTOR — HINJEWADI PHASE 1 → PHASE 3")
    print("="*55)

    # Step 1: Setup
    create_dataset_template()

    # Step 2: Load data
    # ── DEMO: uses synthetic data ──────────────────────────────
    # ── REAL: comment out generate_synthetic_data() and use: ──
    #    df = pd.read_csv(CONFIG["dataset_path"])
    df = generate_synthetic_data(n=2000)

    # Step 3: Train
    models, encoders = train_models(df)

    # Step 4: Predict right now
    print("\n" + "="*55)
    predict_all_apps()

    # Step 5: Predict for specific scenario
    print("──── Scenario: Rainy Monday 8AM with Uber surge ────")
    predict_all_apps(
        hour=8, is_weekend=0, is_raining=1,
        weather_condition="Heavy Rain",
        uber_surge_active=1, uber_surge_multiplier=2.0,
        ola_surge_active=1,  ola_surge_multiplier=1.4,
    )

    # Step 6: Charts
    print("📈 Generating charts...")
    plot_hourly_fares(models, encoders, is_raining=0, is_weekend=0, day_name="Monday")
    plot_hourly_fares(models, encoders, is_raining=1, is_weekend=0, day_name="Tuesday")
    plot_cheapest_heatmap(models, encoders)
    plot_avg_fare_bar(df)
    print_cheapest_summary(df)

    print("\n✅ Done! Files: hourly_fares.png, cheapest_heatmap.png, avg_fare_by_hour.png")


# ════════════════════════════════════════════════════
# HOW TO LOG REAL DATA (run this every day)
# ════════════════════════════════════════════════════
#
# Every time you open all 3 apps, just call:
#
# log_fare_entry(
#     ola_fare=72.0,
#     uber_fare=95.0,  uber_surge_active=1, uber_surge_multiplier=1.6,
#     rapido_fare=55.0,
#     is_raining=0
# )
#
# After ~60-90 days of logging (~2-3x per day), you'll have
# real enough data to replace the synthetic data entirely.
# ════════════════════════════════════════════════════