"""
=====================================================
  AUTO FARE PREDICTOR — LINEAR REGRESSION
  Blue Ridge Tower 20 → Capgemini Phase 3, Pune
  Distance: 6.1 km
=====================================================
pip install pandas numpy scikit-learn matplotlib
"""

import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── CONFIGURATION ─────────────────────────────────────────────
DISTANCE_KM = 6.1  # Blue Ridge Tower 20 → Capgemini Phase 3
CSV_FILE    = "fare_data.csv"

# ── CREATE BLANK CSV TEMPLATE ─────────────────────────────────
def create_csv_template():
    """Creates empty CSV with correct column headers."""
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['day', 'hour', 'ola_price', 'uber_price', 'rapido_price', 'distance'])
        print(f"✅ Created blank CSV: {CSV_FILE}")
    else:
        print(f"📂 CSV exists: {CSV_FILE}")


# ── MANUAL LOGGING FUNCTION ───────────────────────────────────
def log_fare(ola_price, uber_price, rapido_price):
    """
    Call this every time you check all 3 apps.
    
    Example:
        log_fare(ola_price=75, uber_price=92, rapido_price=58)
    """
    now  = datetime.now()
    day  = now.strftime("%A")
    hour = now.hour
    
    with open(CSV_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([day, hour, ola_price, uber_price, rapido_price, DISTANCE_KM])
    
    print(f"✅ Logged at {hour:02d}:00 on {day}")
    print(f"   Ola: ₹{ola_price}  |  Uber: ₹{uber_price}  |  Rapido: ₹{rapido_price}")


# ── GENERATE SYNTHETIC DATA (for testing) ─────────────────────
def generate_synthetic_data(n_rows=500):
    """
    Generates fake data for testing. Delete this once you have real data.
    Each row = one fare check at a specific hour and day.
    """
    np.random.seed(42)
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    rows = []
    
    # Base prices for 6.1 km route
    base = {'ola': 65, 'uber': 78, 'rapido': 50}
    
    for _ in range(n_rows):
        day   = np.random.choice(days)
        hour  = np.random.randint(0, 24)
        
        # Peak hours have higher prices
        is_peak = (7 <= hour <= 10) or (17 <= hour <= 21)
        surge   = 1.4 if is_peak else 1.0
        
        # Weekend slight premium
        is_weekend = day in ['Saturday', 'Sunday']
        weekend_add = 8 if is_weekend else 0
        
        # Calculate fares with randomness
        ola    = base['ola'] * surge + weekend_add + np.random.normal(0, 5)
        uber   = base['uber'] * surge + weekend_add + np.random.normal(0, 5)
        rapido = base['rapido'] * surge + weekend_add + np.random.normal(0, 5)
        
        # Minimum fare floor
        ola    = max(ola, 40)
        uber   = max(uber, 45)
        rapido = max(rapido, 35)
        
        rows.append([day, hour, round(ola, 2), round(uber, 2), round(rapido, 2), DISTANCE_KM])
    
    df = pd.DataFrame(rows, columns=['day', 'hour', 'ola_price', 'uber_price', 'rapido_price', 'distance'])
    df.to_csv(CSV_FILE, index=False)
    print(f"✅ Generated {n_rows} rows of synthetic data → {CSV_FILE}")
    return df


# ── TRAIN LINEAR REGRESSION MODELS ────────────────────────────
def train_models(df):
    """
    Trains 3 separate Linear Regression models:
    - One for Ola
    - One for Uber
    - One for Rapido
    
    Features: day (encoded), hour, distance
    """
    print("\n📊 Training Linear Regression models...\n")
    
    # Encode 'day' as numbers (Monday=0, Tuesday=1, etc.)
    le = LabelEncoder()
    df['day_encoded'] = le.fit_transform(df['day'])
    
    # Features
    X = df[['day_encoded', 'hour', 'distance']]
    
    models = {}
    
    for app, price_col in [('ola', 'ola_price'), ('uber', 'uber_price'), ('rapido', 'rapido_price')]:
        y = df[price_col]
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Linear Regression
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Evaluate
        predictions = model.predict(X_test)
        mae  = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2   = r2_score(y_test, predictions)
        
        models[app] = model
        
        print(f"  {app.upper():8s} → MAE: ₹{mae:.2f}  |  RMSE: ₹{rmse:.2f}  |  R²: {r2:.3f}")
    
    print(f"\n💡 MAE = average error in ₹ (lower is better)")
    print(f"   R²  = 1.0 is perfect, 0.0 is random guessing")
    
    return models, le


# ── PREDICT CHEAPEST APP ──────────────────────────────────────
def predict_cheapest(models, label_encoder, hour=None, day=None):
    """
    Predicts fare for all 3 apps at a given hour and day.
    Shows which app is cheapest.
    """
    now  = datetime.now()
    hour = hour if hour is not None else now.hour
    day  = day  if day  is not None else now.strftime("%A")
    
    # Encode the day
    day_encoded = label_encoder.transform([day])[0]
    
    # Prepare input
    X_input = pd.DataFrame([[day_encoded, hour, DISTANCE_KM]], 
                           columns=['day_encoded', 'hour', 'distance'])
    
    # Predict for each app
    predictions = {}
    for app, model in models.items():
        predictions[app] = round(model.predict(X_input)[0], 2)
    
    # Find cheapest
    cheapest = min(predictions, key=predictions.get)
    
    # Display
    print(f"\n{'='*50}")
    print(f"  Day      : {day}")
    print(f"  Hour     : {hour:02d}:00")
    print(f"  Distance : {DISTANCE_KM} km")
    print(f"{'='*50}")
    print(f"  {'App':10s} {'Predicted Fare':>16s}")
    print(f"  {'-'*30}")
    
    for app, fare in sorted(predictions.items(), key=lambda x: x[1]):
        tag = "  ← 🏆 CHEAPEST" if app == cheapest else ""
        print(f"  {app.title():10s} ₹{fare:>14.2f}{tag}")
    
    print(f"{'='*50}\n")
    
    return predictions, cheapest


# ── VISUALIZATION: Hourly Fare Chart ──────────────────────────
def plot_hourly_fares(models, label_encoder, day="Monday"):
    """Line chart showing predicted fare for each app across 24 hours."""
    colors = {'ola': '#28a745', 'uber': '#222222', 'rapido': '#ff6b00'}
    
    hours = list(range(24))
    day_encoded = label_encoder.transform([day])[0]
    
    results = {app: [] for app in models.keys()}
    
    for hour in hours:
        X_input = pd.DataFrame([[day_encoded, hour, DISTANCE_KM]], 
                               columns=['day_encoded', 'hour', 'distance'])
        for app, model in models.items():
            fare = model.predict(X_input)[0]
            results[app].append(round(fare, 2))
    
    # Plot
    plt.figure(figsize=(14, 6))
    for app, fares in results.items():
        plt.plot(hours, fares, label=app.title(), color=colors[app], 
                 linewidth=2.5, marker='o', markersize=5)
    
    # Highlight peak hours
    plt.axvspan(7, 10, alpha=0.1, color='red', label='Morning Peak')
    plt.axvspan(17, 21, alpha=0.1, color='blue', label='Evening Peak')
    
    plt.title(f"Predicted Auto Fare by Hour — {day}\nBlue Ridge Tower 20 → Capgemini Phase 3 ({DISTANCE_KM} km)", 
              fontsize=13, fontweight='bold')
    plt.xlabel("Hour of Day", fontsize=12)
    plt.ylabel("Fare (₹)", fontsize=12)
    plt.xticks(hours, [f"{h}:00" for h in hours], rotation=45, ha='right', fontsize=8)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("hourly_fares.png", dpi=150)
    plt.show()
    print("📊 Saved → hourly_fares.png")


# ── VISUALIZATION: Cheapest App Heatmap ───────────────────────
def plot_cheapest_heatmap(models, label_encoder):
    """Heatmap showing which app is cheapest at each day×hour."""
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    app_to_num = {'ola': 0, 'uber': 1, 'rapido': 2}
    
    heatmap_data = np.zeros((7, 24))
    fare_data    = np.zeros((7, 24))
    
    for d_idx, day in enumerate(days):
        day_encoded = label_encoder.transform([day])[0]
        
        for hour in range(24):
            X_input = pd.DataFrame([[day_encoded, hour, DISTANCE_KM]], 
                                   columns=['day_encoded', 'hour', 'distance'])
            
            fares = {}
            for app, model in models.items():
                fares[app] = model.predict(X_input)[0]
            
            cheapest = min(fares, key=fares.get)
            heatmap_data[d_idx][hour] = app_to_num[cheapest]
            fare_data[d_idx][hour]    = round(fares[cheapest], 1)
    
    # Plot
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['#28a745', '#222222', '#ff6b00'])
    
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.imshow(heatmap_data, cmap=cmap, aspect='auto', vmin=-0.5, vmax=2.5)
    
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h}:00" for h in range(24)], rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(7))
    ax.set_yticklabels(days, fontsize=10)
    ax.set_title(f"Cheapest App by Hour & Day\nBlue Ridge Tower 20 → Capgemini Phase 3 ({DISTANCE_KM} km)", 
                 fontsize=13, fontweight='bold')
    
    # Add fare amounts inside cells
    for d in range(7):
        for h in range(24):
            ax.text(h, d, f"₹{fare_data[d][h]:.0f}", 
                    ha='center', va='center', fontsize=7, color='white', fontweight='bold')
    
    # Legend
    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor='#28a745', label='Ola'),
        Patch(facecolor='#222222', label='Uber'),
        Patch(facecolor='#ff6b00', label='Rapido')
    ]
    ax.legend(handles=legend, loc='upper right', bbox_to_anchor=(1.08, 1), fontsize=11)
    
    plt.tight_layout()
    plt.savefig("cheapest_heatmap.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("🗺️  Saved → cheapest_heatmap.png")


# ── MAIN EXECUTION ────────────────────────────────────────────
if __name__ == "__main__":
    
    print("="*55)
    print("  AUTO FARE PREDICTOR — LINEAR REGRESSION")
    print(f"  Blue Ridge Tower 20 → Capgemini Phase 3")
    print(f"  Distance: {DISTANCE_KM} km")
    print("="*55)
    
    # Step 1: Create CSV template
    create_csv_template()
    
    # Step 2: Generate data (REPLACE THIS with real data later)
    # Once you have real data, comment out the line below and just use:
    # df = pd.read_csv(CSV_FILE)
    df = generate_synthetic_data(n_rows=500)
    
    # Step 3: Train models
    models, label_encoder = train_models(df)
    
    # Step 4: Predict for current time
    print("\n" + "="*55)
    print("  PREDICTION FOR CURRENT HOUR")
    print("="*55)
    predict_cheapest(models, label_encoder)
    
    # Step 5: Test specific scenarios
    print("── Prediction for Monday 8AM ──")
    predict_cheapest(models, label_encoder, hour=8, day="Monday")
    
    print("── Prediction for Friday 6PM ──")
    predict_cheapest(models, label_encoder, hour=18, day="Friday")
    
    # Step 6: Generate charts
    print("\n📈 Generating visualizations...")
    plot_hourly_fares(models, label_encoder, day="Monday")
    plot_cheapest_heatmap(models, label_encoder)
    
    print("\n✅ Done!")
    print(f"   📄 {CSV_FILE}")
    print("   📊 hourly_fares.png")
    print("   🗺️  cheapest_heatmap.png")


# ════════════════════════════════════════════════════════
#  HOW TO LOG REAL DATA
# ════════════════════════════════════════════════════════
# Every time you check all 3 apps, run:
#
#   log_fare(ola_price=75, uber_price=92, rapido_price=58)
#
# After 100+ real entries, comment out generate_synthetic_data()
# and switch to: df = pd.read_csv(CSV_FILE)
# ════════════════════════════════════════════════════════


