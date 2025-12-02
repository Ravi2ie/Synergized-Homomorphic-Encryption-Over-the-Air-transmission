# app_complete.py
import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import uuid
import io
import base64
import matplotlib.pyplot as plt

# Crypto libs
try:
    import tenseal as ts
    TENSEAL_AVAILABLE = True
except Exception as e:
    ts = None
    TENSEAL_AVAILABLE = False

from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization

# Map libs (folium fallback)
try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except Exception:
    FOLIUM_AVAILABLE = False

# ----------------------
# 1. Driver Data Simulation
# ----------------------
def simulate_driver_data(num_drivers=5, reports_per_driver=1, seed=None):
    if seed is not None:
        np.random.seed(seed)
    data = []
    start_time = datetime.now()
    for driver_id in range(1, num_drivers + 1):
        device_id = str(uuid.uuid4())[:8]
        for i in range(reports_per_driver):
            timestamp = start_time + timedelta(minutes=i*5)
            lat = np.random.uniform(12.90, 13.00)
            lon = np.random.uniform(77.50, 77.60)
            rides = np.random.randint(1, 6)
            earnings = round(np.random.uniform(100, 500), 2)
            data.append({
                "driver_id": driver_id,
                "device_id": device_id,
                "timestamp": timestamp,
                "lat": lat,
                "lon": lon,
                "rides": rides,
                "earnings": earnings
            })
    return pd.DataFrame(data)

# ----------------------
# 2. OTA Helper Functions
# ----------------------
def apply_power_scaling(local_data, path_losses):
    scaled = local_data / path_losses
    scaled *= np.sum(local_data) / np.sum(scaled)
    return scaled

def simulate_ota_transmission(transmitted_signals, num_repeats, noise_std):
    num_users = len(transmitted_signals)
    received_matrix = np.zeros((num_users, num_repeats))
    for i in range(num_repeats):
        noise = np.random.normal(0, noise_std, size=num_users)
        received_matrix[:, i] = transmitted_signals + noise
    return received_matrix

def denoise_received_signals(received_matrix):
    return np.mean(received_matrix, axis=1)

def aggregate_ota_signals(denoised_signals):
    total = np.sum(denoised_signals)
    avg = total / len(denoised_signals)
    return total, avg

# ----------------------
# 3. HE-only Aggregation (TenSEAL CKKS)
# ----------------------
def he_setup_context():
    # Return context and a dict of parameters for visualization
    if not TENSEAL_AVAILABLE:
        return None, {"poly_modulus_degree": None, "coeff_mod_bit_sizes": None, "global_scale": None}
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.generate_galois_keys()
    context.global_scale = 2**40
    params = {
        "poly_modulus_degree": 8192,
        "coeff_mod_bit_sizes": [60, 40, 40, 60],
        "global_scale": "2**40 (approx 1.0995e12)",
        "security_approx": "≈128-bit (depends on params)"
    }
    return context, params

def he_aggregation(local_data):
    context, params = he_setup_context()
    if context is None:
        raise RuntimeError("TenSEAL not available in this environment.")
    t0 = time.perf_counter()
    encrypted_data = [ts.ckks_vector(context, [float(x)]) for x in local_data]
    enc_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    encrypted_sum = encrypted_data[0]
    for enc in encrypted_data[1:]:
        encrypted_sum += enc
    agg_time = time.perf_counter() - t1

    t2 = time.perf_counter()
    decrypted_sum = encrypted_sum.decrypt()[0]
    dec_time = time.perf_counter() - t2

    decrypted_avg = decrypted_sum / len(local_data)
    return {
        "enc_list": encrypted_data,
        "encrypted_sum": encrypted_sum,
        "decrypted_sum": decrypted_sum,
        "decrypted_avg": decrypted_avg,
        "times": {"enc_time": enc_time, "agg_time": agg_time, "dec_time": dec_time},
        "params": params
    }

# ----------------------
# 4. OTA-only Aggregation
# ----------------------
def ota_only_aggregation(local_data, path_losses, num_repeats=50, noise_std=1.0):
    t0 = time.perf_counter()
    transmitted_signals = apply_power_scaling(local_data, path_losses)
    received_matrix = simulate_ota_transmission(transmitted_signals, num_repeats, noise_std)
    denoised_signals = denoise_received_signals(received_matrix)
    ota_sum, ota_avg = aggregate_ota_signals(denoised_signals)
    t1 = time.perf_counter()
    return {
        "transmitted": transmitted_signals,
        "denoised": denoised_signals,
        "ota_sum": ota_sum,
        "ota_avg": ota_avg,
        "times": {"ota_time": t1 - t0},
        "received_matrix": received_matrix
    }

# ----------------------
# 5. Hybrid HE + OTA Aggregation
# ----------------------
def hybrid_he_ota(local_data, path_losses, num_repeats=50, noise_std=1.0):
    # OTA part
    t0 = time.perf_counter()
    transmitted_signals = apply_power_scaling(local_data, path_losses)
    received_matrix = simulate_ota_transmission(transmitted_signals, num_repeats, noise_std)
    denoised_signals = denoise_received_signals(received_matrix)
    t1 = time.perf_counter()

    # HE part on denoised signals
    if not TENSEAL_AVAILABLE:
        raise RuntimeError("TenSEAL not available in this environment.")
    context, params = he_setup_context()
    t2 = time.perf_counter()
    encrypted_ota = [ts.ckks_vector(context, [float(x)]) for x in denoised_signals]
    t3 = time.perf_counter()
    encrypted_sum = encrypted_ota[0]
    for enc in encrypted_ota[1:]:
        encrypted_sum += enc
    t4 = time.perf_counter()
    decrypted_sum = encrypted_sum.decrypt()[0]
    decrypted_avg = decrypted_sum / len(local_data)
    t5 = time.perf_counter()

    return {
        "encrypted_list": encrypted_ota,
        "encrypted_sum": encrypted_sum,
        "decrypted_sum": decrypted_sum,
        "decrypted_avg": decrypted_avg,
        "times": {
            "ota_time": t1 - t0,
            "he_enc_time": t3 - t2,
            "he_agg_time": t4 - t3,
            "he_dec_time": t5 - t4
        },
        "denoised": denoised_signals,
        "params": params,
        "received_matrix": received_matrix
    }

# ----------------------
# 6. Metrics computation helpers
# ----------------------
def absolute_error(true_value, received_value):
    return float(np.abs(received_value - true_value))

def relative_error(true_value, received_value):
    if np.isclose(true_value, 0.0):
        return float(np.nan)
    return float(np.abs(received_value - true_value) / np.abs(true_value) * 100)

def mse(true_value, received_value):
    return float((received_value - true_value) ** 2)

def rmse(true_value, received_value):
    return float(np.sqrt(mse(true_value, received_value)))

def bias(true_value, received_value):
    return float(received_value - true_value)

def hmac_verification_rate(true_value, received_value, atol=1.0):
    # simple closeness test used as placeholder for HMAC success/fail demonstration
    return 1.0 if np.isclose(true_value, received_value, atol=atol).all() else 0.0

def availability_uptime_percent(noise_std):
    return float(max(0, 100 - noise_std*10))

# ----------------------
# 7. OTA Signing & Verification (RSA)
# ----------------------
def generate_rsa_keypair():
    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pub = priv.public_key()
    return priv, pub

def sign_payload_rsa(private_key, payload_bytes):
    signature = private_key.sign(
        payload_bytes,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
        hashes.SHA256()
    )
    return signature

def verify_payload_rsa(public_key, payload_bytes, signature):
    try:
        public_key.verify(
            signature,
            payload_bytes,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256()
        )
        return True
    except Exception:
        return False

# ----------------------
# 8. Utility: ciphertext size estimate + energy model
# ----------------------
def estimate_ciphertext_sizes(enc_list):
    sizes = []
    for enc in enc_list:
        try:
            b = enc.serialize()
            sizes.append(len(b))
        except Exception:
            # fallback conservative estimate
            sizes.append(4096)
    return sizes

def energy_estimate(method, num_items):
    # simplistic energy model (arbitrary units / relative)
    if method == "HE":
        return num_items * 1.5  # expensive per encrypt
    if method == "OTA":
        return num_items * 0.3  # cheaper transmit
    if method == "Hybrid":
        return num_items * 1.0
    return num_items * 0.5

# ===================== DRIVER VISUALIZATION FUNCTIONS =====================

def plot_driver_dashboard(df):
    """Driver performance dashboard with key metrics."""
    if len(df) == 0:
        st.warning("No data for dashboard")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Driver Performance Dashboard", fontsize=16, fontweight='bold')
    
    # 1. Total Earnings by Driver (Bar chart)
    ax = axes[0, 0]
    driver_earnings = df.groupby('driver_id')['earnings'].sum().sort_values(ascending=False)
    colors = plt.cm.viridis(np.linspace(0, 1, len(driver_earnings)))
    ax.barh(range(len(driver_earnings)), driver_earnings.values, color=colors, edgecolor='black', linewidth=1)
    ax.set_yticks(range(len(driver_earnings)))
    ax.set_yticklabels([f"Driver {d}" for d in driver_earnings.index])
    ax.set_xlabel("Total Earnings (₹)", fontsize=11)
    ax.set_title("Total Earnings by Driver", fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # 2. Total Rides by Driver (Pie chart)
    ax = axes[0, 1]
    driver_rides = df.groupby('driver_id')['rides'].sum()
    ax.pie(driver_rides.values, labels=[f"D{d}" for d in driver_rides.index], autopct='%1.1f%%', startangle=90, colors=colors)
    ax.set_title("Ride Distribution", fontsize=12, fontweight='bold')
    
    # 3. Avg Rides per Driver (Bar chart)
    ax = axes[1, 0]
    avg_rides = df.groupby('driver_id')['rides'].mean().sort_values(ascending=False)
    ax.bar(range(len(avg_rides)), avg_rides.values, color='skyblue', edgecolor='black', alpha=0.7)
    ax.set_xticks(range(len(avg_rides)))
    ax.set_xticklabels([f"D{d}" for d in avg_rides.index])
    ax.set_ylabel("Avg Rides per Report", fontsize=11)
    ax.set_title("Average Rides per Driver", fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # 4. Efficiency (Earnings per Ride)
    ax = axes[1, 1]
    driver_efficiency = (df.groupby('driver_id')['earnings'].sum() / df.groupby('driver_id')['rides'].sum()).sort_values(ascending=False)
    ax.barh(range(len(driver_efficiency)), driver_efficiency.values, color='mediumpurple', edgecolor='black', linewidth=1)
    ax.set_yticks(range(len(driver_efficiency)))
    ax.set_yticklabels([f"D{d}" for d in driver_efficiency.index])
    ax.set_xlabel("Earnings per Ride (₹)", fontsize=11)
    ax.set_title("Driver Efficiency", fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def plot_earnings_timeline(df):
    """Timeline of earnings over time by driver."""
    if len(df) == 0:
        st.warning("No data for timeline")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    df_sorted = df.sort_values('timestamp')
    for driver_id in sorted(df_sorted['driver_id'].unique()):
        driver_data = df_sorted[df_sorted['driver_id'] == driver_id]
        ax.plot(range(len(driver_data)), driver_data['earnings'].values, marker='o', label=f"Driver {driver_id}", linewidth=2, markersize=6)
    
    ax.set_xlabel("Report Index", fontsize=11)
    ax.set_ylabel("Earnings (₹)", fontsize=11)
    ax.set_title("Earnings Timeline by Driver", fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def plot_ride_distribution(df):
    """Ride distribution across drivers."""
    if len(df) == 0:
        st.warning("No data for ride distribution")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Total rides per driver (bar)
    ax = axes[0]
    driver_rides = df.groupby('driver_id')['rides'].sum().sort_values(ascending=False)
    colors = plt.cm.Set3(np.linspace(0, 1, len(driver_rides)))
    ax.bar(range(len(driver_rides)), driver_rides.values, color=colors, edgecolor='black', linewidth=1)
    ax.set_xticks(range(len(driver_rides)))
    ax.set_xticklabels([f"D{d}" for d in driver_rides.index])
    ax.set_ylabel("Total Rides", fontsize=11)
    ax.set_title("Total Rides by Driver", fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Right: Pie chart of ride distribution
    ax = axes[1]
    ax.pie(driver_rides.values, labels=[f"Driver {d}" for d in driver_rides.index], autopct='%1.1f%%', startangle=90)
    ax.set_title("Ride Distribution (%)", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def plot_driver_correlation_heatmap(df):
    """Correlation heatmap of driver metrics."""
    if len(df) < 2:
        st.warning("Not enough data for correlation heatmap")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Compute driver-level summary stats (use metrics that have variation)
    driver_stats = df.groupby('driver_id').agg({
        'earnings': ['mean', 'sum', 'min', 'max'],
        'rides': ['mean', 'sum'],
        'lat': ['mean', 'std'],
        'lon': ['mean', 'std']
    })
    
    # Flatten column names
    driver_stats.columns = ['_'.join(col).strip() for col in driver_stats.columns.values]
    
    # Replace inf and nan with 0 to avoid issues in correlation
    driver_stats = driver_stats.replace([np.inf, -np.inf], 0).fillna(0)
    
    # Correlation matrix
    corr_data = driver_stats.corr()
    
    # Replace any remaining NaN in correlation with 0
    corr_data = corr_data.fillna(0)
    
    # Create heatmap
    im = ax.imshow(corr_data.values, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(corr_data)))
    ax.set_yticks(np.arange(len(corr_data)))
    ax.set_xticklabels(corr_data.columns, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(corr_data.columns, fontsize=8)
    
    # Add correlation values as text
    for i in range(len(corr_data)):
        for j in range(len(corr_data)):
            val = corr_data.values[i, j]
            text_color = 'white' if abs(val) > 0.5 else 'black'
            text = ax.text(j, i, f'{val:.2f}',
                          ha="center", va="center", color=text_color, fontsize=7, fontweight='bold')
    
    ax.set_title("Driver Metrics Correlation Heatmap", fontsize=14, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, label='Correlation Coefficient')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ----------------------
# 9. Visualizations helpers
# ----------------------
def plot_histograms(local_data, quantized_data):
    fig, ax = plt.subplots(1,2, figsize=(10,4))
    ax[0].hist(local_data, bins=8, alpha=0.7, label="Raw earnings")
    ax[0].set_title("Raw Earnings Distribution")
    ax[1].hist(quantized_data, bins=8, alpha=0.7, color='orange', label="Quantized earnings")
    ax[1].set_title("Quantized Earnings Distribution")
    st.pyplot(fig)

def plot_radar(metrics_dict):
    labels = ['Privacy', 'Integrity', 'Availability']
    categories = list(metrics_dict.keys())
    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    for cat in categories:
        values = metrics_dict[cat]
        values += values[:1]
        ax.plot(angles, values, label=cat)
        ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    ax.set_title("Privacy / Integrity / Availability Radar")
    ax.legend()
    st.pyplot(fig)

def show_map(df):
    if FOLIUM_AVAILABLE:
        m = folium.Map(location=[12.95, 77.55], zoom_start=12)
        for _, row in df.iterrows():
            folium.CircleMarker([row['lat'], row['lon']], radius=4, color="blue", fill=True).add_to(m)
        st_folium(m, width=700, height=400)
    else:
        # fallback scatter plot
        fig, ax = plt.subplots(figsize=(6,6))
        ax.scatter(df['lon'], df['lat'], s=20)
        ax.set_title("Driver Locations (scatter fallback)")
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        st.pyplot(fig)

# ----------------------
# 11. Driver Analytics & Heatmaps
# ----------------------
def plot_driver_earnings_heatmap(df):
    """Heatmap of driver earnings (driver_id x time)."""
    # Pivot: rows=driver_id, cols=timestamp
    if len(df) == 0:
        st.warning("No data to plot heatmap")
        return
    
    # Create a simplified heatmap: driver earnings
    pivot_data = df.groupby('driver_id')['earnings'].agg(['mean', 'sum', 'std', 'count'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot_data.values.reshape(1, -1), aspect='auto', cmap='YlGn', interpolation='nearest')
    ax.set_yticks([0])
    ax.set_yticklabels(['Earnings Stats'])
    ax.set_xticks(range(len(pivot_data)))
    ax.set_xticklabels([f"D{i+1}" for i in range(len(pivot_data))], rotation=0)
    ax.set_title("Driver Earnings Heatmap (Mean)", fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean Earnings (₹)', rotation=270, labelpad=20)
    
    # Add values on heatmap
    for i, (idx, row) in enumerate(pivot_data.iterrows()):
        ax.text(i, 0, f"₹{row['mean']:.0f}", ha='center', va='center', color='black', fontweight='bold')
    
    st.pyplot(fig)
    st.dataframe(pivot_data.round(2))

def plot_driver_activity_heatmap(df):
    """Heatmap of rides per driver (activity matrix)."""
    if len(df) == 0:
        st.warning("No data for activity heatmap")
        return
    
    # Group by driver and time (if multiple timestamps per driver)
    activity_pivot = df.pivot_table(values='rides', index='driver_id', aggfunc='sum', fill_value=0)
    
    fig, ax = plt.subplots(figsize=(10, max(4, len(activity_pivot) * 0.5)))
    im = ax.imshow(activity_pivot.values, aspect='auto', cmap='Blues', interpolation='nearest')
    ax.set_yticks(range(len(activity_pivot)))
    ax.set_yticklabels([f"Driver {i+1}" for i in range(len(activity_pivot))])
    ax.set_xticks([0])
    ax.set_xticklabels(['Total Rides'])
    ax.set_title("Driver Activity Heatmap (Total Rides)", fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Number of Rides', rotation=270, labelpad=20)
    
    # Add values
    for i, (idx, row) in enumerate(activity_pivot.iterrows()):
        ax.text(0, i, f"{int(row.values[0])}", ha='center', va='center', color='white', fontweight='bold')
    
    st.pyplot(fig)

def plot_geographic_heatmap(df):
    """Geographic heatmap: earnings by location (lat/lon bins)."""
    if len(df) == 0:
        st.warning("No location data for geographic heatmap")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create 2D histogram of earnings at each location
    h = ax.hist2d(df['lon'], df['lat'], bins=10, weights=df['earnings'], cmap='RdYlGn', cmin=0)
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    ax.set_title("Geographic Earnings Heatmap (Weighted by Earnings)", fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(h[3], ax=ax)
    cbar.set_label('Total Earnings (₹)', rotation=270, labelpad=20)
    
    st.pyplot(fig)

def plot_driver_performance_dashboard(df):
    """Multi-panel driver performance metrics."""
    if len(df) == 0:
        st.warning("No data for performance dashboard")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel 1: Total Earnings by Driver
    driver_earnings = df.groupby('driver_id')['earnings'].sum().sort_values(ascending=False)
    axes[0, 0].barh(range(len(driver_earnings)), driver_earnings.values, color='steelblue')
    axes[0, 0].set_yticks(range(len(driver_earnings)))
    axes[0, 0].set_yticklabels([f"Driver {i+1}" for i in range(len(driver_earnings))])
    axes[0, 0].set_xlabel("Total Earnings (₹)")
    axes[0, 0].set_title("Total Earnings by Driver", fontweight='bold')
    for i, v in enumerate(driver_earnings.values):
        axes[0, 0].text(v, i, f" ₹{v:.0f}", va='center')
    
    # Panel 2: Average Rides per Driver
    driver_rides = df.groupby('driver_id')['rides'].mean()
    axes[0, 1].bar(range(len(driver_rides)), driver_rides.values, color='coral')
    axes[0, 1].set_xticks(range(len(driver_rides)))
    axes[0, 1].set_xticklabels([f"D{i+1}" for i in range(len(driver_rides))])
    axes[0, 1].set_ylabel("Avg Rides per Report")
    axes[0, 1].set_title("Average Rides per Driver", fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Panel 3: Earnings Distribution (box plot)
    earnings_by_driver = [df[df['driver_id'] == d]['earnings'].values for d in sorted(df['driver_id'].unique())]
    bp = axes[1, 0].boxplot(earnings_by_driver, labels=[f"D{i+1}" for i in range(len(earnings_by_driver))], patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightgreen')
    axes[1, 0].set_ylabel("Earnings (₹)")
    axes[1, 0].set_title("Earnings Distribution by Driver", fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Panel 4: Efficiency (earnings per ride)
    driver_efficiency = (df.groupby('driver_id')['earnings'].sum() / df.groupby('driver_id')['rides'].sum()).sort_values(ascending=False)
    axes[1, 1].barh(range(len(driver_efficiency)), driver_efficiency.values, color='mediumpurple')
    axes[1, 1].set_yticks(range(len(driver_efficiency)))
    axes[1, 1].set_yticklabels([f"Driver {i+1}" for i in range(len(driver_efficiency))])
    axes[1, 1].set_xlabel("Earnings per Ride (₹/ride)")
    axes[1, 1].set_title("Driver Efficiency (Earnings/Ride)", fontweight='bold')
    for i, v in enumerate(driver_efficiency.values):
        axes[1, 1].text(v, i, f" ₹{v:.1f}", va='center')
    
    plt.tight_layout()
    st.pyplot(fig)

def plot_earnings_timeline(df):
    """Time-series plot of earnings over reports."""
    if len(df) == 0:
        st.warning("No data for timeline")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sort by timestamp and plot earnings for each driver
    df_sorted = df.sort_values('timestamp')
    for driver_id in sorted(df_sorted['driver_id'].unique()):
        driver_data = df_sorted[df_sorted['driver_id'] == driver_id]
        ax.plot(range(len(driver_data)), driver_data['earnings'].values, marker='o', label=f"Driver {driver_id}", linewidth=2)
    
    ax.set_xlabel("Report Index", fontsize=11)
    ax.set_ylabel("Earnings (₹)", fontsize=11)
    ax.set_title("Earnings Timeline (per Driver)", fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    
    st.pyplot(fig)

def plot_ride_distribution(df):
    """Ride count distribution across drivers."""
    if len(df) == 0:
        st.warning("No data for ride distribution")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Total rides per driver
    total_rides = df.groupby('driver_id')['rides'].sum().sort_values(ascending=False)
    ax1.bar(range(len(total_rides)), total_rides.values, color='skyblue', edgecolor='navy')
    ax1.set_xticks(range(len(total_rides)))
    ax1.set_xticklabels([f"D{i+1}" for i in range(len(total_rides))])
    ax1.set_ylabel("Total Rides")
    ax1.set_title("Total Rides by Driver", fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Pie chart: ride distribution
    ax2.pie(total_rides.values, labels=[f"Driver {i+1}" for i in range(len(total_rides))], autopct='%1.1f%%', startangle=90)
    ax2.set_title("Ride Distribution (%)", fontweight='bold')
    
    st.pyplot(fig)

def plot_driver_correlation_heatmap(df):
    """Correlation heatmap: earnings vs rides vs location variation."""
    if len(df) < 2:
        st.warning("Not enough data for correlation heatmap")
        return
    
    # Compute summary stats per driver
    driver_stats = df.groupby('driver_id').agg({
        'earnings': ['mean', 'std'],
        'rides': ['mean', 'std'],
        'lat': 'std',
        'lon': 'std'
    }).round(3)
    
    # Flatten column names
    driver_stats.columns = ['_'.join(col).strip() for col in driver_stats.columns.values]
    
    # Correlation matrix
    corr_matrix = driver_stats.corr()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr_matrix.values, aspect='auto', cmap='coolwarm', vmin=-1, vmax=1)
    
    ax.set_xticks(range(len(corr_matrix)))
    ax.set_yticks(range(len(corr_matrix)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(corr_matrix.columns, fontsize=9)
    ax.set_title("Driver Metrics Correlation Heatmap", fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation', rotation=270, labelpad=20)
    
    # Add correlation values to heatmap
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            text = ax.text(j, i, f'{corr_matrix.values[i, j]:.2f}', ha='center', va='center', 
                          color='white' if abs(corr_matrix.values[i, j]) > 0.5 else 'black', fontsize=8)
    
    plt.tight_layout()
    st.pyplot(fig)

# ----------------------
# 10. Streamlit UI
# ----------------------
st.set_page_config(layout="wide", page_title="HE + OTA Hybrid Simulator — Complete")
st.title("HE / OTA / Hybrid Aggregation Simulator — Complete")

# left sidebar: simulation parameters
with st.sidebar:
    st.header("Simulation Settings")
    num_drivers = st.number_input("Number of Drivers", min_value=1, max_value=500, value=8, step=1)
    reports_per_driver = st.number_input("Reports per Driver", min_value=1, max_value=50, value=1, step=1)
    noise_std = st.slider("OTA Noise Std", min_value=0.0, max_value=5.0, value=1.0)
    num_repeats = st.slider("OTA repeats (averaging)", min_value=1, max_value=200, value=50)
    seed = st.number_input("Random seed (0 = random)", min_value=0, value=0)
    st.markdown("---")
    st.header("Security & Attacks")
    enable_attacks = st.checkbox("Simulate Attacks (tamper/jamming)", value=False)
    enable_ota_sign = st.checkbox("Enable OTA Signing + Verification (RSA)", value=True)
    st.markdown("---")
    st.header("Real-time options")
    realtime = st.checkbox("Enable real-time simulation (auto-run)", value=False)
    realtime_steps = st.number_input("Real-time steps", min_value=1, max_value=50, value=5)
    realtime_delay = st.slider("Step delay (s)", min_value=0.1, max_value=2.0, value=0.5)
    st.markdown("---")
    st.header("Export / Presentation")
    enable_export = st.checkbox("Enable CSV Export", value=True)

# Session state for keys and simple persisted state
if "rsa_keys" not in st.session_state:
    st.session_state.rsa_keys = generate_rsa_keypair()
if "run_id" not in st.session_state:
    st.session_state.run_id = 0

# Run simulation button
if st.button("Run Simulation") or realtime:
    st.session_state.run_id += 1
    if seed == 0:
        seed_val = None
    else:
        seed_val = int(seed)

    # Data
    df = simulate_driver_data(num_drivers, reports_per_driver, seed=seed_val)
    st.subheader("Simulated Driver Data")
    st.dataframe(df)

    LOCAL_DATA = df['earnings'].values.astype(float)
    PATH_LOSSES = np.random.uniform(0.8, 1.2, size=len(LOCAL_DATA))

    # Store results for final dashboard
    all_results = []

    # If realtime: run multiple steps and update UI
    steps = realtime_steps if realtime else 1
    for step in range(steps):
        # optionally alter data a bit each step to simulate live changes
        if steps > 1 and step > 0:
            # small random walk in earnings and GPS
            df['earnings'] = df['earnings'] * (1 + np.random.normal(0, 0.01, size=len(df)))
            df['lat'] += np.random.normal(0, 0.0005, size=len(df))
            df['lon'] += np.random.normal(0, 0.0005, size=len(df))
            LOCAL_DATA = df['earnings'].values.astype(float)

        # HE pipeline
        he_result = None
        he_times = {"enc_time": np.nan, "agg_time": np.nan, "dec_time": np.nan}
        he_cipher_sizes = []
        he_decrypted_sum = np.nan
        he_avg = np.nan
        he_params = {}
        if TENSEAL_AVAILABLE:
            try:
                t_he_start = time.perf_counter()
                he_res = he_aggregation(LOCAL_DATA)
                t_he_total = time.perf_counter() - t_he_start
                he_result = he_res
                he_times = he_res["times"]
                he_decrypted_sum = float(he_res["decrypted_sum"])
                he_avg = float(he_res["decrypted_avg"])
                he_params = he_res.get("params", {})
                he_cipher_sizes = estimate_ciphertext_sizes(he_res["enc_list"])
            except Exception as e:
                st.warning(f"HE pipeline failed: {e}")
        else:
            st.info("TenSEAL not installed — HE pipeline skipped. Install 'tenseal' to enable HE.")

        # OTA pipeline
        t_ota_start = time.perf_counter()
        ota_res = ota_only_aggregation(LOCAL_DATA, PATH_LOSSES, num_repeats=num_repeats, noise_std=noise_std)
        t_ota_total = time.perf_counter() - t_ota_start
        ota_sum = float(ota_res["ota_sum"])
        ota_avg = float(ota_res["ota_avg"])
        ota_times = ota_res["times"]
        ota_received_matrix = ota_res.get("received_matrix")

        # Hybrid pipeline
        hybrid_res = None
        hybrid_times = {}
        hybrid_decrypted_sum = np.nan
        hybrid_avg = np.nan
        hybrid_cipher_sizes = []
        if TENSEAL_AVAILABLE:
            try:
                t_h_start = time.perf_counter()
                hybrid_res = hybrid_he_ota(LOCAL_DATA, PATH_LOSSES, num_repeats=num_repeats, noise_std=noise_std)
                t_h_total = time.perf_counter() - t_h_start
                hybrid_times = hybrid_res["times"]
                hybrid_decrypted_sum = float(hybrid_res["decrypted_sum"])
                hybrid_avg = float(hybrid_res["decrypted_avg"])
                hybrid_cipher_sizes = estimate_ciphertext_sizes(hybrid_res["encrypted_list"])
            except Exception as e:
                st.warning(f"Hybrid pipeline failed: {e}")

        # Apply attacks if requested
        if enable_attacks:
            # OTA tamper: add bias to OTA result
            ota_sum += np.random.uniform(30, 150)
            ota_avg = ota_sum / len(LOCAL_DATA)
            # Jamming: increase noise effective
            # (we don't re-run OTA here; we just reflect degraded availability)
            # HE tamper: corrupt the HE decrypted_sum (simulate ciphertext tampering)
            if he_result is not None:
                he_decrypted_sum += np.random.uniform(-50, 50)
                he_avg = he_decrypted_sum / len(LOCAL_DATA)
            if hybrid_res is not None:
                hybrid_decrypted_sum += np.random.uniform(-30, 30)
                hybrid_avg = hybrid_decrypted_sum / len(LOCAL_DATA)

        # OTA signing: create a signed "update" payload example and verify
        ota_signature_ok = None
        ota_signature_size = 0
        if enable_ota_sign:
            priv, pub = st.session_state.rsa_keys
            payload = f"AGG_UPDATE|sum={ota_sum:.2f}|avg={ota_avg:.2f}|ts={datetime.utcnow().isoformat()}".encode()
            signature = sign_payload_rsa(priv, payload)
            ota_signature_size = len(signature)
            ota_signature_ok = verify_payload_rsa(pub, payload, signature)
        else:
            ota_signature_ok = None

        # Compute true sums / avg (plaintext baseline)
        true_sum = float(np.sum(LOCAL_DATA))
        true_avg = float(np.mean(LOCAL_DATA))

        # Metrics compute for each method
        metrics_results = {}
        availability = availability_uptime_percent(noise_std) - (10 if enable_attacks else 0)
        availability = max(0.0, availability)

        # HE metrics (if available)
        if not np.isnan(he_decrypted_sum):
            metrics_results["HE"] = {
                "Sum": {
                    "Value": he_decrypted_sum,
                    "Absolute Error": absolute_error(true_sum, he_decrypted_sum),
                    "Relative Error (%)": relative_error(true_sum, he_decrypted_sum),
                    "RMSE": rmse(true_sum, he_decrypted_sum),
                    "Bias": bias(true_sum, he_decrypted_sum),
                },
                "Average": {
                    "Value": he_avg,
                    "Absolute Error": absolute_error(true_avg, he_avg),
                    "Relative Error (%)": relative_error(true_avg, he_avg),
                },
                "Times": he_times,
                "CommBytesAvg": np.mean(he_cipher_sizes) if len(he_cipher_sizes) else np.nan,
                "EnergyEst": energy_estimate("HE", len(LOCAL_DATA)),
                "Availability": availability,
                "Params": he_params
            }
        else:
            metrics_results["HE"] = None

        # OTA metrics
        metrics_results["OTA"] = {
            "Sum": {
                "Value": ota_sum,
                "Absolute Error": absolute_error(true_sum, ota_sum),
                "Relative Error (%)": relative_error(true_sum, ota_sum),
                "RMSE": rmse(true_sum, ota_sum),
                "Bias": bias(true_sum, ota_sum),
            },
            "Average": {
                "Value": ota_avg,
                "Absolute Error": absolute_error(true_avg, ota_avg),
                "Relative Error (%)": relative_error(true_avg, ota_avg),
            },
            "Times": ota_times,
            "CommBytesAvg": ota_received_matrix.nbytes if ota_received_matrix is not None else np.nan,
            "SignatureOK": ota_signature_ok,
            "SignatureSize": ota_signature_size,
            "EnergyEst": energy_estimate("OTA", len(LOCAL_DATA)),
            "Availability": availability
        }

        # Hybrid metrics
        if not np.isnan(hybrid_decrypted_sum):
            metrics_results["Hybrid"] = {
                "Sum": {
                    "Value": hybrid_decrypted_sum,
                    "Absolute Error": absolute_error(true_sum, hybrid_decrypted_sum),
                    "Relative Error (%)": relative_error(true_sum, hybrid_decrypted_sum),
                    "RMSE": rmse(true_sum, hybrid_decrypted_sum),
                    "Bias": bias(true_sum, hybrid_decrypted_sum),
                },
                "Average": {
                    "Value": hybrid_avg,
                    "Absolute Error": absolute_error(true_avg, hybrid_avg),
                    "Relative Error (%)": relative_error(true_avg, hybrid_avg),
                },
                "Times": hybrid_times,
                "CommBytesAvg": np.mean(hybrid_cipher_sizes) if len(hybrid_cipher_sizes) else np.nan,
                "EnergyEst": energy_estimate("Hybrid", len(LOCAL_DATA)),
                "Availability": availability,
                "Params": hybrid_res.get("params", {}) if hybrid_res is not None else {}
            }
        else:
            metrics_results["Hybrid"] = None

        # Append to results list for export / dashboard
        all_results.append({
            "step": step,
            "true_sum": true_sum, "true_avg": true_avg,
            "he_sum": he_decrypted_sum, "he_avg": he_avg,
            "ota_sum": ota_sum, "ota_avg": ota_avg,
            "hybrid_sum": hybrid_decrypted_sum, "hybrid_avg": hybrid_avg,
            "metrics": metrics_results,
            "timestamp": datetime.utcnow().isoformat()
        })

        # Show per-step outputs (only final step if realtime)
        if (not realtime) or (realtime and step == steps-1):
            st.subheader("Results (final step shown)" if realtime else "Results")
            st.write(f"True sum: **{true_sum:.2f}**, True avg: **{true_avg:.2f}**")
            # show method metrics
            for method in ["HE", "OTA", "Hybrid"]:
                st.write("----")
                st.markdown(f"### {method} Metrics")
                m = metrics_results.get(method)
                if m is None:
                    st.write("Pipeline not available / failed")
                    continue
                st.write("**Sum**:", {k: v for k, v in m["Sum"].items() if k in ["Value", "Absolute Error", "Relative Error (%)", "RMSE", "Bias"]})
                st.write("**Average**:", m["Average"])
                st.write("Times (s):", m["Times"])
                st.write("Comm bytes (avg):", m.get("CommBytesAvg"))
                st.write("Energy estimate (arbitrary units):", m.get("EnergyEst"))
                if method == "OTA":
                    st.write("OTA signature OK:", m.get("SignatureOK"), "Signature size bytes:", m.get("SignatureSize"))

            # HE parameters display (from he_params / hybrid)
            st.markdown("### HE Parameters (CKKS)")
            if he_params:
                st.json(he_params)
            else:
                st.info("HE parameters not available (TenSEAL not installed or HE failed).")

            # Plots
            st.markdown("### Distribution Plots")
            plot_histograms(LOCAL_DATA, np.round(LOCAL_DATA))

            # ===== DRIVER ANALYTICS SECTION =====
            st.markdown("---")
            st.markdown("## 📊 Driver Analytics & Visualizations")
            
            # Create tabs for different driver visualizations
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📈 Driver Dashboard",
                "📉 Earnings Timeline",
                "🚗 Ride Distribution",
                "🔥 Correlation Heatmap",
                "📍 Geographic Heatmap"
            ])
            
            with tab1:
                st.subheader("Driver Performance Dashboard")
                try:
                    plot_driver_dashboard(df)
                except Exception as e:
                    st.error(f"Dashboard plot error: {e}")
            
            with tab2:
                st.subheader("Earnings Timeline Over Reports")
                try:
                    plot_earnings_timeline(df)
                except Exception as e:
                    st.error(f"Timeline plot error: {e}")
            
            with tab3:
                st.subheader("Ride Distribution Analysis")
                try:
                    plot_ride_distribution(df)
                except Exception as e:
                    st.error(f"Ride distribution plot error: {e}")
            
            with tab4:
                st.subheader("Driver Metrics Correlation")
                try:
                    plot_driver_correlation_heatmap(df)
                except Exception as e:
                    st.error(f"Correlation heatmap error: {e}")
            
            with tab5:
                st.subheader("Geographic Heatmap (Earnings Density)")
                try:
                    # Create 2D heatmap of earnings by location
                    fig, ax = plt.subplots(figsize=(10, 7))
                    scatter = ax.scatter(df['lon'], df['lat'], c=df['earnings'], s=df['rides']*50, 
                                        cmap='RdYlGn', alpha=0.7, edgecolors='black', linewidth=1)
                    ax.set_xlabel("Longitude", fontsize=11)
                    ax.set_ylabel("Latitude", fontsize=11)
                    ax.set_title("Geographic Heatmap: Earnings Density & Ride Volume", fontsize=14, fontweight='bold')
                    cbar = plt.colorbar(scatter, ax=ax, label='Earnings (₹)')
                    ax.grid(alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Geographic heatmap error: {e}")
            
            st.markdown("---")

            # Radar plot: simplified mapping privacy/integrity/availability
            radar_metrics = {}
            # map to 0..1 scale: privacy (HE high ->1, OTA->0), integrity ~ signature ok, availability from metric
            radar_metrics["HE"] = [1.0 if metrics_results["HE"] else 0.0, 1.0, metrics_results["HE"]["Availability"] / 100.0 if metrics_results["HE"] else 0.0]
            radar_metrics["OTA"] = [0.0, 1.0 if metrics_results["OTA"]["SignatureOK"] else 0.4, metrics_results["OTA"]["Availability"] / 100.0]
            radar_metrics["Hybrid"] = [0.8 if metrics_results["Hybrid"] else 0.0, 1.0, metrics_results["Hybrid"]["Availability"] / 100.0 if metrics_results["Hybrid"] else 0.0]
            plot_radar(radar_metrics)

            # Map
            st.markdown("### Driver Locations")
            show_map(df)

            # Small summary table
            summary_rows = []
            for method in ["HE", "OTA", "Hybrid"]:
                m = metrics_results.get(method)
                if m:
                    summary_rows.append({
                        "Method": method,
                        "Sum (Value)": m["Sum"]["Value"],
                        "Sum AbsError": m["Sum"]["Absolute Error"],
                        "Avg AbsError": m["Average"]["Absolute Error"],
                        "CommBytesAvg": m.get("CommBytesAvg"),
                        "EnergyEst": m.get("EnergyEst"),
                        "Availability%": m.get("Availability")
                    })
            st.table(pd.DataFrame(summary_rows))

        # if running realtime, wait then continue to next step
        if realtime and step < steps-1:
            time.sleep(realtime_delay)

    # End of step loop

    # Export option
    if enable_export:
        # flatten all_results to CSV
        rows = []
        for r in all_results:
            rows.append({
                "step": r["step"],
                "timestamp": r["timestamp"],
                "true_sum": r["true_sum"], "true_avg": r["true_avg"],
                "he_sum": r["he_sum"], "he_avg": r["he_avg"],
                "ota_sum": r["ota_sum"], "ota_avg": r["ota_avg"],
                "hybrid_sum": r["hybrid_sum"], "hybrid_avg": r["hybrid_avg"]
            })
        df_export = pd.DataFrame(rows)
        csv = df_export.to_csv(index=False).encode()
        st.download_button("Download results CSV", csv, "results.csv", "text/csv")

    st.success("Simulation run(s) complete.")
