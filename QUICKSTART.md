# Quick Start Guide - Integrated gant.py

## Installation

### 1. Install Required Dependencies
```bash
pip install streamlit pandas numpy matplotlib cryptography
```

### 2. Install Optional (for HE support)
```bash
pip install tenseal folium streamlit-folium
```

### 3. Verify Installation
```bash
python -m py_compile d:\Data\Built_projects\Cryptography\gant.py
echo Syntax OK if no errors above
```

## Running the Application

### Start Streamlit App
```bash
cd "d:\Data\Built projects\Cryptography"
streamlit run gant.py
```

Browser will open at: `http://localhost:8501`

## Usage Instructions

### Mode 1: Attack Simulator 🚨

**Step 1: Select "Attack Simulator" from sidebar**

**Step 2: Configure Dataset**
- Set number of drivers: 2-500
- Set reports per driver: 1-200
- Customize center coordinates (latitude/longitude)
- Click **"Generate Simulated Dataset"**

**Step 3: View Dataset**
- See preview of top 200 rows
- Verify driver IDs, earnings, locations

**Step 4: Analyze Attacks**
- Navigate through 5 attack tabs
- Click **"Replay [Attack] GIF"** to display animation
- View dynamic metrics:
  - **Privacy Breach Count**: Number of leaked records
  - **Privacy Loss ε**: Differential privacy impact (higher = worse)
  - **Key Exposure Score**: Cryptographic key vulnerability (0-1 scale)
  - **Tampering Detection Rate**: Probability of detection (0-1 scale)
  - **Availability Uptime %**: System uptime (0-100%)

**Step 5: Read Analysis**
- **Definition**: Technical description of attack
- **Impact**: Real-world consequences

---

### Mode 2: HE/OTA/Hybrid Aggregation 🔐

**Step 1: Select "HE/OTA/Hybrid Aggregation" from sidebar**

**Step 2: Configure in Sidebar**

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| Number of Drivers | 1-500 | 8 | Participants in aggregation |
| Reports per Driver | 1-50 | 1 | Data points per driver |
| OTA Noise Std | 0.0-5.0 | 1.0 | Transmission noise level |
| OTA repeats | 1-200 | 50 | Averaging iterations |
| Random seed | 0+ | 0 | 0 = random, else reproducible |
| Simulate Attacks | ✓/✗ | ✗ | Inject tampering |
| RSA Signing | ✓/✗ | ✓ | Enable signature verification |
| CSV Export | ✓/✗ | ✓ | Enable results download |

**Step 3: Run Simulation**
- Click **"Run Aggregation Simulation"**
- Wait 2-5 seconds for computation
- View dataset preview

**Step 4: Analyze Results**

For each method (HE, OTA, Hybrid):
- **True Value**: Plaintext baseline (reference)
- **Computed Value**: Method-specific result
- **Absolute Error**: Difference from baseline
- **Relative Error (%)**: Percentage deviation
- **RMSE**: Root mean squared error
- **Computation Times**: Encryption/aggregation/decryption (seconds)
- **Communication Bytes**: Ciphertext/signal size
- **Energy Estimate**: Relative power consumption
- **Availability %**: System uptime under attack

**Step 5: View Driver Analytics**

4 visualization tabs:
1. **📈 Driver Dashboard**: 4-panel performance view (earnings, rides, efficiency)
2. **📉 Earnings Timeline**: Time-series of earnings by driver
3. **🚗 Ride Distribution**: Total rides per driver (bar + pie)
4. **🔥 Correlation Heatmap**: Earnings vs rides vs location variation

**Step 6: Export Results**
- Click **"Download results CSV"** (if enabled)
- Contains: timestamp, true_sum, true_avg, he_sum, ota_sum, hybrid_sum, etc.

---

## Example Scenarios

### Scenario A: Quick Attack Review
1. Select Attack Simulator
2. Use default settings (12 drivers, 6 reports each)
3. Click "Generate Dataset"
4. View each attack tab (2-5 seconds per tab)
5. Compare attack metrics

**Time**: ~2 minutes

---

### Scenario B: HE vs OTA Comparison
1. Select HE/OTA/Hybrid mode
2. Keep default settings
3. Run simulation (ensure TenSEAL installed)
4. Compare error metrics across 3 methods
5. Identify best trade-off (accuracy vs speed)

**Time**: ~1 minute

---

### Scenario C: Attack Impact Analysis
1. Select HE/OTA/Hybrid mode
2. Enable "Simulate Attacks (tamper/jamming)"
3. Run simulation
4. Compare results with/without attacks
5. Observe error degradation

**Time**: ~2 runs, 2 minutes total

---

## Troubleshooting

### Issue: "GIF not found"
**Solution**: Ensure `gifs/` folder exists with 5 attack GIF files

### Issue: "TenSEAL not installed"
**Solution**: Install via `pip install tenseal` or skip HE mode

### Issue: "Streamlit API error - use_container_width"
**Solution**: Already fixed in this version - update Streamlit: `pip install --upgrade streamlit`

### Issue: "Port 8501 already in use"
**Solution**: Stop other Streamlit instances or use: `streamlit run gant.py --server.port 8502`

### Issue: Slow GIF playback or no animation
**Solution**: Ensure GIF has `loop=0` parameter (already applied in code)

---

## Tips & Tricks

1. **Reproducible Results**: Set random seed > 0 in HE/OTA mode
2. **Batch Testing**: Run multiple simulations with seed 1, 2, 3... for statistics
3. **Compare Attacks**: Open dataset once, switch tabs to see instant metric updates
4. **Export Workflow**: Run simulation → adjust parameters → download CSV → analyze in Excel/Python
5. **Performance Tuning**: 
   - Reduce OTA repeats (50 → 10) for 5x speedup
   - Reduce number of drivers (100 → 10) for 100x speedup

---

## Data Flow Diagram

```
ATTACK SIMULATOR PATH:
┌─────────────────┐
│  User Settings  │ (drivers, reports, lat/lon)
└────────┬────────┘
         │
         ▼
┌──────────────────────┐
│ simulate_driver_data │ ← Generate synthetic dataset
└────────┬─────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│ Compute 5 Attack Metrics Functions   │
│ • privacy_breach_count               │
│ • privacy_loss_epsilon               │
│ • key_exposure_score                 │
│ • tampering_detection_rate           │
│ • availability_uptime_percent        │
└────────┬─────────────────────────────┘
         │
         ▼
┌───────────────────┐
│ Display in Tabs   │ ← Load GIFs, show metrics
└───────────────────┘

HE/OTA/HYBRID PATH:
┌─────────────────┐
│  User Settings  │ (drivers, reports, OTA params, attack flag)
└────────┬────────┘
         │
         ▼
┌──────────────────────┐
│ simulate_driver_data │ ← Generate synthetic dataset
└────────┬─────────────┘
         │
         ├─────────────────┬────────────────────┬─────────────────┐
         ▼                 ▼                    ▼                 ▼
    ┌────────┐        ┌────────┐         ┌─────────┐        ┌─────────┐
    │   HE   │        │  OTA   │         │ Hybrid  │        │  True   │
    │ Crypto │        │ Transmit│        │HE+OTA   │        │ Baseline│
    └────┬───┘        └────┬───┘         └────┬────┘        └────┬────┘
         │                 │                  │                  │
         └─────────────────┴──────────────────┴──────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  Compute Error Metrics│
              │  • Absolute Error     │
              │  • Relative Error (%) │
              │  • RMSE               │
              │  • Bias               │
              └───────┬───────────────┘
                      │
                      ▼
         ┌────────────────────────────┐
         │ Display Results + Analytics│
         │ + Driver Visualizations    │
         │ + CSV Export               │
         └────────────────────────────┘
```

---

## Support

- **Documentation**: See `INTEGRATION_SUMMARY.md`
- **Code Comments**: Inline documentation in `gant.py`
- **Error Messages**: Streamlit UI provides helpful error descriptions
- **Dependencies**: Check `requirements.txt` (generated via `pip freeze`)

---

**Ready to explore cryptographic security and attack scenarios? Start the app now!** 🚀
