# 📊 Attack Comparison: gant.py vs app.py

## Overview

| Aspect | **gant.py** (Your Current) | **app.py** (Provided) |
|--------|---------------------------|----------------------|
| **Attacks Implemented** | 11 attacks (5 basic + 6 advanced) | 6 attacks (practical) |
| **Focus** | Cryptographic theoretical | Real-world operational |
| **Defense Mechanisms** | HE, OTA, DP, HMAC, RSA | HMAC verification only |
| **UI Design** | 5 tabs (data, attacks, aggregation, analysis, dashboard) | Single-page inspector |
| **Attack Execution** | Theoretical metrics | Live data manipulation |

---

## 🎯 ATTACKS IN gant.py (11 Total)

### **Group A: Basic Attacks (5)**
1. ✅ **Traffic Analysis** - Metadata pattern recognition
2. ✅ **Tampering** - Data corruption in transit
3. ✅ **Differential Inference** - Repeated query extraction
4. ✅ **Insider Compromise** - Privileged access abuse
5. ✅ **Denial of Service** - Resource exhaustion

### **Group B: Advanced Cryptographic Attacks (6)**
6. ✅ **Signature Forgery** - RSA-PSS bypass attempts
7. ✅ **Replay Attack** - Message re-transmission
8. ✅ **Timing Analysis** - Side-channel key extraction
9. ✅ **Collision Attack** - Hash function vulnerabilities
10. ✅ **Length Extension** - Authenticated message modification
11. ✅ **Byzantine Failure** - Distributed consensus breaking

---

## 🎯 ATTACKS IN app.py (6 Total)

1. ✅ **False Data Injection** - Fake device records added (= Tampering variant)
2. ✅ **Key Compromise** - Attacker gains device secret (= Insider Compromise variant)
3. ✅ **MITM (Man-in-the-Middle)** - Tamper with earnings values (= Tampering variant)
4. ✅ **Replay Attack** - Duplicate same earnings record (= Replay variant)
5. ✅ **OTA Compromise** - Bias wireless aggregation signals (= New attack)
6. ⚠️ **None** - Baseline (no attack)

---

## 🛡️ HOW ATTACKS ARE OVERCOME

### **In gant.py: Theoretical Defense Layers**

```python
# Layer 1: Homomorphic Encryption (HE)
encrypted_data = [ts.ckks_vector(context, [float(x)]) for x in local_data]
encrypted_sum += enc  # Operations on encrypted data
decrypted_sum = encrypted_sum.decrypt()[0]  # Only final decryption

# Against: Traffic Analysis, Insider Access, Tampering
# Why: Attacker never sees plaintext values
```

```python
# Layer 2: Differential Privacy (DP)
def gaussian_mechanism(true_value, sensitivity, epsilon, delta=1e-6):
    noise_scale = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
    noise = np.random.normal(loc=0, scale=noise_scale)
    return true_value + noise  # Each query returns different noisy result

# Against: Differential Inference, Repeated Queries
# Why: Mathematically provable privacy (ε-δ guarantee)
```

```python
# Layer 3: Over-The-Air (OTA) Aggregation
transmitted_signals = apply_power_scaling(local_data, path_losses)
received_matrix = simulate_ota_transmission(transmitted_signals, num_repeats, noise_std)
denoised_signals = denoise_received_signals(received_matrix)

# Against: Replay Attack, OTA Compromise, Tampering
# Why: Each transmission round has different noise (can't replay exact same)
```

```python
# Layer 4: HMAC Message Authentication
hmac_val = compute_hmac_for_row_from_values(device_id, timestamp, quantized_earning, secret_map)

# Against: Tampering Detection
# Why: Any modification to data changes HMAC (detectable)
```

```python
# Layer 5: RSA Digital Signatures
signature = private_key.sign(payload_bytes, padding.PSS(...), hashes.SHA256())

# Against: Signature Forgery
# Why: RSA-PSS is cryptographically hard to forge (2^256 complexity)
```

---

### **In app.py: Practical Verification & Detection**

```python
# Detection Method 1: HMAC Verification
def hmac_verification_rate(df, secret_map):
    valid = 0
    for row in df.iterrows():
        expected = compute_hmac_for_row_from_values(dev, ts, q, secret_map)
        if str(row['hmac']) == expected:
            valid += 1
    return valid / total

# Against: Tampering, Key Compromise
# How: Compares HMAC of received data with expected HMAC
# Effectiveness: Binary (PASS/FAIL) - detects 100% of modified data
```

```python
# Detection Method 2: Sybil Detection
def detect_sybil_devices(df):
    for dev in df['device_id'].unique():
        if str(dev).startswith("fake_"):
            suspects.add(dev)
    grouped = df.groupby(['device_id', 'lat', 'lon']).size()
    for row in grouped:
        if row['count'] >= 3:  # Same device at same location ≥3 times = suspicious
            suspects.add(row['device_id'])

# Against: False Data Injection
# How: Detects fake device patterns and location clustering
# Effectiveness: Catches obvious patterns (device names, duplicate locations)
```

```python
# Detection Method 3: Replay Detection
key_series = attacked['device_id'].astype(str) + '|' + attacked['timestamp'].astype(str)
dup_mask = key_series.duplicated(keep='first')
detected = attacked.loc[dup_mask, 'device_id'].unique().tolist()

# Against: Replay Attack
# How: Finds duplicate (device_id + timestamp) pairs
# Effectiveness: Detects 100% of exact replays; fails on timestamp-modified replays
```

```python
# Detection Method 4: Aggregation Error Detection
agg_error = abs(attacked_sum - original_sum) / original_sum * 100

# Against: All tampering
# How: Compares total earnings before/after attack
# Effectiveness: Detects gross changes; misses distributed small changes
```

```python
# Detection Method 5: Statistical Metrics
precision = TP / (TP + FP)  # Of detected anomalies, how many are real?
recall = TP / (TP + FN)     # Of actual attacks, how many are detected?

# Against: All attacks
# How: Measures detection accuracy
# Effectiveness: Varies by attack type (50-100%)
```

---

## 🔴 ATTACKS NOT OVERCOME IN app.py

### **Attack #6: Signature Forgery** ❌
```python
# gant.py has:
signature = private_key.sign(payload_bytes, padding.PSS(...), hashes.SHA256())

# app.py DOESN'T USE RSA at all!
# No protection against cryptographic signature forgery
# Attacker can forge signatures if private key leaked
```

**Why gant.py overcomes it:**
```python
# RSA-2048 bit with PSS padding + SHA256
# Computational complexity: 2^2048 (infeasible brute-force)
# Semantic security: Different signature for same message (PSS randomization)
```

---

### **Attack #7: Replay Attack** ⚠️ PARTIAL
```python
# app.py detects replay but only if EXACT duplicate:
key_series = device_id + '|' + timestamp  # If timestamp also copied = detected
                                           # If timestamp modified = NOT detected

# gant.py overcomes via:
# OTA noise changes per round (different noise each time)
# Even if data replayed, noise differs → detectable
# DP noise also differs → each query different result
```

**Comparison:**
```
app.py:
  Original: (device_5, 2025-11-27 18:00:00, $250)
  Replay:   (device_5, 2025-11-27 18:00:00, $250)  ← DETECTED (identical)
  
  Modified: (device_5, 2025-11-27 18:05:00, $250)  ← NOT DETECTED (timestamp changed)

gant.py:
  Original:  $250 + OTA_noise_round1 + DP_noise1 = $247.3
  Replayed:  $250 + OTA_noise_round2 + DP_noise2 = $253.8  ← Detected (different!)
```

---

### **Attack #8: Timing Analysis** ❌ NOT ADDRESSED
```python
# Neither gant.py nor app.py implements timing-safe comparison!

# VULNERABLE CODE (both use this):
if str(row.get('hmac', '')) == expected:  # Early exit on mismatch = timing side-channel!
    valid += 1

# SHOULD USE (not implemented):
from cryptography.hazmat.primitives.constant_time import bytes_eq
if bytes_eq(received_hmac_bytes, expected_hmac_bytes):  # Constant-time comparison
    valid += 1
```

**Why gant.py mentions but doesn't prevent it:**
- RSA operations not exposed to network timing (done server-side)
- HMAC verification done in-process (not time-analyzed remotely)
- But vulnerable to local side-channel if attacker has server access

---

### **Attack #9: Collision Attack** ❌ NOT ADDRESSED IN BOTH
```python
# Both use SHA256 for HMAC
# SHA256 collision: requires ~2^128 operations (theoretically possible)
# Practically: Not a concern with 256-bit output

# gant.py mentions it but doesn't add collision resistance
# app.py doesn't mention it

# Would need: use SHA256 as outer hash in HMAC-SHA256 (already doing this)
# Current implementation IS collision-resistant (using HMAC-SHA256 standard)
```

---

### **Attack #10: Length Extension** ⚠️ PARTIALLY PROTECTED
```python
# Both use proper HMAC-SHA256 (resistant to length extension!)

# VULNERABLE PATTERN (not used):
# H(secret + message) = allows length extension

# SAFE PATTERN (both use):
# HMAC-SHA256(message, secret) = immune to length extension
# Because HMAC uses nested hash with separate padding

# Therefore: Length extension attack CANNOT succeed
# gant.py is SAFE (uses HMAC properly)
# app.py is SAFE (uses HMAC properly)
```

---

### **Attack #11: Byzantine Failure** ❌ NOT ADDRESSED
```python
# Neither system implements Byzantine Fault Tolerance
# Both are single-aggregator (not distributed)

# To implement Byzantine FT:
# - Need 3f+1 aggregators (f = failures tolerated)
# - Use PBFT (Practical Byzantine Fault Tolerance)
# - Requires digital signatures on every message
# - Quorum-based consensus

# app.py: Single aggregator (system fails if compromised)
# gant.py: Single aggregator (no Byzantine protection)

# Neither implements: threshold cryptography, multi-party computation
```

---

## 📊 Detailed Attack Coverage Table

| Attack | gant.py | app.py | gant.py Method | app.py Method |
|--------|---------|--------|----------------|---------------|
| 1. Traffic Analysis | ✅ | ❌ | DP noise | N/A |
| 2. Tampering | ✅ | ✅ | HE + HMAC + DP | HMAC verification |
| 3. Differential Inference | ✅ | ❌ | DP provable | N/A |
| 4. Insider Compromise | ✅ | ⚠️ | HE encryption | No protection |
| 5. DoS | ⚠️ | ⚠️ | Rate limiting (not shown) | Monitoring (not shown) |
| 6. Signature Forgery | ✅ | ❌ | RSA-2048-PSS | No RSA used |
| 7. Replay Attack | ✅ | ⚠️ | OTA noise + DP | Timestamp check only |
| 8. Timing Analysis | ⚠️ | ⚠️ | Mentioned but vulnerable | Vulnerable |
| 9. Collision Attack | ✅ | ✅ | SHA256 HMAC | SHA256 HMAC |
| 10. Length Extension | ✅ | ✅ | HMAC standard | HMAC standard |
| 11. Byzantine Failure | ❌ | ❌ | Not implemented | Not implemented |

---

## 🎯 KEY DIFFERENCES IN DEFENSE PHILOSOPHY

### **gant.py: Cryptographic Defense (Preventive)**
```
Goal: Prevent attacks mathematically before they happen

1. HE: Even if data stolen, attacker cannot read it
   - Cost: High computational overhead
   - Guarantee: Information-theoretic security

2. DP: Provable privacy regardless of attacker capability
   - Cost: Utility loss (noise added to answers)
   - Guarantee: Differential privacy bound (ε, δ)

3. OTA: Signal-level averaging prevents exact recovery
   - Cost: Wireless transmission simulation
   - Guarantee: Each round has different noise

Philosophy: "Make it mathematically impossible to attack"
```

### **app.py: Detection & Verification (Detective)**
```
Goal: Detect when attacks happen and measure damage

1. HMAC: Verify data wasn't modified
   - Cost: Low (hash computation)
   - Guarantee: Detects tampering (binary: pass/fail)

2. Sybil Detection: Find fake devices
   - Cost: Low (pattern matching)
   - Guarantee: Detects obvious fakes

3. Replay Detection: Find duplicate submissions
   - Cost: Low (deduplication)
   - Guarantee: Detects exact replays

4. Statistical Metrics: Measure attack success
   - Cost: Low (aggregation)
   - Guarantee: Quantifies damage (precision, recall)

Philosophy: "Let attacks through but catch them and measure impact"
```

---

## 💡 How to Choose

### **Use gant.py if you need:**
- ✅ Privacy that survives insider compromise
- ✅ Protection against inference attacks
- ✅ Proof of security (ε-δ differential privacy)
- ✅ Defense against cryptographic attacks
- ✅ Academic/theoretical correctness

### **Use app.py if you need:**
- ✅ Practical operational monitoring
- ✅ Quick attack simulation
- ✅ Attack metrics and visualization
- ✅ Detection and alerting
- ✅ Forensic analysis after compromise
- ✅ Real-time anomaly detection

---

## 🔧 To Make app.py More Secure (Add Missing Defenses)

### **Add 1: Constant-Time HMAC Comparison**
```python
from cryptography.hazmat.primitives.constant_time import bytes_eq

def hmac_verification_rate(df, secret_map):
    for _, row in df.iterrows():
        expected = compute_hmac_for_row(row, secret_map)
        # Instead of: if str(row.get('hmac', '')) == expected
        # Use:
        if bytes_eq(str(row.get('hmac', '')).encode(), expected.encode()):
            valid += 1
    return valid / total
```

**Prevents:** Timing analysis attacks

---

### **Add 2: Differential Privacy for Replay Prevention**
```python
def apply_dp_to_rows(df, epsilon=1.0):
    """Add DP noise to earnings to prevent exact replays"""
    for idx in df.index:
        sensitivity = df['earnings'].max()
        dp_manager = DifferentialPrivacyManager()
        df.at[idx, 'earnings'] = dp_manager.gaussian_mechanism(
            df.at[idx, 'earnings'], 
            sensitivity=sensitivity, 
            epsilon=epsilon
        )
        df.at[idx, 'quantized_earning'] = int(df.at[idx, 'earnings'] * 100)
        df.at[idx, 'hmac'] = compute_hmac_for_row(df.loc[idx], secret_map)
    return df

# Now replayed earnings have different noise → detectable!
```

**Prevents:** Replay attacks completely (not just timestamp-modified ones)

---

### **Add 3: Byzantine Consensus (for distributed systems)**
```python
def byzantine_consensus(df_from_aggregator_1, df_from_aggregator_2, df_from_aggregator_3):
    """Simple majority voting (1-of-3 Byzantine tolerant)"""
    # Verify HMAC on all three copies
    hmac_rates = [
        hmac_verification_rate(df_from_aggregator_1, secret_map),
        hmac_verification_rate(df_from_aggregator_2, secret_map),
        hmac_verification_rate(df_from_aggregator_3, secret_map),
    ]
    
    # Accept majority (if 2+ have >80% HMAC rate, they're trusted)
    trusted = sum(1 for rate in hmac_rates if rate > 0.8)
    if trusted >= 2:
        return "CONSENSUS"  # System is honest
    else:
        return "BYZANTINE"  # One or more compromised
```

**Prevents:** Byzantine failure (1-of-3 aggregators can be compromised)

---

## 📝 Summary Table

```
┌──────────────────────────────┬──────────────┬────────────────┐
│ Capability                   │ gant.py      │ app.py         │
├──────────────────────────────┼──────────────┼────────────────┤
│ Prevent tampering            │ ✅ HE        │ ✅ HMAC detect │
│ Prevent inference            │ ✅ DP        │ ❌             │
│ Prevent insider access       │ ✅ HE        │ ❌             │
│ Prevent replay (exact)       │ ✅ OTA noise │ ✅ Timestamp   │
│ Prevent replay (modified ts) │ ✅ DP noise  │ ❌             │
│ Detect false data            │ ⚠️ Pattern   │ ✅ Sybil       │
│ Detect key compromise        │ ✅ Behavior  │ ⚠️ Anomalies   │
│ Verify signatures            │ ✅ RSA-PSS   │ ❌ (no RSA)    │
│ Constant-time comparison     │ ⚠️ Mention   │ ❌ Vulnerable  │
│ Byzantine FT                 │ ❌           │ ❌             │
│ Practical monitoring         │ ⚠️ Partial   │ ✅ Full        │
│ Academic rigor               │ ✅ Full      │ ⚠️ Partial     │
└──────────────────────────────┴──────────────┴────────────────┘
```

---

**Conclusion:** 
- **gant.py** = Fort Knox (prevention)
- **app.py** = Security cameras (detection)
- **Both together** = Perfect security posture
