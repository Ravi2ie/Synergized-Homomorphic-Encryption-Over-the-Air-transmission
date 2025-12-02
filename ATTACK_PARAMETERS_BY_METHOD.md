# Attack Parameters Analysis: HE vs OTA vs Hybrid

## Overview

This document explains how attack parameters are scaled and adapted for different cryptographic methods (HE, OTA, Hybrid) in the secure aggregation platform.

---

## Key Principles

### 1. **HE (Homomorphic Encryption)**
- **Strength**: Data encrypted before any computation; attacker cannot see values
- **Weakness**: Vulnerable to attacks that don't require plaintext access (Sybil, replay, blind injection)
- **Attack Effectiveness**: LOWER for data-tampering attacks, HIGHER for structural attacks

### 2. **OTA (Over-the-Air Aggregation)**
- **Strength**: Wireless channel aggregation; lightweight, fast
- **Weakness**: Vulnerable to wireless channel attacks, biasing, MITM
- **Attack Effectiveness**: HIGHER for wireless-level attacks

### 3. **Hybrid (HE + OTA)**
- **Strength**: OTA denoise → HE encrypt → aggregate; dual defense layers
- **Weakness**: Complexity; attack must overcome both mechanisms
- **Attack Effectiveness**: LOWER for most attacks due to dual protection

---

## Attack-by-Attack Analysis

### **1. False Data Injection**
**Goal**: Inject fake devices with forged HMACs to inflate aggregation results

#### Parameter Scalers:
| Method | n_fake_multiplier | Reasoning |
|--------|-------------------|-----------|
| HE | 1.8x | Encrypted values prevent validation; attacker needs more fakes to meaningfully impact encrypted sum |
| OTA | 1.2x | Wireless aggregation detects anomalies through signal patterns; moderate increase |
| Hybrid | 2.0x | Must pass both OTA anomaly detection AND HE validation; highest threshold |

#### Impact Across Methods:
- **HE**: Effectiveness 40% → More fakes needed but doesn't bypass encryption
- **OTA**: Effectiveness 70% → Fakes detected through signal analysis
- **Hybrid**: Effectiveness 30% → Dual defense most resistant

#### Example (Intensity 5):
```
HE:     n_fake = 3 × 1.8 × 1.0 = 5-6 fake devices
OTA:    n_fake = 3 × 1.2 × 1.0 = 3-4 fake devices  
Hybrid: n_fake = 3 × 2.0 × 1.0 = 6 fake devices
```

---

### **2. Key Compromise**
**Goal**: Steal device keys and use them to forge legitimate-looking requests

#### Parameter Scalers:
| Method | multiplier_scale | Reasoning |
|--------|------------------|-----------|
| HE | 0.7x | Compromised keys don't help decrypt ciphertexts; impact REDUCED |
| OTA | 1.0x | Keys directly control OTA signal parameters; FULL impact |
| Hybrid | 0.8x | HE provides additional protection layer even with compromised keys |

#### Impact Across Methods:
- **HE**: Effectiveness 30% → Keys compromised but encryption still protects data
- **OTA**: Effectiveness 80% → Keys control signal parameters directly
- **Hybrid**: Effectiveness 40% → HE limits damage from key compromise

#### Example Impact:
```
With compromised keys and intensity 5:
HE:     Modified earnings factor = 1.5 × 0.7 = 1.05x (minimal change to encrypted values)
OTA:    Modified earnings factor = 1.5 × 1.0 = 1.5x (full MITM-like effect)
Hybrid: Modified earnings factor = 1.5 × 0.8 = 1.2x (protected by HE encryption)
```

---

### **3. Man-in-the-Middle (MITM)**
**Goal**: Intercept and tamper with earnings values during transmission

#### Parameter Scalers:
| Method | tamper_multiplier_scale | Reasoning |
|--------|------------------------|-----------|
| HE | 0.3x | Cannot tamper encrypted values; must intercept BEFORE encryption (very hard) |
| OTA | 1.2x | Wireless channel is unencrypted; HIGH vulnerability |
| Hybrid | 0.4x | Values encrypted; OTA layer adds protection |

#### Impact Across Methods:
- **HE**: Effectiveness 10% → Near impossible; encryption happens client-side
- **OTA**: Effectiveness 85% → Open wireless channel highly vulnerable
- **Hybrid**: Effectiveness 15% → Must breach both layers

#### Example (Intensity 5, tamper_fraction base=0.05):
```
HE:     tamper_fraction = 0.05 × 1.0 × 0.3 = 0.015 (very limited impact)
OTA:    tamper_fraction = 0.05 × 1.0 × 1.2 = 0.06 (moderate tampering)
Hybrid: tamper_fraction = 0.05 × 1.0 × 0.4 = 0.02 (protected)
```

---

### **4. Replay Attack**
**Goal**: Duplicate legitimate earnings submissions multiple times

#### Parameter Scalers:
| Method | max_dup_multiplier | Reasoning |
|--------|-------------------|-----------|
| HE | 1.0x | Replayed ciphertexts are identical; server aggregates them as separate submissions |
| OTA | 1.5x | Wireless channel can carry duplicates; moderate increase needed |
| Hybrid | 1.2x | Sequence numbers in HE context help detect; slight increase |

#### Impact Across Methods:
- **HE**: Effectiveness 60% → Ciphertexts replayable; server doesn't know
- **OTA**: Effectiveness 70% → Wireless duplicates harder to detect
- **Hybrid**: Effectiveness 50% → Hybrid has some replay detection

#### Example (Intensity 5, replay_fraction base=0.1, max_dup base=2):
```
HE:     max_dup = 2 × 1.0 = 2 duplicates allowed
OTA:    max_dup = 2 × 1.5 = 3 duplicates allowed
Hybrid: max_dup = 2 × 1.2 = 2-3 duplicates allowed
```

---

### **5. OTA Compromise**
**Goal**: Bias wireless aggregation signals to manipulate results

#### Parameter Scalers:
| Method | bias_ineffective | Reasoning |
|--------|-----------------|-----------|
| HE | TRUE | HE encrypts data BEFORE wireless transmission; wireless attacks cannot affect encrypted values |
| OTA | FALSE (1.0x) | Direct attack on OTA channel; MAXIMUM effectiveness |
| Hybrid | TRUE | HE encryption layer protects values from wireless biasing |

#### Impact Across Methods:
- **HE**: Effectiveness 0% → Encrypted values cannot be biased
- **OTA**: Effectiveness 90% → Wireless channel is primary target
- **Hybrid**: Effectiveness 0% → Encryption protects from wireless attacks

#### Example:
```
HE:     ❌ INEFFECTIVE - Values encrypted before OTA transmission
OTA:    ✅ HIGHLY EFFECTIVE - Direct control of wireless signals
Hybrid: ❌ INEFFECTIVE - OTA biasing cannot affect encrypted values
```

---

## Intensity Scaling (1-10)

All attacks apply intensity-based scaling:

```
intensity_factor = 0.5 + (intensity - 1) × 0.167

Intensity 1:  factor = 0.50 (weak attack)
Intensity 5:  factor = 1.00 (baseline)
Intensity 10: factor = 1.50 (strong attack)
```

This factor is then multiplied with method-specific scalers.

---

## Detection Likelihood

| Attack | HE | OTA | Hybrid |
|--------|-----|-----|---------|
| false_data_injection | 30% | 50% | 60% |
| key_compromise | 40% | 25% | 35% |
| mitm | 80% | 30% | 75% |
| replay | 35% | 50% | 45% |
| ota_compromise | 100% | 25% | 100% |

**Note**: Detection probability is reduced by increasing intensity (stronger attacks are harder to detect).

---

## Practical Examples

### Scenario 1: Low Intensity (2/10) Attack on Each Method

#### False Data Injection:
```
Intensity factor: 0.5 + (2-1) × 0.167 = 0.667

HE:     n_fake = 3 × 1.8 × 0.667 = 3-4 fakes
        → Effectiveness: 40% × 0.667 = 27% (weak)

OTA:    n_fake = 3 × 1.2 × 0.667 = 2-3 fakes
        → Effectiveness: 70% × 0.667 = 47% (medium)

Hybrid: n_fake = 3 × 2.0 × 0.667 = 4 fakes
        → Effectiveness: 30% × 0.667 = 20% (weak)
```

### Scenario 2: High Intensity (9/10) MITM Attack

```
Intensity factor: 0.5 + (9-1) × 0.167 = 1.833

HE:     tamper_fraction = 0.05 × 1.833 × 0.3 = 0.027
        → Effectiveness: 10% × 1.833 = 18% (still weak due to encryption)
        → Detection: 80% - (9 × 5%) = 35% (hard to hide)

OTA:    tamper_fraction = 0.05 × 1.833 × 1.2 = 0.110
        → Effectiveness: 85% × 1.833 = 156% clamped to 100% (VERY EFFECTIVE)
        → Detection: 30% - (9 × 5%) = 0% (likely undetected)

Hybrid: tamper_fraction = 0.05 × 1.833 × 0.4 = 0.036
        → Effectiveness: 15% × 1.833 = 27% (moderate defense)
        → Detection: 75% - (9 × 5%) = 30% (fairly detectable)
```

---

## UI Features

### Attack Impact Analysis Tab
Located in **Tab 3: Aggregation Analysis** → New section shows:

1. **Attack Selector**: Choose from 5 attack types
2. **Intensity Slider**: Set attack intensity (1-10)
3. **Effectiveness Comparison**: % impact on HE vs OTA vs Hybrid
4. **Detection Probability**: Likelihood of detection by each method
5. **Risk Levels**: HIGH RISK (>60%), MEDIUM (30-60%), LOW (<30%)
6. **Parameter Breakdown**: Exact scaled parameters for each method
7. **Visual Charts**: 
   - Bar chart of effectiveness across methods
   - Detection vs Success probability comparison

### Key Visual Indicators:
- 🔴 Red: High effectiveness (>60%) - Attack likely succeeds
- 🟡 Yellow: Medium effectiveness (30-60%) - Attack has decent chance
- 🟢 Green: Low effectiveness (<30%) - Attack likely fails

---

## Implementation Details

### `scale_attack_params()` Function
```python
scale_attack_params(attack_type, intensity, crypto_method)
```

**Returns**: Dictionary with scaled parameters ready for attack execution

**Usage**:
```python
# Example: Scale false_data_injection for HE at intensity 7
params = scale_attack_params("false_data_injection", 7, "HE")
attacked_df, meta, smap = false_data_injection(df, secret_map, **params)
```

### `estimate_attack_impact()` Function
```python
estimate_attack_impact(attack_type, crypto_method, intensity)
```

**Returns**: Dictionary with:
- `effectiveness`: % (0-100)
- `detection_probability`: % (0-100)
- `success_likelihood`: % (100 - detection)
- `recommendation`: "HIGH/MEDIUM/LOW RISK"

---

## Key Takeaways

1. **HE** is best at protecting against MITM and tampering but vulnerable to structural attacks
2. **OTA** is fast but vulnerable to wireless attacks and key compromise
3. **Hybrid** provides best defense overall by combining both protections
4. **Attack parameters automatically scale** based on crypto method strength
5. **Higher intensity** increases attack effectiveness but also detection likelihood
6. **OTA-specific attacks** (like OTA compromise) are ineffective against encrypted methods

---

## References

- **ATTACK_PARAMETERS**: Configuration dictionary in `gant.py` lines 337-396
- **scale_attack_params()**: Function in `gant.py` lines 398-438
- **estimate_attack_impact()**: Function in `gant.py` lines 440-483
- **UI Section**: "Attack Parameter Analysis" in Tab 3 (lines ~2120-2220)

