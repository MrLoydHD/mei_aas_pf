# DGA Family Classification - Feasibility Analysis

## Executive Summary

**Question:** Can we categorize DGA domains by their algorithm family (like Conficker, CryptoLocker, Necurs)?

**Answer:** Yes, but your current dataset needs to be replaced or augmented with family-labeled data.

---

## Current System Status

### What You Have Now
```
Current Dataset Structure:
┌─────────────────────────────────────────┐
│  class    │  domain                     │
├─────────────────────────────────────────┤
│  dga      │  xk8d3jf9a2                 │
│  dga      │  brothernerveplacebring     │
│  legit    │  google                     │
│  legit    │  amazon                     │
└─────────────────────────────────────────┘

Total: 674,900 domains
- DGA: 337,500
- Legitimate: 337,399

❌ NO FAMILY LABELS
```

### What You Need for Family Classification
```
Required Dataset Structure:
┌──────────────────────────────────────────────────────┐
│  family       │  domain                              │
├──────────────────────────────────────────────────────┤
│  conficker    │  arusjrkmf                           │
│  cryptolocker │  xdqyohajtk                          │
│  necurs       │  lkjhgfdsa                           │
│  emotet       │  vbnmqwert                           │
│  legitimate   │  google                              │
└──────────────────────────────────────────────────────┘

✅ FAMILY LABELS INCLUDED
```

---

## Available Datasets with Family Labels

### 1. chrmor/DGA_domains_dataset (Recommended)
**URL:** https://github.com/chrmor/DGA_domains_dataset

| Attribute | Value |
|-----------|-------|
| Total Domains | 675,000 |
| DGA Domains | 337,500 |
| Legitimate Domains | 337,500 |
| DGA Families | 25 |
| Domains per Family | 13,500 |
| License | Free for research |
| Format | CSV |

**DGA Families Included:**
- 13 time-dependent algorithms
- 12 time-independent algorithms
- Families sourced from Netlab 360 OpenData

**Why This is Perfect:**
- Same size as your current dataset
- Already split 50/50 DGA/legit
- Family labels included
- Free to use for research
- Well-documented

### 2. Mendeley DGA Dataset
**URL:** https://data.mendeley.com/datasets/nhvyvytn2h/1

| Attribute | Value |
|-----------|-------|
| Total Domains | 4,090,661 |
| DGA Domains | 3,092,348 |
| Legitimate Domains | 998,313 |
| DGA Families | 160 |
| Morphological Types | 5 |
| Source | DGArchive |

**Morphological Categories:**
1. Random (unintelligible characters)
2. Pseudo-words (fake word patterns)
3. Dictionary-based
4. Arithmetic-based
5. Hybrid

### 3. Netlab 360 (No Longer Free)
**URL:** https://data.netlab.360.com/dga/

Previously the gold standard for DGA data, now requires commercial license.
Contact: netlab@360.cn for pricing.

### 4. DGArchive (Restricted Access)
**URL:** https://dgarchive.caad.fkie.fraunhofer.de/

| Attribute | Value |
|-----------|-------|
| DGA Families | 92+ |
| Access | Password protected |
| Requirement | Contact maintainer |

---

## Implementation Options

### Option A: Replace Dataset (Recommended)

**Steps:**
1. Download [chrmor/DGA_domains_dataset](https://github.com/chrmor/DGA_domains_dataset)
2. Replace your current `dga_websites.csv` and `legit_websites.csv`
3. Modify model to output family classification instead of binary
4. Retrain with multi-class classification

**Model Changes Required:**
```python
# Current: Binary Classification
output = Dense(1, activation='sigmoid')  # DGA or Legit

# New: Multi-class Classification (26 classes: 25 DGA + 1 Legit)
output = Dense(26, activation='softmax')  # Family classification
```

**Pros:**
- Comprehensive family detection
- Higher value intelligence
- Helps identify specific malware

**Cons:**
- Requires model retraining
- More complex output handling

### Option B: Hybrid Approach

Keep your current binary model, add a second model for family classification:

```
Domain Input
     │
     ▼
┌─────────────────────┐
│  Binary Model       │  ──→  "Is this DGA?" (Yes/No)
│  (Your Current)     │
└─────────────────────┘
     │
     │ (if DGA)
     ▼
┌─────────────────────┐
│  Family Model       │  ──→  "Which family?" (Conficker, Necurs, etc.)
│  (New Model)        │
└─────────────────────┘
```

**Pros:**
- Keep existing binary model
- Add family classification as enhancement
- Backward compatible

**Cons:**
- Two models to maintain
- More complexity

### Option C: Rule-Based Classification

Use regex patterns to classify your existing DGA detections:

```python
DGA_PATTERNS = {
    "conficker": r"^[a-z]{8,11}$",  # 8-11 lowercase letters
    "cryptolocker": r"^[a-z]{12,15}$",  # 12-15 lowercase letters
    "necurs": r"^[a-z]{14,18}$",  # 14-18 lowercase letters
    "emotet": r"^[a-z0-9]{16}$",  # 16 alphanumeric
    # ... more patterns
}

def classify_family(domain):
    for family, pattern in DGA_PATTERNS.items():
        if re.match(pattern, domain):
            return family
    return "unknown"
```

**Pros:**
- No retraining needed
- Immediate implementation
- Explainable rules

**Cons:**
- Less accurate than ML
- Many false positives/negatives
- Only catches known patterns

---

## Comparison: Current vs Enhanced System

| Feature | Current System | With Family Classification |
|---------|---------------|---------------------------|
| Output | DGA / Legitimate | Conficker / Necurs / Emotet / ... / Legitimate |
| Classes | 2 | 26+ |
| Intelligence Value | Medium | High |
| Threat Attribution | ❌ No | ✅ Yes |
| Incident Response | "You have DGA" | "You have Emotet infection" |
| Model Complexity | Low | Medium |

---

## Recommended Implementation Plan

### Phase 1: Data Preparation
1. Download [chrmor/DGA_domains_dataset](https://github.com/chrmor/DGA_domains_dataset)
2. Analyze family distribution
3. Merge with your legitimate domains if needed

### Phase 2: Model Architecture
1. Modify CNN-LSTM architecture for multi-class output
2. Use categorical cross-entropy loss
3. Implement class weighting for imbalanced families

### Phase 3: Training
1. Train new family classification model
2. Evaluate per-family precision/recall
3. Handle "unknown" DGA families

### Phase 4: Integration
1. Update API endpoints to return family predictions
2. Modify frontend to display family information
3. Update browser extension to show family alerts

### Phase 5: Enhancement
1. Add regex rules as fallback
2. Implement confidence thresholds
3. Create family-specific threat intelligence

---

## Code Example: Multi-Class DGA Model

```python
import tensorflow as tf
from tensorflow.keras import layers, Model

def build_family_classifier(vocab_size, max_length, num_families):
    """
    Build a CNN-LSTM model for DGA family classification.

    Args:
        vocab_size: Number of unique characters
        max_length: Maximum domain length
        num_families: Number of DGA families + 1 (legitimate)
    """
    inputs = layers.Input(shape=(max_length,))

    # Character embedding
    x = layers.Embedding(vocab_size, 128)(inputs)

    # CNN for local pattern detection
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)

    # LSTM for sequential patterns
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(32))(x)

    # Dense layers
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    # Multi-class output
    outputs = layers.Dense(num_families, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Usage
num_families = 26  # 25 DGA families + 1 legitimate
model = build_family_classifier(
    vocab_size=128,
    max_length=63,
    num_families=num_families
)
```

---

## API Response Example

### Current Response
```json
{
  "domain": "xk8d3jf9a2.com",
  "is_dga": true,
  "confidence": 0.95
}
```

### Enhanced Response with Family Classification
```json
{
  "domain": "xk8d3jf9a2.com",
  "is_dga": true,
  "confidence": 0.95,
  "family": {
    "name": "conficker",
    "confidence": 0.87,
    "description": "Conficker worm DGA",
    "threat_level": "high",
    "first_seen": "2008",
    "iocs": ["http://conficker.example/iocs"]
  },
  "alternative_families": [
    {"name": "necurs", "confidence": 0.08},
    {"name": "emotet", "confidence": 0.03}
  ]
}
```

---

## Conclusion

**Is DGA family classification possible with your project?**

✅ **YES**, but you need to:
1. Replace or augment your dataset with family-labeled data
2. Modify your model for multi-class classification
3. Update your API and frontend

**Recommended Next Steps:**
1. Download the [chrmor dataset](https://github.com/chrmor/DGA_domains_dataset) (free, 25 families)
2. Decide between Option A (replace) or Option B (hybrid)
3. Implement and test

---

## References

- [chrmor/DGA_domains_dataset](https://github.com/chrmor/DGA_domains_dataset) - Free dataset with 25 DGA families
- [Mendeley DGA Dataset](https://data.mendeley.com/datasets/nhvyvytn2h/1) - 160 families, 4M+ domains
- [Netlab 360 OpenData](https://data.netlab.360.com/dga/) - Commercial (formerly free)
- [DGArchive](https://dgarchive.caad.fkie.fraunhofer.de/) - 92+ families (restricted access)
- [GitHub - ericyoc/gen_dga_regex_and_yara_rules_poc](https://github.com/ericyoc/gen_dga_regex_and_yara_rules_poc) - DGA pattern analysis
