# DGA Detection System

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![React](https://img.shields.io/badge/React-18.2-61dafb)
![FastAPI](https://img.shields.io/badge/FastAPI-0.123-009688)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)

## Introduction

This repository contains a full-stack **Domain Generation Algorithm (DGA) Detection System** developed for the **Aprendizagem Aplicada à Segurança (AAS)** course.

The system detects domains generated algorithmically by botnets and command-and-control (C2) servers, distinguishing between legitimate domain names and random/algorithmic ones using machine learning.

**Features:**
- **6 ML Models**: Random Forest, LSTM, XGBoost, Gradient Boosting, Transformer, DistilBERT
- **Family Classification**: Identify 25 DGA malware families with threat intelligence
- **Ensemble Voting**: Weighted voting based on model accuracy for higher confidence
- FastAPI REST backend with Prometheus metrics
- React dashboard for visualization and analysis
- Chrome browser extension for real-time protection
- **Monitoring**: Prometheus + Grafana for metrics and dashboards
- **Load Testing**: k6 scripts for performance validation

**Authors:**
* Hugo Correia - 108215
* Joaquim Rosa - 109089

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Chrome Extension                           │
│  - Real-time URL monitoring                                 │
│  - Visual warnings for DGA domains                          │
│  - Statistics tracking                                      │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP/REST
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   React Dashboard                            │
│  - Detection statistics & charts                            │
│  - Domain scanner interface                                 │
│  - Model performance metrics                                │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI Backend                            │
│  - /predict - Single domain classification                  │
│  - /predict/batch - Batch classification                    │
│  - /predict/family - Family classification                  │
│  - /metrics - Prometheus metrics                            │
└────────────┬─────────────────────────────────┬──────────────┘
             │                                 │
             ▼                                 ▼
┌────────────────────────────┐   ┌────────────────────────────┐
│     Binary Detection       │   │   Family Classification    │
│  ┌────────┐ ┌────────┐    │   │  ┌────────┐ ┌────────┐    │
│  │   RF   │ │  LSTM  │    │   │  │   RF   │ │  LSTM  │    │
│  └────────┘ └────────┘    │   │  └────────┘ └────────┘    │
│  ┌────────┐ ┌────────┐    │   │  ┌────────┐ ┌────────┐    │
│  │ XGBoost│ │   GB   │    │   │  │ XGBoost│ │   GB   │    │
│  └────────┘ └────────┘    │   │  └────────┘ └────────┘    │
│  ┌────────┐ ┌────────┐    │   │  ┌────────┐ ┌────────┐    │
│  │Transf. │ │DistilB │    │   │  │Transf. │ │DistilB │    │
│  └────────┘ └────────┘    │   │  └────────┘ └────────┘    │
└────────────────────────────┘   └────────────────────────────┘
             │                                 │
             └─────────────┬───────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Monitoring Stack                           │
│  ┌─────────────┐        ┌─────────────┐                     │
│  │ Prometheus  │───────>│   Grafana   │                     │
│  │   :9090     │        │   :3001     │                     │
│  └─────────────┘        └─────────────┘                     │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

* Python 3.12 or higher
* Node.js 18+ (for React dashboard)
* Chrome/Chromium browser (for extension)
* Docker & Docker Compose (optional, for containerized deployment)

## Installation

### 1. Clone and Setup

```bash
git clone https://github.com/MrLoydHD/mei_aas_pf.git
cd mei_aas_pf
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# or
.\venv\Scripts\Activate   # Windows
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

## Docker Deployment

For containerized deployment, use Docker Compose:

```bash
# First time: train the models
make train

# Build and start all services
make start

# Or manually:
docker compose build
docker compose up -d
```

**Available Commands:**

| Command | Description |
|---------|-------------|
| `make build` | Build all Docker images |
| `make up` | Start all services (frontend + backend) |
| `make down` | Stop all services |
| `make logs` | View logs from all services |
| `make train` | Train ML models |
| `make clean` | Remove all containers and images |
| `make status` | Show status of all services |
| `make restart` | Restart all services |

**Services:**
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### Monitoring Stack (Prometheus + Grafana)

Start the monitoring stack alongside the main services:

```bash
# Start monitoring (Prometheus + Grafana)
make monitoring-up

# Stop monitoring
make monitoring-down

# View monitoring logs
make monitoring-logs
```

**Monitoring Services:**
- **Prometheus**: http://localhost:9090 - Metrics collection and querying
- **Grafana**: http://localhost:3001 - Dashboards and visualization (login: admin/admin)
- **API Metrics**: http://localhost:8000/metrics - Raw Prometheus metrics

The Grafana dashboard is pre-configured with:
- Total predictions and DGA detection rate
- Request latency (p50, p95, p99)
- Predictions by model, source, and result
- Family classification breakdown
- Threat level distribution

### Load Testing (k6)

Run load tests to validate API performance:

```bash
# Quick smoke test (10 iterations)
make test-smoke

# Standard load test (5 minutes, up to 20 VUs)
make test-load

# Stress test (12 minutes, up to 100 VUs)
make test-stress

# Run via Docker (no local k6 installation needed)
make test-docker
```

**Test Types:**
| Test | Duration | VUs | Purpose |
|------|----------|-----|---------|
| Smoke | 10 iterations | 1 | Quick validation |
| Load | 5 minutes | 10-20 | Normal load simulation |
| Stress | 12 minutes | 20-100 | Find breaking point |

## Dataset

The project uses multiple datasets located in the `data/raw/` folder:

| File | Records | Description |
|------|---------|-------------|
| `dga_websites.csv` | 337,500 | DGA-generated domains (binary detection) |
| `legit_websites.csv` | 337,398 | Legitimate domains (binary detection) |
| `dga_domains_full.csv` | ~1M | DGA domains with family labels (25 families) |
| `ddns_links.csv` | 34,000+ | Dynamic DNS providers for DDNS detection |
| `words.txt` | 354,986 | English dictionary for feature extraction |

**Sources:**
- DDNS providers: [alexandrosmagos/dyn-dns-list](https://github.com/alexandrosmagos/dyn-dns-list)

## Usage

### Step 1: Train the Models

```bash
# Train all models (RF, LSTM, XGBoost, GB, Transformer, DistilBERT)
python -m src.ml.train --all-models

# Train with smaller sample (faster, for testing)
python -m src.ml.train --all-models --sample-size 50000

# Train only specific models
python -m src.ml.train --rf-only
python -m src.ml.train --lstm-only --epochs 30
python -m src.ml.train --xgb-only
python -m src.ml.train --gb-only
python -m src.ml.train --transformer-only
python -m src.ml.train --distilbert-only

# Train family classifiers
python -m src.ml.train --family-only
```

Training outputs:
- `models/random_forest.joblib` - Trained RF model
- `models/lstm/` - Trained LSTM model
- `models/xgboost.joblib` - Trained XGBoost model
- `models/gradient_boosting.joblib` - Trained GB model
- `models/transformer/` - Trained Transformer model
- `models/distilbert/` - Trained DistilBERT model
- `models/family_classifier_*.joblib` - Family classifiers
- `models/plots/` - Confusion matrices, feature importance, training curves
- `models/results_summary.json` - Performance metrics

### Step 2: Start the Backend API

```bash
# Start FastAPI server
python -m src.api.main

# Or with uvicorn directly
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

API will be available at:
- http://localhost:8000 - API root
- http://localhost:8000/docs - Swagger UI
- http://localhost:8000/redoc - ReDoc documentation

### Step 3: Start the React Dashboard

```bash
cd frontend
npm run dev
```

Dashboard available at http://localhost:3000

### Step 4: Install Chrome Extension

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable "Developer mode" (top right)
3. Click "Load unpacked"
4. Select the `extension/` folder
5. The DGA Detector icon will appear in your toolbar

**Note:** You need to generate PNG icons from the SVG files in `extension/icons/`. You can:
- Use an online SVG to PNG converter
- Or open `extension/icons/generate_icons.html` in a browser and save the canvases

### Step 5: Configure Google OAuth (Optional)

For syncing detection data between the dashboard and extension:

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable the Google+ API
4. Go to Credentials → Create Credentials → OAuth 2.0 Client ID
5. Create **two** OAuth clients:

   **Web Application** (for dashboard):
   - Authorized JavaScript origins: `http://localhost:3000`
   - Authorized redirect URIs: `http://localhost:3000`

   **Chrome Extension** (for browser extension):
   - Application type: Chrome Extension
   - Item ID: Your extension ID (found in `chrome://extensions`)

6. Copy the environment files and add your client IDs:
   ```bash
   cp .env.example .env
   cp frontend/.env.example frontend/.env
   ```

7. Update `extension/manifest.json`:
   - Replace `YOUR_CHROME_EXTENSION_CLIENT_ID` with your Chrome extension OAuth client ID
   - Replace `YOUR_EXTENSION_PUBLIC_KEY` with your extension's public key

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with model status |
| `/predict` | POST | Classify single domain |
| `/predict/batch` | POST | Classify multiple domains |
| `/predict/detailed` | POST | Classify with feature analysis |
| `/predict/family` | POST | Classify with malware family detection |
| `/stats` | GET | Detection statistics |
| `/stats/families` | GET | Family detection statistics |
| `/models` | GET | Model information |
| `/families` | GET | DGA family information |
| `/metrics` | GET | Prometheus metrics |
| `/extension/check` | POST | Lightweight endpoint for extension |
| `/auth/google` | POST | Google OAuth authentication |
| `/auth/me` | GET | Get current user info |
| `/auth/stats` | GET | Get user's detection stats |
| `/auth/sync` | POST | Sync extension stats to server |

### Example API Usage

```bash
# Single domain prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"domain": "suspicious123abc.net"}'

# Batch prediction
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{"domains": ["google.com", "xk23jf9sd.net", "facebook.com"]}'

# Family classification (with threat intelligence)
curl -X POST "http://localhost:8000/predict/family" \
  -H "Content-Type: application/json" \
  -d '{"domain": "xk23jf9sd.net"}'

# Select specific model
curl -X POST "http://localhost:8000/predict?model_type=lstm" \
  -H "Content-Type: application/json" \
  -d '{"domain": "suspicious123abc.net"}'
```

### Model Selection

You can specify which model to use via query parameters:

| Parameter | Values | Description |
|-----------|--------|-------------|
| `model_type` | `auto`, `rf`, `lstm`, `xgb`, `gb`, `transformer`, `distilbert` | Binary detection model |
| `family_model_type` | `auto`, `rf`, `lstm`, `xgb`, `gb`, `transformer`, `distilbert` | Family classification model |

- `auto` (default): Uses the best available model (priority: DistilBERT > Transformer > LSTM > XGBoost > GB > RF)
- For family classification, `auto` uses ensemble voting with weighted accuracy

## Model Comparison

### Binary Detection Models

| Model | Type | Approach | Pros |
|-------|------|----------|------|
| **Random Forest** | Tree-based | Handcrafted features (entropy, n-grams, dictionary) | Fast (~1ms), interpretable, no GPU |
| **XGBoost** | Tree-based | Gradient boosting on handcrafted features | Fast, handles imbalanced data well |
| **Gradient Boosting** | Tree-based | Scikit-learn gradient boosting | Robust, good generalization |
| **LSTM** | Neural Network | CNN + Bidirectional LSTM on characters | Learns patterns automatically |
| **Transformer** | Neural Network | Self-attention on character sequences | Captures long-range dependencies |
| **DistilBERT** | Neural Network | Fine-tuned language model | Highest accuracy, pre-trained knowledge |

### Family Classification

The system can identify 25 DGA malware families with threat intelligence:

| Threat Level | Families |
|--------------|----------|
| **Critical** | Cryptolocker, Locky, WannaCry, Emotet |
| **High** | Conficker, Zeus, Necurs, Ramnit, Dyre, Tinba |
| **Medium** | Bamital, Banjori, Murofet, Rovnix, Symmi, Tempedreve |
| **Low** | Ranbyus, Shiotob, Sisron, Suppobox, Vawtrak |

Each family includes:
- Description and behavior
- First seen date
- Malware type (ransomware, banking trojan, botnet, etc.)
- Threat level assessment

## Project Structure

```
mei_aas_pf/
├── data/                    # Dataset files
│   └── raw/
│       ├── dga_websites.csv
│       ├── legit_websites.csv
│       ├── dga_domains_full.csv  # Extended DGA dataset with families
│       ├── ddns_links.csv        # 34k+ DDNS providers
│       └── words.txt
├── src/
│   ├── ml/                  # Machine learning module
│   │   ├── features.py      # Feature extraction
│   │   ├── random_forest_model.py
│   │   ├── lstm_model.py
│   │   ├── xgboost_model.py
│   │   ├── gradient_boosting_model.py
│   │   ├── transformer_model.py
│   │   ├── distilbert_model.py
│   │   ├── family_classifier.py  # Family classification models
│   │   └── train.py         # Training script
│   ├── api/                 # FastAPI backend
│   │   ├── main.py          # API endpoints + Prometheus metrics
│   │   ├── models.py        # Pydantic schemas
│   │   ├── database.py      # SQLite logging
│   │   └── auth.py          # Authentication
│   └── utils/               # Utility functions
├── models/                  # Trained models (generated)
├── frontend/                # React dashboard
│   ├── src/
│   │   ├── components/ui/   # shadcn/ui components
│   │   ├── pages/
│   │   │   ├── Dashboard/   # Dashboard page & components
│   │   │   ├── Scanner/     # Scanner page & components
│   │   │   └── Models/      # Models page & components
│   │   ├── services/        # API client
│   │   └── types/           # TypeScript types
│   └── package.json
├── extension/               # Chrome extension
│   ├── manifest.json
│   ├── background.js        # Service worker
│   ├── content.js           # Warning overlay
│   ├── popup.html/js        # Extension popup
│   └── icons/               # Extension icons
├── monitoring/              # Monitoring stack
│   ├── prometheus.yml       # Prometheus configuration
│   └── grafana/
│       ├── provisioning/    # Auto-provisioning configs
│       └── dashboards/      # Pre-built dashboards
├── tests/
│   └── load/                # k6 load test scripts
│       ├── k6-load-test.js
│       └── k6-smoke-test.js
├── Dockerfile.backend       # Backend Docker image
├── Dockerfile.frontend      # Frontend Docker image
├── compose.yml              # Docker orchestration (includes monitoring)
├── nginx.conf               # Nginx configuration
├── Makefile                 # Build commands
├── requirements.txt         # Python dependencies
└── README.md
```

## Results

Performance metrics from training on the full dataset:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Random Forest | 89.34% | 92.91% | 85.18% | 88.88% | 95.89% |
| CNN-LSTM | **95.88%** | **96.20%** | **95.53%** | **95.86%** | **99.27%** |

The CNN-LSTM model outperforms Random Forest across all metrics, achieving nearly 96% accuracy and 99% ROC-AUC.

## Security Considerations

- This tool is for **educational and defensive purposes only**
- The extension monitors URLs locally; no data is sent to external servers
- API runs on localhost by default
- For production deployment, add proper authentication and HTTPS

## Future Work

- [ ] Implement online learning for new threats
- [ ] Add DNS query monitoring
- [ ] Support for Firefox extension
- [ ] Add alerting rules for Prometheus
- [x] Containerize with Docker
- [x] Add Google OAuth for dashboard/extension sync
- [x] Add DGA family classification (25 families)
- [x] Add multiple ML models (XGBoost, GB, Transformer, DistilBERT)
- [x] Add Prometheus metrics and Grafana dashboards
- [x] Add k6 load testing scripts
- [x] DDNS provider detection (34k+ providers)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- [DGA Detection with Machine Learning](https://www.endgame.com/blog/technical-blog/using-deep-learning-detect-dgas)
- [Character-level CNNs for DGA Detection](https://arxiv.org/abs/1611.00791)
- Woodbridge et al., "Predicting Domain Generation Algorithms with Long Short-Term Memory Networks"
