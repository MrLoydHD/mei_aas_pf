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
- Two ML models: Random Forest (interpretable) and LSTM (deep learning)
- FastAPI REST backend for real-time predictions
- React dashboard for visualization and analysis
- Chrome browser extension for real-time protection

**Authors:**
* Hugo Correia

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
│  - /stats - Detection statistics                            │
│  - /models - Model information                              │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   ML Models                                  │
│  - Random Forest: Handcrafted features                      │
│  - CNN-LSTM: Character-level deep learning                  │
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
docker-compose build
docker-compose up -d
```

**Available Commands:**

| Command | Description |
|---------|-------------|
| `make build` | Build all Docker images |
| `make up` | Start all services |
| `make down` | Stop all services |
| `make logs` | View logs from all services |
| `make train` | Train ML models |
| `make clean` | Remove all containers and images |

**Services:**
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## Dataset

The project uses a balanced DGA detection dataset located in the `data/raw/` folder:

| File | Records | Description |
|------|---------|-------------|
| `dga_websites.csv` | 337,500 | DGA-generated domains |
| `legit_websites.csv` | 337,398 | Legitimate domains |
| `words.txt` | 354,986 | English dictionary for feature extraction |

## Usage

### Step 1: Train the Models

```bash
# Train both Random Forest and LSTM models
python -m src.ml.train

# Train with smaller sample (faster, for testing)
python -m src.ml.train --sample-size 50000

# Train only Random Forest
python -m src.ml.train --rf-only

# Train only LSTM
python -m src.ml.train --lstm-only --epochs 30
```

Training outputs:
- `models/random_forest.joblib` - Trained RF model
- `models/lstm/` - Trained LSTM model
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

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Classify single domain |
| `/predict/batch` | POST | Classify multiple domains |
| `/predict/detailed` | POST | Classify with feature analysis |
| `/stats` | GET | Detection statistics |
| `/models` | GET | Model information |
| `/extension/check` | POST | Lightweight endpoint for extension |

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
```

## Model Comparison

### Random Forest

**Approach:** Handcrafted features including:
- Domain length, entropy, character distribution
- N-gram analysis (bigrams, trigrams)
- Dictionary word presence
- Vowel/consonant ratios

**Pros:**
- Fast inference (~1ms per domain)
- Interpretable features
- No GPU required

### LSTM (CNN-LSTM Hybrid)

**Approach:** Character-level deep learning:
- Character embeddings
- 1D CNN for local pattern detection
- Bidirectional LSTM for sequential patterns

**Pros:**
- Learns features automatically
- Better at capturing complex patterns
- Higher accuracy potential

## Project Structure

```
mei_aas_pf/
├── data/                    # Dataset files
│   └── raw/
│       ├── dga_websites.csv
│       ├── legit_websites.csv
│       └── words.txt
├── src/
│   ├── ml/                  # Machine learning module
│   │   ├── features.py      # Feature extraction
│   │   ├── random_forest_model.py
│   │   ├── lstm_model.py
│   │   └── train.py         # Training script
│   ├── api/                 # FastAPI backend
│   │   ├── main.py          # API endpoints
│   │   ├── models.py        # Pydantic schemas
│   │   └── database.py      # SQLite logging
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
├── Dockerfile.backend       # Backend Docker image
├── Dockerfile.frontend      # Frontend Docker image
├── docker-compose.yml       # Docker orchestration
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

- [ ] Add more DGA family-specific detection
- [ ] Implement online learning for new threats
- [ ] Add DNS query monitoring
- [ ] Support for Firefox extension
- [x] Containerize with Docker

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- [DGA Detection with Machine Learning](https://www.endgame.com/blog/technical-blog/using-deep-learning-detect-dgas)
- [Character-level CNNs for DGA Detection](https://arxiv.org/abs/1611.00791)
- Woodbridge et al., "Predicting Domain Generation Algorithms with Long Short-Term Memory Networks"
