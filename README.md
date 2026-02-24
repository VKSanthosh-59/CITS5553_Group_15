# ESRI Comparison Tool

A full-stack tool for comparing original vs deep-learning (DL) imputed geochemical assay data for mineral exploration in Western Australia.

**Frontend:** React + Vite + TailwindCSS + Plotly.js  
**Backend:** FastAPI + Pandas + Matplotlib + PyProj

---

## � Project Structure

```
CITS5553_Group_15/
├── backend-esri/            # FastAPI backend
│   ├── app/
│   │   ├── main.py          # FastAPI app, CORS, router registration
│   │   ├── models/
│   │   │   └── schemas.py   # Pydantic response models
│   │   ├── routers/
│   │   │   ├── data.py      # /api/data endpoints (column extraction)
│   │   │   └── analysis.py  # /api/analysis (stats, plots, comparison, export)
│   │   └── services/
│   │       ├── io_service.py    # CSV/DBF/ZIP parsing, encoding detection
│   │       └── comparisons.py  # Grid comparison methods (mean, median, max)
│   └── requirements.txt
├── frontend-esri/           # React + Vite frontend
│   ├── src/
│   │   ├── main.tsx                  # Entry point
│   │   ├── ESRI3DComparisonApp.tsx   # Main application component
│   │   ├── api/
│   │   │   ├── data.ts       # API client for data endpoints
│   │   │   └── analysis.ts   # API client for analysis endpoints
│   │   └── index.css         # Tailwind directives + base styles
│   ├── package.json
│   ├── vite.config.ts        # Vite config (API proxy to backend)
│   ├── tailwind.config.js
│   └── tsconfig.json
├── experimental/            # Standalone analysis scripts & notebooks
│   ├── comparisons.py       # Extended comparison methods (7 methods)
│   ├── clean_parquet_lib.py  # Data cleaning & Parquet I/O library
│   ├── bench_clean_parquet_batch.py  # Pipeline benchmarking
│   └── *.ipynb              # EDA notebooks
├── data/                    # Sample geospatial data
├── Documentation/           # Project documentation
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** with `pip` and `venv`
- **Node.js 20+** with `npm 9+`

### 1. Start the Backend

```bash
cd backend-esri

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate       # macOS / Linux
venv\Scripts\activate          # Windows (cmd)
.\venv\Scripts\Activate.ps1    # Windows (PowerShell)

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn app.main:app --reload
```

Backend runs at **http://127.0.0.1:8000**

### 2. Start the Frontend

```bash
cd frontend-esri

# Install dependencies
npm install

# Run the dev server
npm run dev
```

Frontend runs at **http://localhost:5173**

### 3. Open the App

Navigate to [http://localhost:5173](http://localhost:5173) in your browser.

> **Note:** Start the backend first, then the frontend.

---

## 🔗 Frontend ↔ Backend Integration

- **CORS** is enabled in the backend for `http://localhost:5173` and `http://localhost:5174`.
- The Vite dev server **proxies** all `/api/*` requests to `http://localhost:8000` automatically.
- No manual URL configuration is needed in development.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Backend health check |
| `POST` | `/api/data/columns` | Extract column names from uploaded CSV/DBF/ZIP files |
| `POST` | `/api/analysis/summary` | Compute summary statistics (count, mean, median, max, std) |
| `POST` | `/api/analysis/plots` | Generate histograms + QQ plot as base64 PNGs |
| `POST` | `/api/analysis/plots-data` | Return plot data as JSON (for interactive Plotly charts) |
| `POST` | `/api/analysis/comparison` | Run grid-based comparison and return heatmap arrays |
| `POST` | `/api/analysis/export/plots` | Export selected plots as a ZIP of PNGs |

---

## 🏗️ Production Build (Frontend)

```bash
cd frontend-esri
npm run build
npm run preview
```

---

## ❓ Troubleshooting

### Frontend

| Issue | Solution |
|-------|----------|
| `"vite is not recognized"` | Run `npm install` first. If it persists, delete `node_modules` and `package-lock.json`, then reinstall. |
| Port 5173 already in use | Run `npm run dev -- --port 5174` or kill the process using port 5173. |

### Backend

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Ensure you run `uvicorn` from inside `backend-esri/`, not the project root. |
| Port 8000 already in use | Run `uvicorn app.main:app --reload --port 8001` |
| `No module named 'simpledbf'` | Run `pip install -r requirements.txt` inside your virtual environment. |

---

## 📄 License

University of Western Australia — CITS5553 Capstone Project (Group 15)
