# Tech Stack

## Language
- Python 3.8-3.10

## Web Framework
- Flask (backend API and template rendering)

## Core Libraries
- **CausalNex**: Bayesian Network structure learning and inference (forked/embedded in project)
- **NetworkX**: Graph data structures and algorithms
- **PyVis**: Interactive network visualization
- **PyTorch**: Deep learning backend for neural causal discovery methods
- **scikit-learn**: Data preprocessing, label encoding
- **pandas/numpy**: Data manipulation
- **pgmpy**: Probabilistic graphical models

## Frontend
- HTML templates (Jinja2)
- JavaScript (vanilla)
- vis.js network visualization library
- tom-select for UI components

## Causal Discovery Algorithms
Available in `causaldiscovery_backend/castle/algorithms/`:
- PC (constraint-based)
- GES (score-based)
- NOTEARS (gradient-based, continuous optimization)
- LiNGAM (linear non-Gaussian)
- ANM (additive noise models)
- DAG-GNN, GraNDAG, GOLEM (neural methods)

## Common Commands

### Install Dependencies
```bash
pip install -r causaldiscovery_backend/requirements.txt
```

### Run Backend API Server
```bash
python app.py
# Runs on http://127.0.0.1:5002
```

### Run Web Application
```bash
cd causal_web_app-main_merged_v4
python app.py
# Runs on http://127.0.0.1:5001
```

## Configuration
- Logging configured to `causaldiscovery_backend/logs/app.log`
- File uploads stored in `causaldiscovery_backend/uploads/`
- Static graph outputs in `static/` directories


## Code Style
代码修改尽可能简洁但满足需求，不要增加给出需求之外的功能