---
inclusion: always
---

# Project Structure

## Two-Server Architecture

| Server | Location | Port | Purpose |
|--------|----------|------|---------|
| Backend API | `app.py` (root) | 5002 | Causal discovery algorithms, returns graph structure |
| Web App | `causal_web_app-main_merged_v4/app.py` | 5001 | UI, graph editing, version control |

## Directory Layout

### Backend (`causaldiscovery_backend/`)
- `algorithm/` - Algorithm implementations (CIR, AVICI, gCastle, self_compatibility)
- `castle/algorithms/` - Algorithm wrappers (PC, GES, NOTEARS, LiNGAM, ANM, DAG-GNN)
- `castle/backend/` - PyTorch/MindSpore backends
- `causalnex/` - Embedded CausalNex (structure learning, inference, network, discretiser)
- `enitity/` - Entity classes: `CausalGraph.py`, `Node.py`, `Edge.py`
- `util/discover_pure_analysis.py` - Main discovery interface
- `uploads/` - Uploaded data files

### Web App (`causal_web_app-main_merged_v4/`)
- `templates/` - Jinja2 templates (`main.html`, `index.html`, `versions.html`)
- `static/script.js` - Frontend JavaScript logic
- `static/variety_jsons/` - Stored graph JSON files
- `static/variety_htmls/` - Generated PyVis visualizations
- `static/variety_layouts/` - Layout persistence
- `static/version_control/` - Version snapshots per variety
- `lib/` - Frontend libraries (vis.js, tom-select)

### Data
- `data/datasets/` - Sample CSV files
- `lib/causalnex/` - Additional CausalNex copy

## Key Patterns

### Graph Data Flow
1. CSV upload → `Causal_Discovery.discover_file()` → StructureModel
2. StructureModel → JSON (`variety_jsons/`)
3. JSON → `CausalGraph` → PyVis → HTML output

### Graph JSON Schema
```json
{
  "nodes": [["node_name", "type", "source"], ...],
  "edges": [["start", "end", "relation", "source"], ...]
}
```

### Pending Changes Pattern
Modifications tracked in `pending_changes` list before commit, enabling accept/reject workflow.

## Development Guidelines

- Backend changes: modify `causaldiscovery_backend/` modules
- Frontend changes: modify `causal_web_app-main_merged_v4/` templates and static files
- New algorithms: add to `castle/algorithms/` following existing wrapper patterns
- Graph entities: use `CausalGraph`, `Node`, `Edge` classes from `enitity/`
- Logs written to `causaldiscovery_backend/logs/app.log`
