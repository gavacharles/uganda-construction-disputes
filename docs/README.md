# Uganda Construction Disputes Explorer

Static interactive page (D3) for the ULII construction and arbitration registers.

- `index.html` — the page; switch between 2015–2025 and 2015–2026 at the top.
- `data/<scope>/*.json` — aggregated tables only (counts, shares, test results; no case text or party names).

Rebuild the data: `python -m src.viz.export_dashboard` (after `src.ingest.registers` and `src.temporal.register_temporal`).

Preview locally: `cd docs && python3 -m http.server 8931`, then open http://127.0.0.1:8931.
Publish: GitHub → Settings → Pages → Deploy from branch → `main` / `docs`.
