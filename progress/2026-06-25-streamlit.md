# 2026-06-25 Streamlit Progress

## Done

- Added `app.py` Streamlit workbench.
- The workbench supports:
  - local skin search;
  - database evidence, ignored evidence, or manual simulated evidence;
  - sales action report display;
  - evidence structure view;
  - skin metadata table;
  - JSON export.
- Added `tests/test_streamlit_app.py`.

## Validation

- `python -m unittest discover -s tests -v` passed.
- `app.build_payload(...)` was smoke-tested against local `data/wzry_skins/skins.sqlite3`.

## Run

```bash
python -m streamlit run app.py --server.port 8501
```
