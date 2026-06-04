# Episode log analysis

Put `.xlsx` logs in `episode logs/`. PDFs go to `episode report/`.

From the repo root:

```powershell
python analysis/spawn_spatial_report.py "data/episode logs/your_log.xlsx" -o "data/episode report"
```

Quote the path — the folder name has a space (`episode logs`).

Console taxonomy only (no PDF):

```powershell
python analysis/spawn_spatial_report.py "data/episode logs/your_log.xlsx" --taxonomy-only
```

Optional: If you have multiple phases in one spreadsheet, add `--spawn-phase #` flag to filter to one curriculum phase. (i.e '--spawn-phase 4' to view only phase 4 in a spreadsheet with multiple phases)
