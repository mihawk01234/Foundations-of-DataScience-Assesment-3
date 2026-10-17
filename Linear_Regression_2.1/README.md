# FIFA World Cup 2026: Linear Regression 2.1 (goal difference)

## Run order

```bash
python task2.1_part1.py   # merge, data checks and corrections, EDA, VIF -> task2.1_model_dataset.csv
python task2.1_part2.py   # train and evaluate the model                -> results/lr21_*, figures/lr21_*
```

Packages: pandas, numpy, matplotlib, seaborn, statsmodels, scikit-learn.

## Data

| File | Contents |
|---|---|
| `matches.csv` | Results of the 104 matches (score, penalties, status, match xG) |
| `match_prediction_features.csv` | One row per match with pre-match information (FIFA rank, Elo, squad profile, rest days, averages from earlier WC 2026 matches, venue) |
| `squad_values_transfermarkt.csv` | Transfermarkt squad values (final 26-man squads, June 2026) for the 48 teams. These replace the values in the features file |
| `task2.1_model_dataset.csv` | Model dataset: 104 rows, 8 explanatory variables, target `goal_diff` |

## Data corrections (in task2.1_part1.py)

- **Switzerland 0–0 Colombia (Round of 16):** recorded as a normal-time result, but FIFA's official record shows Switzerland won 4–3 on penalties. The result type is corrected; the goal difference stays 0.
- **Squad values:** 16 teams were more than 25% off Transfermarkt's June 2026 values (e.g. Qatar €121m vs €20m), so all 48 are replaced with `squad_values_transfermarkt.csv`.
- **First-match placeholders:** before a team's first match the "previous match" fields use defaults (0 goals conceded, 10 rest days). Both teams have them on matchday 1, so the differences are 0.

## Report files

| File | Contents |
|---|---|
| `task2.1_part2_report.docx` | Two-page write-up of objectives 4–7 (training, evaluation, plots, influential variables and limitations) with the charts as editable Excel charts |
| `task2.1_part2_charts.xlsx` | The Excel workbook behind those charts |
