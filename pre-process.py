import pandas as pd
import json


excel_file = "dataset.xlsx"
sheets = ["F1", "F2"]

for sheet in sheets:
    df = pd.read_excel(excel_file, sheet_name=sheet)
    # colonna A (index 0) e C (index 2)
    data = []
    for _, row in df.iterrows():
        entry = {
            "ID": row.iloc[0],
            "TEXT": row.iloc[1]
        }
        data.append(entry)
    # Salva in JSON
    with open(f"{sheet}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)