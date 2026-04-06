import pandas as pd


def parse_csv(file_path: str) -> str:
    ext = file_path.split(".")[-1].lower()
    if ext == "csv":
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)
    return df.to_string(index=False)