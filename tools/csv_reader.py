from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import pandas as pd


class CSVLoadError(Exception):
    """Raised when a CSV file cannot be loaded or validated."""


@dataclass
class CSVLoadResult:
    dataframe: pd.DataFrame
    filename: str
    rows: int
    columns: list[str]
    missing_cells: int
    encoding: str


def load_csv(source) -> CSVLoadResult:
    """
    Load a CSV file and return its data with basic metadata.

    Supports file paths and uploaded file-like objects.
    """

    try:
        if isinstance(source, (str, Path)):
            path = Path(source)

            if not path.exists():
                raise CSVLoadError(f"File not found: {path}")

            if not path.is_file():
                raise CSVLoadError(f"Not a regular file: {path}")

            raw_data = path.read_bytes()
            filename = path.name

        elif hasattr(source, "getvalue"):
            raw_data = source.getvalue()
            filename = getattr(source, "name", "uploaded.csv")

        elif hasattr(source, "read"):
            raw_data = source.read()
            filename = getattr(source, "name", "uploaded.csv")

        else:
            raise CSVLoadError("Unsupported input. Provide a file path or file object.")

        if isinstance(raw_data, str):
            raw_data = raw_data.encode("utf-8")

        if not raw_data:
            raise CSVLoadError("The CSV file is empty.")

        try:
            dataframe = pd.read_csv(BytesIO(raw_data), encoding="utf-8-sig")
            encoding = "utf-8-sig"

        except UnicodeDecodeError:
            dataframe = pd.read_csv(BytesIO(raw_data), encoding="latin-1")
            encoding = "latin-1"

        if dataframe.empty and len(dataframe.columns) == 0:
            raise CSVLoadError("The CSV contains no readable columns.")

        if len(dataframe.columns) == 0:
            raise CSVLoadError("The CSV contains no columns.")

        return CSVLoadResult(
            dataframe=dataframe,
            filename=filename,
            rows=len(dataframe),
            columns=dataframe.columns.tolist(),
            missing_cells=int(dataframe.isna().sum().sum()),
            encoding=encoding,
        )

    except CSVLoadError:
        raise

    except Exception as error:
        raise CSVLoadError(f"Could not load CSV: {error}") from error