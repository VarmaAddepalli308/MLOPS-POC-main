"""Configures pytest for the stock project.

* Puts src/ on sys.path so tests can `from utils import ...` exactly like the notebooks do.
* Provides `spark` and `load_fixture` fixtures backed by Databricks Connect (same pattern as the fraud
  reference). They SKIP when databricks-connect / a workspace is not available, so the pure-Python unit
  tests still run in CI and on a laptop.
"""
import csv
import json
import os
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def spark():
    """SparkSession via Databricks Connect (serverless fallback). Skips when unavailable."""
    try:
        from databricks.connect import DatabricksSession
        from databricks.sdk import WorkspaceClient
    except ImportError:
        pytest.skip("databricks-connect not installed (run `uv sync --dev`)")
    try:
        conf = WorkspaceClient().config
        if not (conf.serverless_compute_id or conf.cluster_id or os.environ.get("SPARK_REMOTE")):
            os.environ["DATABRICKS_SERVERLESS_COMPUTE_ID"] = "auto"
        return DatabricksSession.builder.getOrCreate()
    except Exception as e:  # no credentials / unreachable workspace
        pytest.skip(f"Databricks Connect session unavailable: {e}")


@pytest.fixture()
def load_fixture(spark):
    """Callable that loads a JSON or CSV file from fixtures/ into a Spark DataFrame."""

    def _loader(filename: str):
        path = ROOT / "fixtures" / filename
        suffix = path.suffix.lower()
        if suffix == ".json":
            return spark.createDataFrame(json.loads(path.read_text()))
        if suffix == ".csv":
            with path.open(newline="") as f:
                return spark.createDataFrame(list(csv.DictReader(f)))
        raise ValueError(f"Unsupported fixture type for: {filename}")

    return _loader
