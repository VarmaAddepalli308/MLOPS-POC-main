import re

from utils import FEATURE_COLS, load_model_params, load_project, load_yaml, model_name, table

KNOWN_MODELS = {"elastic_net", "random_forest", "extra_trees", "hist_gradient_boosting", "gradient_boosting", "xgboost", "lightgbm", "catboost"}


def test_project_config_sections_present():
    cfg = load_project()
    for k in ["data", "tables", "schemas", "model", "evaluation", "monitoring", "serving"]:
        assert k in cfg
    assert cfg["model"]["selection_metric"] in {"val_rmse", "val_mae", "val_r2", "val_directional_accuracy"}
    assert cfg["model"]["validation_fraction"] + cfg["model"]["test_fraction"] < 0.9


def test_every_table_schema_is_bootstrapped():
    cfg = load_project()
    for key, qualified in cfg["tables"].items():
        assert qualified.split(".")[0] in cfg["schemas"], key
    assert model_name(cfg, "c").split(".")[1] in cfg["schemas"]


def test_names_are_catalog_qualified():
    cfg = load_project()
    assert table(cfg, "inference_log", "stock_demo_test") == "stock_demo_test.serving.stock_inference_log"
    assert model_name(cfg, "stock_demo_prod") == "stock_demo_prod.models.stock_price_regressor"


def test_model_params_known_and_non_empty():
    grids = load_model_params()
    assert grids and set(grids) <= KNOWN_MODELS
    assert all(isinstance(v, list) and v for g in grids.values() for v in g.values())


def test_drift_features_are_model_features():
    assert set(load_project()["monitoring"]["drift_features"]) <= set(FEATURE_COLS)


def test_train_py_factories_match_model_params():
    src = open("src/train.py").read()
    section = src.split("MODEL_FACTORIES = {", 1)[1].split("unknown =", 1)[0]
    declared = set(re.findall(r'^\s+"(\w+)": lambda', section, flags=re.M))
    assert declared == KNOWN_MODELS - {"xgboost", "lightgbm", "catboost"}
    assert all(name in src for name in ("XGBRegressor", "LGBMRegressor", "CatBoostRegressor"))


def test_live_config_complete_and_consistent():
    live = load_yaml("live.yml")["live_data"]
    for k in ["provider", "symbols", "max_symbols", "universe_table", "universe_lookback_days",
              "overlap_days", "max_lookback_days", "auto_adjust", "min_symbol_success_ratio",
              "price_mismatch_tolerance", "max_staleness_days"]:
        assert k in live, k
    assert live["universe_table"].split(".")[0] in load_project()["schemas"]
    assert 0 < live["min_symbol_success_ratio"] <= 1 and live["overlap_days"] >= 1


def test_live_first_run_is_yahoo_bootstrap_without_file_source():
    cfg = load_project()
    live = load_yaml("live.yml")["live_data"]
    assert live["provider"] == "yfinance"
    assert live["symbols"]
    assert "source_file" not in cfg["data"]
    assert "volume_path" not in cfg["data"]
    live_ingestion = open("src/ingest_live.py").read()
    assert "if bronze_exists" in live_ingestion
    assert "bootstrapping historical data from Yahoo Finance" in live_ingestion


def test_training_job_refreshes_live_data_before_features():
    job = open("resources/training_job.yml").read()
    assert job.index("task_key: ingest_live_data") < job.index("task_key: build_features")
    assert "../src/ingest_live.py" in job
