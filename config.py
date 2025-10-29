from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class AppConfig:
    data_dir: Path = Path("./data").resolve()
    cache_prices_dir: Path = Path("./data/prices").resolve()
    default_start: str = "2015-01-01"
    default_end: str = "2100-01-01"  # effectively "until today" by default
    default_risk_free: float = 0.02  # 2% annual risk-free; configurable via CLI
    max_retries: int = 3
    timeout_sec: int = 20

CONFIG = AppConfig()
