"""설정 파일 로더 및 검증"""
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Config:
    """전체 설정을 담는 데이터 클래스"""
    exchange: dict = field(default_factory=dict)
    trading: dict = field(default_factory=dict)
    leverage: dict = field(default_factory=dict)
    risk: dict = field(default_factory=dict)
    strategy: dict = field(default_factory=dict)
    data: dict = field(default_factory=dict)
    telegram: dict = field(default_factory=dict)
    logging: dict = field(default_factory=dict)

    def get(self, dotpath: str, default: Any = None) -> Any:
        """점 표기법으로 중첩 설정 접근: config.get('risk.daily_max_loss_pct')"""
        keys = dotpath.split(".")
        val = self.__dict__
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                return default
            if val is None:
                return default
        return val


def load_config(path: str = "config/settings.yaml") -> Config:
    """YAML 설정 파일 로드 및 검증"""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"설정 파일 없음: {path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # 필수 필드 검증
    required = ["exchange", "trading", "risk"]
    for key in required:
        if key not in raw:
            raise ValueError(f"필수 설정 누락: {key}")

    # API 키 검증
    if raw["exchange"].get("api_key") == "YOUR_API_KEY":
        raise ValueError("API 키를 설정하세요 (config/settings.yaml)")

    return Config(**raw)
