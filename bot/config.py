"""Configuration loading.

Configs can come from a JSON file, environment variables or CLI flags - later
sources win.  Keeping it JSON (rather than YAML) means no extra dependency.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from bot.brokers.paper import BrokerConfig
from bot.core.engine import EngineConfig
from bot.core.risk import RiskConfig
from bot.scaling.engine import ScalingConfig


@dataclass
class BotConfig:
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    scaling: ScalingConfig = field(default_factory=ScalingConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)

    # ------------------------------------------------------------- loading --
    @classmethod
    def load(cls, path: Optional[str] = None) -> "BotConfig":
        data: Dict[str, Any] = {}
        if path:
            target = Path(path)
            if not target.exists():
                raise FileNotFoundError(f"config file not found: {path}")
            data = json.loads(target.read_text())
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BotConfig":
        data = data or {}
        scaling_data = dict(data.get("scaling") or {})
        return cls(
            broker=BrokerConfig(**data.get("broker", {})),
            risk=RiskConfig(**data.get("risk", {})),
            scaling=ScalingConfig.from_dict(scaling_data),
            engine=EngineConfig(**data.get("engine", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "broker": asdict(self.broker),
            "risk": asdict(self.risk),
            "scaling": self.scaling.to_dict(),
            "engine": asdict(self.engine),
        }

    def save(self, path: str) -> str:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2))
        return str(target)


def oanda_credentials() -> Dict[str, str]:
    return {
        "token": os.environ.get("OANDA_TOKEN", ""),
        "account": os.environ.get("OANDA_ACCOUNT", ""),
        "host": os.environ.get("OANDA_HOST", "api-fxpractice.oanda.com"),
    }


def has_oanda_credentials() -> bool:
    creds = oanda_credentials()
    return bool(creds["token"] and creds["account"])
