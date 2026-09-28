"""Strategy registry - importing this package registers every strategy."""

from bot.strategies.base import REGISTRY, Strategy, available, build, register

# Importing the modules is what registers them via the @register decorator.
from bot.strategies import breakout, ma_crossover, rsi_reversion  # noqa: F401,E402

__all__ = ["Strategy", "REGISTRY", "build", "available", "register"]
