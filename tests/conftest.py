"""Test session setup: no live web search from tests."""
import os

os.environ["LIVE_MARKET_PRICING"] = "false"
