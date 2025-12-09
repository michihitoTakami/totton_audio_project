from configparser import ConfigParser
from pathlib import Path


CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "systemd"
    / "journald.conf.d"
    / "magicbox.conf"
)


def _lower_option(option: str) -> str:
    return option.lower()


def load_config() -> ConfigParser:
    parser = ConfigParser()
    parser.optionxform = _lower_option  # keep option names case-insensitive
    parser.read(CONFIG_PATH)
    return parser


def test_template_exists():
    assert CONFIG_PATH.exists(), "journald設定テンプレートが存在しません"


def test_limits_and_retention():
    parser = load_config()
    journal = parser["Journal"]

    assert journal["storage"] == "persistent"
    assert journal["compress"] == "yes"
    assert journal["systemmaxuse"] == "200M"
    assert journal["systemkeepfree"] == "512M"
    assert journal["systemmaxfilesize"] == "16M"
    assert journal["maxretentionsec"] == "14day"


def test_rate_limit_and_forwarding():
    parser = load_config()
    journal = parser["Journal"]

    assert journal["ratelimitintervalsec"] == "30s"
    assert journal["ratelimitburst"] == "500"
    assert journal["forwardtosyslog"] == "no"
