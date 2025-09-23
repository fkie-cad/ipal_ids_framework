import pytest

from .conftest import (
    check_command_output,
    check_tensorflow_support,
    check_with_validation_file,
    metaids,
)

supports_alerts = ["MinMax", "SteadyTime", "Histogram", "GeCo"]


@pytest.mark.parametrize("idsname", supports_alerts)
def test_alerts(idsname):
    check_tensorflow_support(idsname)

    args = [
        "--retrain",
        "--train.ipal",
        "misc/ipal/train.ipal",
        "--live.ipal",
        "misc/ipal/test.ipal",
        "--config",
        f"misc/configs/{idsname}.config",
        "--alerts",
        "-",
    ]
    errno, stdout, stderr = metaids(args)

    check_with_validation_file(
        f"{idsname}.alerts",
        stdout.decode("utf-8").replace("\n", ""),
        test_alerts.__name__,
    )
