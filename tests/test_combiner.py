import pytest

from .conftest import metaids
from .conftest import COMBINERNAMES
from .conftest import check_with_validation_file


@pytest.mark.parametrize("combinername", COMBINERNAMES)
def test_default_combiner_config(combinername):
    errno, stdout, stderr = metaids(["--combiner.default.config", combinername])

    assert combinername in stdout.decode()
    assert errno == 0
    check_with_validation_file(
        "{}.config".format(combinername),
        stdout.decode("utf-8").replace("\n", ""),
        test_default_combiner_config.__name__,
    )


@pytest.mark.parametrize("combinername", COMBINERNAMES)
def test_default_config_combiner(combinername):
    args = [
        "--retrain",
        "--train.ipal",
        "misc/ipal/train.ipal",
        "--train.combiner",
        "misc/ipal/train-combiner.ipal",
        "--live.ipal",
        "misc/ipal/test.ipal",
        "--config",
        "misc/configs/combiner-ids.config",
        "--combiner.config",
        "misc/configs/combiner-{}.config".format(combinername),
        "--output",
        "-",
    ]

    errno, stdout, stderr = metaids(args)

    check_with_validation_file(
        "{}-stderr.ipal".format(combinername),
        stderr.decode("utf-8"),
        test_default_config_combiner.__name__,
        normalize_data=False,
    )

    check_with_validation_file(
        "{}.ipal".format(combinername),
        stdout.decode("utf-8"),
        test_default_config_combiner.__name__,
    )
