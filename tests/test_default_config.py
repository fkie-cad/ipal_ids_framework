import pytest

from .conftest import metaids
from .conftest import IDSNAMES
from .conftest import check_with_validation_file


@pytest.mark.parametrize("idsname", IDSNAMES)
def test_default_config_ipal(idsname):

    args = [
        "--retrain",
        "--train.ipal",
        "misc/ipal/train.ipal",
        "--live.ipal",
        "misc/ipal/test.ipal",
        "--config",
        "misc/configs/{}.config".format(idsname),
        "--output",
        "-",
    ]

    errno, stdout, stderr = metaids(args)

    check_with_validation_file(
        "{}-stderr.ipal".format(idsname),
        stderr.decode("utf-8"),
        test_default_config_ipal.__name__,
        normalize_data=False,
    )

    check_with_validation_file(
        "{}.ipal".format(idsname),
        stdout.decode("utf-8"),
        test_default_config_ipal.__name__,
    )


@pytest.mark.parametrize("idsname", IDSNAMES)
def test_default_config_state(idsname):

    args = [
        "--retrain",
        "--train.state",
        "misc/ipal/train.ipal",
        "--live.state",
        "misc/ipal/test.ipal",
        "--config",
        "misc/configs/{}.config".format(idsname),
        "--output",
        "-",
    ]

    errno, stdout, stderr = metaids(args)

    check_with_validation_file(
        "{}-stderr.state".format(idsname),
        stderr.decode("utf-8"),
        test_default_config_state.__name__,
        normalize_data=False,
    )

    check_with_validation_file(
        "{}.state".format(idsname),
        stdout.decode("utf-8"),
        test_default_config_state.__name__,
    )
