import pytest

from .conftest import (
    COMBINERNAMES,
    check_command_output,
    check_with_validation_file,
    metaids,
)


@pytest.mark.parametrize("combinername", COMBINERNAMES)
def test_default_combiner_config(combinername):
    args = ["--combiner.default.config", combinername]
    errno, stdout, stderr = metaids(args)

    check_command_output(
        returncode=errno,
        args=args,
        stdout=stdout,
        stderr=stderr,
        expectedcode=0,
        expected_stdout=[f"{combinername}"],  # check if combinername is in stdout
        check_for=["ERROR"],  # check if an IPAL error appears
    )

    check_with_validation_file(
        f"{combinername}.config",
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
        f"misc/configs/combiner-{combinername}.config",
        "--output",
        "-",
    ]

    errno, stdout, stderr = metaids(args)

    check_command_output(
        returncode=errno,
        args=args,
        stdout=stdout,
        stderr=stderr,
        expectedcode=0,
        check_for=["ERROR"],  # check if an IPAL error appears
    )

    check_with_validation_file(
        f"{combinername}-stderr.ipal",
        stderr.decode("utf-8"),
        test_default_config_combiner.__name__,
        normalize_data=False,
    )

    check_with_validation_file(
        f"{combinername}.ipal",
        stdout.decode("utf-8"),
        test_default_config_combiner.__name__,
    )
