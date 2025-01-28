import pytest

from .conftest import (
    BATCHIDSNAMES,
    check_command_output,
    check_with_validation_file,
    metaids,
)


def test_not_batchids():
    args = [
        "--retrain",
        "--train.ipal",
        "misc/ipal/train.ipal",
        "--live.ipal",
        "misc/ipal/test.ipal",
        "--config",
        "misc/configs/DummyIDS.config",
        "--live.batch",
        "5",
        "--output",
        "-",
    ]
    errno, stdout, stderr = metaids(args)

    check_command_output(
        returncode=errno,
        args=args,
        stdout=stdout,
        stderr=stderr,
        expectedcode=1,
        expected_stderr=[
            r"ERROR:ipal-iids:'Dummy' is not a BatchIDS, can't do batching! Aborting."
        ],  # only checks if that string shows up
        expected_stdout=[b""],
    )


@pytest.mark.parametrize("batchidsname", BATCHIDSNAMES)
def test_batching(batchidsname):
    args = [
        "--retrain",
        "--train.ipal",
        "misc/ipal/train.ipal",
        "--live.ipal",
        "misc/ipal/test.ipal",
        "--config",
        f"misc/configs/{batchidsname}.config",
        "--live.batch",
        "5",
        "--output",
        "-",
    ]

    errno, stdout, stderr = metaids(args)

    check_with_validation_file(
        f"{batchidsname}-stderr.state",
        stderr.decode("utf-8"),
        test_batching.__name__,
        normalize_data=False,
    )

    check_with_validation_file(
        f"{batchidsname}.state",
        stdout.decode("utf-8"),
        test_batching.__name__,
    )

    check_command_output(
        returncode=errno,
        args=args,
        stdout=stdout,
        stderr=stderr,
        expectedcode=0,
        check_for=["ERROR"],  # check if an IPAL error appears
    )
