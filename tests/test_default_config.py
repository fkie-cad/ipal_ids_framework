import pytest

from .conftest import (
    IDSNAMES,
    _filter_tensorflow_errors,
    check_command_output,
    check_with_validation_file,
    metaids,
)


@pytest.mark.parametrize("idsname", IDSNAMES)
def test_default_config_ipal(idsname):
    args = [
        "--retrain",
        "--train.ipal",
        "misc/ipal/train.ipal",
        "--live.ipal",
        "misc/ipal/test.ipal",
        "--config",
        f"misc/configs/{idsname}.config",
        "--output",
        "-",
    ]

    errno, stdout, stderr = metaids(args)
    stderr = _filter_tensorflow_errors(stderr)

    check_with_validation_file(
        f"{idsname}-stderr.ipal",
        stderr.decode("utf-8"),
        test_default_config_ipal.__name__,
        normalize_data=False,
    )

    check_with_validation_file(
        f"{idsname}.ipal",
        stdout.decode("utf-8"),
        test_default_config_ipal.__name__,
    )

    ids_without_ipal_support = ["InvariantRules", "PASAD", "Seq2SeqNN", "TABOR", "GeCo"]
    ids_without_ipal_support = [x.lower() for x in ids_without_ipal_support]

    expected_state_error = (
        r"ERROR:ipal\-iids:Required argument: \['train.state', 'live.state'\] for IDS"
    )

    if idsname.lower() in ids_without_ipal_support:
        check_command_output(
            returncode=errno,
            args=args,
            stdout=stdout,
            stderr=stderr,
            expectedcode=1,
            expected_stderr=[expected_state_error],
        )
    else:
        check_command_output(
            returncode=errno,
            args=args,
            stdout=stdout,
            stderr=stderr,
            expectedcode=0,
            check_for=["ERROR"],
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
        f"misc/configs/{idsname}.config",
        "--output",
        "-",
    ]

    errno, stdout, stderr = metaids(args)
    stderr = _filter_tensorflow_errors(stderr)

    check_with_validation_file(
        f"{idsname}-stderr.state",
        stderr.decode("utf-8"),
        test_default_config_state.__name__,
        normalize_data=False,
    )

    check_with_validation_file(
        f"{idsname}.state",
        stdout.decode("utf-8"),
        test_default_config_state.__name__,
    )

    ids_without_state_support = [
        "Kitsune",
        "DTMC",
        "InterArrivalTimeMean",
        "InterArrivalTimeRange",
    ]

    ids_without_state_support = [x.lower() for x in ids_without_state_support]

    expected_ipal_error = (
        r"ERROR:ipal\-iids:Required argument: \['train.ipal', 'live.ipal'\] for IDS"
    )

    if idsname.lower() in ids_without_state_support:
        check_command_output(
            returncode=errno,
            args=args,
            stdout=stdout,
            stderr=stderr,
            expectedcode=1,
            expected_stderr=[expected_ipal_error],
        )
    else:
        check_command_output(
            returncode=errno,
            args=args,
            stdout=stdout,
            stderr=stderr,
            expectedcode=0,
            check_for=["ERROR"] if idsname.lower() == "TABOR" else None,
        )
