import pytest

from .conftest import (
    IDSNAMES,
    _filter_tensorflow_errors,
    check_command_output,
    check_with_validation_file,
    metaids,
)


def test_metaids_empty():
    args = []
    errno, stdout, stderr = metaids(args)

    check_command_output(
        returncode=errno,
        args=args,
        stdout=stdout,
        stderr=stderr,
        expectedcode=1,
        expected_stderr=[
            r"ERROR:ipal-iids:no IDS configuration provided, exiting\n"
        ],  # only checks if that string shows up
        expected_stdout=[b""],
        check_for=["WARNING"],
    )


STATIC_COMMANDS = [["-h"], ["--version"]]


@pytest.mark.parametrize("args", STATIC_COMMANDS)
def test_metaids_static_commands(args):
    errno, stdout, stderr = metaids(args)

    check_command_output(
        returncode=errno,
        args=args,
        stdout=stdout,
        stderr=stderr,
        expectedcode=0,
        expected_stderr=[b""],
        check_for=["WARNING", "ERROR"],
    )


@pytest.mark.parametrize("idsname", IDSNAMES)
def test_get_default_config(idsname):
    args = ["--default.config", idsname]
    errno, stdout, stderr = metaids(args)
    stderr = _filter_tensorflow_errors(stderr)

    check_with_validation_file(
        f"{idsname}.config",
        stdout.decode("utf-8").replace("\n", ""),
        test_get_default_config.__name__,
    )

    check_command_output(
        returncode=errno,
        args=args,
        stdout=stdout,
        stderr=stderr,
        expectedcode=0,
        expected_stdout=[f"{idsname}"],  # check if idsname is in stdout
        check_for=["ERROR"],  # check if an IPAL error appears
    )
