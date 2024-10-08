from .conftest import check_command_output, check_with_validation_file, metaids


def test_file_loading():
    args = [
        "--retrain",
        "--train.ipal",
        "misc/ipal/train.ipal",
        "--live.ipal",
        "misc/ipal/test.ipal",
        "--config",
        "misc/file_loading/ids.config",
        "--extra.config",
        "misc/file_loading/extra.config",
        "--combiner.config",
        "misc/file_loading/combiner.config",
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
        check_for=["ERROR"],
    )

    check_with_validation_file(
        "ExtraIDS-stderr.ipal",
        stderr.decode("utf-8"),
        test_file_loading.__name__,
        normalize_data=False,
    )

    check_with_validation_file(
        "ExtraIDS.ipal",
        stdout.decode("utf-8"),
        test_file_loading.__name__,
    )
