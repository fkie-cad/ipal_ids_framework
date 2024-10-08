from __future__ import annotations

import json
import re
from pathlib import Path
from subprocess import PIPE, Popen
from typing import List, Union

# Exclude output paths
collect_ignore = ["snapshots"]

METAIDS = "./ipal-iids"
IDSNAMES = [
    # "Autoregression", # Deprecated
    "BLSTM",
    "DecimalPlaces",
    "DecisionTree",
    "DummyIDS",
    "ExistsIDS",
    "ExtraTrees",
    "Histogram",
    "IsolationForest",
    "Kitsune",
    "MinMax",
    "NaiveBayes",
    "OptimalIDS",
    "RandomForest",
    "SVM",
    "SteadyTime",
    "InterArrivalTimeMean",
    "InterArrivalTimeRange",
]

BATCHIDSNAMES = ["DummyBatchIDS"]

COMBINERNAMES = [
    "Any",
    "Gurobi",
    "Heuristic",
    "LSTM",
    "LogisticRegression",
    "MLP",
    "Matrix",
    "SVM",
]


def metaids(args):
    """
    Call the ipal-iids command with the given arguments.

    :param args: List of arguments for the ipal-iids command.
    :return: A tuple containing the return code, stdout, and stderr from the command.
    """
    current_file_path = Path(__file__).resolve()
    parent_dir = current_file_path.parent.parent

    p = Popen([METAIDS] + args, stdout=PIPE, stderr=PIPE, cwd=parent_dir)
    stdout, stderr = p.communicate()
    return p.returncode, stdout, stderr


########################
# Helper methods
########################


def normalize(content):
    normalized = ""
    for line in content.splitlines():
        normalized += f"{json.dumps(json.loads(line), indent=4, ensure_ascii=False, sort_keys=True)}\n"
    return normalized


def assert_file_contents_equal(validation_path: Path, output_path: Path):
    validation_content = validation_path.read_text().splitlines()
    output_content = output_path.read_text().splitlines()

    if len(validation_content) != len(output_content):
        print(f"{'=' * 25} Start of error {'=' * 25}")
        print("Output lengths don't match, test failed!")
        print("Validation file content:")
        print("\n".join(validation_content))
        print("\nOutput file content:")
        print("\n".join(output_content))
        print(f"{'=' * 25}End of error{'=' * 25}")
        assert False, (
            f"Line count mismatch: {len(validation_content)} lines in validation file "
            f"vs {len(output_content)} lines in output file."
        )

    for i, (val, out) in enumerate(zip(validation_content, output_content)):
        if "###IGNORE-LINE###" not in val and val != out:
            print(f"{'=' * 25} Start of error {'=' * 25}")
            print("Output line didn't match, test failed!")
            print(f"Validation file content (Line {i + 1}): {val}")
            print(f"Output file content (Line {i + 1}): {out}")
            print(f"{'=' * 26} End of error {'=' * 26}")
            assert False, (
                f"Line {i + 1} mismatch:\n" f"Validation: {val}\n" f"Output: {out}"
            )


def calculate_and_create_paths(filename: str, prefix: str):
    if prefix:
        prefix += "_"
    base_path = Path(__file__).parent / "snapshots"
    output_path = base_path / "output" / f"{prefix}{filename}"
    tmp_path = base_path / "tmp" / f"{prefix}{filename}"
    validation_path = base_path / "validation" / f"{prefix}{filename}"
    output_path.parent.mkdir(exist_ok=True)
    tmp_path.parent.mkdir(exist_ok=True)
    validation_path.parent.mkdir(exist_ok=True)
    return output_path, tmp_path, validation_path


def check_with_validation_file(
    filename: str, content: str, prefix: str = "", normalize_data: bool = True
):
    print(f"Processing validation file: {filename}")
    print(f"Content length: {len(content)} characters")
    print(f"Prefix used: '{prefix}'")
    print(f"Normalization: {'enabled' if normalize_data else 'disabled'}")

    output_path, tmp_path, validation_path = calculate_and_create_paths(
        filename, prefix
    )

    tmp_path.write_text(content)

    if normalize_data:
        content = normalize(content)

    output_path.write_text(content)

    assert validation_path.is_file(), (
        f"Validation file for '{filename}' does not exist. "
        f"Test output can be found under: {output_path}"
    )

    assert_file_contents_equal(validation_path, output_path)


def check_command_output(
    returncode: int,
    args: List[str],
    stdout: bytes,
    stderr: bytes,
    expectedcode: int = 0,
    expected_stderr: List[Union[bytes, str]] | None = None,
    expected_stdout: List[Union[bytes, str]] | None = None,
    check_for: List[str] | None = None,
):
    """
    Checks returncode and, if given, stderr and stdout, printing debug information if needed.

    :param returncode: Code to check.
    :param args: List of command arguments ran.
    :param stdout: The command's stdout.
    :param stderr: The command's stderr.
    :param expectedcode: Expected return code, default is 0.
    :param expected_stderr: Expected stderr, given as a list of bytes and/or regex strings, default is to not check.
    :param expected_stdout: Expected stdout, given as a list of bytes and/or regex strings, default is to not check.
    :param check_for: Checks stderr for certain strings, used to check for "ERROR" or "WARNING"
    """

    if check_for:
        for phrase in check_for:
            if phrase.encode() in stderr:
                print_command_failure(
                    returncode,
                    expectedcode,
                    args,
                    stdout,
                    stderr,
                    "stderr",
                    expected_stderr,
                )
                assert False, f'"{phrase}" found in stderr when it was not expected'

    if expected_stderr and not match_output(stderr, expected_stderr):
        print_command_failure(
            returncode, expectedcode, args, stdout, stderr, "stderr", expected_stderr
        )
        assert False, "stderr did not match any expected patterns or bytes"

    if expected_stdout and not match_output(stdout, expected_stdout):
        print_command_failure(
            returncode, expectedcode, args, stdout, stderr, "stdout", expected_stdout
        )
        assert False, "stdout did not match any expected patterns or bytes"

    if returncode != expectedcode:
        print_command_failure(
            returncode, expectedcode, args, stdout, stderr, "returncode"
        )
        assert returncode == expectedcode


def print_command_failure(
    returncode: int,
    expectedcode: int,
    args: List[str],
    stdout: bytes,
    stderr: bytes,
    stream_type: str,
    expected: List[Union[bytes, str]] | None = None,
):
    """
    Print detailed information when a command fails.

    :param returncode: The actual return code.
    :param expectedcode: The expected return code.
    :param args: List of command arguments.
    :param stdout: The command's stdout.
    :param stderr: The command's stderr.
    :param stream_type: Type of stream ("stderr", "stdout", or "returncode") that caused the failure.
    :param expected: The expected patterns or bytes for the failing stream (stdout or stderr).
    """
    print(f"{'=' * 20} COMMAND FAILED {'=' * 20}")

    if stream_type == "returncode":
        print(f"Unexpected returncode: {returncode}, expected {expectedcode}")
    else:
        print(f"Unexpected {stream_type}.")
        if expected:
            print(f"Expected one of the following {stream_type}:")
            for exp in expected:
                print(f"  - {repr(exp)}")

    print(f"Command args: {' '.join(args)}")
    print("-- STDOUT --")
    print(stdout.decode("utf-8"))
    print("-- STDERR --")
    print(stderr.decode("utf-8"))


def match_output(output: bytes, expected: List[Union[bytes, str]]) -> bool:
    """
    Check if the output matches any of the expected patterns or bytes.

    :param output: The actual output to check.
    :param expected: List of expected patterns or bytes.
    :return: True if a match is found, otherwise False.
    """
    for exp in expected:
        if isinstance(exp, bytes) and output == exp:
            return True
        elif isinstance(exp, str) and re.search(exp, output.decode("utf-8")):
            return True
    return False
