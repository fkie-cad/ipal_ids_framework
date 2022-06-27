import json
from pathlib import Path

from subprocess import Popen, PIPE

METAIDS = "./ipal-iids"
IDSNAMES = [
    "Autoregression",
    "BLSTM",
    "DecisionTree",
    "Dummy",
    "ExtraTrees",
    "Histogram",
    "IsolationForest",
    "MinMax",
    "NaiveBayes",
    "Optimal",
    "RandomForest",
    "SVM",
    "Steadytime",
    "inter-arrival-mean",
    "inter-arrival-range",
]


def metaids(args):
    p = Popen([METAIDS] + args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p.communicate()
    return p.returncode, stdout, stderr


########################
# Helper methods
########################


def normalize(content):
    normalized = ""
    for line in content.splitlines():
        normalized += (
            json.dumps(json.loads(line), indent=4, ensure_ascii=False, sort_keys=True)
            + "\n"
        )
    return normalized


def assert_file_contents_equal(validation_path: Path, output_path: Path):
    validation_content = validation_path.read_text().splitlines()
    output_content = output_path.read_text().splitlines()
    for val, out in zip(validation_content, output_content):
        assert "###IGNORE-LINE###" in val or val == out


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
    print(filename, content, prefix)
    output_path, tmp_path, validation_path = calculate_and_create_paths(
        filename, prefix
    )

    tmp_path.write_text(content)

    if normalize_data:
        content = normalize(content)

    output_path.write_text(content)
    if not validation_path.is_file():
        validation_path.write_text("== new file ==\n" + content)

    assert_file_contents_equal(validation_path, output_path)
