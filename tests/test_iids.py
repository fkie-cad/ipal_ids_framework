import pytest

from .conftest import metaids
from .conftest import IDSNAMES
from .conftest import check_with_validation_file


def test_metaids_empty():
    errno, stdout, stderr = metaids([])
    assert stdout == b""
    assert errno == 1
    assert b"ERROR:ipal-iids:no IDS configuration provided, exiting\n" in stderr


@pytest.mark.parametrize("idsname", IDSNAMES)
def test_get_default_config(idsname):
    errno, stdout, stderr = metaids(["--default.config", idsname])

    assert idsname in stdout.decode()
    assert errno == 0
    check_with_validation_file(
        "{}.config".format(idsname),
        stdout.decode("utf-8").replace("\n", ""),
        test_get_default_config.__name__,
    )
