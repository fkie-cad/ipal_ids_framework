from .conftest import metaids
from .conftest import file_eq


def test_metaids_empty():
    errno, stdout, stderr = metaids([])
    assert stdout == b""
    assert errno == 1
    assert b"ERROR:IDS:no IDS configuration provided, exiting\n" in stderr
