from services.ocr_asr.main import GID_REGEX, _extract_coords
from services.orchestrator.main import _rrf, _weights


def test_gid_regex():
    assert GID_REGEX.match("ANBB00001А")
    assert not GID_REGEX.match("ANBB0001A")


def test_extract_coords():
    lat, lon = _extract_coords("широта 55.75 долгота 37.61")
    assert lat == 55.75
    assert lon == 37.61


def test_rrf_weights():
    lists = {"visual": [{"gid": "A"}, {"gid": "B"}], "semantic": [{"gid": "B"}]}
    weights = _weights(has_image=True, has_text=True, has_ocr_gid=False)
    scores = _rrf(lists, weights)
    assert scores["B"] > 0
    assert "A" in scores
