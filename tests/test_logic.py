from services.ocr_asr.main import STRICT_GID_REGEX, _extract_coords, _find_gid
from services.orchestrator.main import _rrf, _weights


def test_gid_regex_and_normalization():
    assert STRICT_GID_REGEX.match("ANBB00001А")
    assert _find_gid("found gid: ANВВ00001А") == "ANBB00001А"
    assert _find_gid("wrong gid: ANBB0001A") is None


def test_extract_coords_with_keywords():
    lat, lon = _extract_coords("широта 55.75 долгота 37.61")
    assert lat == 55.75
    assert lon == 37.61


def test_extract_coords_comma_decimal_and_bounds():
    lat, lon = _extract_coords("координаты 55,75 37,61")
    assert lat == 55.75
    assert lon == 37.61

    lat2, lon2 = _extract_coords("широта 155.75 долгота 237.61")
    assert lat2 is None
    assert lon2 is None


def test_rrf_weights():
    lists = {"visual": [{"gid": "A"}, {"gid": "B"}], "semantic": [{"gid": "B"}]}
    weights = _weights(has_image=True, has_text=True, has_ocr_gid=False)
    scores = _rrf(lists, weights)
    assert scores["B"] > 0
    assert "A" in scores
