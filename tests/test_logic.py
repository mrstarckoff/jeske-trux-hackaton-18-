from services.ocr_asr.main import STRICT_GID_REGEX, _extract_addresses, _extract_coords, _find_gid
from services.orchestrator.main import _rrf, _weights


def test_gid_regex_and_normalization():
    assert STRICT_GID_REGEX.match("ANBB00001А")
    assert _find_gid("found gid: ANВВ00001А") == "ANBB00001А"
    assert _find_gid("wrong gid: ANBB0001A") is None


def test_extract_coords_with_keywords_and_bounds():
    lat, lon = _extract_coords("широта 55.75 долгота 37.61")
    assert lat == 55.75 and lon == 37.61

    lat2, lon2 = _extract_coords("широта 155.75 долгота 237.61")
    assert lat2 is None and lon2 is None


def test_extract_addresses_fallback():
    text = "Объект: улица Ленина 15, рядом остановка"
    vals = _extract_addresses(text)
    assert isinstance(vals, list)
    assert len(vals) >= 1


def test_rrf_weights_multimodal_case():
    weights = _weights(has_image=True, has_text=True, has_ocr_gid=False)
    assert weights == {"ocr": 0.0, "visual": 0.4, "semantic": 0.3, "lexical": 0.15, "geo": 0.15}

    lists = {"visual": [{"gid": "A"}], "semantic": [{"gid": "B"}], "lexical": [{"gid": "A"}]}
    scores = _rrf(lists, weights)
    assert scores["A"] > 0 and scores["B"] > 0
