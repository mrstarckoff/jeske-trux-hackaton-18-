from app.services.feature_extraction import extract_priority_gid
from app.services.retrieval import platt_scaling, weighted_rrf


def test_gid_regex_priority() -> None:
    gid = extract_priority_gid(['noise', 'ABCD12345A'])
    assert gid == 'ABCD12345A'


def test_weighted_rrf_and_platt() -> None:
    rankings = {
        'visual': ['A', 'B', 'C'],
        'semantic': ['B', 'C', 'A'],
    }
    weights = {'visual': 0.4, 'semantic': 0.6}
    scores = weighted_rrf(rankings, weights, k=1)
    assert scores['B'] > scores['A']
    assert 0 <= platt_scaling(scores['B']) <= 100
