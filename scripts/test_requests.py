import json

import requests

BASE = "http://localhost:8000"


def run() -> None:
    cases = [
        (
            "text+coords",
            {"text": "щит около Тверской", "lat": 55.757, "lon": 37.615},
            None,
        ),
        (
            "audio transcript",
            {},
            {"audio": ("sample.txt", "широта 55.757 долгота 37.615 рядом gid anbb00001а", "text/plain")},
        ),
    ]

    for name, data, files in cases:
        r = requests.post(f"{BASE}/search", data=data, files=files, timeout=20)
        print(f"[{name}] status={r.status_code}")
        print(json.dumps(r.json(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
