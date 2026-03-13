import json
import requests

BASE = "http://localhost:8000"


def _print_case(name: str, response: requests.Response) -> None:
    print(f"\n[{name}] status={response.status_code}")
    try:
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
    except Exception:
        print(response.text)


def run() -> None:
    # 1) text + coords
    r1 = requests.post(
        f"{BASE}/search",
        data={"text": "щит около Тверской", "lat": "55.757", "lon": "37.615"},
        timeout=20,
    )
    _print_case("text+coords", r1)

    # 2) audio transcript with gid + coords
    audio_payload = "широта 55.757 долгота 37.615 рядом gid anbb00001а"
    r2 = requests.post(
        f"{BASE}/search",
        files={"audio": ("sample.txt", audio_payload, "text/plain")},
        timeout=20,
    )
    _print_case("audio transcript", r2)

    # 3) image-only mock (payload includes textual gid for OCR stub)
    image_bytes = b"detected marker ANBB00002A from billboard"
    r3 = requests.post(
        f"{BASE}/search",
        files={"image": ("sample.jpg", image_bytes, "image/jpeg")},
        timeout=20,
    )
    _print_case("image only", r3)

    # 4) full multimodal
    r4 = requests.post(
        f"{BASE}/search",
        data={"text": "цифровой экран у ТТК", "lat": "55.81", "lon": "37.64"},
        files={
            "audio": ("sample.txt", "долгота 37.64 широта 55.81", "text/plain"),
            "image": ("sample.jpg", b"no gid here", "image/jpeg"),
        },
        timeout=20,
    )
    _print_case("full multimodal", r4)


if __name__ == "__main__":
    run()
