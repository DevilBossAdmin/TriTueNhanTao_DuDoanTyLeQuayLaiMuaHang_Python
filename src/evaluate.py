"""Read saved evaluation artifacts and print a compact report."""
import json
from .config import METRICS_PATH


def main() -> None:
    if not METRICS_PATH.exists():
        raise FileNotFoundError("Chưa có metrics. Hãy chạy `python -m src.train`.")
    payload = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
