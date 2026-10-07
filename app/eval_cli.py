import asyncio
import json
from .evaluation import evaluate_benchmark

if __name__ == "__main__":
    report = asyncio.run(evaluate_benchmark())
    print(json.dumps(report, indent=2, sort_keys=True))
