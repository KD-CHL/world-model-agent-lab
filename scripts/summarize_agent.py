"""Export a common task/skill summary from an Agent event log."""
import argparse
import json
from pathlib import Path
from wmal.analysis.agent_summary import summarize_agent_events
from wmal.logging.manifest import atomic_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    records = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    summary = summarize_agent_events(records)
    atomic_json(args.output, summary)
    print(json.dumps(summary['task_status_counts'], ensure_ascii=False))


if __name__ == '__main__': main()
