"""Reserved CLI for run_agent; not implemented."""
import argparse

def main():
    parser = argparse.ArgumentParser(description="run_agent: design placeholder, not implemented")
    parser.add_argument("--config", help="Future configuration path")
    parser.parse_args()
    parser.exit(2, "NOT IMPLEMENTED: consult docs/07_delivery_plan.md\n")

if __name__ == "__main__":
    main()
