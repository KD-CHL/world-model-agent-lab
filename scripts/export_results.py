"""Reserved CLI for export_results; not implemented."""
import argparse

def main():
    parser = argparse.ArgumentParser(description="export_results: design placeholder, not implemented")
    parser.add_argument("--config", help="Future configuration path")
    parser.parse_args()
    parser.exit(2, "NOT IMPLEMENTED: consult docs/07_delivery_plan.md\n")

if __name__ == "__main__":
    main()
