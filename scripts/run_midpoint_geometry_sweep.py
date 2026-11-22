"""Run or archive the accumulated midpoint step/horizon sweep."""
from argparse import ArgumentParser
from pathlib import Path
from diagnostics.studies.midpoint_geometry_sweep import archive_record
from studies.midpoint_geometry_sweep import RUN_ID, run_sweep


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/midpoint_geometry')/RUN_ID)
    parser.add_argument('--archive-only', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.archive_only:
        archive_record(args.output)
    else:
        run_sweep(root, args.output)


if __name__ == '__main__':
    main()
