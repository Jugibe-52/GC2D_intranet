"""Run or recover the paired section 4.2 benchmark without opening Jupyter."""
from argparse import ArgumentParser
import json
from pathlib import Path
from studies.bm4_numpy_jax_io import prepare_study, run_study, publish_study


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('run', 'publish'))
    parser.add_argument('run_id')
    parser.add_argument('--cycles', type=int, nargs='+', default=[10, 30, 90])
    parser.add_argument('--local-results', action='store_true', help='Keep remote results as a local recovery cache without publishing yet.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if Path(args.run_id).name != args.run_id or args.run_id in ('.', '..'):
        parser.error('Use a simple run identifier, not a path.')
    directory = root / 'outputs/studies/bm4_numpy_jax' / args.run_id
    if args.operation == 'publish':
        print(publish_study(directory))
    else:
        if (directory / 'study.json').exists():
            record = json.loads((directory / 'study.json').read_text())
            if record['cycles'] != args.cycles:
                parser.error('The saved run uses different cycle counts.')
        else:
            prepare_study(root, directory, cycles=tuple(args.cycles))
        print(run_study(root, directory, publish=not args.local_results))


if __name__ == '__main__':
    main()
