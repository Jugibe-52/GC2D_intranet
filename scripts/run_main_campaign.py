"""Prepare, run or export the agreed campaign without executing a notebook."""
from argparse import ArgumentParser
from pathlib import Path
from studies.main_campaign import CampaignConfig, RHOS
from studies.main_campaign_io import prepare_campaign, run_campaign, finalize_campaign


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('prepare', 'run', 'view', 'finalize'))
    parser.add_argument('directory', type=Path)
    parser.add_argument('--cycles', type=int, default=5000)
    parser.add_argument('--pilot', action='store_true', help='Select rho=0.30; set --cycles explicitly for a short pilot.')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--local-only', action='store_true', help='Keep Modal results locally without uploading to B2.')
    parser.add_argument('--allow-partial', action='store_true',
        help='Publish verified results while explicitly showing recorded failures without saved archives.')
    args = parser.parse_args()
    if args.allow_partial and args.operation not in ('view', 'finalize'):
        parser.error('--allow-partial applies only to view or finalize.')
    root = Path(__file__).resolve().parents[1]
    directory = args.directory.resolve()
    if args.operation == 'prepare':
        prepare_campaign(directory, root, config=CampaignConfig(cycles=args.cycles),
            particle_path=root.parent / 'TFM_redaccion/datos/conjuntos_particulas_rho030.json',
            rhos=(.3,) if args.pilot else RHOS)
        print(directory / 'campaign.json')
    elif args.operation == 'run':
        report = run_campaign(directory, root, max_workers=args.workers, publish=not args.local_only)
        if report['status'] == 'failed':
            raise SystemExit('Some cases failed; inspect progress.json and recover their original calls.')
    elif args.operation == 'finalize':
        print(finalize_campaign(directory, allow_partial=args.allow_partial))
    else:
        from visualization.main_campaign import export_campaign
        print(export_campaign(directory, allow_partial=args.allow_partial))


if __name__ == '__main__':
    main()
