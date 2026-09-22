"""Load, reuse and assemble matched saved probe runs without reintegration."""

from pathlib import Path

from diagnostics.poincare_probe import load_poincare_probe, save_poincare_probe
from studies.poincare_region_probe import (
    RegionProbeSettings, integrate_region_probes, region_probe_contract, region_probe_panel,
)


def calculate_region_probe_run(potential, settings, directory, *, contract):
    """Reuse a complete matching run or calculate and persist the requested batch."""
    directory = Path(directory)
    if (directory / 'COMPLETE.json').exists():
        result, _ = load_poincare_probe(directory, expected_contract=contract)
        print(f'{settings.method}: loaded the complete matching probe run.')
        return result

    def report(done, total):
        if done % (250 * settings.steps_per_cycle) == 0 or done == total:
            print(f'{settings.method}: {done // settings.steps_per_cycle}/{settings.cycles} cycles', flush=True)

    result = integrate_region_probes(potential, settings, progress=report)
    save_poincare_probe(directory, result, contract=contract)
    print(f'{settings.method}: saved {result["step_count"]:,} complete steps.')
    return result


def load_region_probe_batch(directory, *, background):
    """Read a batch or the original single-particle RK4 record and verify its context.

    Software hashes describe the saved calculation and are not replaced by the
    viewer's current source. The physical field, method and background file hashes
    must still match. This permits viewing earlier completed calculations.
    """
    result, metadata = load_poincare_probe(directory)
    contract = metadata['contract']
    controls = dict(contract['settings'])
    if 'particle_id' in controls:
        controls['particle_ids'] = (controls.pop('particle_id'),)
        controls['initial_xy_over_L'] = (controls['initial_xy_over_L'],)
        controls['colors'] = (controls.pop('color'),)
    controls['method'] = contract['method']
    settings = RegionProbeSettings(**controls)
    current = region_probe_contract(settings, snapshot_sha256=contract['snapshot_sha256'],
                                    provenance=contract['field_provenance'], background=background)
    for key in ('background_run_id', 'background_sha256'):
        if current[key] != contract[key]:
            raise ValueError(f'The probe was computed against a different context: {key}')
    return settings, result, metadata


def load_saved_region_panels(backgrounds, batch_directories):
    """Assemble one panel per method from checksum-verified, cycle-aligned batches."""
    panels = {}
    for method, background in backgrounds.items():
        batches = []
        for directory in batch_directories[method]:
            settings, result, _ = load_region_probe_batch(directory, background=background)
            batches.append((settings, result))
        panels[method] = region_probe_panel(background, batches)
    return panels
