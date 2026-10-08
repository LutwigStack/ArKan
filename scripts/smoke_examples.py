#!/usr/bin/env python3
"""Run already-built CPU examples with deadlines and check their observable results."""
import argparse
import math
from pathlib import Path
import re
import subprocess


def run_example(directory, name):
    result = subprocess.run([str(directory / name)], check=True, capture_output=True, text=True, timeout=30)
    print(result.stdout, end='')
    if re.search(r'(?i)(?<!\w)(nan|[+-]?inf)(?!\w)', result.stdout):
        raise AssertionError(f'{name}: non-finite result')
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('target/debug/examples'))
    parser.add_argument('--serde', action='store_true')
    args = parser.parse_args()
    directory = args.directory.resolve()
    basic = run_example(directory, 'basic')
    output = re.search(r'Output: \[0\]=([^,]+), \[1\]=([^,]+),', basic)
    if output is None or not all(math.isfinite(float(value)) for value in output.groups()):
        raise AssertionError('basic: missing finite single-sample inference output')
    batch = re.search(r'Output sample \[0\]: (\S+)', basic)
    if batch is None or not math.isfinite(float(batch[1])):
        raise AssertionError('basic: missing finite batch inference output')
    baked = run_example(directory, 'baked_inference')
    if 'Mean abs error across 5 samples:' not in baked or 'Done.' not in baked:
        raise AssertionError('baked_inference: inference comparison did not complete')
    if args.serde and 'Round-trip output identical: true' not in baked:
        raise AssertionError('baked_inference: serialization round-trip differed')


if __name__ == '__main__':
    main()
