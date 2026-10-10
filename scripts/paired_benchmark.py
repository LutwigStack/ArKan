#!/usr/bin/env python3
"""Freeze a paired plan, emit its schedule, and analyze retained kernel JSONL.

This stdlib tool never launches benchmarks or qualifies application correctness.
"""
import hashlib
import json
import random
import re
import argparse
import os
import tempfile
from fractions import Fraction as F
from pathlib import Path

CASES = ('one-layer', 'two-layer', 'wide-hidden', 'deep-hidden')
LO, HI = F(97, 100)**4, F(103, 100)**4
METHOD = dict(bundles=32, bootstrap=10000, seed=20261010, block_length=4,
              calibration_min_ns=100000000, calibration_max_ns=250000000,
              max_iterations=1 << 24, targets=[2, 3],
              ci_indices=[249, 9750], threshold_percent=3)
KERNEL = {'case', 'iterations', 'warmup', 'elapsed_nanos', 'output_bits',
          'affinity_before', 'affinity_after'}

def need(ok, message='invalid input'):
    if not ok:
        raise ValueError(message)

def integer(x, low=0, high=(1 << 128)-1):
    need(type(x) is int and low <= x <= high, 'integer/type/range')
    return x

def keys(x, wanted):
    need(type(x) is dict and set(x) == set(wanted), 'object fields')

def digest(x):
    need(type(x) is str and re.fullmatch('[0-9a-f]{64}', x), 'digest')
    return x

def pairs(rows):
    result = {}
    for key, value in rows:
        need(key not in result, 'duplicate JSON key')
        result[key] = value
    return result

def loads(raw):
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: need(False, 'nonfinite JSON'))

def rich(row, path=None):
    keys(row, ('path', 'bytes', 'sha256'))
    p = Path(row['path'])
    need(type(row['path']) is str and p.is_absolute() and str(p) == row['path'])
    need('..' not in p.parts and (path is None or p == path), 'path')
    integer(row['bytes'], 1, 1 << 30)
    digest(row['sha256'])
    return p

def schedule():
    for b in range(32):
        order = list(range(4))
        order = order[b % 4:] + order[:b % 4]
        if (b // 4) % 2:
            order.reverse()
        for c in order:
            for slot, role in enumerate('ABBA' if (b+c) % 2 == 0 else 'BAAB'):
                yield b, c, slot, role

def kernel(row, c, n, w, cases=CASES, cpu=2):
    need(row['case'] == cases[c], 'case')
    need(integer(row['iterations'], 1, 1 << 24) == n)
    need(integer(row['warmup'], 0, 1 << 24) == w)
    integer(row['elapsed_nanos'], 1)
    bits = row['output_bits']
    need(type(bits) is list and len(bits) == 4)
    for word in bits:
        integer(word, 0, (1 << 32)-1)
    for k in ('affinity_before', 'affinity_after'):
        need(type(row[k]) is list and len(row[k]) == 1)
        need(integer(row[k][0]) == cpu, 'child affinity')
    need(integer(row['exit_code']) == 0 and row['reaped'] is True, 'child terminal status')

def parse_campaign(raw, campaign, authority_sha256, plan_sha256, binaries, cases=CASES, cpu=2):
    """Pure exact JSONL parser; output bits are the lossless checksum."""
    need(campaign in ('null', 'paired'))
    need(type(raw) is bytes and len(raw) <= 1048576 and raw.endswith(b'\n'), 'raw size/termination')
    rows = [loads(line) for line in raw.splitlines()]
    need(len(rows) == 514, '514 complete records required')
    head, end = rows[0], rows[-1]
    keys(head, ('type', 'schema', 'campaign', 'authority_sha256', 'plan_sha256', 'binaries', 'calibration'))
    need(head['type'] == 'header' and integer(head['schema']) == 1 and head['campaign'] == campaign)
    need(digest(head['authority_sha256']) == authority_sha256 and digest(head['plan_sha256']) == plan_sha256)
    keys(binaries, ('A', 'B'))
    for item in binaries.values():
        rich(item)
    expected = {'A': binaries['A'], 'B': binaries['A'] if campaign == 'null' else binaries['B']}
    need(head['binaries'] == expected, 'logical binary identities')
    # Validate copied header descriptors too: Python equality must not accept bool bytes.
    keys(head['binaries'], ('A', 'B'))
    for item in head['binaries'].values():
        rich(item)
    calibration = head['calibration']
    need(type(calibration) is list and 4 <= len(calibration) <= 100, 'calibration count')
    selected, outputs, offset = [], [], 0
    for c in range(4):
        n = 1
        while True:
            need(offset < len(calibration), 'calibration prefix')
            r = calibration[offset]
            keys(r, KERNEL | {'binary_sha256', 'exit_code', 'reaped'})
            kernel(r, c, n, 8, cases, cpu)
            need(digest(r['binary_sha256']) == binaries['A']['sha256'], 'A-only calibration')
            if n == 1:
                bits = r['output_bits']
            need(r['output_bits'] == bits, 'calibration output changed')
            offset += 1
            if r['elapsed_nanos'] >= 100000000:
                need(r['elapsed_nanos'] <= 250000000, 'calibration overshoot')
                selected.append(n)
                outputs.append(bits)
                break
            n *= 2
            need(n <= 1 << 24, 'calibration exhausted')
    need(offset == len(calibration), 'extra calibration probes')
    for row, (b, c, slot, role) in zip(rows[1:-1], schedule()):
        keys(row, KERNEL | {'type', 'campaign', 'bundle', 'case_id', 'slot', 'role', 'binary_sha256', 'exit_code', 'reaped'})
        need(row['type'] == 'chunk' and row['campaign'] == campaign)
        need(integer(row['bundle']) == b and integer(row['case_id']) == c and integer(row['slot']) == slot and row['role'] == role, 'schedule')
        n = selected[c]
        kernel(row, c, n, max(8, (n+9)//10), cases, cpu)
        need(digest(row['binary_sha256']) == expected[role]['sha256'], 'chunk binary')
        need(row['output_bits'] == outputs[c], 'full output bits differ')
    keys(end, ('type', 'campaign', 'rows', 'status'))
    need(end['type'] == 'terminal' and end['campaign'] == campaign and integer(end['rows']) == 512 and end['status'] == 'COMPLETE')
    return dict(campaign=campaign, cases=list(cases), cpu=cpu, calibration=calibration, rows=rows[1:-1])

def middle(values):
    values = sorted(values)
    i = len(values)//2
    return values[i-1]*values[i]

def summarize(campaign):
    """Exact power-four estimates. Shared circular indices across cases."""
    by_case = [[] for _ in CASES]
    for b in range(32):
        for c in range(4):
            rows = [r for r in campaign['rows'] if r['bundle'] == b and r['case_id'] == c]
            a = [r['elapsed_nanos'] for r in rows if r['role'] == 'A']
            z = [r['elapsed_nanos'] for r in rows if r['role'] == 'B']
            by_case[c].append(F(z[0]*z[1], a[0]*a[1]))
    rng = random.Random(20261010)
    ranks, ordered = [], []
    for values in by_case:
        order = sorted(range(32), key=values.__getitem__)
        rank = [0]*32
        for k, b in enumerate(order):
            rank[b] = k
        ranks.append(rank)
        ordered.append([values[b] for b in order])
    boot = [[] for _ in CASES]
    for _ in range(10000):
        indices = [(start+d) % 32 for start in [rng.randrange(32) for _ in range(8)] for d in range(4)]
        for c in range(4):
            rr = sorted(ranks[c][b] for b in indices)
            boot[c].append(ordered[c][rr[15]]*ordered[c][rr[16]])
    answer = []
    for c, x in enumerate(by_case):
        ci = sorted(boot[c])
        halves = [middle(x[:16]), middle(x[16:])]
        orders = [middle([x[b] for b in range(32) if (b+c) % 2 == parity]) for parity in (0, 1)]
        low, high = ci[249], ci[9750]
        answer.append(dict(case=campaign['cases'][c], point=middle(x), lower=low, upper=high,
                           halves=halves, orders=orders, width=high/low,
                           half_drift=max(halves)/min(halves), order_effect=max(orders)/min(orders)))
    return answer

def noise(summary):
    return all(r['width'] <= HI and r['half_drift'] <= HI and r['order_effect'] <= HI for r in summary)

def decide(summary, null=False, prerequisites=True):
    valid = prerequisites and noise(summary)
    equivalent = all(r['lower'] >= LO and r['upper'] <= HI for r in (summary if null else summary[:2]))
    if null:
        equivalent = equivalent and all(r['lower'] <= 1 <= r['upper'] for r in summary)
    slow = [i for i, r in enumerate(summary) if r['lower'] > HI and all(v > HI for v in r['halves'])]
    nonreg = all(r['point'] <= HI and r['upper'] <= HI for r in summary)
    eligible = [] if null else [i for i in (2, 3) if summary[i]['point'] <= LO and summary[i]['upper'] < 1 and all(v < 1 for v in summary[i]['halves'])]
    status = ('INVALID_STOP' if not valid else
              ('PASS' if equivalent else 'INVALID_STOP') if null else
              'REJECT' if slow else 'INVALID_STOP' if not equivalent else
              'DEFER' if not nonreg or not eligible else 'PASS')
    return dict(status=status, structural_valid=prerequisites, noise_valid=noise(summary),
                control_equivalent=equivalent, nonreg=nonreg, eligible=eligible if status == 'PASS' else [],
                verified_slow=slow if valid and not null else [])

def cpu_guards(records, authority_sha256, selected_cpu=2):
    need(type(records) is list and len(records) in (4, 6), 'guard count')
    anchor, fixed, previous_periods = None, None, 0
    for i, g in enumerate(records):
        need(type(g) is dict and integer(g['schema']) == 1)
        need(g['status'] == ('BEFORE_NATIVE' if i % 2 == 0 else 'QUALIFIED'))
        need(digest(g['AUTHORITY_TIMING_sha256']) == authority_sha256)
        cpu = g['CPU']
        keys(cpu, ('quota_us', 'period_us', 'cgroup_path', 'cpuset_effective',
                   'affinity', 'nr_periods', 'nr_throttled', 'throttled_usec'))
        need(integer(cpu['quota_us'], -1) != 0)
        integer(cpu['period_us'], 1)
        for key in ('cgroup_path', 'cpuset_effective'):
            need(type(cpu[key]) is str and cpu[key], 'CPU identity')
        need(type(cpu['affinity']) is list and cpu['affinity'])
        mask = [integer(x) for x in cpu['affinity']]
        need(mask == sorted(set(mask)) and selected_cpu in mask, 'parent affinity')
        for k in ('nr_periods', 'nr_throttled', 'throttled_usec'):
            integer(cpu[k])
        need(cpu['nr_periods'] >= previous_periods, 'CPU counter reversed')
        previous_periods = cpu['nr_periods']
        identity = {k: cpu[k] for k in ('quota_us', 'period_us', 'cgroup_path', 'cpuset_effective', 'affinity')}
        now = (cpu['nr_throttled'], cpu['throttled_usec'])
        if anchor is None:
            anchor, fixed = now, identity
        need(now == anchor and identity == fixed, 'CPU/throttle changed')


def serial(value):
    if isinstance(value, F):
        return [value.numerator, value.denominator]
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, list):
        return [serial(v) for v in value]
    return value

def null_report(campaign, guards, authority_sha256):
    """Pure pre-B gate; caller writes complete report create-only."""
    need(campaign['campaign'] == 'null' and len(guards) == 4)
    cpu_guards(guards, authority_sha256, campaign['cpu'])
    summary = summarize(campaign)
    return serial(dict(schema=1, campaign='null', candidate_chunks=0,
                       scale='B/A ratio power4; fractions [numerator,denominator]',
                       **decide(summary, null=True), cases=summary))

def validate_plan(plan):
    keys(plan, ('schema', 'method', 'cases', 'binaries', 'cpu'))
    need(integer(plan['schema']) == 1, 'schema')
    need(json.dumps(plan['method'], sort_keys=True) == json.dumps(METHOD, sort_keys=True), 'frozen typed method')
    cases = plan['cases']
    need(type(cases) is list and len(cases) == 4)
    need(all(type(c) is str and re.fullmatch('[A-Za-z0-9_-]{1,80}', c) for c in cases), 'four case names')
    need(len(set(cases)) == 4, 'unique cases')
    integer(plan['cpu'], 0, 65535)
    keys(plan['binaries'], ('A', 'B'))
    for row in plan['binaries'].values():
        rich(row)
    return plan


def record(path):
    p = Path(path).resolve(strict=True)
    raw = p.read_bytes()
    return dict(path=str(p), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def analyze(plan_raw, null_raw, paired_raw, guards):
    plan = validate_plan(loads(plan_raw))
    head = loads(null_raw.splitlines()[0])
    auth, plan_sha = digest(head['authority_sha256']), hashlib.sha256(plan_raw).hexdigest()
    count = 4 if paired_raw is None else 6
    need(type(guards) is list and len(guards) == count, 'phase guard count')
    cpu_guards(guards, auth, plan['cpu'])
    args = (auth, plan_sha, plan['binaries'], plan['cases'], plan['cpu'])
    null = parse_campaign(null_raw, 'null', *args)
    nr = null_report(null, guards[:4], auth)
    if paired_raw is None:
        return dict(schema=1, status='INVALID_STOP', reason='null_gate' if nr['status'] != 'PASS' else 'candidate_not_run', candidate_chunks=0, eligible=[], null=nr)
    need(nr['status'] == 'PASS', 'B measured without null PASS')
    paired = parse_campaign(paired_raw, 'paired', *args)
    need(paired['calibration'] == null['calibration'], 'calibration changed')
    summary = summarize(paired)
    return serial(dict(schema=1, candidate_chunks=512, **decide(summary), cases=summary, null=nr,
                       scale='B/A ratio power4; fractions [numerator,denominator]',
                       qualification='Numeric result only; correctness, native receipts, resources and source/profile acceptance remain separate'))


def publish(path, value):
    """Exclusive atomic publication; an interrupted attempt may leave an owned temp."""
    output = Path(path)
    raw = (json.dumps(value, sort_keys=True, separators=(',', ':'))+'\n').encode()
    need(len(raw) <= 65536, 'report size')
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix='.paired-', delete=False) as stream:
            tmp = Path(stream.name)
            stream.write(raw)
        os.link(tmp, output)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    p = actions.add_parser('plan', help='freeze cases/binary identities before data')
    p.add_argument('--base', required=True)
    p.add_argument('--candidate', required=True)
    p.add_argument('--cases', nargs=4, required=True, metavar='CASE')
    p.add_argument('--cpu', type=int, default=2)
    p.add_argument('--output', required=True)
    p = actions.add_parser('schedule', help='print all 512 prospective chunks')
    p.add_argument('--plan', required=True)
    p = actions.add_parser('analyze', help='analyze retained complete null and optional paired streams')
    p.add_argument('--plan', required=True)
    p.add_argument('--null', required=True)
    p.add_argument('--paired')
    p.add_argument('--guards', required=True, help='JSON array: calibration/null/(paired) before+after controls')
    p.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    if args.action == 'plan':
        plan = validate_plan(dict(schema=1, method=METHOD, cases=args.cases, cpu=args.cpu,
                                  binaries=dict(A=record(args.base), B=record(args.candidate))))
        publish(args.output, plan)
        return 0
    plan_raw = Path(args.plan).read_bytes()
    need(len(plan_raw) <= 65536, 'plan size')
    plan = validate_plan(loads(plan_raw))
    if args.action == 'schedule':
        for b, c, s, role in schedule():
            print(json.dumps(dict(bundle=b, case_id=c, case=plan['cases'][c], slot=s, role=role), separators=(',', ':')))
        return 0
    try:
        result = analyze(plan_raw, Path(args.null).read_bytes(),
                         Path(args.paired).read_bytes() if args.paired else None,
                         loads(Path(args.guards).read_bytes()))
    except (ValueError, KeyError, TypeError, IndexError, OSError) as error:
        result = dict(schema=1, status='INVALID_STOP', reason=str(error), eligible=[])
    publish(args.output, result)
    print(json.dumps(dict(status=result['status'], eligible=result['eligible'])))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
