import contextlib
import copy
import hashlib
import io
import json
import random
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import paired_benchmark as p


class PairedTests(unittest.TestCase):
    def setUp(self):
        self.binaries = {r: dict(path='/tmp/'+r, bytes=1, sha256=r.lower()*64) for r in 'AB'}
        self.plan = dict(schema=1, method=p.METHOD, cases=list(p.CASES),
                         binaries=self.binaries, cpu=2)
        self.plan_raw = (json.dumps(self.plan)+'\n').encode()
        self.plan_sha = hashlib.sha256(self.plan_raw).hexdigest()
        cpu = dict(quota_us=400000, period_us=100000, cgroup_path='/',
                   cpuset_effective='0-4', affinity=list(range(5)),
                   nr_periods=100, nr_throttled=1, throttled_usec=10)
        self.guards = [dict(schema=1, status='BEFORE_NATIVE' if i % 2 == 0 else 'QUALIFIED',
                            AUTHORITY_TIMING_sha256='c'*64, CPU=copy.deepcopy(cpu)) for i in range(6)]

    def records(self, phase, targets=96000000):
        cal = [dict(case=case, iterations=1, warmup=8, elapsed_nanos=100000000,
                    output_bits=[1, 2, 3, 4], affinity_before=[2], affinity_after=[2],
                    binary_sha256='a'*64, exit_code=0, reaped=True) for case in p.CASES]
        binary = self.binaries if phase == 'paired' else dict(A=self.binaries['A'], B=self.binaries['A'])
        rows = [dict(type='header', schema=1, campaign=phase, authority_sha256='c'*64,
                     plan_sha256=self.plan_sha, binaries=binary, calibration=cal)]
        for b, c, slot, role in p.schedule():
            row = dict(cal[c], type='chunk', campaign=phase, bundle=b, case_id=c, slot=slot, role=role,
                       binary_sha256=binary[role]['sha256'])
            if phase == 'paired' and role == 'B' and c >= 2:
                row['elapsed_nanos'] = targets
            rows.append(row)
        rows.append(dict(type='terminal', campaign=phase, rows=512, status='COMPLETE'))
        return rows

    @staticmethod
    def raw(rows):
        return ('\n'.join(json.dumps(row) for row in rows)+'\n').encode()

    def parse(self, rows, phase):
        return p.parse_campaign(self.raw(rows), phase, 'c'*64, self.plan_sha, self.binaries)

    def test_schedule_balances_positions_and_has_fixed_order(self):
        rows = list(p.schedule())
        self.assertEqual(len(rows), 512)
        self.assertEqual([c for b, c, s, r in rows if b == 0 and s == 0], [0, 1, 2, 3])
        self.assertEqual([c for b, c, s, r in rows if b == 4 and s == 0], [3, 2, 1, 0])
        for c in range(4):
            starts = [r for b, case, s, r in rows if case == c and s == 0]
            self.assertEqual(starts.count('A'), 16)
            for slot in range(4):
                self.assertEqual(sum(r == 'A' for b, case, s, r in rows if case == c and s == slot), 16)

    def test_complete_streams_and_null_before_candidate(self):
        null, paired = self.raw(self.records('null')), self.raw(self.records('paired'))
        result = p.analyze(self.plan_raw, null, paired, self.guards)
        self.assertEqual((result['status'], result['eligible']), ('PASS', [2, 3]))
        stopped = p.analyze(self.plan_raw, null, None, self.guards[:4])
        self.assertEqual((stopped['candidate_chunks'], stopped['reason']), (0, 'candidate_not_run'))
        bad = self.records('null')
        for row in bad[1:-1]:
            if row['role'] == 'B':
                row['elapsed_nanos'] = 120000000
        with self.assertRaises(ValueError):
            p.analyze(self.plan_raw, self.raw(bad), paired, self.guards)

        # Portable interface: predeclared labels, CPU 7 and an unlimited quota.
        plan = copy.deepcopy(self.plan)
        plan.update(cases=['control-small', 'control-large', 'target-small', 'target-large'], cpu=7)
        plan_raw = (json.dumps(plan)+'\n').encode()
        plan_sha = hashlib.sha256(plan_raw).hexdigest()
        campaigns = []
        for phase in ('null', 'paired'):
            rows = self.records(phase)
            rows[0]['plan_sha256'] = plan_sha
            for row in rows[0]['calibration'] + rows[1:-1]:
                row['case'] = plan['cases'][p.CASES.index(row['case'])]
                row['affinity_before'] = row['affinity_after'] = [7]
            campaigns.append(self.raw(rows))
        guards = copy.deepcopy(self.guards)
        for guard in guards:
            guard['CPU'].update(quota_us=-1, cpuset_effective='6-9', affinity=[6, 7, 8, 9])
        portable = p.analyze(plan_raw, *campaigns, guards)
        self.assertEqual((portable['status'], portable['eligible']), ('PASS', [2, 3]))
        self.assertEqual([row['case'] for row in portable['cases']], plan['cases'])
        self.assertEqual([row['case'] for row in portable['null']['cases']], plan['cases'])

    def test_strict_protocol_rejects_type_identity_shape_and_order_mutants(self):
        for key, value in [('bundles', 32.0), ('bootstrap', True), ('targets', [2.0, 3])]:
            plan = copy.deepcopy(self.plan); plan['method'][key] = value
            with self.assertRaises(ValueError): p.validate_plan(plan)
        for key, value in [('elapsed_nanos', False), ('elapsed_nanos', 0), ('iterations', True),
                           ('reaped', 1), ('exit_code', False), ('warmup', 9), ('bundle', 1),
                           ('output_bits', [True, 2, 3, 4]), ('output_bits', [1, 2, 3]),
                           ('affinity_after', [0]), ('unknown', 0), ('binary_sha256', 'b'*64)]:
            with self.subTest(key=key, value=value):
                rows = self.records('null'); rows[1][key] = value
                with self.assertRaises(ValueError): self.parse(rows, 'null')
        for change in ('missing', 'extra', 'calibration', 'binary'):
            rows = self.records('null')
            if change == 'missing': rows.pop(3)
            elif change == 'extra': rows.insert(3, rows[3])
            elif change == 'calibration': rows[0]['calibration'].append(rows[0]['calibration'][0])
            else: rows[0]['binaries']['B'] = self.binaries['B']
            with self.assertRaises(ValueError): self.parse(rows, 'null')
        for text in ('{"a":1,"a":2}', '{"a":NaN}'):
            with self.assertRaises(ValueError): p.loads(text)

    def test_cpu_guards_use_frozen_identity_and_unthrottled_anchor(self):
        p.cpu_guards(self.guards, 'c'*64)
        for key, value in [('quota_us', True), ('quota_us', 1000000), ('affinity', [2]),
                           ('nr_periods', 99), ('nr_throttled', 2), ('throttled_usec', 11)]:
            guards = copy.deepcopy(self.guards); guards[-1]['CPU'][key] = value
            with self.assertRaises(ValueError): p.cpu_guards(guards, 'c'*64)

    def test_shared_moving_block_bootstrap_matches_direct_fraction_oracle(self):
        rows = self.records('paired')
        for row in rows[1:-1]:
            if row['role'] == 'B':
                row['elapsed_nanos'] = 99000000 + (row['bundle'] % 5)*10000
        summary = p.summarize(self.parse(rows, 'paired'))
        x = [Fraction((99000000+(b % 5)*10000)**2, 100000000**2) for b in range(32)]
        rng = random.Random(20261010); samples = []
        for _ in range(10000):
            indices = [(start+d) % 32 for start in [rng.randrange(32) for _ in range(8)] for d in range(4)]
            selected = sorted(x[b] for b in indices)
            samples.append(selected[15]*selected[16])
        samples.sort()
        for row in summary:
            self.assertEqual((row['lower'], row['upper']), (samples[249], samples[9750]))
        self.assertEqual(summary[0]['point'], sorted(x)[15]*sorted(x)[16])

    def test_exact_thresholds_null_and_slowdown_precedence(self):
        summary = p.summarize(self.parse(self.records('null'), 'null'))
        self.assertEqual(p.decide(summary, null=True)['status'], 'PASS')
        summary[2].update(point=p.LO, upper=Fraction(999, 1000), halves=[Fraction(999, 1000)]*2)
        self.assertEqual(p.decide(summary)['status'], 'PASS')
        summary[2]['point'] += Fraction(1, 10**12)
        self.assertEqual(p.decide(summary)['status'], 'DEFER')
        summary[0].update(point=p.HI+1, lower=p.HI+1, upper=p.HI+1, halves=[p.HI+1]*2)
        self.assertEqual(p.decide(summary)['status'], 'REJECT')
        summary[0]['width'] = p.HI+Fraction(1, 10**12)
        self.assertEqual(p.decide(summary)['status'], 'INVALID_STOP')

    def test_cli_and_atomic_exclusive_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); a = root/'A'; b = root/'B'; a.write_bytes(b'A'); b.write_bytes(b'B')
            out = root/'plan.json'
            self.assertEqual(p.main(['plan', '--base', str(a), '--candidate', str(b),
                                     '--cases', *p.CASES, '--output', str(out)]), 0)
            before = out.read_bytes()
            with self.assertRaises(FileExistsError): p.publish(out, {'replacement': True})
            self.assertEqual(out.read_bytes(), before)
            with patch.object(p.os, 'link', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt): p.publish(root/'interrupted.json', {})
            self.assertFalse((root/'interrupted.json').exists())
            self.assertFalse(list(root.glob('.paired-*')))
            with contextlib.redirect_stdout(io.StringIO()) as stream:
                self.assertEqual(p.main(['schedule', '--plan', str(out)]), 0)
            self.assertEqual(len(stream.getvalue().splitlines()), 512)


if __name__ == '__main__':
    unittest.main()
