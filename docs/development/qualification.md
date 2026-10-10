# Guarded validation command receipts

Proposed repository paths: `scripts/check_command.py`, `scripts/test_check_command.py`,
`scripts/test_cancellation.py`, and `docs/development/qualification.md`.
This utility requires POSIX and Python 3.11+
and adds no dependencies. Existing Criterion benches and example smoke checks remain
the application workflows; this records an already selected foreground command.

Public command:

```text
python3 scripts/check_command.py --manifest /path/to/check.json --output /path/to/new-evidence
python3 -B -m unittest discover -s scripts -p test_check_command.py -v
python3 -B scripts/test_cancellation.py -v
```

Manifest schema 1 has `cwd`, `argv` (a string vector with an absolute executable
path), positive finite `timeout_seconds`, and nonempty `inputs` (objects with
`path` and SHA256 `sha256`). Relative `cwd` and input paths resolve against the
manifest's parent directory. Record the actual toolchain, library/caller build,
lock, features, linked dependencies, source snapshot, executable, test source and
prior review report identities in the input list when the command depends on them.
It is the manifest author's responsibility to enumerate the complete inputs.
The utility never discovers a Cargo dependency closure or supplies independent
qualification, compilation, correctness, ownership or acceptance review.

The output directory must be absent. It contains an exact `manifest.json` copy,
binary `stdout.log`/`stderr.log` when launch was attempted, and `receipt.json` with
argv/cwd, manifest SHA, wall timestamps, elapsed seconds, PID, child exit status,
outcome and SHA/byte identities of retained payloads. Launched commands also record
`group_cleanup`: TERM/KILL delivery, direct-child reaping, process-group presence
after cleanup, and cleanup elapsed seconds. Outcomes distinguish PASS,
FAIL, REFUSED, TIMEOUT, INTERRUPTED, INPUT_CHANGED and ERROR. A zero child exit
is PASS only when pre/post immutable input and manifest checks also pass.
Duplicate JSON keys and nonfinite deadlines refuse launch. Output is written
directly to disk, preserving partial output without unbounded capture in memory.
The utility returns zero only for PASS. It never retries or overwrites its output.
Passing evidence requires both a complete PASS receipt and normal wrapper exit
zero; a receipt by itself never certifies success.
An existing evidence directory is a consumed attempt; choosing another directory
does not authorize a repeat of an application experiment.

Children must join their descendants and must not detach process groups. Handled
INT/TERM signals only record a pending interruption, so they cannot abandon an
unregistered child inside Popen. After handle registration the wait checks the
pending signal at most every 100 ms. Every postlaunch exit runs cleanup before
restoring the caller's handlers. Cleanup sends TERM to the whole owned process
group, allows a one-second grace when TERM was delivered, and sends KILL even
when the leader already exited. It reaps the direct child and checks group
disappearance within a shared six-second cleanup budget. A successful direct
child that leaves its group present produces ERROR. Failure to confirm reaping
or group disappearance also produces ERROR, preserving the observed cleanup
fields rather than implying that cleanup completed.

Deferred INT/TERM handling remains active through retained-output hashing.
Finalization restores both original handlers in a finally, including on hashing
failure, then accounts for pending interruption before publishing the receipt.
This restoration is the signal handoff: subsequent signals use the caller's
original behavior. With the normal CLI dispositions, SIGTERM during receipt
publication terminates nonzero and can leave an incomplete receipt; it does not
become handled success. Hashing/write failures also remain nonpassing. Require
the receipt and wrapper exit together even if a receipt was already written.

The utility cannot waitpid orphan descendants. A terminated orphan zombie may
keep its process group present until its adoptive parent reaps it; this prevents
a confirmed cleanup receipt within the budget and is reported as ERROR. Group
absence is a bounded observation after termination, not a claim about escaped
sessions or universal OS process ownership. Cleanup time and pre/post hashing
count in receipt elapsed time, but extend beyond the command deadline. OS launch
and uninterruptible kernel I/O are not bounded by the polling interval. This is
a validation utility, not the full scientific campaign supervisor:
that driver creates independent caller sessions and needs separate supervision.
File hashing before and after does not prevent a hostile or transient change
between guards. A SIGKILL of the utility or machine failure can leave partial files
without a final receipt; that is incomplete evidence, never a passing check.

The focused suite uses tiny Python fixture children only. It covers nonzero child
status/partial stdout and stderr, ordinary and TERM-resistant timeout/reaping,
immutable mismatch before side effects, postlaunch mutation, failed launch,
duplicate manifest keys and refusal to overwrite or repeat an existing attempt.
Four additional real Linux fixtures exercise a TERM-resistant same-group descendant
on timeout and waiting interruption, and actual SIGTERM immediately before Popen
returns and after handle registration. Those fixtures temporarily become their
own orphan subreaper, reap their own children on RED and GREEN, then restore that
per-process setting. The production utility does not adopt orphan descendants.
The focused suite does not execute ArKan or prove Rust/application correctness.
Three finalization fixtures additionally cover real SIGTERM during successful
stdout hashing and receipt publication, and restoration of both original handlers
when stdout hashing raises an error.

## Prepare pinned A/B sources

`prepare_ab.py` stages an already selected recipe without launching commands:

```text
python3 scripts/prepare_ab.py --recipe /absolute/path/recipe.json --output /absolute/path/new-prepared
python3 -B -m unittest discover -s scripts -p test_prepare_ab.py -v
```

A schema-1 recipe lists source files with byte counts and SHA256 hashes, optional
role-specific postimages, auxiliary files, reusable argv vectors, profiles, cases
and a method descriptor. The script copies the inventory into `source-A/` and
`source-B/`, applies only the pinned postimages, and writes expanded commands to
`plan.json`. It does not parse patches or rewrite command paths. Supply the intended
fresh snapshot, output and dependency paths in the recipe.

For example, this minimal recipe stages one unchanged file for both roles and
serializes a command without running it. Replace the size and hash with the actual
file descriptor; `/absolute/source` must exist and the output must be absent.

```json
{
  "schema": 1,
  "source_root": "/absolute/source",
  "inventory": [{"path": "src/lib.rs", "bytes": 25, "sha256": "<64 lowercase hex digits>"}],
  "overlays": [],
  "files": [],
  "role_files": [],
  "vectors": {"compiler": ["/absolute/rustc"]},
  "profiles": {},
  "commands": [{"id": "A-list", "profile": null,
    "argv_parts": [{"vector": "compiler"}, ["--version"]],
    "cwd": "/absolute/source", "timeout_seconds": 15}],
  "cases": ["selected-case"],
  "method": null
}
```

An overlay names its roles and inventory path, the exact base `preimage_sha256`,
and a `postimage` descriptor with absolute `path`, `bytes` and `sha256`.
A selected profile declares exact feature names, required codegen values and
direct extern names. Profiled rustc vectors use canonical separate `-C`, `--cfg`
and `--extern` arguments. The helper rejects conflicting protected options,
duplicate features or externs, unsafe paths, overlapping roots and output names,
changed inputs, unknown fields and existing output namespaces.

Run preparation under a trusted parent directory with no competing writer.
Source, input, cwd and output paths cannot have symlink ancestors; the serialized
executable may be a symlink but must be a canonical absolute lexical path.
Executable identity and dependency closure remain the recorder's responsibility.
After checking every input and output, the helper atomically publishes the complete
`PREPARED.json` marker with a create-only link. Initial validation can fail before
creating a namespace; replay refusal preserves an existing namespace and marker.
After preparation creates its own namespace, failure or handled INT/TERM can leave
partial files without a passing marker. SIGKILL or machine failure can leave partial
files; inspect the complete marker and its pinned plan before use.

`PREPARED` certifies staging integrity only. It does not certify compilation,
application correctness, measurements or adoption. Pass selected expanded commands
and their complete inputs to `check_command.py` for separate execution receipts.
The self-contained seven-test suite covers binary snapshots, profiles, path and
preimage rejection, replay refusal, and failure/cancellation before and after marker
publication. A separate workspace integration check reproduced a pinned recipe's
32 argv vectors, 320 role files and four auxiliary files byte for byte; it launched
no application command and did not adopt that recipe's candidate.

## Reuse a paired timing method

`paired_benchmark.py` freezes a small pre-data plan, emits the exact prospective
schedule, and analyzes retained measurements without launching a benchmark:

```text
python3 scripts/paired_benchmark.py plan --base /absolute/A-kernel --candidate /absolute/B-kernel --cases control0 control1 target0 target1 --cpu 2 --output /new/plan.json
python3 scripts/paired_benchmark.py schedule --plan /new/plan.json
python3 scripts/paired_benchmark.py analyze --plan /new/plan.json --null /run/null.jsonl --paired /run/paired.jsonl --guards /run/guards.json --output /new/analysis.json
python3 -B -m unittest discover -s scripts -p test_paired_benchmark.py -v
```

The first two cases are controls and the last two are the only eligible targets.
The plan pins both program byte identities, four unique case names, selected CPU
and the fixed method. Plan and result publication is atomic and create-only;
require a complete result and normal command exit zero together for numeric PASS.
Interrupted attempts can leave partial files and never authorize a retry.

Collectors may import `schedule`, `parse_campaign` and `null_report` from this
module. The collector must run A-only doubling calibration (eight untimed calls,
first batch reaching 100 ms, at most 250 ms, N at most 2^24), then freeze
W=max(8,ceil(N/10)). Each campaign uses 32 four-case bundles and four ABBA/BAAB
chunks per case, balanced 16/16. Complete the same-binary A/A null campaign and
require its `null_report` PASS **before launching any candidate chunk**. The
analyzer checks that prerequisite but cannot establish launch chronology by itself.
Do not rerun, change thresholds or recalibrate with B after seeing a result.

Each raw JSONL stream has a header, 512 ordered chunks and a terminal record.
Headers contain `type`, `schema`, `campaign`, `authority_sha256`, `plan_sha256`,
`binaries` and the complete ordered `calibration` list. Authority is the collector's
pre-data contract hash, not an invented external approval. Calibration records
and chunks contain the kernel's `case`, `iterations`, `warmup`, `elapsed_nanos`,
four raw-u32 `output_bits`, and actual `affinity_before`/`affinity_after`, plus
`binary_sha256`, `exit_code` and `reaped`. Chunks also contain `type`, `campaign`,
`bundle`, `case_id`, `slot` and logical `role`. The terminal has `type`, `campaign`,
`rows` (512) and `status` (`COMPLETE`). The complete output words must remain
identical; booleans, duplicate keys, extra fields and incomplete streams reject.

`guards.json` is the collector's ordered array of calibration/null/paired
before/after controls: six records, or four when the null gate stops execution.
Each has schema 1, alternating `BEFORE_NATIVE`/`QUALIFIED` status,
`AUTHORITY_TIMING_sha256` matching the header, and `CPU` with quota, period,
cgroup, cpuset, parent affinity and cumulative period/throttle counters.
CPU identity must stay fixed, the selected child CPU must belong to the parent
mask, period counts cannot decrease and throttle counters cannot change.
Omit `--paired` for a null-only stop; the report records zero candidate chunks.

Statistics reuse exact rational arithmetic on fourth powers of B/A ratios and
10,000 shared circular moving-block bootstrap draws (length four, seed 20261010).
All gates use the unchanged 3% bounds, including interval width relative to its
lower bound, half drift and order effect. Controls require equivalence; every
case requires nonregression. A target needs at least 3% median gain, upper CI below
one and both half estimates below one. Structural/null/noise failure produces
`INVALID_STOP`; validated material slowdown produces `REJECT` before ordinary
control equivalence; unresolved nonregression or no eligible target produces
`DEFER`. Only the remaining case is numeric `PASS`.

These are approximate per-case moving-bootstrap intervals under local stationarity,
not a simultaneous 95% guarantee. Each predeclared target uses a nominal 97.5%
upper endpoint; a union bound gives nominal 5% family error if those approximate
bootstrap coverage assumptions hold. Native receipts, compiler/features,
source/dependency identities, resource limits and independent correctness/memory
review remain with `check_command.py` and the collector. This utility neither
discovers that closure nor approves a code change. CI exercises synthetic protocol
and exact-statistic oracles; it takes no performance measurements.
