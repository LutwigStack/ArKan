# Releasing

A release is a checklist, not a judgement call.

## Before you tag

Run everything. All of it must pass on a **quiet machine** — see
`tasks/07-baked-simd-int-path/results/measurement-hygiene.md` for why a loaded box
invalidates any number you collect.

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
cargo test --doc
cargo check --lib --no-default-features
cargo build --benches
cargo build --examples
cargo publish --dry-run
```

And the three example crates, which are **not** workspace members and therefore are not
covered by any of the commands above:

```bash
for d in sinusoid mnist game2048; do (cd "examples/$d" && cargo build) || echo "FAILED: $d"; done
```

That gap is real history, not paranoia: `benches/optimizer.rs` and `examples/game2048` both
rotted into a non-compiling state precisely because no root-level command built them.

## Then

1. **CHANGELOG.** Move `## [x.y.z] - Unreleased` to a dated heading. It must include the bad
   news — known limitations, accuracy tails, things measured but not implemented. A release
   note that only lists wins is how the last four releases happened.
2. **Version.** `version` in `Cargo.toml` matches the CHANGELOG heading. Pre-1.0, so a minor
   bump carries breaking changes.
3. **README.** The install snippets name a version that will exist after this publish. If the
   crate is still unpublished, the "not published yet" note must still be accurate — or
   removed, if this release fixes that.
4. **MSRV.** If any dependency floor moved, update `rust-version`, the README MSRV table, and
   the `msrv` CI matrix together. There is no committed `Cargo.lock`, so these drift on their
   own.

```bash
git tag -a vX.Y.Z -m "vX.Y.Z"
git push origin vX.Y.Z
cargo publish
```

## What blocks a release

- Any command above failing.
- A claim in README, `docs/BENCHMARKS.md` or the rustdoc that the code cannot do. The
  README's fences are compiled as doctests (`ReadmeDoctests` in `src/lib.rs`), so API drift
  is caught — but a *performance* or *accuracy* claim is not checked by anything. Verify
  those by re-measuring, not by trusting the previous release's numbers.
- A correctness fix that changes observable behaviour without a CHANGELOG entry.

## What does NOT block a release

Documented limitations. `BakedModel` being slower than f32 at batch=1, its worst-case tail
at >=1 sigma, the absence of out-of-grid diagnostics — these are fine to ship as long as they
are stated where a user will see them before depending on them. Shipping a known limitation
honestly is not the failure mode this project has; overstating is.
