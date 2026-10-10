"""Self-contained integrity tests; no historical experiment paths or children."""
import copy
import io
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest import mock

import prepare_ab as p


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "lib.rs").write_bytes(b"pub fn value() -> u32 { 1 }\n")
        (self.source / "data.bin").write_bytes(bytes([0, 255, 128, 13, 10]))
        self.post = self.root / "candidate.rs"
        self.post.write_bytes(b"pub fn value() -> u32 { 2 }\n")
        self.wrapper = self.root / "wrapper.template"
        self.wrapper.write_bytes(b'#[path = "source-ROLE/lib.rs"] mod app;\n')
        self.method = self.root / "method.json"
        self.method.write_bytes(b'{"sample_size":100,"warm_seconds":5}\n')
        self.output = self.root / "prepared"
        inventory = [dict(p.row(self.source / name, (self.source / name).read_bytes()),
                          path=name) for name in ("lib.rs", "data.bin")]
        self.recipe = dict(
            schema=1, source_root=str(self.source), inventory=inventory,
            overlays=[dict(roles=["B"], path="lib.rs",
                           preimage_sha256=inventory[0]["sha256"],
                           postimage=p.row(self.post, self.post.read_bytes()))],
            files=[],
            role_files=[dict(path="unit-{role}.rs",
                             source=p.row(self.wrapper, self.wrapper.read_bytes()),
                             token="ROLE")],
            vectors=dict(
                compiler=["/qualified/rustc"],
                flags=["-C", "opt-level=3", "-C", "debug-assertions=off",
                       "-C", "overflow-checks=off", "--cfg", 'feature="default"',
                       "--cfg", 'feature="parallel"', "--cfg", 'feature="serde"'],
                dependencies=["--extern", "rayon=/qualified/librayon.rlib"]),
            profiles=dict(release=dict(
                codegen=["opt-level=3", "debug-assertions=off", "overflow-checks=off"],
                features=["default", "parallel", "serde"], externs=["rayon"])),
            commands=[dict(
                id="A-release", profile="release",
                argv_parts=[dict(vector="compiler"), dict(vector="flags"),
                            dict(vector="dependencies"),
                            ["--test", "--cfg", "cfg(docsrs,test)", "unit-A.rs",
                             "-o", "/future/qualified-tests"]],
                cwd=str(self.source), timeout_seconds=120)],
            cases=["whole-order4", "whole-order5"],
            method=p.row(self.method, self.method.read_bytes()))
        self.initial = {path: path.read_bytes() for path in
                        (self.source / "lib.rs", self.source / "data.bin",
                         self.post, self.wrapper, self.method)}

    def unchanged(self):
        for path, data in self.initial.items():
            self.assertEqual(path.read_bytes(), data)

    def refused(self, recipe=None, output=None):
        with self.assertRaises((ValueError, OSError)):
            p.prepare(self.recipe if recipe is None else recipe,
                      self.output if output is None else output)
        self.assertFalse((self.output / "PREPARED.json").exists())
        self.unchanged()

    def test_lossless_binary_roles_profile_and_complete_marker(self):
        real_link = p.os.link
        def exclusive_link(source, destination, **kwargs):
            self.assertFalse(Path(destination).exists())
            marker = p.strict(Path(source).read_bytes())
            self.assertEqual(marker["plan"], p.row(
                self.output / "plan.json", (self.output / "plan.json").read_bytes()))
            for item in json.loads((self.output / "plan.json").read_bytes())["outputs"]:
                self.assertEqual(p.row(Path(item["path"]), Path(item["path"]).read_bytes()),
                                 item)
            return real_link(source, destination, **kwargs)
        with mock.patch.object(p.os, "link", side_effect=exclusive_link) as linked:
            plan = p.prepare(self.recipe, self.output)
        self.assertEqual(linked.call_count, 1)
        self.assertEqual((self.output / "source-A/lib.rs").read_bytes(),
                         self.initial[self.source / "lib.rs"])
        self.assertEqual((self.output / "source-B/lib.rs").read_bytes(),
                         self.initial[self.post])
        for role in "AB":
            self.assertEqual((self.output / f"source-{role}/data.bin").read_bytes(),
                             self.initial[self.source / "data.bin"])
            self.assertEqual((self.output / f"unit-{role}.rs").read_bytes(),
                             self.initial[self.wrapper].replace(b"ROLE", role.encode()))
        self.assertEqual(plan["commands"], [dict(
            id="A-release", profile="release", schema=1,
            argv=["/qualified/rustc", "-C", "opt-level=3", "-C", "debug-assertions=off",
                  "-C", "overflow-checks=off", "--cfg", 'feature="default"',
                  "--cfg", 'feature="parallel"', "--cfg", 'feature="serde"',
                  "--extern", "rayon=/qualified/librayon.rlib", "--test", "--cfg",
                  "cfg(docsrs,test)", "unit-A.rs", "-o", "/future/qualified-tests"],
            cwd=str(self.source), timeout_seconds=120)])
        self.assertEqual(p.strict((self.output / "PREPARED.json").read_bytes()),
                         dict(schema=1, plan=p.row(
                             self.output / "plan.json", (self.output / "plan.json").read_bytes())))
        self.assertFalse((self.output / ".PREPARED.tmp").exists())
        self.unchanged()
        executable = self.root / "executable"
        executable.write_bytes(b"not executed\n")
        alias = self.root / "executable-alias"
        alias.symlink_to(executable)
        recipe = copy.deepcopy(self.recipe)
        recipe["vectors"]["compiler"] = [str(alias)]
        plan = p.prepare(recipe, self.root / "executable-alias-output")
        self.assertEqual(plan["commands"][0]["argv"][0], str(alias))
        with self.assertRaisesRegex(ValueError, "symlink"):
            p.absolute(str(alias))
        self.unchanged()

    def test_replay_preserves_existing_namespace(self):
        self.output.mkdir()
        marker = self.output / "PREPARED.json"
        marker.write_bytes(b"retained")
        with self.assertRaisesRegex(ValueError, "already exists"):
            p.prepare(self.recipe, self.output)
        self.assertEqual(marker.read_bytes(), b"retained")
        self.assertEqual(list(self.output.iterdir()), [marker])
        self.unchanged()

    def test_reject_preimage_duplicate_token_unknown_fields_and_profiles(self):
        mutations = [
            lambda r: r["overlays"][0].update(preimage_sha256="0" * 64),
            lambda r: r["inventory"].append(copy.deepcopy(r["inventory"][0])),
            lambda r: r["overlays"].append(copy.deepcopy(r["overlays"][0])),
            lambda r: r["role_files"][0].update(token="absent"),
            lambda r: r.update(undeclared=True),
            lambda r: r["commands"][0].update(profile="debug"),
            lambda r: r["vectors"].update(dependencies=[]),
            lambda r: r["vectors"]["flags"].remove("debug-assertions=off"),
            lambda r: r["vectors"]["flags"].extend(["-C", "debug-assertions=on"]),
            lambda r: r["commands"][0].update(timeout_seconds=True),
            lambda r: r["inventory"][0].update(path="../escape"),
            lambda r: r["vectors"].update(compiler=["relative/rustc"]),
            lambda r: r["vectors"].update(compiler=[str(self.root) + "/../rustc"]),
            lambda r: r["files"].append(dict(path="PREPARED.json/child",
                                            source=r["overlays"][0]["postimage"])),
            lambda r: r["files"].append(dict(path="plan.json/child",
                                            source=r["overlays"][0]["postimage"])),
            lambda r: r["files"].append(dict(path=".PREPARED.tmp/child",
                                            source=r["overlays"][0]["postimage"])),
            lambda r: r["files"].append(dict(path="source-A",
                                            source=r["overlays"][0]["postimage"])),
            lambda r: r["files"].extend([
                dict(path="aux", source=r["overlays"][0]["postimage"]),
                dict(path="aux/child", source=r["overlays"][0]["postimage"])]),
            lambda r: r["files"].extend([
                dict(path="aux/child", source=r["overlays"][0]["postimage"]),
                dict(path="aux", source=r["overlays"][0]["postimage"])]),
            lambda r: r["vectors"]["flags"].append("-Copt-level=0"),
            lambda r: r["vectors"]["flags"].append("--codegen=opt-level=0"),
            lambda r: r["vectors"]["flags"].extend(["--codegen", "opt-level=0"]),
            lambda r: r["vectors"]["flags"].append('--cfg=feature="gpu"'),
            lambda r: r["vectors"]["flags"].append(
                "--extern=rayon=/different/librayon.rlib"),
            lambda r: r["vectors"]["flags"].extend(["--cfg", 'feature="gpu"']),
            lambda r: r["vectors"]["flags"].extend(["--cfg", 'feature = "gpu"']),
            lambda r: r["vectors"]["flags"].extend(["--cfg", ' feature="gpu"']),
            lambda r: r["vectors"]["flags"].extend(["--cfg", 'feature="serde"']),
            lambda r: r["commands"][0]["argv_parts"].append(["-C"]),
            lambda r: r["commands"][0]["argv_parts"].append(["--cfg"]),
            lambda r: r["commands"][0]["argv_parts"].append(["--extern"]),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                recipe = copy.deepcopy(self.recipe)
                mutate(recipe)
                self.refused(recipe)
                self.assertFalse(self.output.exists())

    def test_reject_source_output_overlap_and_symlink_ancestors(self):
        self.refused(output=self.source / "nested")
        outside = self.root / "outside"
        outside.mkdir()
        link = self.root / "link"
        link.symlink_to(outside, target_is_directory=True)
        self.refused(output=link / "prepared")
        self.assertEqual(list(outside.iterdir()), [])
        candidate = self.root / "candidate-link"
        candidate.symlink_to(self.post)
        recipe = copy.deepcopy(self.recipe)
        recipe["overlays"][0]["postimage"]["path"] = str(candidate)
        self.refused(recipe)

    def test_mid_preparation_and_closed_temp_failure_never_publish(self):
        real_write = p.write_new
        for fail_at in ("source-A/data.bin", ".PREPARED.tmp"):
            with self.subTest(fail_at=fail_at):
                destination = self.root / ("partial-" + fail_at.replace("/", "-"))
                def interrupted(path, data):
                    real_write(path, data)
                    if str(path.relative_to(destination)) == fail_at:
                        raise InterruptedError("injected cancellation")
                with mock.patch.object(p, "write_new", side_effect=interrupted):
                    with self.assertRaisesRegex(InterruptedError, "cancellation"):
                        p.prepare(self.recipe, destination)
                self.assertTrue(destination.is_dir())
                self.assertTrue(any(destination.iterdir()))
                self.assertFalse((destination / "PREPARED.json").exists())
                self.unchanged()
        destination = self.root / "partial-after-link"
        real_link = p.os.link
        linked = False
        def publish(source, target, **kwargs):
            nonlocal linked
            result = real_link(source, target, **kwargs)
            linked = True
            return result
        def cancel_after_link():
            if linked:
                raise InterruptedError("injected cancellation after publication")
        with mock.patch.object(p.os, "link", side_effect=publish):
            with self.assertRaisesRegex(InterruptedError, "after publication"):
                p.prepare(self.recipe, destination, cancel_after_link)
        self.assertTrue(destination.is_dir())
        self.assertTrue((destination / ".PREPARED.tmp").is_file())
        self.assertFalse((destination / "PREPARED.json").exists())
        self.unchanged()

    def test_cli_deferred_sigterm_cancels_before_publication(self):
        recipe_file = self.root / "recipe.json"
        recipe_file.write_bytes(p.encode(self.recipe))
        handlers = {}
        def register(signum, handler):
            handlers[signum] = handler
            return signal.SIG_DFL
        real_write = p.write_new
        delivered = False
        def write_and_cancel(path, data):
            nonlocal delivered
            real_write(path, data)
            if not delivered:
                delivered = True
                handlers[signal.SIGTERM](signal.SIGTERM, None)
        stderr = io.StringIO()
        with mock.patch.object(p.signal, "signal", side_effect=register), \
             mock.patch.object(p, "write_new", side_effect=write_and_cancel), \
             mock.patch.object(p.sys, "argv", [
                 "prepare_ab.py", "--recipe", str(recipe_file),
                 "--output", str(self.output)]), \
             mock.patch.object(p.sys, "stderr", stderr):
            self.assertEqual(p.main(), 1)
        self.assertIn("canceled", stderr.getvalue())
        self.assertTrue(self.output.is_dir())
        self.assertFalse((self.output / "PREPARED.json").exists())
        self.unchanged()

    def test_duplicate_json_and_nonfinite_recipe_are_rejected(self):
        for data in (b'{"schema":1,"schema":1}', b'{"timeout":NaN}',
                     b'{"timeout":Infinity}'):
            with self.assertRaises(ValueError):
                p.strict(data)


if __name__ == "__main__":
    unittest.main()
