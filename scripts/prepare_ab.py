"""Prepare pinned A/B sources and argv vectors; never execute or approve them."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import signal
import sys

MAX_RECIPE = 128 * 1024
MAX_FILE = 4 * 1024 * 1024
MAX_OUTPUT = 24 * 1024 * 1024
MAX_INPUT = 12 * 1024 * 1024
ROLES = ("A", "B")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def fields(value, names):
    require(type(value) is dict and set(value) == set(names), "unexpected fields")


def text(value):
    require(type(value) is str and 0 < len(value) <= 4096 and "\0" not in value,
            "invalid string")
    return value


def identifier(value):
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", text(value)),
            "invalid identifier")
    return value


def strict(data):
    require(len(data) <= MAX_RECIPE, "recipe too large")
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "duplicate JSON key")
            out[key] = value
        return out
    def constant(value):
        raise ValueError("nonfinite JSON: " + value)
    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode()


def relative(value):
    value = text(value)
    path = PurePosixPath(value)
    require(not path.is_absolute() and str(path) == value and
            ".." not in path.parts and value != ".", "unsafe relative path")
    return value


def lexical_absolute(value):
    path = Path(text(value))
    require(path.is_absolute() and str(path) == value and ".." not in path.parts,
            "unsafe absolute path")
    return path


def absolute(value):
    path = lexical_absolute(value)
    for item in (path, *path.parents):
        require(not item.is_symlink(), "symlink path")
    return path


def row(path, data):
    return dict(path=str(path), bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest())


def read_record(record):
    fields(record, ("path", "bytes", "sha256"))
    require(type(record["bytes"]) is int and 0 <= record["bytes"] <= MAX_FILE,
            "invalid byte count")
    require(type(record["sha256"]) is str and
            re.fullmatch(r"[0-9a-f]{64}", record["sha256"]), "invalid digest")
    path = absolute(record["path"])
    require(path.is_file() and path.stat().st_size == record["bytes"],
            "input size changed")
    with path.open("rb") as stream:
        data = stream.read(MAX_FILE + 1)
    require(row(path, data) == record, "input changed")
    return data


def strings(values, limit=2048):
    require(type(values) is list and len(values) <= limit, "invalid string vector")
    return [text(value) for value in values]


def expand_commands(vectors, profiles, commands):
    require(type(vectors) is dict and len(vectors) <= 64, "invalid vectors")
    vectors = {identifier(name): strings(value) for name, value in vectors.items()}
    require(type(profiles) is dict and len(profiles) <= 16, "invalid profiles")
    for name, profile in profiles.items():
        identifier(name)
        fields(profile, ("features", "codegen", "externs"))
        for key in profile:
            values = strings(profile[key])
            require(len(values) == len(set(values)), "duplicate profile value")
    require(type(commands) is list and 0 < len(commands) <= 64, "invalid commands")
    out, ids = [], set()
    for command in commands:
        fields(command, ("id", "profile", "argv_parts", "cwd", "timeout_seconds"))
        name = identifier(command["id"])
        require(name not in ids, "duplicate command")
        ids.add(name)
        require(type(command["argv_parts"]) is list and
                len(command["argv_parts"]) <= 256, "invalid argv parts")
        argv = []
        for part in command["argv_parts"]:
            if type(part) is dict:
                fields(part, ("vector",))
                require(text(part["vector"]) in vectors, "unknown vector")
                argv.extend(vectors[part["vector"]])
            else:
                argv.extend(strings(part))
        require(0 < len(argv) <= 2048, "invalid argv")
        lexical_absolute(argv[0])
        cwd = absolute(command["cwd"])
        timeout = command["timeout_seconds"]
        require(type(timeout) in (int, float) and math.isfinite(timeout) and
                0 < timeout <= 3600, "invalid timeout")
        selected = command["profile"]
        if selected is not None:
            require(type(selected) is str and selected in profiles, "unknown profile")
            require(not any(value.startswith("@") for value in argv[1:]),
                    "response files are not allowed in profiled commands")
            profile = profiles[selected]
            options = {"-C": [], "--cfg": [], "--extern": []}
            i = 1
            while i < len(argv):
                value = argv[i]
                require(not (value.startswith("-C") and value != "-C") and
                        not value.startswith("--codegen") and
                        not (value.startswith("--cfg") and value != "--cfg") and
                        not (value.startswith("--extern") and value != "--extern"),
                        "noncanonical protected flag")
                if value in options:
                    require(i + 1 < len(argv) and not argv[i + 1].startswith("-"),
                            "missing protected flag value")
                    options[value].append(argv[i + 1])
                    i += 2
                else:
                    i += 1
            codegen, cfg = options["-C"], options["--cfg"]
            externs = [value.split("=", 1)[0] for value in options["--extern"]]
            require(len(externs) == len(set(externs)), "duplicate direct extern")
            require(all(codegen.count(value) == 1 and all(item == value for item in codegen
                        if item.split("=", 1)[0] == value.split("=", 1)[0])
                        for value in profile["codegen"]), "profile codegen missing or conflicting")
            features = []
            for value in cfg:
                if re.match(r"feature\b", value.lstrip()):
                    match = re.fullmatch(r'feature="([^"]+)"', value)
                    require(match is not None, "noncanonical feature cfg")
                    features.append(match[1])
            require(len(features) == len(set(features)) and
                    set(features) == set(profile["features"]), "profile feature mismatch")
            require(all(value in externs for value in profile["externs"]),
                    "direct extern missing")
        out.append(dict(id=name, profile=selected, schema=1, argv=argv,
                        cwd=str(cwd), timeout_seconds=timeout))
    return out


def render(recipe, output):
    fields(recipe, ("schema", "source_root", "inventory", "overlays", "files",
                    "role_files", "vectors", "profiles", "commands", "cases", "method"))
    require(type(recipe["schema"]) is int and recipe["schema"] == 1, "unsupported schema")
    source = absolute(recipe["source_root"])
    output = absolute(str(output))
    require(source.is_dir() and output.parent.is_dir(), "missing root")
    require(source != output and source not in output.parents and
            output not in source.parents, "source/output overlap")
    inputs, outputs = {}, {}
    def read(record):
        path = absolute(record["path"])
        require(path != output and output not in path.parents, "input/output overlap")
        data = read_record(record)
        require(str(path) not in inputs or inputs[str(path)] == record,
                "conflicting input")
        inputs[str(path)] = record
        require(sum(item["bytes"] for item in inputs.values()) <= MAX_INPUT,
                "inputs too large")
        return data
    def put(name, data):
        name = relative(name)
        require(PurePosixPath(name).parts[0] not in
                ("plan.json", "PREPARED.json", ".PREPARED.tmp"), "reserved output")
        require(all(name != other and not name.startswith(other + "/") and
                    not other.startswith(name + "/") for other in outputs),
                "output file/ancestor collision")
        outputs[name] = data
        require(sum(map(len, outputs.values())) <= MAX_OUTPUT, "outputs too large")
    inventory = recipe["inventory"]
    require(type(inventory) is list and 0 < len(inventory) <= 256, "invalid inventory")
    base = {}
    for item in inventory:
        fields(item, ("path", "bytes", "sha256"))
        name = relative(item["path"])
        require(name not in base, "duplicate source path")
        base[name] = read(dict(item, path=str(source / name)))
    roles = {role: dict(base) for role in ROLES}
    require(type(recipe["overlays"]) is list and len(recipe["overlays"]) <= 64,
            "invalid overlays")
    seen = set()
    for overlay in recipe["overlays"]:
        fields(overlay, ("roles", "path", "preimage_sha256", "postimage"))
        names = strings(overlay["roles"], 2)
        require(names and len(set(names)) == len(names) and set(names) <= set(ROLES),
                "invalid overlay roles")
        name = relative(overlay["path"])
        data = read(overlay["postimage"])
        for role in names:
            require(name in base and (role, name) not in seen, "ambiguous overlay")
            require(hashlib.sha256(base[name]).hexdigest() == overlay["preimage_sha256"],
                    "overlay preimage changed")
            seen.add((role, name))
            roles[role][name] = data
    for role in ROLES:
        for name, data in roles[role].items():
            put("source-" + role + "/" + name, data)
    require(type(recipe["files"]) is list and len(recipe["files"]) <= 32, "invalid files")
    for item in recipe["files"]:
        fields(item, ("path", "source"))
        put(item["path"], read(item["source"]))
    require(type(recipe["role_files"]) is list and len(recipe["role_files"]) <= 16,
            "invalid role files")
    for item in recipe["role_files"]:
        fields(item, ("path", "source", "token"))
        require(type(item["path"]) is str and item["path"].count("{role}") == 1 and
                "{" not in item["path"].replace("{role}", "") and
                "}" not in item["path"].replace("{role}", ""), "invalid role filename")
        data = read(item["source"])
        token = item["token"]
        if token is not None:
            token = text(token).encode()
            require(data.count(token) == 1, "ambiguous role token")
        for role in ROLES:
            put(item["path"].replace("{role}", role),
                data if token is None else data.replace(token, role.encode()))
    cases = strings(recipe["cases"], 64)
    require(cases and len(cases) == len(set(cases)), "invalid cases")
    for case in cases:
        identifier(case)
    if recipe["method"] is not None:
        read(recipe["method"])
    commands = expand_commands(recipe["vectors"], recipe["profiles"], recipe["commands"])
    plan = dict(schema=1, recipe_sha256=hashlib.sha256(encode(recipe)).hexdigest(),
                inputs=[inputs[name] for name in sorted(inputs)],
                outputs=[row(output / name, data) for name, data in sorted(outputs.items())],
                commands=commands, profiles=recipe["profiles"], cases=cases, method=recipe["method"])
    require(len(encode(plan)) <= 256 * 1024, "plan too large")
    return outputs, plan


def write_new(path, data):
    absolute(str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    absolute(str(path))
    with path.open("xb") as stream:
        stream.write(data)


def prepare(recipe, output, check_cancel=lambda: None, recipe_record=None):
    output = absolute(str(output))
    require(not output.exists(), "output already exists")
    check_cancel()
    outputs, plan = render(recipe, output)
    if recipe_record is not None:
        require(output not in absolute(recipe_record["path"]).parents,
                "recipe/output overlap")
        read_record(recipe_record)
        existing = {item["path"]: item for item in plan["inputs"]}
        require(recipe_record["path"] not in existing or
                existing[recipe_record["path"]] == recipe_record, "conflicting recipe input")
        existing[recipe_record["path"]] = recipe_record
        plan["inputs"] = [existing[name] for name in sorted(existing)]
        require(sum(item["bytes"] for item in plan["inputs"]) <= MAX_INPUT,
                "inputs too large")
    owned = published = False
    marker = output / "PREPARED.json"
    temporary = output / ".PREPARED.tmp"
    try:
        check_cancel()
        output.mkdir()
        owned = True
        for name, data in sorted(outputs.items()):
            check_cancel()
            write_new(output / name, data)
        for item in plan["inputs"]:
            check_cancel()
            read_record(item)
        for item in plan["outputs"]:
            check_cancel()
            read_record(item)
        payload = encode(plan)
        require(len(payload) <= 256 * 1024, "plan too large")
        write_new(output / "plan.json", payload)
        check_cancel()
        write_new(temporary, encode(dict(schema=1, plan=row(output / "plan.json", payload))))
        check_cancel()
        # link publishes the closed, complete marker without replacing an existing name.
        os.link(temporary, marker, follow_symlinks=False)
        published = True
        check_cancel()
        temporary.unlink()
    except BaseException:
        if owned and published:
            marker.unlink(missing_ok=True)
        raise
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    previous, pending, completed = {}, [], False
    def canceled(signum, frame):
        # Keep the ownership assignment after the atomic link uninterrupted.
        pending.append(signum)
    def check_cancel():
        if pending:
            raise InterruptedError("preparation canceled")
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, canceled)
        path = absolute(str(args.recipe))
        require(path.is_file() and path.stat().st_size <= MAX_RECIPE, "invalid recipe file")
        with path.open("rb") as stream:
            data = stream.read(MAX_RECIPE + 1)
        prepare(strict(data), args.output, check_cancel, row(path, data))
        completed = True
        check_cancel()
        return 0
    except (ValueError, OSError) as error:
        if completed:
            (args.output / "PREPARED.json").unlink(missing_ok=True)
        print(str(error), file=sys.stderr)
        return 1
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
