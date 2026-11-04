#!/usr/bin/env python3
"""Run an AI-Toolbox / Studio Toolbox pipeline preset without the UI.

A preset is the MOV -> EXR, Fix Missing Frames, Expansion Studio and
Lens Studio tabs' settings saved under a name from the Pipeline tab
(~/.ai_toolbox/presets/NAME.json). This runs one input -- a movie or a
frame folder -- through the preset's steps, one after another, each
step's output folder feeding the next, exactly as the tabs would.

    python pipeline.py list
    python pipeline.py show --preset NAME
    python pipeline.py run --preset NAME --input shot.mov [--steps exr fixframes upscale organic]
                           [--json result.json] [--quiet]

Exit code 0 when every step wrote its output, 1 otherwise. --json
writes {"ok": bool, "outputs": {step: folder}, "error": str|null} for
the calling tool. Run it with the Python the app itself uses (the
launcher remembers it in ~/.ai_toolbox/python); the step models need
that environment. The app is a Tk program, so this needs a logged-in
desktop session on the Mac even though no window is shown.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

STEP_KEYS = ("exr", "fixframes", "upscale", "organic")


def _app_class():
    import ai_toolbox
    return ai_toolbox.App


def cmd_list(_args):
    App = _app_class()
    names = App._pipeline_list()
    if not names:
        print(f"No presets in {App.PIPELINE_DIR} -- save one from the "
              "Pipeline tab.")
        return 0
    for n in names:
        try:
            d = App._pipeline_preset_read(n)
            steps = " -> ".join(d.get("steps") or [])
        except Exception as ex:
            steps = f"(unreadable: {ex})"
        print(f"{n:30s} {steps}")
    return 0


def _read_preset(App, name):
    try:
        return App._pipeline_preset_read(name)
    except FileNotFoundError:
        names = App._pipeline_list()
        sys.exit(f"no preset {name!r} in {App.PIPELINE_DIR}"
                 + (f" -- have: {', '.join(names)}" if names else
                    " -- save one from the Pipeline tab"))
    except Exception as ex:
        sys.exit(f"could not read preset {name!r}: {ex}")


def cmd_show(args):
    App = _app_class()
    d = _read_preset(App, args.preset)
    print(json.dumps(d, indent=1))
    return 0


def cmd_run(args):
    App = _app_class()
    preset = _read_preset(App, args.preset)
    steps = list(args.steps) if args.steps else None
    for s in steps or []:
        if s not in STEP_KEYS:
            sys.exit(f"unknown step {s!r}; choose from {', '.join(STEP_KEYS)}")
    if not os.path.exists(args.input):
        sys.exit(f"input not found: {args.input}")

    app = App(modules=["pipeline"], title="Pipeline")
    app.withdraw()
    result = {"ok": False, "outputs": {}, "error": None,
              "preset": preset.get("name"), "input": os.path.abspath(args.input)}

    def on_log(msg):
        if not args.quiet:
            print(msg, flush=True)
        if msg.startswith("✖") and not result["error"]:
            result["error"] = msg[1:].strip()

    def on_done(ok, outputs):
        result["ok"] = bool(ok)
        result["outputs"] = dict(outputs)
        app.after(100, app.destroy)

    try:
        app._pipeline_run(preset, args.input, steps, on_log=on_log,
                          on_done=on_done)
    except Exception as ex:
        result["error"] = str(ex)
        print(f"✖ {ex}", file=sys.stderr)
        app.destroy()
    else:
        app.mainloop()

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=1)
    return 0 if result["ok"] else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="name the saved presets")
    sp = sub.add_parser("show", help="print a preset's JSON")
    sp.add_argument("--preset", required=True, help="preset name or .json path")
    rp = sub.add_parser("run", help="run a preset on an input")
    rp.add_argument("--preset", required=True, help="preset name or .json path")
    rp.add_argument("--input", required=True, help="a movie, or a frame folder")
    rp.add_argument("--steps", nargs="+", metavar="STEP",
                    help="override the preset's steps: " + " ".join(STEP_KEYS))
    rp.add_argument("--json", metavar="FILE", help="write the result here")
    rp.add_argument("--quiet", action="store_true", help="no log on stdout")
    args = ap.parse_args(argv)
    return {"list": cmd_list, "show": cmd_show, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
