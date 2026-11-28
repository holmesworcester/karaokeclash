"""Open the current operator desk with a provisioned Karaoke Playground runtime."""
import argparse
import os
from pathlib import Path
import sys


def present_desk(app):
    desk = app.window
    display = desk.get_display()
    operator_monitor = display.get_monitor_at_window(desk.get_window())
    for index in range(display.get_n_monitors()):
        if display.get_monitor(index) != operator_monitor:
            app.stage.fullscreen_on_monitor(desk.get_screen(), index)
            app.stage.full = True
            app.show_stage()
            break
    desk.present()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("installation", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare-carriers", action="store_true", help="transcribe all playable backings")
    modes.add_argument("--audit-carriers", action="store_true", help="compare all carrier scores with their audio")
    parser.add_argument("--workers", type=int, default=2, help="parallel CPU transcription workers")
    parser.add_argument("--report", type=Path, help="write the library measurements as JSON")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    report = args.report.resolve() if args.report else None
    root = args.installation.resolve()
    desk_modules = Path(__file__).resolve().parent / "linux-four"
    os.chdir(root)
    sys.path[:0] = [str(desk_modules), str(root / "linux-four")]
    # Add Song starts the importer in a child interpreter.
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [*sys.path[:2], os.environ.get("PYTHONPATH")]))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("PIPEWIRE_LATENCY", "64/48000")
    os.environ.setdefault("PIPEWIRE_RATE", "1/48000")
    if (root / "karaoke-playground-4").is_file():
        data_home = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
        os.environ.setdefault("KARAOKE_DATA", str(data_home / "karaoke-playground-4"))

    if args.prepare_carriers or args.audit_carriers:
        from prepare_carriers import run
        run(args.workers, args.audit_carriers, report)
        return

    from desk import Desk, Gtk

    app = Desk()
    present_desk(app)
    Gtk.main()


if __name__ == "__main__":
    main()
