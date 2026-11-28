"""The application menu and shell resolve to one desk, using the installed runtime."""
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest

import gi
gi.require_version("GioUnix", "2.0")
from gi.repository import Gio, GioUnix

ROOT = Path(__file__).resolve().parents[1]


class InstalledDesk(unittest.TestCase):
    def test_menu_and_shell_launch_the_shared_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            runtime = tmp / "packaged runtime"
            (runtime / "linux-four").mkdir(parents=True)
            (runtime / "linux-four/audio.py").touch()
            probe = tmp / "commands"
            probe.mkdir()
            capture = tmp / "arguments"
            jack = probe / "pw-jack"
            jack.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$LAUNCH_CAPTURE"\n')
            jack.chmod(0o755)
            env = {
                **os.environ,
                "XDG_BIN_HOME": str(tmp / "installed commands"),
                "XDG_DATA_HOME": str(tmp / "data"),
                "XDG_CONFIG_HOME": str(tmp / "config"),
                "PATH": str(probe) + os.pathsep + os.environ["PATH"],
                "LAUNCH_CAPTURE": str(capture),
            }
            env.pop("KARAOKE_RUNTIME", None)
            cli = Path(env["XDG_BIN_HOME"]) / "karaoke-playground-4"
            cli.parent.mkdir()
            # Reinstall over an earlier menu command and a broken legacy symlink.
            cli.symlink_to(tmp / "old-desk")
            subprocess.run([ROOT / "scripts/install-linux-four", runtime], env=env, check=True)
            self.assertEqual(cli.resolve(), ROOT / "scripts/karaoke-playground-4")
            expected = ["/usr/bin/python3", str(ROOT / "preview.py"), str(runtime)]
            subprocess.run([cli, "--help"], env=env, check=True)
            self.assertEqual(capture.read_text().splitlines(), expected + ["--help"])

            capture.unlink()
            entry = Path(env["XDG_DATA_HOME"]) / "applications/karaoke-playground-4.desktop"
            info = GioUnix.DesktopAppInfo.new_from_filename(str(entry))
            context = Gio.AppLaunchContext()
            for key, value in env.items():
                context.setenv(key, value)
            context.unsetenv("KARAOKE_RUNTIME")
            self.assertTrue(info.launch([], context))
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if capture.exists() and capture.read_text().endswith("\n"):
                    break
                time.sleep(0.01)
            self.assertEqual(capture.read_text().splitlines(), expected)
