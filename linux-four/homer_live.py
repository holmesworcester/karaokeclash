"""Configure the existing RVC stream in the Homer-only GPU subprocess."""

from contextlib import redirect_stdout
import math
from pathlib import Path
import sys
from types import SimpleNamespace


class PrimedInput:
    """Give the first converter window silent history and real right context."""

    def __init__(self, source, samples):
        self.source = source
        self.prefix = bytes(samples * 2)

    def read(self, count):
        prefix, self.prefix = self.prefix[:count], self.prefix[count:]
        return prefix + self.source.read(count - len(prefix))


def main(tools, rvc, packets):
    sink = sys.stdout.buffer
    with redirect_stdout(sys.stderr):
        sys.path.insert(0, str(tools))
        import live_character
        import torch
        from framework_rvc import Converter

        fast = packets == 1
        # The runtime owns the stream, GPU lease checks and aligned packet joins.
        # Its constructor binding is local to this Homer-only interpreter.
        class HomerConverter(Converter):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                encoder = self.voices["homer"][0].enc_p
                # RVC splits this projection into mean and log standard deviation.
                # Reduce random latent variation between windows without moving the
                # learned mean or changing checkpoint files.
                with torch.no_grad():
                    encoder.proj.bias[encoder.out_channels:].add_(math.log(.35 if fast else .30))

        live_character.Converter = HomerConverter
        live_character.PROFILES["homer-monitor"] = live_character.PROFILES["fast"]._replace(
            history_ms=180 if fast else 400, guard_ms=30 if fast else 90, future_ms=5 if fast else 10,
            index_rate=.85 if fast else .75, pitch="praat" if fast else "rmvpe",
        )
        args = SimpleNamespace(upstream=rvc / "rvc", models=rvc / "alternate",
                               voice="homer", profile="homer-monitor", hop_ms=20 * int(packets),
                               idle_slots="http://127.0.0.1:8731/slots")
        live_character.run(args, PrimedInput(sys.stdin.buffer, 80 if fast else 160), sink)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]))
