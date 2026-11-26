# Karaoke Playground 4 for Linux

Four independent microphone strips and a separate 2×2 audience display.

[Download the latest Linux build](https://github.com/holmesworcester/karaokeclash/releases/latest).

This x86-64 build targets Ubuntu 26.04 LTS and requires glibc 2.43 or newer. Install the desktop audio and Python dependencies on Ubuntu:

```sh
sudo apt install pipewire-jack pipewire-bin python3-gi-cairo gir1.2-gtk-3.0 python3-numpy ffmpeg
```

Extract the archive and run `./karaoke-playground-4` in a graphical session. Open **Audio I/O** to assign the four capture inputs and stereo return. Open **Audience** for the projected lyrics and press **F11** for fullscreen.

The app opens with an empty song library. Use **Add song** to prepare a recording you supply. Set `KARAOKE_IMPORTS` to the directory of an existing library of prepared imports. Automated song preparation needs a signed-in Codex CLI and the local karaoke skill tools. Live character conversion additionally needs the local voice models and runtime.

Based on Karaoke Playground commit `f4e246b773bb0d6414bb82fb673c178de9cd486f`.
