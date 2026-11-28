"""The built-in vocoder controls shared by the mic strip and its carrier settings menu."""
from gi.repository import Gtk

from dial import Dial


VOCODER_PARAMS = (
    ("Dry / Wet", "%", 0, 100, 100),
    ("Consonants", "%", 0, 100, 50),
    ("Vocal attack", " ms", 1, 80, 5),
    ("Release", " ms", 5, 300, 40),
    ("Formant shift", " st", -12, 12, 0),
    ("Carrier harmonics", "%", 0, 100, 55),
    ("Synth register", "", -2, 1, 0),
    ("Glide", " ms", 0, 240, 100),
)
VOCODER_HINTS = (
    "At 100% only the speech-shaped carrier is heard while a carrier note plays.",
    "Resynthesize S, F and T sounds with noise in the upper speech bands.",
    "How quickly a syllable opens the frequency bands.",
    "How long the bands ring after a syllable. Short times separate rap syllables.",
    "Move the vowel resonances up or down while keeping the carrier notes.",
    "Blend a rounded saw with a brighter saw and pulse for clearer speech.",
    "Transpose the synthesized chord together. Speech range lowers high notes.",
    "Bend connected MIDI melody notes. Zero plays the original scored attacks.",
)
VOCODER_PRESETS = {
    "Tight speech": [100, 75, 2, 20, 0, 80, 1, 100],
    "Vowel stack": [100, 38, 5, 64, 0, 88, 0, 100],
    "Smooth vowels": [100, 30, 12, 95, -2, 35, 0, 100],
}


class VocoderControls(Gtk.Box):
    def __init__(self, mic, lane, strip, audio, presets, save):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=4, margin=4)
        self.mic, self.lane, self.strip, self.audio, self.save = mic, lane, strip, audio, save
        self.values = strip["vocoder"]
        self.controls, self.labels, self.fields = {}, {}, {}
        self.loading = False

        self.menu = Gtk.MenuButton(label="Vocoder…")
        self.popover = Gtk.Popover.new(self.menu)
        self.menu.set_popover(self.popover)
        self.menu.set_no_show_all(True)
        self.inline = Gtk.Box()
        self.inline.set_no_show_all(True)
        self.popover.add(self)
        self.get_accessible().set_name(f"{mic} vocoder controls")

        header = Gtk.Box(spacing=6)
        self.add(header)
        self.preset_label = Gtk.Label(label="Preset")
        header.add(self.preset_label)
        self.preset = Gtk.ComboBoxText()
        self.preset.get_accessible().set_name(f"{mic} vocoder preset")
        self.presets = {"Balanced": [param[-1] for param in VOCODER_PARAMS], **VOCODER_PRESETS}
        for name in (*self.presets, "Custom"):
            self.preset.append(name, name)
        header.add(self.preset)

        rows = [Gtk.Box(spacing=5, homogeneous=True) for _ in range(2)]
        for row in rows:
            self.add(row)
        for i, (name, unit, lo, hi, default) in enumerate(VOCODER_PARAMS):
            label = self.labels[i] = Gtk.Label(label=name)
            if i == 6:
                control = Gtk.ComboBoxText()
                for mode, title in ((0, "Score pitch"), (1, "Speech range"), (-1, "−1 octave"), (-2, "−2 octaves")):
                    control.append(str(mode), title)
                control.set_active_id(str(int(self.values[i])))
                control.connect("changed", lambda w, j=i: self.changed(j, int(w.get_active_id())))
                header.add(label)
                header.add(control)
                setter = lambda v, w=control: w.set_active_id(str(int(v)))
            else:
                control = Dial(f"{mic} vocoder {name.lower()}", lo, hi, self.values[i], default, unit)
                control.connect("value-changed", lambda w, j=i: self.changed(j, round(w.get_value())))
                self.fields[i] = self.add_dial(rows[0] if i < 4 else rows[1], label, control)
                setter = control.set_value
            control.get_accessible().set_name(f"{mic} vocoder {name.lower()}")
            control.set_tooltip_text(VOCODER_HINTS[i])
            self.controls[i] = control
            presets.bind(f"vocoder.{i}", name, "vocoder", lambda j=i: self.values[j], setter)
            presets.attach_control(f"vocoder.{i}", control)

        self.width = Dial(f"{mic} vocoder stereo width", 0, 100, strip["stereo_width"], 100, "%")
        self.width.set_tooltip_text("Scale the recorded stereo placement in the MIDI monitor and vocoder.")
        self.add_dial(rows[1], Gtk.Label(label="Stereo width"), self.width)
        self.width.connect("value-changed", self.change_width)
        presets.bind("stereo_width", "Stereo width", "vocoder", self.width.get_value, self.width.set_value)
        presets.attach_control("stereo_width", self.width)
        audio.call("stereo_width", lane, self.width.get_value())

        self.source = Dial(f"{mic} TAL source vocal", 0, 100, strip["tal_source_level"], 0, "%")
        self.source.set_tooltip_text("Add the original mic, Hear sample or Test vocal to the TAL return.")
        self.source_field = self.add_dial(rows[0], Gtk.Label(label="Source vocal"), self.source)
        self.source.connect("value-changed", self.change_source_level)
        presets.bind("tal_source_level", "Source vocal", "tal", self.source.get_value, self.source.set_value,
                     group_label="TAL Vocoder")
        presets.attach_control("tal_source_level", self.source)
        audio.call("tal_source_level", lane, self.source.get_value())

        self.reset = Gtk.Button(label="Reset")
        self.reset.get_accessible().set_name(f"{mic} reset vocoder controls")
        self.reset.set_tooltip_text("Restore the Balanced preset and full stereo width")
        self.reset.connect("clicked", self.reset_controls)
        header.pack_end(self.reset, False, False, 0)
        self.preset.connect("changed", self.use_preset)
        self.apply()
        self.show_all()
        for widget in (self.preset, self.preset_label, self.reset, self.source_field, *self.fields.values()):
            widget.set_no_show_all(True)

    @staticmethod
    def add_dial(row, label, dial):
        field = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        field.add(label)
        field.add(dial)
        row.pack_start(field, False, False, 0)
        return field

    def changed(self, index, value):
        self.values[index] = value
        if not self.loading:
            self.apply()
            self.save()

    def apply(self):
        self.audio.call("vocoder", self.lane, *self.values[:4])
        self.audio.call("vocoder_shape", self.lane, *self.values[4:7])
        self.audio.call("glide", self.lane, self.values[7])
        name = next((name for name, values in self.presets.items() if values == self.values), "Custom")
        if self.preset.get_active_id() != name:
            self.preset.set_active_id(name)

    def use_preset(self, widget):
        values = self.presets.get(widget.get_active_id())
        if values is None:
            self.apply()
            return
        if values == self.values:
            return
        self.loading = True
        try:
            for i, value in enumerate(values):
                if i == 6:
                    self.controls[i].set_active_id(str(int(value)))
                else:
                    self.controls[i].set_value(value)
        finally:
            self.loading = False
        self.apply()
        self.save()

    def change_width(self, widget):
        value = round(widget.get_value())
        self.strip["stereo_width"] = value
        self.audio.call("stereo_width", self.lane, value)
        self.save()

    def reset_controls(self, *_):
        self.preset.set_active_id("Balanced")
        self.width.set_value(100)

    def change_source_level(self, widget):
        value = round(widget.get_value())
        self.strip["tal_source_level"] = value
        self.audio.call("tal_source_level", self.lane, value)
        self.save()

    def show_for(self, *, inline, harmony, tal, glide):
        parent = self.inline if inline else self.popover
        if self.get_parent() != parent:
            self.popover.popdown()
            self.get_parent().remove(self)
            parent.add(self)
        self.inline.set_visible(inline)
        self.menu.set_visible(not inline)

        group = "Harmony" if harmony else "Carrier" if tal else "Vocoder"
        self.menu.set_label(group + "…")
        self.menu.get_accessible().set_name(f"{self.mic} {group.lower()} settings")
        self.menu.set_tooltip_text("Adjust the scored singing harmony" if harmony else "Adjust the vocoder and its scored carrier")
        mix = "Harmony level" if harmony else "Wet level" if tal else "Dry / Wet"
        self.labels[0].set_text(mix)
        self.controls[0].get_accessible().set_name(f"{self.mic} {group.lower()} {mix.lower()}")
        self.controls[0].set_tooltip_text("Level of the tuned voices." if harmony else
                                          "Level of the vocoded voice and scored bass. Source vocal adds the original singer." if tal else VOCODER_HINTS[0])
        self.source_field.set_visible(tal)
        self.labels[6].set_text("Vocal register" if harmony else "Synth register")
        for widget in (self.preset_label, self.preset, self.reset):
            widget.set_visible(not (harmony or tal))
        for i in (1, 2, 3, 4):
            self.fields[i].set_visible(not (harmony or tal))
        self.fields[5].set_visible(not harmony)
        self.fields[7].set_visible(glide)
