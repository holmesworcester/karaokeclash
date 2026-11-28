"""Edit TAL's VST3 parameters by their stable plug-in IDs."""
import re

from gi.repository import Gtk
from dial import Dial


TITLES = {
    "volume": "Output volume", "mod volume": "Modulator input", "carrier volume": "Carrier input",
    "inputmode": "Input mode", "harmonics": "Harmonics", "esserintensity": "Ess intensity",
    "enveloperelease": "Envelope release", "compand": "Compand", "chorus": "Chorus",
    "sidechain": "Sidechain", "Bypass": "Bypass (mute return)", "unknown": "Unknown (TAL)",
    "noisevolume": "Noise level", "pulsevolume": "Pulse level", "sawvolume": "Saw level",
    "suboscvolume": "Sub oscillator level", "osctranspose": "Transpose", "suboscoctave": "Sub octave",
    "oscsync": "Oscillator sync", "pulsetune": "Pulse tuning", "sawtune": "Saw tuning",
    "pulsefinetune": "Pulse fine tuning", "sawfinetune": "Saw fine tuning", "polymode": "Poly mode",
    "_portamento": "Portamento", "tune": "Master tuning", "panic": "Panic",
}
OSCILLATORS = {"noisevolume", "pulsevolume", "sawvolume", "suboscvolume", "osctranspose",
               "suboscoctave", "oscsync", "pulsetune", "sawtune", "pulsefinetune", "sawfinetune",
               "polymode", "_portamento", "tune", "panic"}


class TalControls(Gtk.Box):
    def __init__(self, audio, lane, saved, save):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=3)
        self.set_tooltip_text("Edit TAL-Vocoder-2 plug-in parameters")
        self.audio, self.lane, self.saved, self.save = audio, lane, saved, save
        self.params = audio.tal_parameters(lane)
        self.bind_control = None
        self.sliders = {}
        self.midi = {}
        for index, param in enumerate(self.params):
            key = str(param.id)
            if key in saved:
                audio.call("tal_set", lane, index, saved[key])
        self.set_no_show_all(True)
        self.connect("show", self.build)

    def store(self, index, value):
        param = self.params[index]
        key = str(param.id)
        if abs(value - param.initial) < 1e-6:
            self.saved.pop(key, None)
        else:
            self.saved[key] = value
        self.audio.call("tal_set", self.lane, index, value)
        self.save()

    def value(self, i):
        return self.saved.get(str(self.params[i].id), self.params[i].initial)

    def scale(self, index, title):
        param = self.params[index]
        if param.name == b"inputmode":
            widget = Gtk.ComboBoxText()
            for label in ("Synth carrier / mic modulator", "Synth carrier / score modulator",
                          "Score carrier / mic modulator", "Mic carrier / score modulator"):
                widget.append_text(label)
            widget.set_active(round(self.value(index)*3))
            widget.connect("changed", lambda w: self.store(index, w.get_active()/3))
            widget.get_accessible().set_name(title)
            widget.set_tooltip_text(param.name.decode())
            self.sliders[index] = widget
            return widget
        def display(value):
            label = self.audio.call("tal_display", self.lane, index, value).decode()
            if param.steps:
                return label
            if re.fullmatch(r"-?\d+\.\d{7}", label):
                return f"{100*value:.1f}%"
            return label or f"{100*value:.1f}%"

        dial = Dial(title, 0, 1, self.value(index), param.initial, "",
                    step=1/param.steps if param.steps else .001, formatter=display)
        dial.set_sensitive(not param.flags & 2)
        dial.connect("value-changed", lambda w: self.store(index, w.get_value()))
        self.sliders[index] = dial
        return dial

    def build(self, *_):
        if self.get_children():
            return
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self.add(box)
        if not self.params:
            box.add(Gtk.Label(label="TAL-Vocoder-2 unavailable"))
            box.show_all()
            return
        header = Gtk.Box(spacing=6)
        header.add(Gtk.Label(label="TAL Vocoder", xalign=0))
        self.stack = Gtk.Stack()
        self.stack.set_vhomogeneous(False)
        tabs = Gtk.StackSwitcher(stack=self.stack)
        tabs.get_accessible().set_name("TAL parameter group")
        header.add(tabs)
        reset = Gtk.Button(label="Reset")
        reset.set_tooltip_text("Reset TAL settings on this pane")
        reset.connect("clicked", self.reset)
        header.pack_end(reset, False, False, 0)
        box.add(header)
        box.add(self.stack)
        groups = {}
        for group in ("Vocoder", "Oscillators", "11-band EQ", "MIDI controllers"):
            grid = Gtk.Grid(column_spacing=5, row_spacing=2)
            self.stack.add_titled(grid, group, {"11-band EQ":"EQ", "MIDI controllers":"MIDI"}.get(group, group))
            groups[group] = grid
        rows = dict.fromkeys(groups, 0)
        first = ("inputmode", "mod volume", "carrier volume", "volume", "esserintensity",
                 "enveloperelease", "harmonics", "compand", "chorus")
        order = {name: i for i, name in enumerate(first)}
        ordered = sorted(enumerate(self.params), key=lambda item:
                         order.get(item[1].name.decode(), len(first) + (item[1].name == b"unknown")))
        for i, param in ordered:
            name = param.name.decode()
            midi = re.fullmatch(r"MIDI CC (\d+)\|(\d+)", name)
            if midi:
                self.midi[tuple(map(int, midi.groups()))] = i
                continue
            group = "Oscillators" if name in OSCILLATORS else "Vocoder"
            title = TITLES.get(name, name)
            if name.startswith("vocoderband") and name[11:].isdigit():
                group, title = "11-band EQ", f"Band {int(name[11:])+1}"
            control = self.scale(i, title)
            if self.bind_control:
                self.bind_control(i, control)
            if name == "inputmode":
                groups[group].attach(Gtk.Label(label=title, xalign=0), 0, 0, 1, 1)
                groups[group].attach(control, 1, 0, 4, 1)
                continue
            row, column = divmod(rows[group], 6)
            top = 1 if group == "Vocoder" else 0
            caption = Gtk.Label(label=title, justify=Gtk.Justification.CENTER)
            caption.set_line_wrap(True)
            caption.set_max_width_chars(11)
            groups[group].attach(caption, column, top+row*2, 1, 1)
            groups[group].attach(control, column, top+row*2+1, 1, 1)
            rows[group] += 1
        if self.midi:
            self.midi_controls(groups["MIDI controllers"])
        box.show_all()

    def midi_controls(self, grid):
        channel = Gtk.SpinButton.new_with_range(1, 16, 1)
        number = Gtk.SpinButton.new_with_range(0, 129, 1)
        for row, (name, widget) in enumerate((("Channel", channel), ("Controller", number))):
            widget.get_accessible().set_name("MIDI " + name.lower())
            grid.attach(Gtk.Label(label=name, xalign=0), 0, row, 1, 1)
            grid.attach(widget, 1, row, 1, 1)
        grid.attach(Gtk.Label(label="Value", xalign=0), 0, 2, 1, 1)
        def select(*_):
            old = grid.get_child_at(1, 2)
            if old:
                self.sliders.pop(self.midi_index, None)
                old.destroy()
            index = self.midi.get((channel.get_value_as_int()-1, number.get_value_as_int()))
            if index is not None:
                self.midi_index = index
                slider = self.scale(index, "MIDI controller value")
                if self.bind_control:
                    self.bind_control(index, slider)
                grid.attach(slider, 1, 2, 1, 1)
                slider.show()
        channel.connect("value-changed", select)
        number.connect("value-changed", select)
        select()

    def restore(self, values):
        for i, param in enumerate(self.params):
            key = str(param.id)
            if key in self.saved or key in values:
                self.apply_parameter(i, values.get(key, param.initial))

    def apply_parameter(self, index, value):
        widget = self.sliders.get(index)
        if isinstance(widget, Gtk.ComboBoxText):
            widget.set_active(round(value*3))
        elif widget:
            widget.set_value(value)
        if abs(self.value(index)-value) > 1e-6:
            self.store(index, value)

    def reset(self, *_):
        self.restore({})
        self.save()
