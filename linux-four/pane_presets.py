"""Named snapshots of pane controls, shared between mic strips in the desk session."""
from copy import deepcopy
from gi.repository import Gtk


class PanePresets(Gtk.MenuButton):
    def __init__(self, bank, save, pane):
        super().__init__(label="Presets…")
        self.bank, self.save = bank, save
        self.bindings = {}
        self.groups = {"pane": "Whole pane"}
        self.get_accessible().set_name(pane + " presets")
        self.set_tooltip_text("Save or recall settings for this pane, an effect, or one control")
        pop = Gtk.Popover.new(self)
        self.set_popover(pop)
        grid = Gtk.Grid(margin=8, column_spacing=6, row_spacing=5)
        pop.add(grid)
        self.scope, self.names = Gtk.ComboBoxText(), Gtk.ComboBoxText()
        self.name = Gtk.Entry(placeholder_text="Name this preset")
        for row, (title, control) in enumerate((("Controls", self.scope), ("Saved presets", self.names),
                                                ("Name", self.name))):
            grid.attach(Gtk.Label(label=title, xalign=0), 0, row, 1, 1)
            grid.attach(control, 1, row, 2, 1)
            control.get_accessible().set_name(pane + " preset " + title.lower())
        self.recall = Gtk.Button(label="Recall")
        self.write = Gtk.Button(label="Save preset")
        grid.attach(self.recall, 1, 3, 1, 1)
        grid.attach(self.write, 2, 3, 1, 1)
        self.recall.connect("clicked", self.apply)
        self.write.connect("clicked", self.store)
        self.name.connect("changed", self.update_save)
        self.name.connect("activate", lambda _: self.store() if self.write.get_sensitive() else None)
        self.scope.connect("changed", lambda _: self.refill())
        self.names.connect("changed", self.selected)
        pop.connect("show", self.open)
        grid.show_all()

    def bind(self, key, label, group, get, set_value, *, group_label=None, in_pane=True):
        if group:
            self.groups[group] = group_label or group.title()
        self.bindings[key] = (label, group, get, set_value, in_pane)

    def attach_control(self, key, widget):
        def show(*_):
            self.get_popover().popup()
            self.scope.set_active_id(key)
            return True
        widget.connect("popup-menu", show)
        widget.connect("button-press-event", lambda w, e: show() if e.button == 3 else False)

    def keys(self, scope):
        if scope in self.bindings:
            return [scope]
        return [k for k, (_, group, _, _, in_pane) in self.bindings.items()
                if (scope == "pane" and in_pane) or (group == scope and in_pane)]

    def open(self, *_):
        scope = self.scope.get_active_id() or "pane"
        self.scope.remove_all()
        for key, label in self.groups.items():
            self.scope.append(key, label)
        for key, (label, group, *_) in self.bindings.items():
            if key != group:
                prefix = self.groups[group] + " · " if group else ""
                self.scope.append(key, prefix + label)
        self.scope.set_active_id(scope)

    def refill(self, name=None):
        self.names.remove_all()
        for key in sorted(self.bank.get(self.scope.get_active_id(), {}), key=str.casefold):
            self.names.append(key, key)
        if not name or not self.names.set_active_id(name):
            self.names.set_active(0)
        self.selected()

    def selected(self, *_):
        name = self.names.get_active_id()
        self.name.set_text(name or "")
        self.recall.set_sensitive(name is not None)
        self.update_save()

    def update_save(self, *_):
        name = self.name.get_text().strip()
        self.write.set_sensitive(bool(name and self.scope.get_active_id()))
        self.write.set_label("Update preset" if name in self.bank.get(self.scope.get_active_id(), {}) else "Save preset")

    def store(self, *_):
        scope, name = self.scope.get_active_id(), self.name.get_text().strip()
        if scope and name:
            self.bank.setdefault(scope, {})[name] = {k: deepcopy(self.bindings[k][2]()) for k in self.keys(scope)}
            self.refill(name)
            self.save()

    def apply(self, *_):
        scope, name = self.scope.get_active_id(), self.names.get_active_id()
        values = self.bank.get(scope, {}).get(name)
        if values is not None:
            for key in self.keys(scope):
                if key in values:
                    self.bindings[key][3](deepcopy(values[key]))
            self.save()
            self.get_popover().popdown()
