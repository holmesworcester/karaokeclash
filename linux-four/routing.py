"""Song assignment controls shared by the transport and the mic strips."""
import cairo
import gi

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
from gi.repository import Gdk, Gtk, Pango

from playlist import Playlist, searchable

SONG = Gtk.TargetEntry.new("application/x-karaoke-song", Gtk.TargetFlags.SAME_APP, 0)


class SongLibrary(Playlist):
    def __init__(self):
        super().__init__("Song library", "Search songs or artists", lambda key: None)

        self.request_rows = {}

    def show_requests(self, records, action):
        # Only user-visible changes rebuild a row, preserving an expanded prompt
        # while the worker moves between preparation stages.
        records = {job: {k: value[k] for k in ("source", "notes", "state")}
                   for job, value in records.items()}
        for job, row in list(self.request_rows.items()):
            if job not in records or row.record != records[job]:
                self.rows.remove(row)
                del self.request_rows[job]
        for job, record in records.items():
            if job in self.request_rows:
                continue
            state = record["state"]
            status = {"queued": "In progress", "running": "In progress", "failed": "Failed",
                      "cancelled": "Cancelled", "cancelling": "Stopping", "invalid": "Unavailable"}[state]
            row = Gtk.ListBoxRow(selectable=False, activatable=False)
            row.record = record
            box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=1)
            heading = Gtk.Box(spacing=5)
            title = Gtk.Label(label=record["source"], xalign=0, ellipsize=Pango.EllipsizeMode.END, width_chars=1)
            heading.pack_start(title, True, True, 0)
            badge = Gtk.Label(label=status)
            badge.get_style_context().add_class("dim-label")
            heading.pack_end(badge, False, False, 0)
            box.add(heading)
            details = Gtk.Box(spacing=3)
            notes = record["notes"]
            if notes:
                prompt = Gtk.Expander()
                preview = Gtk.Label(label=notes, xalign=0, ellipsize=Pango.EllipsizeMode.END,
                                    width_chars=1, max_width_chars=18)
                prompt.set_label_widget(preview)
                prompt.set_label_fill(True)
                full = Gtk.Label(label=notes, xalign=0, wrap=True, selectable=True, max_width_chars=23)
                full.set_line_wrap_mode(Pango.WrapMode.WORD_CHAR)
                prompt.add(full)
                prompt.connect("notify::expanded", lambda w, _, p=preview, n=notes:
                               p.set_text("Prompt" if w.get_expanded() else n))
                details.pack_start(prompt, True, True, 0)
            if state not in ("invalid", "cancelling"):
                again = state in ("failed", "cancelled")
                caption = "Retry" if again else "Cancel"
                control = Gtk.Button.new_from_icon_name("view-refresh-symbolic" if again else "window-close-symbolic",
                                                        Gtk.IconSize.MENU)
                control.set_relief(Gtk.ReliefStyle.NONE)
                control.set_valign(Gtk.Align.START)
                control.get_accessible().set_name(caption)
                control.set_tooltip_text(caption + " import")
                control.connect("clicked", lambda _, j=job, st=state: action(j, st))
                details.pack_end(control, False, False, 0)
            box.add(details)
            description = "\n".join(filter(None, (record["source"], status, notes)))
            row.get_accessible().set_name(description)
            row.terms = searchable(description)
            row.add(box)
            self.rows.insert(row, 0)
            self.request_rows[job] = row
            row.show_all()

    def show_catalog(self, songs, loops):
        entries = []
        for key in loops:
            song = songs[key]
            entries.append((key, song["title"], song.get("artist", ""), ""))
        self.refill(sorted(entries, key=lambda row: (row[1].casefold(), row[2].casefold())))
        for key in loops:
            request = songs[key].get("importRequest")
            if request:
                row = self.items[key]
                prompt = request["notes"]
                if prompt:
                    label = Gtk.Label(label=prompt, xalign=0, max_width_chars=24, ellipsize=Pango.EllipsizeMode.END)
                    row.get_child().get_child().add(label)
                    label.show()
                row.terms += " " + searchable(request["source"] + " " + prompt)
                row.set_tooltip_text("\n".join(filter(None, (row.get_accessible().get_name(), request["source"], prompt))))
        self.rows.invalidate_filter()

    def refill(self, entries):
        super().refill(entries)
        for key, row in self.items.items():
            # Imported song IDs can end in /full. Search the displayed song and artist.
            row.terms = searchable(row.get_accessible().get_name())
            source = Gtk.EventBox()
            content = row.get_child()
            row.remove(content)
            source.add(content)
            row.add(source)
            source.show_all()
            source.connect("button-press-event", lambda _, event, r=row: self.rows.select_row(r) or False)
            source.drag_source_set(Gdk.ModifierType.BUTTON1_MASK, [SONG], Gdk.DragAction.COPY)
            source.connect("drag-begin", self.show_drag_preview)
            source.connect("drag-data-get", self.drag_song, key)
        if self.words:
            self.filter()

    @staticmethod
    def show_drag_preview(source, context):
        # Render the song title and artist as they appear in the library.
        row = source.get_parent()
        width, height = row.get_allocated_width(), row.get_allocated_height()
        scale = row.get_scale_factor()
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width * scale, height * scale)
        surface.set_device_scale(scale, scale)
        cr = cairo.Context(surface)
        cr.set_source_rgb(0.98, 0.98, 0.96)
        cr.paint()
        row.draw(cr)
        cr.set_source_rgba(0.24, 0.33, 0.30, 0.75)
        cr.rectangle(0.5, 0.5, width - 1, height - 1)
        cr.set_line_width(1)
        cr.stroke()
        surface.set_device_offset(10 * scale, 8 * scale)
        Gtk.drag_set_icon_surface(context, surface)

    @staticmethod
    def drag_song(row, context, data, info, time, key):
        data.set(data.get_target(), 8, key.encode())


class SongTarget(Gtk.Button):
    def __init__(self, role, library, assign):
        super().__init__()
        self.role = role
        self.assign = assign
        self.caption = Gtk.Label(ellipsize=Pango.EllipsizeMode.END, max_width_chars=27)
        icons = {
            "Hear": "audio-input-microphone-symbolic",
            "See": "camera-video-symbolic",
            "Backing": "audio-x-generic-symbolic",
        }
        box = Gtk.Box(spacing=7, halign=Gtk.Align.CENTER)
        box.add(Gtk.Image.new_from_icon_name(icons[role], Gtk.IconSize.BUTTON))
        box.add(self.caption)
        self.add(box)
        style = Gtk.CssProvider()
        style.load_from_data(b"""
            button, button:hover, button:active {
                background-image: none;
                background-color: rgba(66, 95, 89, 0.07);
                border: 2px dashed #758a85;
                border-radius: 3px;
                box-shadow: none;
                text-shadow: none;
                min-width: 105px;
                min-height: 38px;
                padding: 4px 12px;
            }
            button:drop(active) {
                background-color: #c3e4d9;
                border-color: #245f4c;
            }
        """)
        self.get_style_context().add_provider(style, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
        self.show_assignment(None)
        self.connect("clicked", lambda *_: assign(library.current))
        self.drag_dest_set(Gtk.DestDefaults.ALL, [SONG], Gdk.DragAction.COPY)
        self.connect("drag-data-received", self.drop)

    def show_assignment(self, title):
        title = title or "Drag here"
        self.caption.set_text(title)
        self.get_accessible().set_name(f"{self.role} · {title}")
        self.set_tooltip_text(f"{self.role} · {title}\nDrop a song or select one in the library and click here.")

    def drop(self, widget, context, x, y, data, info, time):
        key = bytes(data.get_data() or b"").decode()
        Gtk.drag_finish(context, bool(self.assign(key)), False, time)
