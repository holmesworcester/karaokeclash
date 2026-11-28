"""Keyboard-accessible GTK range with a vertically draggable rotary face."""

import math
import gi

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
from gi.repository import Gtk, Gdk, Atk


class Dial(Gtk.Scale):
    def __init__(self, name, lo, hi, value, default, unit, *, step=1, formatter=None):
        adjustment = Gtk.Adjustment(
            value=value, lower=lo, upper=hi, step_increment=step, page_increment=min(hi-lo, 5*step)
        )
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, adjustment=adjustment)
        self.set_draw_value(False)
        self.set_size_request(49, 46)
        self.set_can_focus(True)
        self.default, self.unit, self.origin = default, unit, None
        self.formatter = formatter
        self.set_digits(max(0, math.ceil(-math.log10(step))))
        self.get_accessible().set_name(name)
        self.get_accessible().set_role(Atk.Role.SLIDER)
        self.set_tooltip_text(
            name + " · Drag up or down. Shift for fine control. Double-click to reset."
        )
        self.add_events(
            Gdk.EventMask.POINTER_MOTION_MASK
            | Gdk.EventMask.BUTTON_PRESS_MASK
            | Gdk.EventMask.BUTTON_RELEASE_MASK
        )
        self.connect("value-changed", lambda _: self.queue_draw())

    def do_button_press_event(self, event):
        if event.button != 1:
            return False
        self.grab_focus()
        if event.type == Gdk.EventType.DOUBLE_BUTTON_PRESS:
            self.set_value(self.default)
            self.origin = None
        else:
            self.origin = (event.y_root, self.get_value())
            self.grab_add()
        return True

    def do_motion_notify_event(self, event):
        if self.origin is None:
            return False
        y, v = self.origin
        bounds = self.get_adjustment()
        amount = (y - event.y_root) / 140 * (bounds.get_upper() - bounds.get_lower())
        step = bounds.get_step_increment()
        moved = v + amount * (0.12 if event.state & Gdk.ModifierType.SHIFT_MASK else 1)
        self.set_value(bounds.get_lower() + round((moved - bounds.get_lower()) / step) * step)
        return True

    def do_button_release_event(self, event):
        if self.has_grab():
            self.grab_remove()
        self.origin = None
        return True

    def do_draw(self, ctx):
        w = self.get_allocated_width()
        x, y = w / 2, 17
        value = self.get_value()
        bounds = self.get_adjustment()
        ratio = (value - bounds.get_lower()) / (bounds.get_upper() - bounds.get_lower())
        start, extent = 0.75 * math.pi, 1.5 * math.pi
        ctx.set_line_width(3)
        ctx.set_line_cap(1)
        ctx.set_source_rgb(0.76, 0.77, 0.75)
        ctx.arc(x, y, 12, start, start + extent)
        ctx.stroke()
        ctx.set_source_rgb(0.19, 0.36, 0.36)
        ctx.arc(x, y, 12, start, start + extent * ratio)
        ctx.stroke()
        a = start + extent * ratio
        ctx.move_to(x + 4 * math.cos(a), y + 4 * math.sin(a))
        ctx.line_to(x + 9 * math.cos(a), y + 9 * math.sin(a))
        ctx.stroke()
        txt = (
            ["1/32", "1/16", "1/8", "1/8.", "1/4", "1/2"][round(value)]
            if self.unit == "beat"
            else f"{value:.0f}{self.unit}"
        )
        layout = self.create_pango_layout(self.formatter(value) if self.formatter else txt)
        sx, sy = layout.get_pixel_size()
        from gi.repository import PangoCairo

        ctx.move_to((w - sx) / 2, 30)
        color = self.get_style_context().get_color(Gtk.StateFlags.NORMAL)
        ctx.set_source_rgb(color.red, color.green, color.blue)
        PangoCairo.show_layout(ctx, layout)
        if self.has_visible_focus():
            Gtk.render_focus(
                self.get_style_context(),
                ctx,
                1,
                0,
                w - 2,
                self.get_allocated_height() - 1,
            )
        return True
