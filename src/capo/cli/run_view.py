"""
run_view.py — the single owner of the live run view (real terminals only).

One full-screen prompt_toolkit Application owns the screen for the whole run:
streamed log lines fill a scrollable pane on top, and a persistent one-line input
bar (with slash-command completion + history suggestions) stays pinned at the
bottom so a command can be typed at any moment while logs keep flowing. This
replaces the previous patch_stdout + Rich-console-print mix, where two writers
shared one terminal — which produced the duplicated "live" banner, raw-ANSI
leakage (\x1b[…m showing as text), and a prompt that only woke up on Ctrl+D. Here
prompt_toolkit is the sole renderer; Rich is used only to turn each log line into
an ANSI string the app prints verbatim, so the existing colour scheme is kept.

Ownership rules (the whole point):
  • exactly one terminal writer during a run — this Application;
  • the log streamers feed lines in via RunConsole.feed (sink), they never print;
  • commands (/health, /status, /abort, …) render through the shared interaction
    router (run_console.dispatch_run_command), captured into the log pane.

Scrolling: the mouse wheel / trackpad scroll the log history (the natural gesture,
and the only one that works without dedicated keys on a laptop), as do Shift+↑/↓
(line) and PgUp/PgDn (page); scrolling back to the bottom — or typing a command —
resumes live follow. Wheel scrolling needs mouse reporting on (mouse_support=True),
which means terminal text selection now goes through the usual modifier (Option-
or Fn-drag on macOS terminals); CAPO_RUN_UI=plain turns the whole view off if a
terminal misbehaves. Detach: Ctrl+D or /quit leave the full-screen view but keep
streaming logs in plaintext (Rich → the real terminal, no patch_stdout, so no
ANSI leak); Ctrl+C then stops the run.

This is constructed only in a real TTY. app.py keeps the plain streaming path for
non-terminals (tests, pipes, CI), and CAPO_RUN_UI=plain forces that plain path in
a real terminal too (escape hatch if a specific terminal misbehaves).
"""

from __future__ import annotations

import shutil
import sys
import threading
from collections import deque
from contextlib import contextmanager
from pathlib import Path

from prompt_toolkit import Application
from prompt_toolkit.application import get_app
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
from prompt_toolkit.key_binding.defaults import load_key_bindings
from prompt_toolkit.layout import Float, FloatContainer, HSplit, Layout, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.layout.processors import BeforeInput
from prompt_toolkit.mouse_events import MouseEventType
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.table import Table
from rich.text import Text

from .colors import NOISE, PURPLE_0, THEME, console
from .run_console import (
    SLASH_COMMANDS,
    RunContext,
    SlashCommandLexer,
    _arm_abort,
    _do_abort,
    dispatch_run_command,
    fmt_elapsed,
)

# rows reserved by the fixed chrome (banner + separator + input) plus one row of
# safety margin, so the rendered slice always fits and the newest line stays
# visible above the input bar (which an HSplit pins, so it can never be pushed
# off-screen regardless of log volume).
_RESERVED_ROWS = 4


class _LogControl(FormattedTextControl):
    """Logs-pane control that routes wheel / trackpad scroll to our own scroller.

    The pane is pre-sliced to exactly fit (see RunConsole._render_logs), so the
    content never overflows its window and prompt_toolkit's built-in scroll never
    fires. We intercept SCROLL_UP/DOWN here and step our manual scroll instead;
    every other mouse event falls through (NotImplemented) so clicks, focus and
    text don't change behaviour."""

    def __init__(self, get_text, on_scroll) -> None:
        super().__init__(get_text)
        self._on_scroll = on_scroll

    def mouse_handler(self, mouse_event):
        if mouse_event.event_type == MouseEventType.SCROLL_UP:
            self._on_scroll(-1)
            return None
        if mouse_event.event_type == MouseEventType.SCROLL_DOWN:
            self._on_scroll(+1)
            return None
        return NotImplemented


class RunConsole:
    """Full-screen live run view. Construct in a real terminal only.

    Lifecycle (driven by app.py._launch):
      rc = RunConsole(...); start streamers with sink=rc.feed; rc.run() in a
      daemon thread (blocks until the run ends or the user detaches); on terminal
      state, set the shared stop_event → the view exits and the summary prints.
    """

    def __init__(
        self,
        run_id: str,
        run_dir: Path,
        runs_root: Path,
        stop_event: threading.Event,
        start_time: float,
        abort_event: threading.Event | None = None,
    ) -> None:
        self._run_id = run_id
        self._run_dir = run_dir
        self._stop = stop_event
        self._start = start_time
        self._abort = abort_event or threading.Event()

        # log buffer: one entry per visual line of pre-rendered ANSI. capped so a
        # very long run never grows memory without bound.
        self._lines: deque[str] = deque(maxlen=5000)
        self._lock = threading.Lock()  # guards _lines + the render console

        # scroll state: None = follow the live tail; an int = a frozen top line
        # index the user scrolled back to (new lines don't yank the view).
        self._scroll_top: int | None = None

        # detach state: once detached, logs stream in plaintext to the real
        # terminal (see _enter_plaintext_mode) instead of into the pane.
        self._detached = False
        self._plaintext = False
        self._plain_console: Console | None = None

        # fixed render width, sampled once (resize mid-run is tolerated, not
        # tracked: the input bar stays pinned either way, only wrapping drifts).
        self._cols = max(40, shutil.get_terminal_size((100, 30)).columns)
        self._render = Console(
            theme=THEME, highlight=False, force_terminal=True,
            color_system="truecolor", width=self._cols,
        )

        # the input bar: slash-command completion + history suggestions (same feel
        # as the chat front door), routed through the shared interaction router
        # with command output captured into the log pane. allow_nested_ui is False:
        # no nested /config editor while this full-screen app is active.
        self._ctx = RunContext(
            run_id=run_id, run_dir=run_dir, runs_root=runs_root,
            stop_event=stop_event, start_time=start_time, abort_event=self._abort,
            capture=self._capture_to_pane, allow_nested_ui=False,
        )
        self._buffer = Buffer(
            multiline=False,
            completer=WordCompleter(SLASH_COMMANDS, sentence=True),
            complete_while_typing=True,
            auto_suggest=AutoSuggestFromHistory(),
            history=InMemoryHistory(),
            accept_handler=self._on_accept,
        )
        self._app: Application | None = None

    # === log ingestion (called from the streamer threads) ===

    def feed(self, renderable: object) -> None:
        """Sink for stream_log. While the view owns the screen, render one Rich
        renderable to ANSI and append it; after a detach, print straight to the
        real terminal in plaintext (Rich, coloured, no patch_stdout → no leak).

        A folded log entry may contain internal newlines; we store one deque
        entry per visual line so the slice-render fits the pane exactly."""
        if self._plaintext and self._plain_console is not None:
            self._plain_console.print(renderable, highlight=False)
            return
        with self._lock:
            with self._render.capture() as cap:
                self._render.print(renderable)
            self._lines.extend(cap.get().rstrip("\n").split("\n"))
        self._invalidate()

    def feed_ansi(self, text: str) -> None:
        """Append already-rendered ANSI text (captured command output)."""
        if not text:
            return
        with self._lock:
            self._lines.extend(text.rstrip("\n").split("\n"))
        self._invalidate()

    @contextmanager
    def _capture_to_pane(self):
        """Capture a block of shared-console prints and show them in the pane.

        Runs only on the app thread (command handling), so using the shared Rich
        console's capture() is safe here — log rendering uses the separate render
        console under the lock."""
        with console.capture() as cap:
            yield
        self.feed_ansi(cap.get())

    # === rendering (called on the app thread) ===

    def _avail_rows(self) -> int:
        rows = self._app.output.get_size().rows if self._app else 30
        return max(1, rows - _RESERVED_ROWS)

    def _render_logs(self) -> ANSI:
        avail = self._avail_rows()
        with self._lock:
            total = len(self._lines)
            if self._scroll_top is None:
                top = max(0, total - avail)            # follow the live tail
            else:
                top = max(0, min(self._scroll_top, max(0, total - 1)))
            window = list(self._lines)[top: top + avail]
        return ANSI("\n".join(window))

    def _render_banner(self) -> ANSI:
        """Top bar: live dot + run id on the left, running clock on the right."""
        import time

        grid = Table.grid(expand=True)
        grid.add_column(justify="left", ratio=1)
        grid.add_column(justify="right")
        grid.add_row(
            Text.from_markup(
                f"  [phase.run]●[/] [brand]live[/]  "
                f"[muted]· {self._run_id} · streaming run logs · type a command anytime[/]"
            ),
            Text.from_markup(f"[muted]{fmt_elapsed(time.monotonic() - self._start)}  [/]"),
        )
        return self._to_ansi(grid)

    def _render_sep(self) -> ANSI:
        """Separator rule, or a hint when the user has scrolled off the live tail."""
        if self._scroll_top is None:
            return self._to_ansi(Text("─" * self._cols, style="muted"))
        hint = " ⏸ scrolled — wheel / Shift+↑↓ / PgUp·PgDn · scroll to bottom for live "
        line = (hint + "─" * max(0, self._cols - len(hint)))[: self._cols]
        return self._to_ansi(Text(line, style="accent"))

    def _to_ansi(self, renderable: object) -> ANSI:
        with self._lock:
            with self._render.capture() as cap:
                self._render.print(renderable)
            return ANSI(cap.get().rstrip("\n"))

    # === input + scrolling ===

    def _on_accept(self, buff: Buffer) -> bool:
        action = dispatch_run_command(buff.text, self._ctx)  # the one shared router
        if action == "abort":
            self._stop.set()        # router already SIGINT'd the orchestrator
            self._exit()
        elif action == "detach":
            self._detach()
        elif action == "narrow":
            return True              # ambiguous prefix — keep the line so they type more
        else:
            self._scroll_top = None  # typing a command snaps back to the live tail
        return False                 # falsy → clear the input line for the next one

    def _scroll(self, direction: int, lines: int | None = None) -> None:
        """Scroll the log pane up/down. lines=None steps ~one page (PgUp/PgDn);
        a small lines value gives line-grained steps (wheel, Shift+↑/↓). Scrolling
        back down past the bottom resumes live follow."""
        avail = self._avail_rows()
        step = lines if lines is not None else max(1, avail - 1)
        with self._lock:
            total = len(self._lines)
            bottom_top = max(0, total - avail)
            cur = bottom_top if self._scroll_top is None else self._scroll_top
            new = cur + direction * step
            self._scroll_top = None if new >= bottom_top else max(0, new)
        self._invalidate()

    def _wheel(self, direction: int) -> None:
        """Mouse/trackpad scroll: a few lines per notch (gentler than a page)."""
        self._scroll(direction, lines=3)

    # === app construction / lifecycle ===

    def _key_bindings(self) -> KeyBindings:
        kb = KeyBindings()

        @kb.add("pageup")
        def _(_e) -> None:
            self._scroll(-1)

        @kb.add("pagedown")
        def _(_e) -> None:
            self._scroll(+1)

        # line-grained scroll for laptops without PgUp/PgDn keys. the input bar is
        # single-line, so Shift+↑/↓ never means "extend selection" here — free to
        # reuse, and easy to reach on every keyboard.
        @kb.add("s-up")
        def _(_e) -> None:
            self._scroll(-1, lines=1)

        @kb.add("s-down")
        def _(_e) -> None:
            self._scroll(+1, lines=1)

        # eager=True so these win over the merged default bindings (whose Ctrl+C
        # raises KeyboardInterrupt and Ctrl+D sends EOF — both would kill the view).
        @kb.add("c-c", eager=True)
        def _(_e) -> None:
            # Ctrl+C shares the /abort arm → confirm path: first press asks, a
            # second commits (so does typing y).
            if self._ctx.abort_pending:
                self._ctx.abort_pending = False
                _do_abort(self._ctx)   # sets events + SIGINT; app.py owns teardown
                self._exit()           # leave the full-screen view
            else:
                _arm_abort(self._ctx)

        @kb.add("c-d", eager=True)
        def _(_e) -> None:  # Ctrl+D detaches; the run keeps going (plaintext stream).
            self._detach()

        return kb

    def _markup_ansi(self, markup: str) -> str:
        with console.capture() as cap:  # app thread only
            console.print(markup, end="")
        return cap.get()

    def _style(self) -> Style:
        return Style.from_dict(
            {"cmd": f"{PURPLE_0} bold", "prompt": f"{PURPLE_0} bold", "sep": NOISE}
        )

    def _build_app(self) -> Application:
        banner = Window(FormattedTextControl(self._render_banner), height=1, dont_extend_height=True)
        logs = Window(_LogControl(self._render_logs, self._wheel), wrap_lines=False, always_hide_cursor=True)
        sep = Window(FormattedTextControl(self._render_sep), height=1, dont_extend_height=True)
        inp = Window(
            BufferControl(
                buffer=self._buffer,
                lexer=SlashCommandLexer(),
                input_processors=[BeforeInput("  › ", style="class:prompt")],
            ),
            height=1, dont_extend_height=True,
        )
        # a completion menu floats over the body (slash-command dropdown).
        root = FloatContainer(
            content=HSplit([banner, logs, sep, inp]),
            floats=[Float(xcursor=True, ycursor=True,
                          content=CompletionsMenu(max_height=8, scroll_offset=1))],
        )
        layout = Layout(root, focused_element=inp)
        # merge prompt_toolkit's defaults (Tab-completion, completion-menu nav,
        # emacs line editing) so the input bar matches the chat front door; our
        # own bindings come last so they take precedence (PgUp/PgDn, Ctrl+C/D).
        keys = merge_key_bindings([load_key_bindings(), self._key_bindings()])
        return Application(
            layout=layout,
            key_bindings=keys,
            style=self._style(),
            full_screen=True,
            refresh_interval=0.5,  # ticks the clock + flushes streamed logs
            mouse_support=True,    # wheel/trackpad scroll (Option-/Fn-drag still selects text)
        )

    def _invalidate(self) -> None:
        app = self._app  # invalidate() is thread-safe; redraw from any thread.
        if app is not None:
            try:
                app.invalidate()
            except Exception:
                pass

    def _exit(self) -> None:
        app = self._app  # called on the app thread (key handler / watcher).
        if app is not None and app.is_running:
            app.exit()

    def _detach(self) -> None:
        self._detached = True  # run() switches to plaintext once the screen restores
        self._exit()

    def _enter_plaintext_mode(self) -> None:
        """After detach: the full-screen app has exited and the real screen is
        restored. Stream the rest of the run in plaintext, writing to the real
        terminal (sys.__stdout__) so we bypass the run-time stdout redirect and
        never touch patch_stdout — no ANSI leak. Ctrl+C still stops the run."""
        out = Console(theme=THEME, highlight=False, file=sys.__stdout__)
        out.print(
            f"\n  [brand]Detached[/]  [muted]— the run keeps going; logs continue below.[/]\n"
            f"  [muted]stop the run: [/][cmd]Ctrl+C[/]    "
            f"[muted]·  health: [/][cmd]capo health {self._run_id}[/]\n"
        )
        with self._lock:
            self._plain_console = out
            self._plaintext = True

    async def _watch_stop(self) -> None:
        import asyncio

        while not self._stop.is_set():
            await asyncio.sleep(0.2)
        self._exit()

    def _pre_run(self) -> None:
        get_app().create_background_task(self._watch_stop())

    def run(self) -> None:
        """Block running the full-screen view until the run ends or the user
        detaches. Any UI-layer crash sets stop_event and returns rather than
        killing the run: the orchestrator runs in the main thread, and the summary
        box still prints afterwards."""
        import asyncio

        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())
        self._app = self._build_app()
        try:
            # handle_sigint=False: signal handlers only install on the main thread,
            # and this runs in a daemon thread (SIGINT reaches the orchestrator).
            self._app.run(pre_run=self._pre_run, handle_sigint=False)
        except KeyboardInterrupt:
            self._detached = True  # safety net: a stray interrupt detaches, never kills
        except Exception:
            self._stop.set()
        # detach (not a run-end / abort) → keep streaming in plaintext.
        if self._detached and not self._stop.is_set():
            self._enter_plaintext_mode()
