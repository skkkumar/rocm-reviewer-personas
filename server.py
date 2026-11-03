#!/usr/bin/env python3
"""Local server for the reviewer app.

Serves index.html and runs "Review as" / "Reply to review" through your Claude Code CLI (`claude -p`),
so the page works end to end without claude.ai and without an API key.

  python server.py              # then open http://127.0.0.1:8765/?pr=59280
  python server.py --port 9000
"""
import argparse, http.server, json, os, shutil, subprocess, tempfile

ROOT = os.path.dirname(os.path.abspath(__file__))
CLAUDE = shutil.which("claude")
# Empty scratch dir so the review is not influenced by any CLAUDE.md or project settings.
SCRATCH = tempfile.mkdtemp(prefix="rocm-review-")


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)

    def log_message(self, fmt, *args):
        if "/api/" in (args[0] if args else ""):
            super().log_message(fmt, *args)

    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        if self.path == "/api/health":
            return self._json(200, {"claude": bool(CLAUDE)})
        return super().do_GET()

    def do_POST(self):
        if self.path != "/api/claude":
            return self._json(404, {"error": "not found"})
        if not CLAUDE:
            return self._json(503, {"error": "claude CLI not found on PATH"})
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        cmd = [CLAUDE, "-p", "--output-format", "stream-json", "--verbose", "--include-partial-messages",
               "--tools", "", "--no-session-persistence", "--strict-mcp-config"]
        if req.get("model"):
            cmd += ["--model", req["model"]]
        proc = subprocess.Popen(cmd, cwd=SCRATCH, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, encoding="utf-8", errors="replace")
        proc.stdin.write(req.get("prompt", "")); proc.stdin.close()
        self.send_response(200); self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Cache-Control", "no-cache"); self.end_headers()
        streamed = False
        try:
            for line in proc.stdout:
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                text = None
                if ev.get("type") == "stream_event":
                    d = ev.get("event", {}).get("delta", {})
                    if d.get("type") == "text_delta":
                        text = d.get("text"); streamed = True
                elif ev.get("type") == "result":
                    if ev.get("is_error"):
                        text = "\n\n**Claude CLI error:** " + str(ev.get("result") or ev.get("subtype"))
                    elif not streamed:
                        text = ev.get("result")
                if text:
                    self.wfile.write(text.encode("utf-8")); self.wfile.flush()
            proc.wait()
            if proc.returncode and not streamed:
                err = proc.stderr.read().strip()[-800:]
                self.wfile.write(f"\n\n**Claude CLI exited with {proc.returncode}.** {err}".encode("utf-8"))
        except (BrokenPipeError, ConnectionResetError):
            pass  # browser pressed Stop
        finally:
            if proc.poll() is None:
                proc.kill()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"Reviewer app on http://127.0.0.1:{a.port}/  (Claude CLI: {CLAUDE or 'not found - LLM modes will offer copy-prompt'})")
    srv.serve_forever()


if __name__ == "__main__":
    main()
