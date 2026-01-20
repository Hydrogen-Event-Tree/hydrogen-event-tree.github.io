import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any


class ReusableHTTPServer(HTTPServer):
    allow_reuse_address = True


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _load_events(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def run(
    events_path: str = "events.json",
    index_path: str = "index.html",
    host: str = "127.0.0.1",
    port: int = 4000,
    max_tries: int = 5,
) -> None:
    events = _load_events(events_path)
    index_html = _read_text(index_path)

    class Handler(BaseHTTPRequestHandler):
        def _respond(self, code: int, content_type: str, body: str) -> None:
            data = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # type: ignore[override]
            if self.path in ("/", "/index.html"):
                self._respond(200, "text/html; charset=utf-8", index_html)
                return

            if self.path == "/events.json":
                try:
                    payload = Path(events_path).read_text(encoding="utf-8")
                except FileNotFoundError:
                    self._respond(404, "text/plain; charset=utf-8", "events.json not found.")
                except OSError as exc:
                    self._respond(500, "text/plain; charset=utf-8", f"Unable to read events file: {exc}")
                else:
                    self._respond(200, "application/json; charset=utf-8", payload)
                return

            self._respond(404, "text/plain; charset=utf-8", "Not found.")

        def log_message(self, format: str, *args: Any) -> None:  # type: ignore[override]
            return  # Silence default logging.

    server = None
    bound_port = port
    last_error: Exception | None = None
    for attempt in range(max_tries):
        try_port = port + attempt
        try:
            server = ReusableHTTPServer((host, try_port), Handler)
            bound_port = try_port
            break
        except OSError as exc:
            last_error = exc
            continue

    if server is None:
        raise last_error if last_error else RuntimeError("Unable to start server.")

    print(
        f"Dashboard available at http://{host}:{bound_port} "
        f"(serving {events_path} with {len(events)} events via {index_path})"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    run()
