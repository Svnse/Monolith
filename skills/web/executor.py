"""Web fetch executor — fetches URL content with optional HTML-to-text extraction.

Safety:
- Only http/https schemes are allowed.
- Hostnames are resolved at request time and blocked if ANY resolved address
  is private/loopback/link-local/multicast/unspecified. This prevents SSRF
  to internal services like 127.0.0.1, 169.254.169.254, 10.x, etc.
"""
import ipaddress
import socket
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser


_MAX_BYTES = 1_000_000  # 1 MB cap on raw download
_DEFAULT_TIMEOUT = 15
_UA = "Monolith/0.1 (web tool)"
_ALLOWED_SCHEMES = ("http", "https")


def _coerce_int(value, default, lo, hi):
    try:
        v = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


def _is_safe_url(url):
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception as exc:
        return False, f"invalid URL ({exc})"
    if parsed.scheme not in _ALLOWED_SCHEMES:
        return False, f"scheme '{parsed.scheme}' not allowed (http/https only)"
    if not parsed.hostname:
        return False, "no hostname"
    hostname = parsed.hostname
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        return False, f"hostname resolution failed ({exc})"
    for info in infos:
        ip_str = info[4][0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_unspecified or ip.is_multicast or ip.is_reserved):
            return False, f"hostname '{hostname}' resolves to non-public IP {ip_str}"
    return True, ""


class _TextExtractor(HTMLParser):
    _BLOCK_TAGS = {"p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "section", "article"}
    _SKIP_TAGS = {"script", "style", "head", "noscript", "iframe"}

    def __init__(self):
        super().__init__()
        self.parts = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        t = tag.lower()
        if t in self._SKIP_TAGS:
            self._skip_depth += 1
        if t in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        t = tag.lower()
        if t in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        if t in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._skip_depth == 0:
            self.parts.append(data)

    def get_text(self):
        text = "".join(self.parts)
        lines = []
        for line in text.split("\n"):
            cleaned = " ".join(line.split())
            if cleaned:
                lines.append(cleaned)
        return "\n".join(lines)


def _fetch(url, timeout):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": _UA,
            "Accept": "*/*",
            "Accept-Encoding": "identity",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        content_type = resp.headers.get("Content-Type", "")
        body = resp.read(_MAX_BYTES + 1)
        truncated = len(body) > _MAX_BYTES
        if truncated:
            body = body[:_MAX_BYTES]
        final_url = resp.geturl()
    return body, content_type, truncated, final_url


def run(cmd: dict, _ctx) -> str:
    verb = str(cmd.get("verb", "")).strip().lower() or "text"
    url = str(cmd.get("url", "")).strip()
    if not url:
        return "[web: no url provided]"
    if verb not in ("text", "fetch"):
        return f"[web: unknown verb '{verb}' - expected text/fetch]"

    ok, reason = _is_safe_url(url)
    if not ok:
        return f"[web: blocked - {reason}]"

    timeout = _coerce_int(cmd.get("timeout", _DEFAULT_TIMEOUT), _DEFAULT_TIMEOUT, 1, 60)
    max_chars = _coerce_int(cmd.get("max_chars", 4000), 4000, 200, 50000)

    try:
        body, content_type, truncated, final_url = _fetch(url, timeout)
    except urllib.error.HTTPError as exc:
        return f"[web: HTTP {exc.code} {exc.reason} for {url}]"
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        return f"[web: connection error - {reason}]"
    except socket.timeout:
        return f"[web: timed out after {timeout}s]"
    except Exception as exc:
        return f"[web: error - {exc}]"

    encoding = "utf-8"
    ct_lower = content_type.lower()
    if "charset=" in ct_lower:
        try:
            encoding = ct_lower.split("charset=", 1)[1].split(";", 1)[0].strip() or "utf-8"
        except Exception:
            encoding = "utf-8"
    try:
        text = body.decode(encoding, errors="replace")
    except LookupError:
        text = body.decode("utf-8", errors="replace")

    if verb == "text":
        is_html = "html" in ct_lower or text.lstrip()[:200].lower().startswith(("<!doctype", "<html", "<!--"))
        if is_html:
            parser = _TextExtractor()
            try:
                parser.feed(text)
                text = parser.get_text()
            except Exception:
                pass

    truncated_note = ""
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n... [text truncated at {max_chars} chars]"
        truncated_note = ", body-truncated"
    elif truncated:
        truncated_note = ", body-truncated"

    redirect_note = ""
    if final_url and final_url != url:
        redirect_note = f", final_url={final_url}"

    return f"[web: {verb} {url}{redirect_note}{truncated_note}, content-type: {content_type}]\n{text}"
