from urllib.parse import urlparse, parse_qs

def extract_youtube_id(url: str) -> str | None:
    try:
        parsed_url = urlparse(url)
        if parsed_url.hostname in ("www.youtube.com", "youtube.com"):
            query = parse_qs(parsed_url.query)
            return query.get("v", [None])[0]
        elif parsed_url.hostname == "youtu.be":
            return parsed_url.path.lstrip("/")
    except Exception:
        pass
    return None

