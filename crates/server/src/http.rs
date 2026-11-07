//! Just enough HTTP/1.1 for a local JSON + SSE API: one request per connection
//! (`Connection: close`), `Content-Length` bodies only.

use std::io::{self, BufRead, BufReader, Read, Write};

const MAX_HEADER_BYTES: usize = 16 * 1024;
pub const MAX_BODY_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug)]
pub struct Request {
    pub method: String,
    pub path: String,
    pub body: Vec<u8>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum ParseError {
    Malformed(&'static str),
    TooLarge,
    /// The peer closed before sending a request line.
    Closed,
}

pub fn read_request(stream: impl Read) -> Result<Request, ParseError> {
    let mut r = BufReader::new(stream);
    let mut header_bytes = 0;
    let mut line = String::new();
    let mut next_line = |r: &mut BufReader<_>, line: &mut String| -> Result<usize, ParseError> {
        line.clear();
        let n = r
            .read_line(line)
            .map_err(|_| ParseError::Malformed("unreadable request"))?;
        header_bytes += n;
        if header_bytes > MAX_HEADER_BYTES {
            return Err(ParseError::TooLarge);
        }
        Ok(n)
    };
    if next_line(&mut r, &mut line)? == 0 {
        return Err(ParseError::Closed);
    }
    let mut parts = line.split_whitespace();
    let (Some(method), Some(target)) = (parts.next(), parts.next()) else {
        return Err(ParseError::Malformed("bad request line"));
    };
    let method = method.to_owned();
    // Query strings carry nothing this API uses.
    let path = target.split('?').next().unwrap_or(target).to_owned();

    let mut content_length = 0usize;
    loop {
        if next_line(&mut r, &mut line)? == 0 {
            return Err(ParseError::Malformed("headers not terminated"));
        }
        let l = line.trim_end();
        if l.is_empty() {
            break;
        }
        if let Some((k, v)) = l.split_once(':')
            && k.trim().eq_ignore_ascii_case("content-length")
        {
            content_length = v
                .trim()
                .parse()
                .map_err(|_| ParseError::Malformed("bad Content-Length"))?;
        }
    }
    if content_length > MAX_BODY_BYTES {
        return Err(ParseError::TooLarge);
    }
    let mut body = vec![0; content_length];
    r.read_exact(&mut body)
        .map_err(|_| ParseError::Malformed("body shorter than Content-Length"))?;
    Ok(Request { method, path, body })
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        413 => "Payload Too Large",
        _ => "Internal Server Error",
    }
}

pub fn respond(w: &mut impl Write, status: u16, content_type: &str, body: &[u8]) -> io::Result<()> {
    write!(
        w,
        "HTTP/1.1 {status} {}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        reason(status),
        body.len()
    )?;
    w.write_all(body)?;
    w.flush()
}

/// Starts a Server-Sent Events response; the body ends when the connection closes.
pub fn start_sse(w: &mut impl Write) -> io::Result<()> {
    w.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n")?;
    w.flush()
}

pub fn sse_event(w: &mut impl Write, data: &str) -> io::Result<()> {
    write!(w, "data: {data}\n\n")?;
    w.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_request_line_headers_and_body() {
        let raw = b"POST /v1/chat/completions?x=1 HTTP/1.1\r\nHost: a\r\ncontent-length: 4\r\n\r\nabcdEXTRA";
        let r = read_request(&raw[..]).unwrap();
        assert_eq!(
            (r.method.as_str(), r.path.as_str(), r.body.as_slice()),
            ("POST", "/v1/chat/completions", &b"abcd"[..])
        );
    }

    #[test]
    fn rejects_oversized_bodies_and_short_bodies() {
        let big = format!(
            "POST / HTTP/1.1\r\nContent-Length: {}\r\n\r\n",
            MAX_BODY_BYTES + 1
        );
        assert_eq!(
            read_request(big.as_bytes()).unwrap_err(),
            ParseError::TooLarge
        );
        let short = b"POST / HTTP/1.1\r\nContent-Length: 10\r\n\r\nabc";
        assert!(matches!(
            read_request(&short[..]),
            Err(ParseError::Malformed(_))
        ));
        assert_eq!(read_request(&b""[..]).unwrap_err(), ParseError::Closed);
    }
}
