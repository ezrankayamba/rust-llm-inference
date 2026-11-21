//! Just enough HTTP/1.1 for a local JSON + SSE API: one request per connection
//! (`Connection: close`), `Content-Length` bodies only.

use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::TcpStream;

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
    /// A body sent with `Transfer-Encoding`, which this server doesn't decode.
    LengthRequired,
    /// The peer closed before sending a request line.
    Closed,
}

pub fn read_request(stream: impl Read) -> Result<Request, ParseError> {
    // The limit applies while a line is read: checked after it, a line that never ends would
    // grow without bound.
    let mut r = BufReader::new(stream.take(MAX_HEADER_BYTES as u64 + 1));
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
    let mut encoded = false;
    loop {
        if next_line(&mut r, &mut line)? == 0 {
            return Err(ParseError::Malformed("headers not terminated"));
        }
        let l = line.trim_end();
        if l.is_empty() {
            break;
        }
        let Some((k, v)) = l.split_once(':') else {
            continue;
        };
        let k = k.trim();
        if k.eq_ignore_ascii_case("content-length") {
            content_length = v
                .trim()
                .parse()
                .map_err(|_| ParseError::Malformed("bad Content-Length"))?;
        } else if k.eq_ignore_ascii_case("transfer-encoding") {
            encoded = true;
        }
    }
    if encoded {
        return Err(ParseError::LengthRequired);
    }
    if content_length > MAX_BODY_BYTES {
        return Err(ParseError::TooLarge);
    }
    // What is buffered past the headers counts towards the body as well.
    r.get_mut().set_limit(content_length as u64);
    let mut body = vec![0; content_length];
    r.read_exact(&mut body)
        .map_err(|_| ParseError::Malformed("body shorter than Content-Length"))?;
    Ok(Request { method, path, body })
}

/// Whether the client has closed the connection (or half-closed it, which looks the same).
/// Toggles the socket's non-blocking flag, which clones share too, so only call this from the
/// thread that writes to the socket.
pub fn peer_gone(stream: &TcpStream) -> bool {
    if stream.set_nonblocking(true).is_err() {
        return false;
    }
    let peeked = stream.peek(&mut [0]);
    let restored = stream.set_nonblocking(false);
    match peeked {
        Ok(n) => n == 0,
        Err(e) if e.kind() == io::ErrorKind::WouldBlock => restored.is_err(),
        Err(_) => true,
    }
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        408 => "Request Timeout",
        411 => "Length Required",
        413 => "Payload Too Large",
        503 => "Service Unavailable",
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
    fn peer_gone_turns_true_only_after_the_client_closes() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (server, _) = listener.accept().unwrap();
        assert!(!peer_gone(&server));
        drop(client);
        // The FIN arrives asynchronously, even on loopback.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while !peer_gone(&server) {
            assert!(std::time::Instant::now() < deadline, "close not seen");
            std::thread::yield_now();
        }
    }

    #[test]
    fn peer_gone_leaves_the_socket_blocking() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let mut client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (mut server, _) = listener.accept().unwrap();
        peer_gone(&server);
        client.write_all(b"x").unwrap();
        let mut b = [0];
        // A socket left non-blocking could fail this read with WouldBlock.
        server.read_exact(&mut b).unwrap();
        assert_eq!(b, *b"x");
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

    /// A header line that never ends: an unbounded reader stands in for a client trickling it.
    #[test]
    fn rejects_an_endless_header_line_without_reading_it_all() {
        let endless = std::io::repeat(b'a');
        let head = &b"GET / HTTP/1.1\r\nX: "[..];
        assert_eq!(
            read_request(head.chain(endless)).unwrap_err(),
            ParseError::TooLarge
        );
    }

    #[test]
    fn rejects_chunked_bodies_as_length_required() {
        let raw = b"POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n4\r\nabcd\r\n0\r\n\r\n";
        assert_eq!(
            read_request(&raw[..]).unwrap_err(),
            ParseError::LengthRequired
        );
    }
}
