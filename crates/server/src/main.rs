//! `rli-server`: OpenAI-compatible HTTP API over one loaded model.

mod api;
mod http;

use api::App;
use rli_models::Engine;
use std::net::{TcpListener, TcpStream};
use std::process::ExitCode;
use std::sync::{Arc, Mutex};
use std::time::Duration;

const USAGE: &str = "\
usage: rli-server -m <model.gguf> [--host 127.0.0.1] [--port 8080] [--ctx 4096] [--gpu-layers N]

Endpoints: GET /healthz, GET /v1/models, POST /v1/chat/completions, POST /v1/completions
";

const READ_TIMEOUT: Duration = Duration::from_secs(30);

struct Args {
    model: String,
    host: String,
    port: u16,
    ctx: usize,
    gpu_layers: Option<usize>,
}

fn parse_args() -> Result<Args, String> {
    let mut it = std::env::args().skip(1);
    let mut a = Args {
        model: String::new(),
        host: "127.0.0.1".into(),
        port: 8080,
        ctx: 4096,
        gpu_layers: None,
    };
    let mut model = None;
    while let Some(flag) = it.next() {
        let mut val = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "-m" | "--model" => model = Some(val()?),
            "--host" => a.host = val()?,
            "--port" => a.port = val()?.parse().map_err(|_| "--port: invalid value")?,
            "-c" | "--ctx" => a.ctx = val()?.parse().map_err(|_| "--ctx: invalid value")?,
            "--gpu-layers" => {
                a.gpu_layers = Some(val()?.parse().map_err(|_| "--gpu-layers: invalid value")?)
            }
            "-h" | "--help" => return Err(String::new()),
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    a.model = model.ok_or("--model is required")?;
    Ok(a)
}

fn serve(app: &App, mut stream: TcpStream) {
    let _ = stream.set_read_timeout(Some(READ_TIMEOUT));
    let result = match http::read_request(&stream) {
        Ok(req) => api::handle(app, req, &mut stream),
        Err(http::ParseError::Closed) => Ok(()),
        Err(http::ParseError::TooLarge) => {
            http::respond(&mut stream, 413, "text/plain", b"request too large")
        }
        Err(http::ParseError::Malformed(m)) => {
            http::respond(&mut stream, 400, "text/plain", m.as_bytes())
        }
    };
    if let Err(e) = result {
        eprintln!("connection error: {e}");
    }
}

fn run(a: Args) -> Result<(), String> {
    #[allow(unused_mut)]
    let mut engine = Engine::load(&a.model).map_err(|e| format!("{}: {e}", a.model))?;
    if let Some(n) = a.gpu_layers {
        rli_models::backend::attach(&mut engine.model, n).map_err(|e| e.to_string())?;
    }
    let session = Mutex::new(engine.new_session(a.ctx));
    let app = Arc::new(App { engine, session });
    let listener = TcpListener::bind((a.host.as_str(), a.port))
        .map_err(|e| format!("bind {}:{}: {e}", a.host, a.port))?;
    let addr = listener.local_addr().map_err(|e| e.to_string())?;
    eprintln!("serving {} on http://{addr}", app.engine.model_name);
    for stream in listener.incoming() {
        match stream {
            Ok(s) => {
                let app = Arc::clone(&app);
                std::thread::spawn(move || serve(&app, s));
            }
            Err(e) => eprintln!("accept failed: {e}"),
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            if !e.is_empty() {
                eprintln!("error: {e}\n");
            }
            eprint!("{USAGE}");
            return ExitCode::FAILURE;
        }
    };
    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
