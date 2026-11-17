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
                  [--threads N] [--repack] [--coarse-acts]

--gpu-layers defaults to all layers when built with a GPU backend; 0 = CPU only.
--repack copies Q4_K weights into a layout x86 decodes faster from, for about their size in
extra memory.
--coarse-acts quantizes CPU prompt activations with one scale per 256 values, not per 32: faster
for K-quant models on ARM, less accurate.
--threads defaults to one CPU thread per logical core.

Endpoints: GET / (chat page), GET /healthz, GET /v1/models, POST /v1/chat/completions,
POST /v1/completions
";

const READ_TIMEOUT: Duration = Duration::from_secs(30);
/// A client that stops reading fails its next write after this long, which ends its generation
/// and frees the shared session.
const WRITE_TIMEOUT: Duration = Duration::from_secs(30);

struct Args {
    model: String,
    host: String,
    port: u16,
    ctx: usize,
    gpu_layers: Option<usize>,
    repack: bool,
    coarse_acts: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut it = std::env::args().skip(1);
    let mut a = Args {
        model: String::new(),
        host: "127.0.0.1".into(),
        port: 8080,
        ctx: 4096,
        gpu_layers: None,
        repack: false,
        coarse_acts: false,
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
            "--repack" => a.repack = true,
            "--coarse-acts" => a.coarse_acts = true,
            "-t" | "--threads" => {
                let n = val()?.parse().map_err(|_| "--threads: invalid value")?;
                rli_models::set_threads(n)?
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
    if let Err(e) = stream.set_write_timeout(Some(WRITE_TIMEOUT)) {
        eprintln!("connection error: {e}");
        return;
    }
    let result = match http::read_request(&stream) {
        Ok(req) => api::handle(app, req, &mut &stream, || http::peer_gone(&stream)),
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
    let mut engine = Engine::load(&a.model).map_err(|e| format!("{}: {e}", a.model))?;
    if let Some(w) = rli_models::backend::attach_default(&mut engine.model, a.gpu_layers)
        .map_err(|e| e.to_string())?
    {
        eprintln!("warning: {w}");
    }
    engine.model.coarse_acts = a.coarse_acts;
    if a.repack {
        match engine.model.repack() {
            0 => eprintln!("warning: --repack does nothing here: it needs Q4_K weights and AVX2"),
            n => eprintln!("repacked:  {} MiB of weights copied", n >> 20),
        }
    }
    if let Some(w) = &engine.template_warning {
        eprintln!("warning: {w}");
    }
    let session = engine.new_session(a.ctx);
    let app = Arc::new(App {
        max_ctx: session.max_ctx,
        session: Mutex::new(session),
        engine,
    });
    let listener = TcpListener::bind((a.host.as_str(), a.port))
        .map_err(|e| format!("bind {}:{}: {e}", a.host, a.port))?;
    let addr = listener.local_addr().map_err(|e| e.to_string())?;
    let backend = app
        .engine
        .model
        .accelerator()
        .map_or("cpu".into(), |(n, l)| format!("{n}, {l} layers"));
    eprintln!(
        "serving {} [{backend}] on http://{addr}",
        app.engine.model_name
    );
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
