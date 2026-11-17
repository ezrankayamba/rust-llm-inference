//! `rli`: one-shot prompts and interactive chat from the terminal.

use rli_models::{Engine, GenParams, Message, SamplerConfig};
use std::io::{BufRead, Write};
use std::process::ExitCode;

const USAGE: &str = "\
usage: rli -m <model.gguf> [options]

  -m, --model <path>        GGUF model file (required)
  -p, --prompt <text>       one-shot prompt; omit for interactive chat
      --raw                 send the prompt as-is, without the chat template
      --stop <text>         end the reply before this text; repeatable
  -s, --system <text>       system message for chat
  -n, --max-tokens <n>      tokens to generate per reply (default 256)
  -c, --ctx <n>             context size (default 4096, capped at the model's)
      --temp <f>            temperature, 0 = greedy (default 0.8)
      --top-k <n>           (default 40, 0 = off)
      --top-p <f>           (default 0.95)
      --repeat-penalty <f>  (default 1.1, 1 = off)
      --seed <n>
      --gpu-layers <n>      layers to run on the GPU (default: all, if this build has a
                            GPU backend; 0 = CPU only)
      --repack              copy Q4_K weights into a layout x86 decodes faster from; costs
                            about their size in extra memory
      --coarse-acts         CPU prompts: one activation scale per 256 values, not per 32;
                            faster for K-quant models on ARM, less accurate
  -t, --threads <n>         CPU threads (default: one per logical core)
      --info               print model metadata and exit
";

struct Args {
    model: String,
    prompt: Option<String>,
    raw: bool,
    system: Option<String>,
    ctx: usize,
    info: bool,
    gpu_layers: Option<usize>,
    repack: bool,
    coarse_acts: bool,
    params: GenParams,
}

fn parse_args() -> Result<Args, String> {
    let mut it = std::env::args().skip(1);
    let mut model = None;
    let mut a = Args {
        model: String::new(),
        prompt: None,
        raw: false,
        system: None,
        ctx: 4096,
        info: false,
        gpu_layers: None,
        repack: false,
        coarse_acts: false,
        params: GenParams {
            max_tokens: 256,
            sampler: SamplerConfig::default(),
            stop: Vec::new(),
        },
    };
    fn num<T: std::str::FromStr>(flag: &str, v: Option<String>) -> Result<T, String> {
        let v = v.ok_or_else(|| format!("{flag} needs a value"))?;
        v.parse()
            .map_err(|_| format!("{flag}: invalid value {v:?}"))
    }
    while let Some(flag) = it.next() {
        let s = &mut a.params.sampler;
        match flag.as_str() {
            "-m" | "--model" => model = it.next(),
            "-p" | "--prompt" => a.prompt = Some(it.next().ok_or("--prompt needs a value")?),
            "-s" | "--system" => a.system = Some(it.next().ok_or("--system needs a value")?),
            "--raw" => a.raw = true,
            "--stop" => {
                let stop = it.next().filter(|s| !s.is_empty());
                a.params
                    .stop
                    .push(stop.ok_or("--stop needs non-empty text")?)
            }
            "--info" => a.info = true,
            "-n" | "--max-tokens" => a.params.max_tokens = num(&flag, it.next())?,
            "-c" | "--ctx" => a.ctx = num(&flag, it.next())?,
            "--temp" => s.temperature = num(&flag, it.next())?,
            "--top-k" => s.top_k = num(&flag, it.next())?,
            "--top-p" => s.top_p = num(&flag, it.next())?,
            "--repeat-penalty" => s.repeat_penalty = num(&flag, it.next())?,
            "--seed" => s.seed = num(&flag, it.next())?,
            "--gpu-layers" => a.gpu_layers = Some(num(&flag, it.next())?),
            "--repack" => a.repack = true,
            "--coarse-acts" => a.coarse_acts = true,
            "-t" | "--threads" => rli_models::set_threads(num(&flag, it.next())?)?,
            "-h" | "--help" => return Err(String::new()),
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    a.model = model.ok_or("--model is required")?;
    Ok(a)
}

fn print_info(e: &Engine) {
    let c = &e.model.cfg;
    eprintln!("model:     {}", e.model_name);
    eprintln!(
        "arch:      {} ({} layers, d={}, heads={}/{} kv, head_dim={}, ff={})",
        c.arch_name, c.n_layer, c.n_embd, c.n_head, c.n_head_kv, c.head_dim, c.n_ff
    );
    if c.n_expert > 0 {
        eprintln!("experts:   {} ({} active)", c.n_expert, c.n_expert_used);
    }
    eprintln!("context:   {} (trained)", c.n_ctx_train);
    eprintln!(
        "vocab:     {}  template: {:?}",
        e.tokenizer.vocab_size(),
        e.template
    );
    if let Some((name, n)) = e.model.accelerator() {
        eprintln!("backend:   {name}, {n}/{} layers offloaded", c.n_layer);
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
    if !a.raw
        && let Some(w) = &engine.template_warning
    {
        eprintln!("warning: {w}");
    }
    if a.info {
        print_info(&engine);
        return Ok(());
    }
    let mut sess = engine.new_session(a.ctx);
    let mut messages: Vec<Message> = a.system.iter().map(|s| Message::new("system", s)).collect();
    let stdout = std::io::stdout();
    let generate = |tokens: &[u32], sess: &mut _| {
        let out = engine.generate(
            sess,
            tokens,
            &a.params,
            || false,
            |t| {
                let mut o = stdout.lock();
                let _ = o.write_all(t.as_bytes());
                let _ = o.flush();
                true
            },
        );
        println!();
        eprintln!(
            "[{} prompt tokens ({} evaluated) in {:.2}s = {:.1} tok/s | {} generated in {:.2}s = {:.1} tok/s | {:?}]",
            out.prompt_tokens,
            out.prompt_evaluated,
            out.prefill_secs,
            out.prompt_evaluated as f64 / out.prefill_secs.max(1e-9),
            out.completion_tokens,
            out.decode_secs,
            out.completion_tokens as f64 / out.decode_secs.max(1e-9),
            out.finish
        );
        out.text
    };

    if let Some(p) = &a.prompt {
        let tokens = if a.raw {
            engine.tokenizer.encode(p, engine.tokenizer.add_bos, false)
        } else {
            messages.push(Message::new("user", p));
            engine.chat_tokens(&messages)
        };
        generate(&tokens, &mut sess);
        return Ok(());
    }

    eprintln!(
        "{} loaded. Type a message; empty line or Ctrl-D to quit.",
        engine.model_name
    );
    let stdin = std::io::stdin();
    loop {
        eprint!("> ");
        let mut line = String::new();
        if stdin
            .lock()
            .read_line(&mut line)
            .map_err(|e| e.to_string())?
            == 0
            || line.trim().is_empty()
        {
            return Ok(());
        }
        messages.push(Message::new("user", line.trim()));
        // The session keeps the previous turns' KV cache; only the new suffix is evaluated.
        let tokens = engine.chat_tokens(&messages);
        let reply = generate(&tokens, &mut sess);
        messages.push(Message::new("assistant", reply.trim()));
    }
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
