mod eval;
mod search;
mod tt;

use cozy_chess::util::{display_uci_move, parse_uci_move};
use cozy_chess::Board;
use search::{Limits, Searcher};
use std::io::{self, BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;

const NAME: &str = "chessbot-engine";
const AUTHOR: &str = "Stefano Baghino";
const DEFAULT_HASH_MB: usize = 64;

enum Cmd {
    NewGame,
    Position(Board, Vec<u64>),
    Go(Limits),
    SetHash(usize),
    Bench(i32),
    Quit,
}

fn parse_position(tokens: &[&str]) -> Option<(Board, Vec<u64>)> {
    let mut i = 0;
    let mut board;
    if tokens.get(i) == Some(&"startpos") {
        board = Board::startpos();
        i += 1;
    } else if tokens.get(i) == Some(&"fen") {
        i += 1;
        let mut fen_parts = Vec::new();
        while i < tokens.len() && tokens[i] != "moves" {
            fen_parts.push(tokens[i]);
            i += 1;
        }
        let fen = fen_parts.join(" ");
        board = Board::from_fen(&fen, false).ok()?;
    } else {
        return None;
    }
    let mut history = Vec::new();
    if tokens.get(i) == Some(&"moves") {
        i += 1;
        while i < tokens.len() {
            let mv = parse_uci_move(&board, tokens[i]).ok()?;
            history.push(board.hash());
            board.play(mv);
            i += 1;
        }
    }
    Some((board, history))
}

fn parse_go(tokens: &[&str]) -> Limits {
    let mut l = Limits::default();
    let mut i = 0;
    let num = |s: Option<&&str>| s.and_then(|v| v.parse::<i64>().ok()).map(|v| v.max(0) as u64);
    while i < tokens.len() {
        match tokens[i] {
            "wtime" => l.wtime = num(tokens.get(i + 1)),
            "btime" => l.btime = num(tokens.get(i + 1)),
            "winc" => l.winc = num(tokens.get(i + 1)),
            "binc" => l.binc = num(tokens.get(i + 1)),
            "movestogo" => l.movestogo = num(tokens.get(i + 1)),
            "movetime" => l.movetime = num(tokens.get(i + 1)),
            "nodes" => l.nodes = num(tokens.get(i + 1)),
            "depth" => l.depth = num(tokens.get(i + 1)).map(|d| d as i32),
            "infinite" => {
                l.infinite = true;
                i += 1;
                continue;
            }
            _ => {
                i += 1;
                continue;
            }
        }
        i += 2;
    }
    l
}

fn search_thread(rx: mpsc::Receiver<Cmd>, stop: Arc<AtomicBool>) {
    let mut searcher = Searcher::new(DEFAULT_HASH_MB, stop.clone());
    let mut board = Board::startpos();
    let mut history: Vec<u64> = Vec::new();
    for cmd in rx {
        match cmd {
            Cmd::NewGame => searcher.new_game(),
            Cmd::Position(b, h) => {
                board = b;
                history = h;
            }
            Cmd::SetHash(mb) => searcher.tt.resize(mb),
            Cmd::Go(limits) => {
                let best = searcher.go(&board, &history, &limits);
                match best {
                    Some(m) => println!("bestmove {}", display_uci_move(&board, m)),
                    None => println!("bestmove 0000"),
                }
                io::stdout().flush().ok();
                stop.store(false, Ordering::Relaxed);
            }
            Cmd::Bench(depth) => {
                let fens = [
                    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
                    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
                    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
                    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
                    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
                ];
                let start = std::time::Instant::now();
                let mut total = 0u64;
                for fen in fens {
                    searcher.new_game();
                    let b = Board::from_fen(fen, false).unwrap();
                    let limits = Limits { depth: Some(depth), ..Default::default() };
                    searcher.go(&b, &[], &limits);
                    total += searcher.nodes_searched();
                }
                let ms = start.elapsed().as_millis().max(1) as u64;
                println!("bench: {} nodes {} nps", total, total * 1000 / ms);
                io::stdout().flush().ok();
            }
            Cmd::Quit => break,
        }
    }
}

fn main() {
    let stop = Arc::new(AtomicBool::new(false));
    let (tx, rx) = mpsc::channel::<Cmd>();
    let worker = {
        let stop = stop.clone();
        thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(move || search_thread(rx, stop))
            .expect("spawn search thread")
    };

    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("bench") {
        let depth = args.get(2).and_then(|d| d.parse().ok()).unwrap_or(10);
        tx.send(Cmd::Bench(depth)).ok();
        tx.send(Cmd::Quit).ok();
        worker.join().ok();
        return;
    }

    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        let tokens: Vec<&str> = line.split_whitespace().collect();
        let Some(&cmd) = tokens.first() else { continue };
        match cmd {
            "uci" => {
                println!("id name {}", NAME);
                println!("id author {}", AUTHOR);
                println!("option name Hash type spin default {} min 1 max 4096", DEFAULT_HASH_MB);
                println!("option name Threads type spin default 1 min 1 max 1");
                println!("uciok");
            }
            "isready" => println!("readyok"),
            "ucinewgame" => {
                tx.send(Cmd::NewGame).ok();
            }
            "setoption" => {
                // setoption name X value Y
                let name_idx = tokens.iter().position(|&t| t == "name");
                let value_idx = tokens.iter().position(|&t| t == "value");
                if let (Some(n), Some(v)) = (name_idx, value_idx) {
                    let name = tokens[n + 1..v].join(" ");
                    let value = tokens[v + 1..].join(" ");
                    if name.eq_ignore_ascii_case("Hash") {
                        if let Ok(mb) = value.parse::<usize>() {
                            tx.send(Cmd::SetHash(mb.clamp(1, 4096))).ok();
                        }
                    }
                }
            }
            "position" => match parse_position(&tokens[1..]) {
                Some((b, h)) => {
                    tx.send(Cmd::Position(b, h)).ok();
                }
                None => println!("info string invalid position"),
            },
            "go" => {
                stop.store(false, Ordering::Relaxed);
                tx.send(Cmd::Go(parse_go(&tokens[1..]))).ok();
            }
            "stop" => stop.store(true, Ordering::Relaxed),
            "quit" => {
                stop.store(true, Ordering::Relaxed);
                tx.send(Cmd::Quit).ok();
                break;
            }
            _ => {}
        }
        io::stdout().flush().ok();
    }
    worker.join().ok();
}
