//! Alpha-beta search: iterative deepening, PVS, TT, null move, LMR, quiescence with SEE.

use crate::eval::{self, Tables};
use crate::nnue::{Accumulator, Network};
use crate::tt::{Bound, TranspositionTable};
use cozy_chess::{
    get_bishop_moves, get_king_moves, get_knight_moves, get_pawn_attacks, get_rook_moves, BitBoard, Board, Color,
    Move, Piece, Rank, Square,
};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub const MATE: i32 = 30000;
pub const MATE_IN_MAX: i32 = MATE - 1000;
pub const MAX_PLY: usize = 128;
const INF: i32 = 32001;

#[derive(Clone, Default, Debug)]
pub struct Limits {
    pub wtime: Option<u64>,
    pub btime: Option<u64>,
    pub winc: Option<u64>,
    pub binc: Option<u64>,
    pub movestogo: Option<u64>,
    pub movetime: Option<u64>,
    pub depth: Option<i32>,
    pub nodes: Option<u64>,
    pub infinite: bool,
    /// `go ponder`: search the expected reply without a clock until `ponderhit` (the shared
    /// ponder flag drops) turns it into a normal timed search, or `stop` ends it.
    pub ponder: bool,
}

pub struct MoveList {
    moves: [Move; 256],
    scores: [i32; 256],
    len: usize,
}

impl MoveList {
    fn new() -> Self {
        MoveList { moves: [Move { from: Square::A1, to: Square::A1, promotion: None }; 256], scores: [0; 256], len: 0 }
    }
    #[inline]
    fn push(&mut self, mv: Move) {
        self.moves[self.len] = mv;
        self.len += 1;
    }
    /// Selection-sort style pick of the best remaining move at index i.
    #[inline]
    fn pick(&mut self, i: usize) -> (Move, i32) {
        let mut best = i;
        for j in i + 1..self.len {
            if self.scores[j] > self.scores[best] {
                best = j;
            }
        }
        self.moves.swap(i, best);
        self.scores.swap(i, best);
        (self.moves[i], self.scores[i])
    }
}

const NONE_EVAL: i32 = -INF - 1;
const HIST_MAX: i32 = 16384;
const CONT_TABLES: usize = 2;
const PIECE_TO: usize = 12 * 64;

pub struct Searcher {
    pub tt: Arc<TranspositionTable>,
    pub silent: bool,
    pub shared_nodes: Arc<AtomicU64>,
    pub thread_id: usize,
    pub tables: Tables,
    pub net: Option<Box<Network>>,
    pub use_nnue: bool,
    accs: Vec<Accumulator>,
    stop: Arc<AtomicBool>,
    /// Shared with the UCI front end: true while a `go ponder` search waits for `ponderhit`.
    pub ponder_flag: Arc<AtomicBool>,
    /// This search started as a ponder and has not received `ponderhit` yet.
    pondering: bool,
    /// Centipawns a draw costs the root side (UCI `Contempt`): positive avoids draws,
    /// negative seeks them. Tapers to zero as material comes off, see `draw_score`.
    pub contempt: i32,
    /// Time limits to apply on `ponderhit`, from the clocks of the `go ponder` command.
    ponder_limits: (Option<Duration>, Option<Duration>),
    killers: [[Option<Move>; 2]; MAX_PLY],
    history: [[[i32; 64]; 64]; 2],
    cont_hist: Vec<i32>,
    capt_hist: Vec<i32>,
    counter: [[Option<Move>; 64]; 12],
    ss_piece_to: [Option<usize>; MAX_PLY + 4],
    ss_eval: [i32; MAX_PLY + 4],
    hashes: Vec<u64>,
    root_index: usize,
    nodes: u64,
    seldepth: usize,
    start: Instant,
    soft_limit: Option<Duration>,
    hard_limit: Option<Duration>,
    node_limit: Option<u64>,
    aborted: bool,
    lmr: [[i32; 64]; 64],
}

fn is_insufficient(board: &Board) -> bool {
    let occ = board.occupied();
    if occ.len() > 3 {
        return false;
    }
    if occ.len() == 2 {
        return true;
    }
    // Three pieces: kings plus one minor.
    !(board.pieces(Piece::Knight) | board.pieces(Piece::Bishop)).is_empty()
}

fn attackers_to(board: &Board, sq: Square, occ: BitBoard) -> BitBoard {
    let queens = board.pieces(Piece::Queen);
    (get_pawn_attacks(sq, Color::White) & board.colored_pieces(Color::Black, Piece::Pawn))
        | (get_pawn_attacks(sq, Color::Black) & board.colored_pieces(Color::White, Piece::Pawn))
        | (get_knight_moves(sq) & board.pieces(Piece::Knight))
        | (get_king_moves(sq) & board.pieces(Piece::King))
        | (get_bishop_moves(sq, occ) & (board.pieces(Piece::Bishop) | queens))
        | (get_rook_moves(sq, occ) & (board.pieces(Piece::Rook) | queens))
}

fn is_capture(board: &Board, mv: Move) -> bool {
    let enemy = board.colors(!board.side_to_move());
    if enemy.has(mv.to) {
        return true;
    }
    if board.piece_on(mv.from) == Some(Piece::Pawn) {
        if let Some(f) = board.en_passant() {
            let ep_sq = Square::new(f, Rank::Sixth.relative_to(board.side_to_move()));
            return mv.to == ep_sq;
        }
    }
    false
}

fn captured_piece(board: &Board, mv: Move) -> Option<Piece> {
    let enemy = board.colors(!board.side_to_move());
    if enemy.has(mv.to) {
        return board.piece_on(mv.to);
    }
    if board.piece_on(mv.from) == Some(Piece::Pawn) && mv.from.file() != mv.to.file() {
        return Some(Piece::Pawn);
    }
    None
}

/// Static exchange evaluation of a move (material gain from the mover's perspective).
pub fn see(board: &Board, mv: Move) -> i32 {
    let stm = board.side_to_move();
    // Castling is encoded as king takes own rook.
    if board.colors(stm).has(mv.to) {
        return 0;
    }
    let mut gain = [0i32; 32];
    let mut d = 0usize;
    let mut occ = board.occupied();
    let victim = captured_piece(board, mv);
    let mut attacker = board.piece_on(mv.from).unwrap();
    gain[0] = victim.map_or(0, eval::piece_value);
    if let Some(p) = mv.promotion {
        gain[0] += eval::piece_value(p) - eval::piece_value(Piece::Pawn);
        attacker = p;
    }
    if victim == Some(Piece::Pawn) && board.piece_on(mv.to).is_none() {
        // en passant: remove the captured pawn from occupancy
        let cap_sq = Square::new(mv.to.file(), mv.from.rank());
        occ ^= cap_sq.bitboard();
    }
    let mut from_bb = mv.from.bitboard();
    let mut side = stm;
    loop {
        d += 1;
        gain[d] = eval::piece_value(attacker) - gain[d - 1];
        if (-gain[d - 1]).max(gain[d]) < 0 {
            break;
        }
        occ ^= from_bb;
        side = !side;
        let att = attackers_to(board, mv.to, occ) & occ & board.colors(side);
        if att.is_empty() {
            break;
        }
        // least valuable attacker
        let mut found = None;
        for p in [Piece::Pawn, Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen, Piece::King] {
            let bb = att & board.pieces(p);
            if !bb.is_empty() {
                found = Some((p, bb.next_square().unwrap()));
                break;
            }
        }
        let (p, sq) = found.unwrap();
        attacker = p;
        from_bb = sq.bitboard();
        if d >= 30 {
            break;
        }
    }
    while d > 1 {
        d -= 1;
        gain[d - 1] = -((-gain[d - 1]).max(gain[d]));
    }
    gain[0]
}

fn gen_moves(board: &Board, list: &mut MoveList, captures_only: bool) {
    let stm = board.side_to_move();
    let enemy = board.colors(!stm);
    let ep_bb = board
        .en_passant()
        .map_or(BitBoard::EMPTY, |f| Square::new(f, Rank::Sixth.relative_to(stm)).bitboard());
    let promo_rank = Rank::Eighth.relative_to(stm).bitboard();
    board.generate_moves(|mut pm| {
        if captures_only {
            let mut mask = enemy;
            if pm.piece == Piece::Pawn {
                mask |= ep_bb | promo_rank;
            }
            pm.to &= mask;
        }
        for mv in pm {
            if captures_only && mv.promotion.is_some() && mv.promotion != Some(Piece::Queen) {
                continue;
            }
            list.push(mv);
        }
        false
    });
}

#[inline]
fn piece_index(board: &Board, sq: Square) -> usize {
    let p = board.piece_on(sq).unwrap() as usize;
    let c = board.color_on(sq).unwrap() as usize;
    c * 6 + p
}

impl Searcher {
    pub fn new(tt: Arc<TranspositionTable>, stop: Arc<AtomicBool>, shared_nodes: Arc<AtomicU64>, thread_id: usize) -> Self {
        let mut lmr = [[0i32; 64]; 64];
        for d in 1..64 {
            for m in 1..64 {
                lmr[d][m] = (0.75 + (d as f64).ln() * (m as f64).ln() / 2.25) as i32;
            }
        }
        let net = Network::load_default();
        let use_nnue = net.is_some();
        Searcher {
            tt,
            silent: thread_id != 0,
            shared_nodes,
            thread_id,
            tables: eval::build_tables(),
            net,
            use_nnue,
            accs: vec![Accumulator::default(); MAX_PLY + 2],
            stop,
            ponder_flag: Arc::new(AtomicBool::new(false)),
            pondering: false,
            ponder_limits: (None, None),
            contempt: 0,
            killers: [[None; 2]; MAX_PLY],
            history: [[[0; 64]; 64]; 2],
            cont_hist: vec![0; CONT_TABLES * PIECE_TO * PIECE_TO],
            capt_hist: vec![0; 12 * 64 * 6],
            counter: [[None; 64]; 12],
            ss_piece_to: [None; MAX_PLY + 4],
            ss_eval: [NONE_EVAL; MAX_PLY + 4],
            hashes: Vec::with_capacity(1024),
            root_index: 0,
            nodes: 0,
            seldepth: 0,
            start: Instant::now(),
            soft_limit: None,
            hard_limit: None,
            node_limit: None,
            aborted: false,
            lmr,
        }
    }

    pub fn static_eval(&self, board: &Board) -> i32 {
        match (&self.net, self.use_nnue) {
            (Some(net), true) => net.evaluate(&net.refresh(board), board.side_to_move()),
            _ => eval::evaluate(&self.tables, board),
        }
    }

    #[inline]
    fn eval_at(&self, board: &Board, ply: usize) -> i32 {
        match (&self.net, self.use_nnue) {
            (Some(net), true) => net.evaluate(&self.accs[ply], board.side_to_move()),
            _ => eval::evaluate(&self.tables, board),
        }
    }

    #[inline]
    fn push_acc(&mut self, board: &Board, mv: Move, ply: usize) {
        if self.use_nnue {
            if let Some(net) = &self.net {
                let next = net.update(&self.accs[ply], board, mv);
                self.accs[ply + 1] = next;
            }
        }
    }

    pub fn nodes_searched(&self) -> u64 {
        self.nodes
    }

    pub fn new_game(&mut self) {
        self.tt.clear();
        self.new_game_local();
    }

    pub fn new_game_local(&mut self) {
        self.killers = [[None; 2]; MAX_PLY];
        self.history = [[[0; 64]; 64]; 2];
        for v in self.cont_hist.iter_mut() {
            *v = 0;
        }
        for v in self.capt_hist.iter_mut() {
            *v = 0;
        }
        self.counter = [[None; 64]; 12];
    }

    fn set_limits(&mut self, board: &Board, limits: &Limits) {
        self.soft_limit = None;
        self.hard_limit = None;
        self.node_limit = limits.nodes;
        self.pondering = false;
        if limits.infinite {
            return;
        }
        if limits.ponder {
            // Compute the clock limits now, apply them when ponderhit arrives; a ponder
            // without clocks gets a second on ponderhit rather than searching forever.
            let mut timed = Limits { ponder: false, ..limits.clone() };
            if timed.wtime.is_none() && timed.btime.is_none() && timed.movetime.is_none() {
                timed.movetime = Some(1000);
            }
            self.set_limits(board, &timed);
            self.ponder_limits = (self.soft_limit, self.hard_limit);
            self.soft_limit = None;
            self.hard_limit = None;
            self.pondering = true;
            return;
        }
        if let Some(mt) = limits.movetime {
            let d = Duration::from_millis(mt.saturating_sub(10).max(1));
            self.soft_limit = Some(d);
            self.hard_limit = Some(d);
            return;
        }
        let (time, inc) = match board.side_to_move() {
            Color::White => (limits.wtime, limits.winc.unwrap_or(0)),
            Color::Black => (limits.btime, limits.binc.unwrap_or(0)),
        };
        if let Some(t) = time {
            let t = t as f64;
            let inc = inc as f64;
            let mtg = limits.movestogo.map_or(24.0, |m| (m as f64).clamp(1.0, 40.0));
            let overhead = 30.0;
            let usable = (t - overhead).max(1.0);
            let soft = (usable / mtg + inc * 0.75).min(usable * 0.8);
            let hard = (soft * 4.0).min(usable * 0.8);
            self.soft_limit = Some(Duration::from_millis(soft.max(1.0) as u64));
            self.hard_limit = Some(Duration::from_millis(hard.max(1.0) as u64));
        }
    }

    /// While pondering, watches for ponderhit: the clock starts and the limits computed
    /// from the `go ponder` command take effect.
    #[inline]
    fn poll_ponderhit(&mut self) {
        if self.pondering && !self.ponder_flag.load(Ordering::Relaxed) {
            self.pondering = false;
            self.start = Instant::now();
            (self.soft_limit, self.hard_limit) = self.ponder_limits;
        }
    }

    #[inline]
    fn check_time(&mut self) {
        if self.stop.load(Ordering::Relaxed) {
            self.aborted = true;
            return;
        }
        if self.nodes & 1023 == 0 {
            self.poll_ponderhit();
        }
        if let Some(n) = self.node_limit {
            if self.nodes >= n {
                self.aborted = true;
                return;
            }
        }
        if self.nodes & 1023 == 0 {
            self.shared_nodes.fetch_add(1024, Ordering::Relaxed);
            if let Some(h) = self.hard_limit {
                if self.start.elapsed() >= h {
                    self.aborted = true;
                }
            }
        }
    }

    /// True if the current position (already pushed as the last entry of `hashes`) occurred earlier,
    /// either in the game or on the current search path. Only positions with the same side to move
    /// can match, so the scan steps back two plies at a time starting from the grandparent.
    fn is_repetition(&self, hash: u64, halfmove: u8) -> bool {
        let len = self.hashes.len();
        let mut i = len as isize - 3;
        let limit = (len as isize - 1 - halfmove as isize).max(0);
        while i >= limit {
            if self.hashes[i as usize] == hash {
                return true;
            }
            i -= 2;
        }
        false
    }

    /// Runs the search and returns the best move. `history` holds hashes of all prior game positions
    /// (excluding the root).
    pub fn go(&mut self, board: &Board, history: &[u64], limits: &Limits) -> Option<Move> {
        self.start = Instant::now();
        self.nodes = 0;
        self.aborted = false;
        self.set_limits(board, limits);
        if self.thread_id == 0 {
            self.tt.new_search();
            self.shared_nodes.store(0, Ordering::Relaxed);
        }
        self.hashes.clear();
        self.hashes.extend_from_slice(history);
        self.root_index = self.hashes.len();
        self.hashes.push(board.hash());
        if self.use_nnue {
            if let Some(net) = &self.net {
                self.accs[0] = net.refresh(board);
            }
        }
        for k in self.killers.iter_mut() {
            *k = [None; 2];
        }
        self.ss_piece_to = [None; MAX_PLY + 4];
        self.ss_eval = [NONE_EVAL; MAX_PLY + 4];

        let mut list = MoveList::new();
        gen_moves(board, &mut list, false);
        if list.len == 0 {
            return None;
        }
        let mut best_move = list.moves[0];
        let mut best_score = -INF;
        let mut prev_score = -INF;
        let mut stable = 0;
        let max_depth = limits.depth.unwrap_or(MAX_PLY as i32 - 1).min(MAX_PLY as i32 - 1);
        let mut depth = 1 + (self.thread_id % 2) as i32;
        while depth <= max_depth {
            self.poll_ponderhit();
            let base_soft = self.soft_limit;
            self.seldepth = 0;
            let mut delta = 18;
            let mut alpha = if depth >= 5 { (best_score - delta).max(-INF) } else { -INF };
            let mut beta = if depth >= 5 { (best_score + delta).min(INF) } else { INF };
            let mut score = best_score;
            loop {
                let s = self.negamax(board, depth, 0, alpha, beta, true, true, None);
                if self.aborted {
                    break;
                }
                if s <= alpha {
                    beta = (alpha + beta) / 2;
                    alpha = (alpha - delta).max(-INF);
                } else if s >= beta {
                    beta = (beta + delta).min(INF);
                } else {
                    score = s;
                    break;
                }
                delta += delta / 2;
                if delta > 1000 {
                    alpha = -INF;
                    beta = INF;
                }
            }
            if self.aborted {
                break;
            }
            let mut new_best = best_move;
            if let Some(e) = self.tt.probe(board.hash()) {
                if let Some(m) = e.best_move() {
                    if board.is_legal(m) {
                        new_best = m;
                    }
                }
            }
            if new_best == best_move {
                stable += 1;
            } else {
                stable = 0;
            }
            best_move = new_best;
            prev_score = if prev_score == -INF { score } else { best_score };
            best_score = score;
            self.print_info(board, depth, best_score);
            if let (Some(base), Some(hard)) = (base_soft, self.hard_limit) {
                let mut factor = match stable {
                    0 => 1.6,
                    1 => 1.3,
                    2 => 1.1,
                    3 => 1.0,
                    _ => 0.85,
                };
                if best_score < prev_score - 30 {
                    factor *= 1.3;
                }
                if depth < 6 {
                    factor = 1.0;
                }
                let limit = Duration::from_secs_f64((base.as_secs_f64() * factor).min(hard.as_secs_f64()));
                if self.start.elapsed() >= limit {
                    break;
                }
            }
            // Stop early only on a mate the search has actually verified: the mating line must fit
            // inside the current depth and the score must have held for a full iteration.
            if best_score.abs() >= MATE_IN_MAX && MATE - best_score.abs() <= depth && best_score == prev_score {
                break;
            }
            depth += 1;
        }
        Some(best_move)
    }

    /// Score of a draw at `ply`, from the side to move: the root side gives up `contempt`
    /// scaled by the material left (full at 32 pieces, nothing at 7, where tablebases
    /// know the truth), the opponent gains it.
    fn draw_score(&self, board: &Board, ply: usize) -> i32 {
        if self.contempt == 0 {
            return 0;
        }
        let pieces = board.occupied().len() as i32;
        let c = self.contempt * (pieces - 7).clamp(0, 25) / 25;
        if ply % 2 == 0 { -c } else { c }
    }

    /// The reply the search expects after `best`, for `bestmove ... ponder`.
    pub fn ponder_move(&self, board: &Board, best: Move) -> Option<Move> {
        let mut b = board.clone();
        b.play_unchecked(best);
        let m = self.tt.probe(b.hash())?.best_move()?;
        if b.is_legal(m) { Some(m) } else { None }
    }

    fn print_info(&self, board: &Board, depth: i32, score: i32) {
        if self.silent {
            return;
        }
        let elapsed = self.start.elapsed();
        let ms = elapsed.as_millis().max(1) as u64;
        let total_nodes = self.shared_nodes.load(Ordering::Relaxed).max(self.nodes);
        let nps = total_nodes * 1000 / ms;
        let score_str = if score.abs() >= MATE_IN_MAX {
            let plies = MATE - score.abs();
            let moves = (plies + 1) / 2;
            format!("mate {}", if score > 0 { moves } else { -moves })
        } else {
            format!("cp {}", score)
        };
        let mut pv = Vec::new();
        let mut b = board.clone();
        let mut seen = Vec::new();
        for _ in 0..depth.max(1) as usize {
            let Some(e) = self.tt.probe(b.hash()) else { break };
            let Some(m) = e.best_move() else { break };
            if !b.is_legal(m) || seen.contains(&b.hash()) {
                break;
            }
            seen.push(b.hash());
            pv.push(cozy_chess::util::display_uci_move(&b, m).to_string());
            b.play_unchecked(m);
        }
        println!(
            "info depth {} seldepth {} score {} nodes {} nps {} hashfull {} time {} pv {}",
            depth,
            self.seldepth,
            score_str,
            total_nodes,
            nps,
            self.tt.hashfull(),
            ms,
            pv.join(" ")
        );
    }

    #[inline]
    fn cont_index(table: usize, prev: usize, cur: usize) -> usize {
        table * PIECE_TO * PIECE_TO + prev * PIECE_TO + cur
    }

    #[inline]
    fn quiet_score(&self, board: &Board, mv: Move, ply: usize) -> i32 {
        let stm = board.side_to_move() as usize;
        let cur = piece_index(board, mv.from) * 64 + mv.to as usize;
        let mut s = self.history[stm][mv.from as usize][mv.to as usize];
        for t in 0..CONT_TABLES {
            if ply >= t + 1 {
                if let Some(prev) = self.ss_piece_to[ply - 1 - t] {
                    s += self.cont_hist[Self::cont_index(t, prev, cur)];
                }
            }
        }
        s
    }

    #[inline]
    fn capt_index(board: &Board, mv: Move, victim: Piece) -> usize {
        (piece_index(board, mv.from) * 64 + mv.to as usize) * 6 + victim as usize
    }

    fn score_moves(&self, board: &Board, list: &mut MoveList, tt_move: Option<Move>, ply: usize) {
        let counter = if ply >= 1 {
            self.ss_piece_to[ply - 1].and_then(|pt| self.counter[pt / 64][pt % 64])
        } else {
            None
        };
        for i in 0..list.len {
            let mv = list.moves[i];
            let s = if Some(mv) == tt_move {
                1 << 30
            } else if let Some(victim) = captured_piece(board, mv) {
                let base = eval::piece_value(victim) * 32 + self.capt_hist[Self::capt_index(board, mv, victim)];
                if see(board, mv) >= 0 {
                    (1 << 28) + base + if mv.promotion == Some(Piece::Queen) { 20000 } else { 0 }
                } else {
                    (1 << 20) + base
                }
            } else if mv.promotion == Some(Piece::Queen) {
                (1 << 28) + 30000
            } else if mv.promotion.is_some() {
                -(1 << 20)
            } else if self.killers[ply][0] == Some(mv) {
                (1 << 27) + 2
            } else if self.killers[ply][1] == Some(mv) {
                (1 << 27) + 1
            } else if counter == Some(mv) {
                1 << 27
            } else {
                self.quiet_score(board, mv, ply)
            };
            list.scores[i] = s;
        }
    }

    #[inline]
    fn gravity(h: &mut i32, bonus: i32) {
        let b = bonus.clamp(-HIST_MAX, HIST_MAX);
        *h += b - *h * b.abs() / HIST_MAX;
    }

    fn update_quiet_stats(&mut self, board: &Board, mv: Move, ply: usize, bonus: i32) {
        let stm = board.side_to_move() as usize;
        Self::gravity(&mut self.history[stm][mv.from as usize][mv.to as usize], bonus);
        let cur = piece_index(board, mv.from) * 64 + mv.to as usize;
        for t in 0..CONT_TABLES {
            if ply >= t + 1 {
                if let Some(prev) = self.ss_piece_to[ply - 1 - t] {
                    let idx = Self::cont_index(t, prev, cur);
                    Self::gravity(&mut self.cont_hist[idx], bonus);
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn negamax(
        &mut self,
        board: &Board,
        mut depth: i32,
        ply: usize,
        mut alpha: i32,
        mut beta: i32,
        pv_node: bool,
        allow_null: bool,
        excluded: Option<Move>,
    ) -> i32 {
        if ply > self.seldepth {
            self.seldepth = ply;
        }
        let in_check = !board.checkers().is_empty();
        if in_check && excluded.is_none() {
            depth += 1;
        }
        if depth <= 0 {
            return self.qsearch(board, ply, alpha, beta);
        }
        self.nodes += 1;
        self.check_time();
        if self.aborted {
            return 0;
        }
        if ply >= MAX_PLY - 1 {
            return self.eval_at(board, ply);
        }
        let root = ply == 0;
        if !root {
            if board.halfmove_clock() >= 100 || is_insufficient(board) || self.is_repetition(board.hash(), board.halfmove_clock()) {
                return self.draw_score(board, ply);
            }
            alpha = alpha.max(-MATE + ply as i32);
            beta = beta.min(MATE - ply as i32 - 1);
            if alpha >= beta {
                return alpha;
            }
        }

        let hash = board.hash();
        let tt_entry = if excluded.is_none() { self.tt.probe(hash) } else { None };
        let mut tt_move = None;
        let mut tt_eval = None;
        let mut tt_score = NONE_EVAL;
        let mut tt_depth = -1;
        let mut tt_bound = Bound::None;
        if let Some(e) = tt_entry {
            tt_move = e.best_move().filter(|&m| board.is_legal(m));
            tt_eval = Some(e.eval as i32);
            tt_score = tt_score_from(e.score as i32, ply);
            tt_depth = e.depth as i32;
            tt_bound = e.bound();
            if !pv_node && tt_depth >= depth {
                match tt_bound {
                    Bound::Exact => return tt_score,
                    Bound::Lower if tt_score >= beta => return tt_score,
                    Bound::Upper if tt_score <= alpha => return tt_score,
                    _ => {}
                }
            }
        }

        let raw_eval = if in_check { NONE_EVAL } else { tt_eval.unwrap_or_else(|| self.eval_at(board, ply)) };
        self.ss_eval[ply] = raw_eval;
        let mut static_eval = raw_eval;
        if !in_check && tt_score != NONE_EVAL {
            let usable = match tt_bound {
                Bound::Exact => true,
                Bound::Lower => tt_score > raw_eval,
                Bound::Upper => tt_score < raw_eval,
                Bound::None => false,
            };
            if usable {
                static_eval = tt_score;
            }
        }
        let improving = !in_check
            && (ply < 2 || self.ss_eval[ply - 2] == NONE_EVAL || raw_eval > self.ss_eval[ply - 2]);

        let stm = board.side_to_move();

        if !pv_node && !in_check && excluded.is_none() {
            let rfp_margin = 75 * depth - if improving { 50 } else { 0 };
            if depth <= 8 && static_eval - rfp_margin >= beta && static_eval < MATE_IN_MAX {
                return static_eval;
            }
            if depth <= 3 && static_eval + 250 * depth <= alpha {
                let s = self.qsearch(board, ply, alpha, beta);
                if s <= alpha {
                    return s;
                }
            }
            let non_pawn = board.colors(stm) & !(board.pieces(Piece::Pawn) | board.pieces(Piece::King));
            if allow_null && depth >= 3 && static_eval >= beta && !non_pawn.is_empty() {
                if let Some(nb) = board.null_move() {
                    let r = 3 + depth / 4 + ((static_eval - beta) / 200).min(3);
                    self.hashes.push(nb.hash());
                    self.accs[ply + 1] = self.accs[ply];
                    self.ss_piece_to[ply] = None;
                    let score = -self.negamax(&nb, depth - 1 - r, ply + 1, -beta, -beta + 1, false, false, None);
                    self.hashes.pop();
                    if self.aborted {
                        return 0;
                    }
                    if score >= beta {
                        return if score >= MATE_IN_MAX { beta } else { score };
                    }
                }
            }
        }

        if depth >= 4 && tt_move.is_none() && excluded.is_none() {
            depth -= 1;
        }

        let mut list = MoveList::new();
        gen_moves(board, &mut list, false);
        self.score_moves(board, &mut list, tt_move, ply);

        let orig_alpha = alpha;
        let mut best_score = -INF;
        let mut best_move: Option<Move> = None;
        let mut quiets_tried: Vec<Move> = Vec::with_capacity(16);
        let mut captures_tried: Vec<(Move, Piece)> = Vec::with_capacity(8);
        let mut moves_searched = 0;
        let mut legal = 0;
        let futility_margin = static_eval + 100 + 110 * depth;
        let lmp_limit = if improving { 4 + 2 * depth * depth } else { 2 + depth * depth };

        for i in 0..list.len {
            let (mv, mscore) = list.pick(i);
            if Some(mv) == excluded {
                continue;
            }
            legal += 1;
            let victim = captured_piece(board, mv);
            let capture = victim.is_some();
            let quiet = !capture && mv.promotion.is_none();
            let is_killer_or_counter = mscore >= (1 << 27) && mscore < (1 << 28);

            if !root && !pv_node && best_score > -MATE_IN_MAX && quiet && !in_check {
                if depth <= 4 && moves_searched >= lmp_limit {
                    continue;
                }
                if depth <= 6 && futility_margin <= alpha {
                    continue;
                }
                if depth <= 3 && mscore < -3000 * depth {
                    continue;
                }
            }
            if !root && !pv_node && capture && depth <= 5 && best_score > -MATE_IN_MAX && mscore < (1 << 21) && see(board, mv) < -60 * depth {
                continue;
            }

            let mut extension = 0;
            if !root
                && depth >= 8
                && excluded.is_none()
                && Some(mv) == tt_move
                && tt_depth >= depth - 3
                && tt_bound != Bound::Upper
                && tt_score.abs() < MATE_IN_MAX
            {
                let sbeta = tt_score - 2 * depth;
                let sdepth = (depth - 1) / 2;
                let s = self.negamax(board, sdepth, ply, sbeta - 1, sbeta, false, false, Some(mv));
                if self.aborted {
                    return 0;
                }
                if s < sbeta {
                    extension = 1;
                    if !pv_node && s < sbeta - 25 {
                        extension = 2;
                    }
                } else if sbeta >= beta {
                    return sbeta;
                }
            }

            let mut child = board.clone();
            child.play_unchecked(mv);
            self.hashes.push(child.hash());
            self.push_acc(board, mv, ply);
            self.ss_piece_to[ply] = Some(piece_index(board, mv.from) * 64 + mv.to as usize);
            let gives_check = !child.checkers().is_empty();
            let new_depth = depth - 1 + extension;

            let mut score;
            if moves_searched == 0 {
                score = -self.negamax(&child, new_depth, ply + 1, -beta, -alpha, pv_node, true, None);
            } else {
                let mut r = 0;
                if depth >= 3 && moves_searched >= 2 && quiet && !in_check {
                    r = self.lmr[depth.min(63) as usize][moves_searched.min(63) as usize];
                    if pv_node {
                        r -= 1;
                    }
                    if is_killer_or_counter {
                        r -= 1;
                    }
                    if gives_check {
                        r -= 1;
                    }
                    if !improving {
                        r += 1;
                    }
                    r -= (mscore / 6000).clamp(-2, 2);
                    r = r.clamp(0, new_depth - 1);
                }
                score = -self.negamax(&child, new_depth - r, ply + 1, -alpha - 1, -alpha, false, true, None);
                if r > 0 && score > alpha && !self.aborted {
                    score = -self.negamax(&child, new_depth, ply + 1, -alpha - 1, -alpha, false, true, None);
                }
                if pv_node && score > alpha && score < beta && !self.aborted {
                    score = -self.negamax(&child, new_depth, ply + 1, -beta, -alpha, true, true, None);
                }
            }
            self.hashes.pop();
            moves_searched += 1;
            if self.aborted {
                return 0;
            }

            if score > best_score {
                best_score = score;
                if score > alpha {
                    best_move = Some(mv);
                    alpha = score;
                    if score >= beta {
                        let bonus = (depth * depth * 4 + depth * 8).min(1500);
                        if quiet {
                            if self.killers[ply][0] != Some(mv) {
                                self.killers[ply][1] = self.killers[ply][0];
                                self.killers[ply][0] = Some(mv);
                            }
                            if ply >= 1 {
                                if let Some(prev) = self.ss_piece_to[ply - 1] {
                                    self.counter[prev / 64][prev % 64] = Some(mv);
                                }
                            }
                            self.update_quiet_stats(board, mv, ply, bonus);
                            for &q in &quiets_tried {
                                self.update_quiet_stats(board, q, ply, -bonus);
                            }
                        } else if let Some(v) = victim {
                            let idx = Self::capt_index(board, mv, v);
                            Self::gravity(&mut self.capt_hist[idx], bonus);
                        }
                        for &(c, v) in &captures_tried {
                            let idx = Self::capt_index(board, c, v);
                            Self::gravity(&mut self.capt_hist[idx], -bonus);
                        }
                        break;
                    }
                }
            }
            if quiet {
                quiets_tried.push(mv);
            } else if let Some(v) = victim {
                captures_tried.push((mv, v));
            }
        }

        if legal == 0 {
            return if excluded.is_some() {
                alpha
            } else if in_check {
                -MATE + ply as i32
            } else {
                0
            };
        }

        if excluded.is_none() {
            let bound = if best_score >= beta {
                Bound::Lower
            } else if best_score > orig_alpha {
                Bound::Exact
            } else {
                Bound::Upper
            };
            self.tt.store(hash, best_move, tt_score_to(best_score, ply), if in_check { 0 } else { raw_eval }, depth, bound);
        }
        best_score
    }

    fn qsearch(&mut self, board: &Board, ply: usize, mut alpha: i32, beta: i32) -> i32 {
        self.nodes += 1;
        self.check_time();
        if self.aborted {
            return 0;
        }
        if ply > self.seldepth {
            self.seldepth = ply;
        }
        if ply >= MAX_PLY - 1 {
            return self.eval_at(board, ply);
        }
        let in_check = !board.checkers().is_empty();
        let hash = board.hash();
        let mut tt_move = None;
        let mut tt_eval = None;
        if let Some(e) = self.tt.probe(hash) {
            tt_move = e.best_move().filter(|&m| board.is_legal(m));
            tt_eval = Some(e.eval as i32);
            let s = tt_score_from(e.score as i32, ply);
            match e.bound() {
                Bound::Exact => return s,
                Bound::Lower if s >= beta => return s,
                Bound::Upper if s <= alpha => return s,
                _ => {}
            }
        }

        let mut best_score;
        let static_eval;
        if in_check {
            best_score = -INF;
            static_eval = 0;
        } else {
            static_eval = tt_eval.unwrap_or_else(|| self.eval_at(board, ply));
            best_score = static_eval;
            if best_score >= beta {
                return best_score;
            }
            if best_score > alpha {
                alpha = best_score;
            }
        }

        let mut list = MoveList::new();
        gen_moves(board, &mut list, !in_check);
        if in_check && list.len == 0 {
            return -MATE + ply as i32;
        }
        self.score_moves(board, &mut list, tt_move, ply);
        let orig_alpha = alpha;
        let mut best_move = None;
        for i in 0..list.len {
            let (mv, mscore) = list.pick(i);
            if !in_check {
                if mscore < (1 << 21) && mscore >= 0 && mscore < (1 << 27) {
                    continue;
                }
                if let Some(victim) = captured_piece(board, mv) {
                    if static_eval + eval::piece_value(victim) + 200 < alpha && mv.promotion.is_none() {
                        continue;
                    }
                }
            }
            let mut child = board.clone();
            child.play_unchecked(mv);
            self.push_acc(board, mv, ply);
            let score = -self.qsearch(&child, ply + 1, -beta, -alpha);
            if self.aborted {
                return 0;
            }
            if score > best_score {
                best_score = score;
                if score > alpha {
                    alpha = score;
                    best_move = Some(mv);
                    if score >= beta {
                        break;
                    }
                }
            }
        }
        let bound = if best_score >= beta {
            Bound::Lower
        } else if best_score > orig_alpha {
            Bound::Exact
        } else {
            Bound::Upper
        };
        self.tt.store(hash, best_move, tt_score_to(best_score, ply), static_eval, 0, bound);
        best_score
    }
}

fn tt_score_to(score: i32, ply: usize) -> i32 {
    if score >= MATE_IN_MAX {
        score + ply as i32
    } else if score <= -MATE_IN_MAX {
        score - ply as i32
    } else {
        score
    }
}

fn tt_score_from(score: i32, ply: usize) -> i32 {
    if score >= MATE_IN_MAX {
        score - ply as i32
    } else if score <= -MATE_IN_MAX {
        score + ply as i32
    } else {
        score
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tt::TranspositionTable;

    fn searcher() -> Searcher {
        Searcher::new(Arc::new(TranspositionTable::new(8)), Arc::new(AtomicBool::new(false)), Arc::new(AtomicU64::new(0)), 0)
    }

    #[test]
    fn ponder_search_waits_for_ponderhit_then_uses_the_clock() {
        let mut s = searcher();
        s.silent = true;
        let flag = Arc::new(AtomicBool::new(true));
        s.ponder_flag = flag.clone();
        let b = Board::default();
        let limits = Limits { wtime: Some(600), btime: Some(600), ponder: true, ..Default::default() };
        let t0 = Instant::now();
        let handle = std::thread::spawn(move || {
            let mv = s.go(&b, &[], &limits);
            (mv, s.nodes)
        });
        std::thread::sleep(Duration::from_millis(300));
        flag.store(false, Ordering::Relaxed); // ponderhit
        let (mv, nodes) = handle.join().unwrap();
        let elapsed = t0.elapsed();
        assert!(mv.is_some());
        assert!(nodes > 1000, "pondered {} nodes", nodes);
        // 300 ms of pondering plus a timed search on a 600 ms clock: well under two seconds,
        // and clearly longer than the 300 ms wait alone.
        assert!(elapsed > Duration::from_millis(300) && elapsed < Duration::from_millis(2000), "{:?}", elapsed);
    }

    #[test]
    fn draw_score_follows_contempt_side_and_material() {
        let mut s = searcher();
        assert_eq!(s.draw_score(&Board::default(), 0), 0);
        s.contempt = 20;
        // 32 pieces: full contempt; the root side (even plies) dislikes the draw, the opponent likes it.
        assert_eq!(s.draw_score(&Board::default(), 0), -20);
        assert_eq!(s.draw_score(&Board::default(), 3), 20);
        // 16 pieces: (16 - 7) / 25 of it.
        let mid = Board::from_fen("r3k2r/pp3ppp/8/8/8/8/PP3PPP/R3K2R w KQkq - 0 1", false).unwrap();
        assert_eq!(mid.occupied().len(), 16);
        assert_eq!(s.draw_score(&mid, 0), -7);
        // Seven pieces or fewer: plain draw.
        let ending = Board::from_fen("8/8/8/3k4/8/8/3PK3/8 w - - 0 1", false).unwrap();
        assert_eq!(s.draw_score(&ending, 0), 0);
        s.contempt = -20;
        assert_eq!(s.draw_score(&Board::default(), 0), 20);
    }

    #[test]
    fn ponder_search_stops_on_stop() {
        let mut s = searcher();
        s.silent = true;
        let stop = Arc::new(AtomicBool::new(false));
        let flag = Arc::new(AtomicBool::new(true));
        s.stop = stop.clone();
        s.ponder_flag = flag.clone();
        let b = Board::default();
        let limits = Limits { wtime: Some(600), btime: Some(600), ponder: true, ..Default::default() };
        let handle = std::thread::spawn(move || s.go(&b, &[], &limits));
        std::thread::sleep(Duration::from_millis(200));
        stop.store(true, Ordering::Relaxed);
        assert!(handle.join().unwrap().is_some());
    }

    #[test]
    fn ponder_move_is_the_expected_reply() {
        let mut s = searcher();
        s.silent = true;
        let b = Board::default();
        let mv = s.go(&b, &[], &Limits { depth: Some(6), ..Default::default() }).unwrap();
        let reply = s.ponder_move(&b, mv).unwrap();
        let mut after = b.clone();
        after.play_unchecked(mv);
        assert!(after.is_legal(reply));
    }

    #[test]
    fn finds_mate_in_one() {
        let mut s = searcher();
        s.silent = true;
        let b = Board::from_fen("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1", false).unwrap();
        let mv = s.go(&b, &[], &Limits { depth: Some(4), ..Default::default() }).unwrap();
        assert_eq!(mv.to_string(), "d1d8");
    }

    fn hashes_after(moves: &[&str]) -> (Vec<u64>, Board) {
        let mut b = Board::default();
        let mut hashes = vec![b.hash()];
        for m in moves {
            b.play(m.parse().unwrap());
            hashes.push(b.hash());
        }
        (hashes, b)
    }

    #[test]
    fn detects_repetition_of_same_side_positions() {
        let mut s = searcher();
        // Knights out and back: the root is the start position for the second time.
        let (hashes, b) = hashes_after(&["g1f3", "g8f6", "f3g1", "f6g8"]);
        s.hashes = hashes;
        assert!(s.is_repetition(b.hash(), 4));
        // Same line one ply shorter: the root has occurred before only with the other side to move.
        let (hashes, b) = hashes_after(&["g1f3", "g8f6", "f3g1"]);
        s.hashes = hashes;
        assert!(!s.is_repetition(b.hash(), 3));
    }

    #[test]
    fn takes_perpetual_check_when_lost() {
        let mut s = searcher();
        s.silent = true;
        // White is a queen and a rook down but can check forever on e8/h5.
        let b = Board::from_fen("6k1/6p1/5p2/8/8/7K/r3Q3/q7 w - - 0 1", false).unwrap();
        let mv = s.go(&b, &[], &Limits { depth: Some(8), ..Default::default() }).unwrap();
        assert_eq!(mv.to_string(), "e2e8");
    }

    #[test]
    fn see_values_captures() {
        let b = Board::from_fen("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", false).unwrap();
        let mv = Move { from: Square::E4, to: Square::D5, promotion: None };
        assert_eq!(see(&b, mv), 100);
        let b = Board::from_fen("4k3/8/2p5/3p4/4P3/8/8/4K3 w - - 0 1", false).unwrap();
        assert_eq!(see(&b, mv), 0);
    }
}
