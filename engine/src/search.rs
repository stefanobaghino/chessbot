//! Alpha-beta search: iterative deepening, PVS, TT, null move, LMR, quiescence with SEE.

use crate::eval::{self, Tables};
use crate::nnue::{Accumulator, Network};
use crate::tt::{Bound, TranspositionTable};
use cozy_chess::{
    get_bishop_moves, get_king_moves, get_knight_moves, get_pawn_attacks, get_rook_moves, BitBoard, Board, Color,
    Move, Piece, Rank, Square,
};
use std::sync::atomic::{AtomicBool, Ordering};
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

pub struct Searcher {
    pub tt: TranspositionTable,
    pub tables: Tables,
    pub net: Option<Box<Network>>,
    pub use_nnue: bool,
    accs: Vec<Accumulator>,
    stop: Arc<AtomicBool>,
    killers: [[Option<Move>; 2]; MAX_PLY],
    history: [[[i32; 64]; 64]; 2],
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

impl Searcher {
    pub fn new(hash_mb: usize, stop: Arc<AtomicBool>) -> Self {
        let mut lmr = [[0i32; 64]; 64];
        for d in 1..64 {
            for m in 1..64 {
                lmr[d][m] = (0.75 + (d as f64).ln() * (m as f64).ln() / 2.25) as i32;
            }
        }
        let net = Network::load_default();
        let use_nnue = net.is_some();
        Searcher {
            tt: TranspositionTable::new(hash_mb),
            tables: eval::build_tables(),
            net,
            use_nnue,
            accs: vec![Accumulator::default(); MAX_PLY + 2],
            stop,
            killers: [[None; 2]; MAX_PLY],
            history: [[[0; 64]; 64]; 2],
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
        self.killers = [[None; 2]; MAX_PLY];
        self.history = [[[0; 64]; 64]; 2];
    }

    fn set_limits(&mut self, board: &Board, limits: &Limits) {
        self.soft_limit = None;
        self.hard_limit = None;
        self.node_limit = limits.nodes;
        if limits.infinite {
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
            let mtg = limits.movestogo.map_or(30.0, |m| (m as f64).clamp(1.0, 40.0));
            let overhead = 30.0;
            let usable = (t - overhead).max(1.0);
            let soft = (usable / mtg + inc * 0.75).min(usable * 0.8);
            let hard = (soft * 4.0).min(usable * 0.8);
            self.soft_limit = Some(Duration::from_millis(soft.max(1.0) as u64));
            self.hard_limit = Some(Duration::from_millis(hard.max(1.0) as u64));
        }
    }

    #[inline]
    fn check_time(&mut self) {
        if self.stop.load(Ordering::Relaxed) {
            self.aborted = true;
            return;
        }
        if let Some(n) = self.node_limit {
            if self.nodes >= n {
                self.aborted = true;
                return;
            }
        }
        if self.nodes & 1023 == 0 {
            if let Some(h) = self.hard_limit {
                if self.start.elapsed() >= h {
                    self.aborted = true;
                }
            }
        }
    }

    fn is_repetition(&self, hash: u64, halfmove: u8) -> bool {
        let len = self.hashes.len();
        // Same side to move => step back two plies at a time.
        let mut i = len as isize - 2;
        let limit = (len as isize - 1 - halfmove as isize).max(0);
        // Positions after the root count once (twofold), earlier ones also once.
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
        self.tt.new_search();
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
        // Age history so old information fades.
        for c in self.history.iter_mut() {
            for f in c.iter_mut() {
                for v in f.iter_mut() {
                    *v /= 2;
                }
            }
        }

        let mut list = MoveList::new();
        gen_moves(board, &mut list, false);
        if list.len == 0 {
            return None;
        }
        let mut best_move = list.moves[0];
        let mut best_score = -INF;
        let max_depth = limits.depth.unwrap_or(MAX_PLY as i32 - 1).min(MAX_PLY as i32 - 1);
        let mut depth = 1;
        while depth <= max_depth {
            self.seldepth = 0;
            let mut delta = 20;
            let mut alpha = if depth >= 5 { (best_score - delta).max(-INF) } else { -INF };
            let mut beta = if depth >= 5 { (best_score + delta).min(INF) } else { INF };
            let mut score = best_score;
            loop {
                let s = self.negamax(board, depth, 0, alpha, beta, true, true);
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
            best_score = score;
            if let Some(e) = self.tt.probe(board.hash()) {
                if let Some(m) = e.best_move() {
                    if board.is_legal(m) {
                        best_move = m;
                    }
                }
            }
            self.print_info(board, depth, best_score);
            if let Some(soft) = self.soft_limit {
                if self.start.elapsed() >= soft {
                    break;
                }
            }
            if best_score.abs() >= MATE_IN_MAX && depth >= 8 {
                break;
            }
            depth += 1;
        }
        Some(best_move)
    }

    fn print_info(&self, board: &Board, depth: i32, score: i32) {
        let elapsed = self.start.elapsed();
        let ms = elapsed.as_millis().max(1) as u64;
        let nps = self.nodes * 1000 / ms;
        let score_str = if score.abs() >= MATE_IN_MAX {
            let plies = MATE - score.abs();
            let moves = (plies + 1) / 2;
            format!("mate {}", if score > 0 { moves } else { -moves })
        } else {
            format!("cp {}", score)
        };
        // Extract PV from the TT.
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
            self.nodes,
            nps,
            self.tt.hashfull(),
            ms,
            pv.join(" ")
        );
    }

    fn score_moves(&self, board: &Board, list: &mut MoveList, tt_move: Option<Move>, ply: usize) {
        let stm = board.side_to_move() as usize;
        for i in 0..list.len {
            let mv = list.moves[i];
            let s = if Some(mv) == tt_move {
                1 << 30
            } else if let Some(victim) = captured_piece(board, mv) {
                let attacker = board.piece_on(mv.from).unwrap();
                let base = eval::piece_value(victim) * 10 - eval::piece_value(attacker) / 10;
                if see(board, mv) >= 0 {
                    (1 << 28) + base + if mv.promotion == Some(Piece::Queen) { 900 } else { 0 }
                } else {
                    (1 << 20) + base
                }
            } else if mv.promotion == Some(Piece::Queen) {
                (1 << 28) + 1000
            } else if mv.promotion.is_some() {
                -(1 << 20)
            } else if self.killers[ply][0] == Some(mv) {
                (1 << 27) + 1
            } else if self.killers[ply][1] == Some(mv) {
                1 << 27
            } else {
                self.history[stm][mv.from as usize][mv.to as usize]
            };
            list.scores[i] = s;
        }
    }

    fn update_history(&mut self, stm: usize, mv: Move, bonus: i32) {
        let h = &mut self.history[stm][mv.from as usize][mv.to as usize];
        let clamped = bonus.clamp(-400, 400);
        *h += clamped - *h * clamped.abs() / 16384;
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
    ) -> i32 {
        if ply > self.seldepth {
            self.seldepth = ply;
        }
        let in_check = !board.checkers().is_empty();
        if in_check {
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
            // Draw detection.
            if board.halfmove_clock() >= 100 || is_insufficient(board) || self.is_repetition(board.hash(), board.halfmove_clock()) {
                return 0;
            }
            // Mate distance pruning.
            alpha = alpha.max(-MATE + ply as i32);
            beta = beta.min(MATE - ply as i32 - 1);
            if alpha >= beta {
                return alpha;
            }
        }

        let hash = board.hash();
        let tt_entry = self.tt.probe(hash);
        let mut tt_move = None;
        let mut tt_eval = None;
        if let Some(e) = tt_entry {
            tt_move = e.best_move().filter(|&m| board.is_legal(m));
            tt_eval = Some(e.eval as i32);
            if !pv_node && e.depth as i32 >= depth {
                let s = tt_score_from(e.score as i32, ply);
                match e.bound() {
                    Bound::Exact => return s,
                    Bound::Lower if s >= beta => return s,
                    Bound::Upper if s <= alpha => return s,
                    _ => {}
                }
            }
        }

        let static_eval = if in_check {
            -INF
        } else {
            tt_eval.unwrap_or_else(|| self.eval_at(board, ply))
        };

        let stm = board.side_to_move();
        let stm_i = stm as usize;

        if !pv_node && !in_check {
            // Reverse futility pruning.
            if depth <= 7 && static_eval - 75 * depth >= beta && static_eval < MATE_IN_MAX {
                return static_eval;
            }
            // Null move pruning.
            let non_pawn = board.colors(stm) & !(board.pieces(Piece::Pawn) | board.pieces(Piece::King));
            if allow_null && depth >= 3 && static_eval >= beta && !non_pawn.is_empty() {
                if let Some(nb) = board.null_move() {
                    let r = 3 + depth / 4 + ((static_eval - beta) / 200).min(3);
                    self.hashes.push(nb.hash());
                    self.accs[ply + 1] = self.accs[ply];
                    let score = -self.negamax(&nb, depth - 1 - r, ply + 1, -beta, -beta + 1, false, false);
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

        // Internal iterative reduction when no TT move.
        if pv_node && depth >= 4 && tt_move.is_none() {
            depth -= 1;
        }

        let mut list = MoveList::new();
        gen_moves(board, &mut list, false);
        if list.len == 0 {
            return if in_check { -MATE + ply as i32 } else { 0 };
        }
        self.score_moves(board, &mut list, tt_move, ply);

        let orig_alpha = alpha;
        let mut best_score = -INF;
        let mut best_move: Option<Move> = None;
        let mut quiets_tried: Vec<Move> = Vec::with_capacity(16);
        let mut moves_searched = 0;
        let futility_margin = static_eval + 100 + 110 * depth;

        for i in 0..list.len {
            let (mv, mscore) = list.pick(i);
            let capture = is_capture(board, mv);
            let quiet = !capture && mv.promotion.is_none();
            let is_killer = mscore >= (1 << 27) && mscore < (1 << 28);

            if !root && !pv_node && best_score > -MATE_IN_MAX && quiet && !in_check {
                // Late move pruning.
                if depth <= 4 && moves_searched >= 4 + depth * depth * 2 {
                    continue;
                }
                // Futility pruning.
                if depth <= 6 && futility_margin <= alpha {
                    continue;
                }
                // History pruning.
                if depth <= 3 && mscore < -2000 * depth {
                    continue;
                }
            }
            // SEE pruning of losing captures at low depth.
            if !root && !pv_node && capture && depth <= 5 && best_score > -MATE_IN_MAX && mscore < (1 << 21) && see(board, mv) < -50 * depth {
                continue;
            }

            let mut child = board.clone();
            child.play_unchecked(mv);
            self.hashes.push(child.hash());
            self.push_acc(board, mv, ply);
            let gives_check = !child.checkers().is_empty();

            let mut score;
            if moves_searched == 0 {
                score = -self.negamax(&child, depth - 1, ply + 1, -beta, -alpha, pv_node, true);
            } else {
                // Late move reductions.
                let mut r = 0;
                if depth >= 3 && moves_searched >= 2 && quiet && !in_check {
                    r = self.lmr[depth.min(63) as usize][moves_searched.min(63) as usize];
                    if pv_node {
                        r -= 1;
                    }
                    if is_killer {
                        r -= 1;
                    }
                    if gives_check {
                        r -= 1;
                    }
                    if mscore > 4000 {
                        r -= 1;
                    } else if mscore < -4000 {
                        r += 1;
                    }
                    r = r.clamp(0, depth - 2);
                }
                score = -self.negamax(&child, depth - 1 - r, ply + 1, -alpha - 1, -alpha, false, true);
                if r > 0 && score > alpha && !self.aborted {
                    score = -self.negamax(&child, depth - 1, ply + 1, -alpha - 1, -alpha, false, true);
                }
                if pv_node && score > alpha && score < beta && !self.aborted {
                    score = -self.negamax(&child, depth - 1, ply + 1, -beta, -alpha, true, true);
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
                        if quiet {
                            if self.killers[ply][0] != Some(mv) {
                                self.killers[ply][1] = self.killers[ply][0];
                                self.killers[ply][0] = Some(mv);
                            }
                            let bonus = depth * depth + depth;
                            self.update_history(stm_i, mv, bonus);
                            for &q in &quiets_tried {
                                self.update_history(stm_i, q, -bonus);
                            }
                        }
                        break;
                    }
                }
            }
            if quiet {
                quiets_tried.push(mv);
            }
        }

        let bound = if best_score >= beta {
            Bound::Lower
        } else if best_score > orig_alpha {
            Bound::Exact
        } else {
            Bound::Upper
        };
        self.tt.store(hash, best_move, tt_score_to(best_score, ply), if in_check { 0 } else { static_eval }, depth, bound);
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
                // Skip losing captures.
                if mscore < (1 << 21) && mscore >= 0 && mscore < (1 << 27) {
                    continue;
                }
                // Delta pruning.
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
