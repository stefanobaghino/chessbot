//! Minimal NNUE: (768 -> HIDDEN)x2 perspective accumulators, SCReLU, single output.
//! Weights are quantised: feature transformer by QA, output layer by QB.

use cozy_chess::{Board, Color, File, Move, Piece, Rank, Square};

pub const HIDDEN: usize = 256;
const QA: i32 = 255;
const QB: i32 = 64;
const SCALE: i32 = 400;
const FEATURES: usize = 768;

static NET_BYTES: &[u8] = include_bytes!("../nets/default.bin");

pub struct Network {
    w0: Vec<[i16; HIDDEN]>,
    b0: [i16; HIDDEN],
    w1: [i16; 2 * HIDDEN],
    b1: i32,
}

#[derive(Clone, Copy)]
pub struct Accumulator {
    v: [[i16; HIDDEN]; 2],
}

impl Default for Accumulator {
    fn default() -> Self {
        Accumulator { v: [[0; HIDDEN]; 2] }
    }
}

fn read_i16(bytes: &[u8], i: usize) -> i16 {
    i16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]])
}

impl Network {
    pub fn load_default() -> Option<Box<Network>> {
        Self::from_bytes(NET_BYTES)
    }

    pub fn from_bytes(b: &[u8]) -> Option<Box<Network>> {
        if b.len() < 4 {
            return None;
        }
        let hidden = i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize;
        if hidden != HIDDEN {
            eprintln!("info string net hidden size {} does not match build {}", hidden, HIDDEN);
            return None;
        }
        let expected = 4 + 2 * (FEATURES * HIDDEN + HIDDEN + 2 * HIDDEN) + 4;
        if b.len() != expected {
            eprintln!("info string net size {} != expected {}", b.len(), expected);
            return None;
        }
        let body = &b[4..];
        let mut w0 = vec![[0i16; HIDDEN]; FEATURES];
        let mut off = 0;
        for row in w0.iter_mut() {
            for (j, x) in row.iter_mut().enumerate() {
                *x = read_i16(body, off + j);
            }
            off += HIDDEN;
        }
        let mut b0 = [0i16; HIDDEN];
        for (j, x) in b0.iter_mut().enumerate() {
            *x = read_i16(body, off + j);
        }
        off += HIDDEN;
        let mut w1 = [0i16; 2 * HIDDEN];
        for (j, x) in w1.iter_mut().enumerate() {
            *x = read_i16(body, off + j);
        }
        off += 2 * HIDDEN;
        let o = 2 * off;
        let b1 = i32::from_le_bytes([body[o], body[o + 1], body[o + 2], body[o + 3]]);
        // A net of all zeros is the placeholder shipped before training.
        if w0.iter().all(|r| r.iter().all(|&x| x == 0)) {
            return None;
        }
        Some(Box::new(Network { w0, b0, w1, b1 }))
    }

    #[inline]
    fn feature(persp: Color, piece: Piece, color: Color, sq: Square) -> usize {
        let rel_color = (color != persp) as usize;
        let rel_sq = if persp == Color::White { sq as usize } else { sq as usize ^ 56 };
        rel_color * 384 + piece as usize * 64 + rel_sq
    }

    #[inline]
    fn add(&self, acc: &mut Accumulator, piece: Piece, color: Color, sq: Square) {
        let fw = Self::feature(Color::White, piece, color, sq);
        let fb = Self::feature(Color::Black, piece, color, sq);
        let (rw, rb) = (&self.w0[fw], &self.w0[fb]);
        for i in 0..HIDDEN {
            acc.v[0][i] = acc.v[0][i].wrapping_add(rw[i]);
            acc.v[1][i] = acc.v[1][i].wrapping_add(rb[i]);
        }
    }

    #[inline]
    fn remove(&self, acc: &mut Accumulator, piece: Piece, color: Color, sq: Square) {
        let fw = Self::feature(Color::White, piece, color, sq);
        let fb = Self::feature(Color::Black, piece, color, sq);
        let (rw, rb) = (&self.w0[fw], &self.w0[fb]);
        for i in 0..HIDDEN {
            acc.v[0][i] = acc.v[0][i].wrapping_sub(rw[i]);
            acc.v[1][i] = acc.v[1][i].wrapping_sub(rb[i]);
        }
    }

    pub fn refresh(&self, board: &Board) -> Accumulator {
        let mut acc = Accumulator { v: [self.b0; 2] };
        for sq in board.occupied() {
            let p = board.piece_on(sq).unwrap();
            let c = board.color_on(sq).unwrap();
            self.add(&mut acc, p, c, sq);
        }
        acc
    }

    /// Accumulator for the position after `mv` is played on `board`.
    pub fn update(&self, parent: &Accumulator, board: &Board, mv: Move) -> Accumulator {
        let mut acc = *parent;
        let stm = board.side_to_move();
        let piece = board.piece_on(mv.from).unwrap();
        if board.colors(stm).has(mv.to) {
            // Castling: king takes own rook.
            let rank = mv.from.rank();
            let (kf, rf) = if mv.to.file() > mv.from.file() { (File::G, File::F) } else { (File::C, File::D) };
            self.remove(&mut acc, Piece::King, stm, mv.from);
            self.remove(&mut acc, Piece::Rook, stm, mv.to);
            self.add(&mut acc, Piece::King, stm, Square::new(kf, rank));
            self.add(&mut acc, Piece::Rook, stm, Square::new(rf, rank));
            return acc;
        }
        self.remove(&mut acc, piece, stm, mv.from);
        if let Some(cap) = board.piece_on(mv.to) {
            self.remove(&mut acc, cap, !stm, mv.to);
        } else if piece == Piece::Pawn && mv.from.file() != mv.to.file() {
            self.remove(&mut acc, Piece::Pawn, !stm, Square::new(mv.to.file(), mv.from.rank()));
        }
        self.add(&mut acc, mv.promotion.unwrap_or(piece), stm, mv.to);
        acc
    }

    pub fn evaluate(&self, acc: &Accumulator, stm: Color) -> i32 {
        let us = &acc.v[stm as usize];
        let them = &acc.v[!stm as usize];
        let mut sum: i64 = 0;
        for i in 0..HIDDEN {
            let a = (us[i] as i32).clamp(0, QA);
            let b = (them[i] as i32).clamp(0, QA);
            sum += (a * a * self.w1[i] as i32) as i64;
            sum += (b * b * self.w1[HIDDEN + i] as i32) as i64;
        }
        let out = sum / QA as i64 + self.b1 as i64;
        (out * SCALE as i64 / (QA * QB) as i64) as i32
    }
}

/// Plays pseudo-random games and checks incremental accumulators against full refreshes.
pub fn selfcheck() {
    let Some(net) = Network::load_default() else {
        println!("selfcheck: no network loaded");
        return;
    };
    let mut seed: u64 = 0x9E3779B97F4A7C15;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let mut checked = 0u64;
    let mut mismatches = 0u64;
    for _ in 0..200 {
        let mut board = Board::startpos();
        let mut acc = net.refresh(&board);
        for _ in 0..300 {
            let mut moves = Vec::new();
            board.generate_moves(|pm| {
                moves.extend(pm);
                false
            });
            if moves.is_empty() || board.halfmove_clock() >= 100 {
                break;
            }
            // Prefer captures and castling sometimes to exercise those paths.
            let mv = moves[(rnd() % moves.len() as u64) as usize];
            acc = net.update(&acc, &board, mv);
            board.play_unchecked(mv);
            let fresh = net.refresh(&board);
            checked += 1;
            if acc.v != fresh.v {
                mismatches += 1;
                if mismatches <= 5 {
                    println!("mismatch after {} in {}", mv, board);
                }
                acc = fresh;
            }
        }
    }
    println!("selfcheck: {} positions, {} mismatches", checked, mismatches);
}

#[allow(dead_code)]
fn rank_of(sq: Square) -> Rank {
    sq.rank()
}
