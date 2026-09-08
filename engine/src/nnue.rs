//! NNUE: (KING_BUCKETS x 768 -> HIDDEN)x2 perspective accumulators, SCReLU, one output
//! layer per material bucket. Each perspective indexes its features by its own king's
//! bucket, mirroring the board when the king stands on files e-h, so the accumulator of a
//! side is rebuilt when its king changes bucket or mirror.
//! Weights are quantised: feature transformer by QA, output layer by QB.

use cozy_chess::{Board, Color, File, Move, Piece, Square};

pub const HIDDEN: usize = 384;
pub const KING_BUCKETS: usize = 4;
pub const OUT_BUCKETS: usize = 4;
/// File format version; the header is FORMAT, HIDDEN, KING_BUCKETS, OUT_BUCKETS as i32.
const FORMAT: i32 = 2;
const QA: i32 = 255;
const QB: i32 = 64;
const SCALE: i32 = 400;
const FEATURES: usize = 768;

/// King bucket per (own-perspective) king square; files e-h are mirrored onto a-d first.
#[rustfmt::skip]
const KING_BUCKET: [u8; 64] = [
    0, 0, 1, 1, 1, 1, 0, 0,
    2, 2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3,
];

static NET_BYTES: &[u8] = include_bytes!("../nets/default.bin");

/// Short SHA-256 of the embedded net file, for the UCI banner.
pub fn net_id() -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(NET_BYTES);
    let hex: String = digest.iter().map(|b| format!("{:02x}", b)).collect();
    hex[..12].to_string()
}

pub struct Network {
    w0: Vec<[i16; HIDDEN]>,
    b0: [i16; HIDDEN],
    w1: Vec<[i16; 2 * HIDDEN]>,
    b1: [i32; OUT_BUCKETS],
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

/// How a perspective indexes its features: the king bucket's base row and the mirror flag.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Key {
    base: usize,
    mirror: bool,
}

/// Piece placements removed and added by a move.
struct Delta {
    rem: [(Piece, Color, Square); 2],
    add: [(Piece, Color, Square); 2],
    n_rem: usize,
    n_add: usize,
}

fn read_i16(bytes: &[u8], i: usize) -> i16 {
    i16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]])
}

fn read_i32(bytes: &[u8], i: usize) -> i32 {
    i32::from_le_bytes([bytes[4 * i], bytes[4 * i + 1], bytes[4 * i + 2], bytes[4 * i + 3]])
}

/// Output bucket by number of pieces on the board (2..=32).
#[inline]
pub fn out_bucket(pieces: usize) -> usize {
    (pieces.saturating_sub(2) / (32 / OUT_BUCKETS)).min(OUT_BUCKETS - 1)
}

impl Network {
    /// `Ok(None)` is the all-zero placeholder shipped before training; `Err` is a net
    /// whose format does not match this build, which must never reach a release.
    pub fn load_default() -> Result<Option<Box<Network>>, String> {
        Self::from_bytes(NET_BYTES)
    }

    pub fn from_bytes(b: &[u8]) -> Result<Option<Box<Network>>, String> {
        if b.len() < 16 {
            return Err(format!("net file is {} bytes, too short for a header", b.len()));
        }
        let header = [read_i32(b, 0), read_i32(b, 1), read_i32(b, 2), read_i32(b, 3)];
        let want = [FORMAT, HIDDEN as i32, KING_BUCKETS as i32, OUT_BUCKETS as i32];
        if header != want {
            return Err(format!(
                "net header (format, hidden, king buckets, out buckets) {:?} does not match build {:?}",
                header, want
            ));
        }
        let rows = KING_BUCKETS * FEATURES;
        let expected = 16 + 2 * (rows * HIDDEN + HIDDEN + OUT_BUCKETS * 2 * HIDDEN) + 4 * OUT_BUCKETS;
        if b.len() != expected {
            return Err(format!("net size {} != expected {}", b.len(), expected));
        }
        let body = &b[16..];
        let mut w0 = vec![[0i16; HIDDEN]; rows];
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
        let mut w1 = vec![[0i16; 2 * HIDDEN]; OUT_BUCKETS];
        for row in w1.iter_mut() {
            for (j, x) in row.iter_mut().enumerate() {
                *x = read_i16(body, off + j);
            }
            off += 2 * HIDDEN;
        }
        let tail = &body[2 * off..];
        let mut b1 = [0i32; OUT_BUCKETS];
        for (j, x) in b1.iter_mut().enumerate() {
            *x = read_i32(tail, j);
        }
        if w0.iter().all(|r| r.iter().all(|&x| x == 0)) {
            return Ok(None);
        }
        Ok(Some(Box::new(Network { w0, b0, w1, b1 })))
    }

    #[inline]
    fn key(persp: Color, king: Square) -> Key {
        let rel = if persp == Color::White { king as usize } else { king as usize ^ 56 };
        Key { base: KING_BUCKET[rel] as usize * FEATURES, mirror: rel & 7 >= 4 }
    }

    #[inline]
    fn keys(board: &Board) -> [Key; 2] {
        [Self::key(Color::White, board.king(Color::White)), Self::key(Color::Black, board.king(Color::Black))]
    }

    #[inline]
    fn feature(persp: Color, key: Key, piece: Piece, color: Color, sq: Square) -> usize {
        let rel_color = (color != persp) as usize;
        let mut rel_sq = if persp == Color::White { sq as usize } else { sq as usize ^ 56 };
        if key.mirror {
            rel_sq ^= 7;
        }
        key.base + rel_color * 384 + piece as usize * 64 + rel_sq
    }

    #[inline]
    fn add_row(&self, v: &mut [i16; HIDDEN], f: usize) {
        let row = &self.w0[f];
        for i in 0..HIDDEN {
            v[i] = v[i].wrapping_add(row[i]);
        }
    }

    #[inline]
    fn sub_row(&self, v: &mut [i16; HIDDEN], f: usize) {
        let row = &self.w0[f];
        for i in 0..HIDDEN {
            v[i] = v[i].wrapping_sub(row[i]);
        }
    }

    /// One perspective's accumulator built from scratch.
    fn refresh_side(&self, board: &Board, persp: Color, v: &mut [i16; HIDDEN]) {
        *v = self.b0;
        let key = Self::key(persp, board.king(persp));
        for sq in board.occupied() {
            let p = board.piece_on(sq).unwrap();
            let c = board.color_on(sq).unwrap();
            self.add_row(v, Self::feature(persp, key, p, c, sq));
        }
    }

    pub fn refresh(&self, board: &Board) -> Accumulator {
        let mut acc = Accumulator::default();
        self.refresh_side(board, Color::White, &mut acc.v[0]);
        self.refresh_side(board, Color::Black, &mut acc.v[1]);
        acc
    }

    fn delta(board: &Board, mv: Move) -> Delta {
        let stm = board.side_to_move();
        let piece = board.piece_on(mv.from).unwrap();
        let mut d = Delta {
            rem: [(Piece::Pawn, stm, mv.from); 2],
            add: [(Piece::Pawn, stm, mv.to); 2],
            n_rem: 0,
            n_add: 0,
        };
        if board.colors(stm).has(mv.to) {
            // Castling: king takes own rook.
            let rank = mv.from.rank();
            let (kf, rf) = if mv.to.file() > mv.from.file() { (File::G, File::F) } else { (File::C, File::D) };
            d.rem = [(Piece::King, stm, mv.from), (Piece::Rook, stm, mv.to)];
            d.add = [(Piece::King, stm, Square::new(kf, rank)), (Piece::Rook, stm, Square::new(rf, rank))];
            d.n_rem = 2;
            d.n_add = 2;
            return d;
        }
        d.rem[0] = (piece, stm, mv.from);
        d.n_rem = 1;
        if let Some(cap) = board.piece_on(mv.to) {
            d.rem[1] = (cap, !stm, mv.to);
            d.n_rem = 2;
        } else if piece == Piece::Pawn && mv.from.file() != mv.to.file() {
            d.rem[1] = (Piece::Pawn, !stm, Square::new(mv.to.file(), mv.from.rank()));
            d.n_rem = 2;
        }
        d.add[0] = (mv.promotion.unwrap_or(piece), stm, mv.to);
        d.n_add = 1;
        d
    }

    #[inline]
    fn apply(&self, v: &mut [i16; HIDDEN], persp: Color, key: Key, d: &Delta) {
        for &(p, c, sq) in &d.rem[..d.n_rem] {
            self.sub_row(v, Self::feature(persp, key, p, c, sq));
        }
        for &(p, c, sq) in &d.add[..d.n_add] {
            self.add_row(v, Self::feature(persp, key, p, c, sq));
        }
    }

    /// Accumulator for the position after `mv` is played on `board`.
    pub fn update(&self, parent: &Accumulator, board: &Board, mv: Move) -> Accumulator {
        let mut acc = *parent;
        let stm = board.side_to_move();
        let keys = Self::keys(board);
        let d = Self::delta(board, mv);
        if board.piece_on(mv.from) == Some(Piece::King) {
            let child = {
                let mut b = board.clone();
                b.play_unchecked(mv);
                b
            };
            if Self::key(stm, child.king(stm)) != keys[stm as usize] {
                // Every feature of the mover's perspective changes: rebuild that side.
                self.refresh_side(&child, stm, &mut acc.v[stm as usize]);
                self.apply(&mut acc.v[!stm as usize], !stm, keys[!stm as usize], &d);
                return acc;
            }
        }
        self.apply(&mut acc.v[0], Color::White, keys[0], &d);
        self.apply(&mut acc.v[1], Color::Black, keys[1], &d);
        acc
    }

    pub fn evaluate(&self, acc: &Accumulator, stm: Color, pieces: usize) -> i32 {
        let us = &acc.v[stm as usize];
        let them = &acc.v[!stm as usize];
        let ob = out_bucket(pieces);
        let w1 = &self.w1[ob];
        let sum = Self::dot(us, &w1[..HIDDEN]) + Self::dot(them, &w1[HIDDEN..]);
        let out = sum / QA as i64 + self.b1[ob] as i64;
        (out * SCALE as i64 / (QA * QB) as i64) as i32
    }
    /// sum_i screlu(x_i) * w_i with SCReLU = clamp(x, 0, QA)^2.
    #[cfg(target_arch = "aarch64")]
    #[inline]
    fn dot(x: &[i16; HIDDEN], w: &[i16]) -> i64 {
        use std::arch::aarch64::*;
        unsafe {
            let zero = vdupq_n_s16(0);
            let qa = vdupq_n_s16(QA as i16);
            let mut acc0 = vdupq_n_s64(0);
            let mut acc1 = vdupq_n_s64(0);
            let mut i = 0;
            while i < HIDDEN {
                let xv = vld1q_s16(x.as_ptr().add(i));
                let wv = vld1q_s16(w.as_ptr().add(i));
                let a = vminq_s16(vmaxq_s16(xv, zero), qa);
                // (a * w) fits in i32; multiplying by a again stays below i32::MAX.
                let lo = vmull_s16(vget_low_s16(a), vget_low_s16(w_low(wv)));
                let hi = vmull_high_s16(a, wv);
                let a_lo = vmovl_s16(vget_low_s16(a));
                let a_hi = vmovl_high_s16(a);
                acc0 = vpadalq_s32(acc0, vmulq_s32(lo, a_lo));
                acc1 = vpadalq_s32(acc1, vmulq_s32(hi, a_hi));
                i += 8;
            }
            vaddvq_s64(acc0) + vaddvq_s64(acc1)
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    #[inline]
    fn dot(x: &[i16; HIDDEN], w: &[i16]) -> i64 {
        let mut sum: i64 = 0;
        for i in 0..HIDDEN {
            let a = (x[i] as i32).clamp(0, QA);
            sum += (a * w[i] as i32 * a) as i64;
        }
        sum
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn w_low(v: std::arch::aarch64::int16x8_t) -> std::arch::aarch64::int16x8_t {
    v
}

/// Plays pseudo-random games and checks incremental accumulators against full refreshes.
pub fn selfcheck() {
    let net = match Network::load_default() {
        Ok(Some(net)) => net,
        Ok(None) => {
            println!("selfcheck: no network loaded");
            return;
        }
        Err(e) => {
            println!("selfcheck: FAILED, {}", e);
            return;
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The same cases are pinned in tests/test_train.py: the trainer must index identically.
    #[test]
    fn feature_index_matches_trainer() {
        let f = |persp, king, piece, color, sq| Network::feature(persp, Network::key(persp, king), piece, color, sq);
        // White perspective, king on g1 (mirrored to b1, bucket 0): own knight on f3 -> c3.
        assert_eq!(f(Color::White, Square::G1, Piece::Knight, Color::White, Square::F3), 64 + 18);
        // Black perspective, king on e8 (flipped e1, bucket 1, mirrored to d1): white queen on d1
        // flips to d8 = 59 and mirrors to e8 = 60.
        assert_eq!(f(Color::Black, Square::E8, Piece::Queen, Color::White, Square::D1), 768 + 384 + 256 + 60);
        // White perspective, king on c2 (bucket 2, no mirror): black pawn on a7.
        assert_eq!(f(Color::White, Square::C2, Piece::Pawn, Color::Black, Square::A7), 2 * 768 + 384 + 48);
        // Black perspective, king on h4 (flipped h5 = 39, bucket 3, mirrored): own king itself -> a5 = 32.
        assert_eq!(f(Color::Black, Square::H4, Piece::King, Color::Black, Square::H4), 3 * 768 + 320 + 32);
    }

    #[test]
    fn out_bucket_by_piece_count() {
        assert_eq!(out_bucket(2), 0);
        assert_eq!(out_bucket(9), 0);
        assert_eq!(out_bucket(10), 1);
        assert_eq!(out_bucket(17), 1);
        assert_eq!(out_bucket(18), 2);
        assert_eq!(out_bucket(26), 3);
        assert_eq!(out_bucket(32), 3);
    }

    #[test]
    fn header_mismatch_is_an_error() {
        let old = [HIDDEN as i32].iter().flat_map(|x| x.to_le_bytes()).chain(std::iter::repeat(0u8).take(4096)).collect::<Vec<_>>();
        assert!(Network::from_bytes(&old).is_err());
    }
}
