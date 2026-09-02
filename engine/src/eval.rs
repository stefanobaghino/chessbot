//! Tapered evaluation based on PeSTO piece-square tables.
//! Scores are from the side to move's perspective, in centipawns.

use cozy_chess::{get_bishop_moves, get_king_moves, get_knight_moves, get_rook_moves, BitBoard, Board, Color, File, Piece, Rank, Square};

pub const MG_VALUE: [i32; 6] = [82, 337, 365, 477, 1025, 0];
pub const EG_VALUE: [i32; 6] = [94, 281, 297, 512, 936, 0];
const PHASE_INC: [i32; 6] = [0, 1, 1, 2, 4, 0];
const TEMPO: i32 = 10;

// Tables are listed from rank 8 (top) to rank 1, a-file first.
#[rustfmt::skip]
const MG_PAWN: [i32; 64] = [
      0,   0,   0,   0,   0,   0,  0,   0,
     98, 134,  61,  95,  68, 126, 34, -11,
     -6,   7,  26,  31,  65,  56, 25, -20,
    -14,  13,   6,  21,  23,  12, 17, -23,
    -27,  -2,  -5,  12,  17,   6, 10, -25,
    -26,  -4,  -4, -10,   3,   3, 33, -12,
    -35,  -1, -20, -23, -15,  24, 38, -22,
      0,   0,   0,   0,   0,   0,  0,   0,
];
#[rustfmt::skip]
const EG_PAWN: [i32; 64] = [
      0,   0,   0,   0,   0,   0,   0,   0,
    178, 173, 158, 134, 147, 132, 165, 187,
     94, 100,  85,  67,  56,  53,  82,  84,
     32,  24,  13,   5,  -2,   4,  17,  17,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   8,   8,  10,  13,   0,   2,  -7,
      0,   0,   0,   0,   0,   0,   0,   0,
];
#[rustfmt::skip]
const MG_KNIGHT: [i32; 64] = [
    -167, -89, -34, -49,  61, -97, -15, -107,
     -73, -41,  72,  36,  23,  62,   7,  -17,
     -47,  60,  37,  65,  84, 129,  73,   44,
      -9,  17,  19,  53,  37,  69,  18,   22,
     -13,   4,  16,  13,  28,  19,  21,   -8,
     -23,  -9,  12,  10,  19,  17,  25,  -16,
     -29, -53, -12,  -3,  -1,  18, -14,  -19,
    -105, -21, -58, -33, -17, -28, -19,  -23,
];
#[rustfmt::skip]
const EG_KNIGHT: [i32; 64] = [
    -58, -38, -13, -28, -31, -27, -63, -99,
    -25,  -8, -25,  -2,  -9, -25, -24, -52,
    -24, -20,  10,   9,  -1,  -9, -19, -41,
    -17,   3,  22,  22,  22,  11,   8, -18,
    -18,  -6,  16,  25,  16,  17,   4, -18,
    -23,  -3,  -1,  15,  10,  -3, -20, -22,
    -42, -20, -10,  -5,  -2, -20, -23, -44,
    -29, -51, -23, -15, -22, -18, -50, -64,
];
#[rustfmt::skip]
const MG_BISHOP: [i32; 64] = [
    -29,   4, -82, -37, -25, -42,   7,  -8,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -16,  37,  43,  40,  35,  50,  37,  -2,
     -4,   5,  19,  50,  37,  37,   7,  -2,
     -6,  13,  13,  26,  34,  12,  10,   4,
      0,  15,  15,  15,  14,  27,  18,  10,
      4,  15,  16,   0,   7,  21,  33,   1,
    -33,  -3, -14, -21, -13, -12, -39, -21,
];
#[rustfmt::skip]
const EG_BISHOP: [i32; 64] = [
    -14, -21, -11,  -8, -7,  -9, -17, -24,
     -8,  -4,   7, -12, -3, -13,  -4, -14,
      2,  -8,   0,  -1, -2,   6,   0,   4,
     -3,   9,  12,   9, 14,  10,   3,   2,
     -6,   3,  13,  19,  7,  10,  -3,  -9,
    -12,  -3,   8,  10, 13,   3,  -7, -15,
    -14, -18,  -7,  -1,  4,  -9, -15, -27,
    -23,  -9, -23,  -5, -9, -16,  -5, -17,
];
#[rustfmt::skip]
const MG_ROOK: [i32; 64] = [
     32,  42,  32,  51, 63,  9,  31,  43,
     27,  32,  58,  62, 80, 67,  26,  44,
     -5,  19,  26,  36, 17, 45,  61,  16,
    -24, -11,   7,  26, 24, 35,  -8, -20,
    -36, -26, -12,  -1,  9, -7,   6, -23,
    -45, -25, -16, -17,  3,  0,  -5, -33,
    -44, -16, -20,  -9, -1, 11,  -6, -71,
    -19, -13,   1,  17, 16,  7, -37, -26,
];
#[rustfmt::skip]
const EG_ROOK: [i32; 64] = [
    13, 10, 18, 15, 12,  12,   8,   5,
    11, 13, 13, 11, -3,   3,   8,   3,
     7,  7,  7,  5,  4,  -3,  -5,  -3,
     4,  3, 13,  1,  2,   1,  -1,   2,
     3,  5,  8,  4, -5,  -6,  -8, -11,
    -4,  0, -5, -1, -7, -12,  -8, -16,
    -6, -6,  0,  2, -9,  -9, -11,  -3,
    -9,  2,  3, -1, -5, -13,   4, -20,
];
#[rustfmt::skip]
const MG_QUEEN: [i32; 64] = [
    -28,   0,  29,  12,  59,  44,  43,  45,
    -24, -39,  -5,   1, -16,  57,  28,  54,
    -13, -17,   7,   8,  29,  56,  47,  57,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
     -1, -18,  -9,  10, -15, -25, -31, -50,
];
#[rustfmt::skip]
const EG_QUEEN: [i32; 64] = [
     -9,  22,  22,  27,  27,  19,  10,  20,
    -17,  20,  32,  41,  58,  25,  30,   0,
    -20,   6,   9,  49,  47,  35,  19,   9,
      3,  22,  24,  45,  57,  40,  57,  36,
    -18,  28,  19,  47,  31,  34,  39,  23,
    -16, -27,  15,   6,   9,  17,  10,   5,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -33, -28, -22, -43,  -5, -32, -20, -41,
];
#[rustfmt::skip]
const MG_KING: [i32; 64] = [
    -65,  23,  16, -15, -56, -34,   2,  13,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
     -9,  24,   2, -16, -20,   6,  22, -22,
    -17, -20, -12, -27, -30, -25, -14, -36,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -14, -14, -22, -46, -44, -30, -15, -27,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -15,  36,  12, -54,   8, -28,  24,  14,
];
#[rustfmt::skip]
const EG_KING: [i32; 64] = [
    -74, -35, -18, -18, -11,  15,   4, -17,
    -12,  17,  14,  17,  17,  38,  23,  11,
     10,  17,  23,  15,  20,  45,  44,  13,
     -8,  22,  24,  27,  26,  33,  26,   3,
    -18,  -4,  21,  24,  27,  23,   9, -11,
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    -27, -11,   4,  13,  14,   4,  -5, -17,
    -53, -34, -21, -11, -28, -14, -24, -43,
];

/// Precomputed tables: PSTs with material folded in, plus pawn-structure masks.
pub struct Tables {
    pub mg: [[[i32; 64]; 6]; 2],
    pub eg: [[[i32; 64]; 6]; 2],
    passed_mask: [[BitBoard; 64]; 2],
    adjacent_files: [BitBoard; 8],
    king_zone: [BitBoard; 64],
}

const FILE_BBS: [BitBoard; 8] = [
    File::A.bitboard(), File::B.bitboard(), File::C.bitboard(), File::D.bitboard(),
    File::E.bitboard(), File::F.bitboard(), File::G.bitboard(), File::H.bitboard(),
];

pub fn build_tables() -> Tables {
    let mg_src = [&MG_PAWN, &MG_KNIGHT, &MG_BISHOP, &MG_ROOK, &MG_QUEEN, &MG_KING];
    let eg_src = [&EG_PAWN, &EG_KNIGHT, &EG_BISHOP, &EG_ROOK, &EG_QUEEN, &EG_KING];
    let mut t = Tables {
        mg: [[[0; 64]; 6]; 2],
        eg: [[[0; 64]; 6]; 2],
        passed_mask: [[BitBoard::EMPTY; 64]; 2],
        adjacent_files: [BitBoard::EMPTY; 8],
        king_zone: [BitBoard::EMPTY; 64],
    };
    for p in 0..6 {
        for sq in 0..64 {
            // Source tables are laid out from a8; cozy squares index from a1.
            let white_idx = sq ^ 56;
            t.mg[0][p][sq] = MG_VALUE[p] + mg_src[p][white_idx];
            t.eg[0][p][sq] = EG_VALUE[p] + eg_src[p][white_idx];
            t.mg[1][p][sq] = MG_VALUE[p] + mg_src[p][sq];
            t.eg[1][p][sq] = EG_VALUE[p] + eg_src[p][sq];
        }
    }
    for f in 0..8 {
        let mut bb = BitBoard::EMPTY;
        if f > 0 {
            bb |= FILE_BBS[f - 1];
        }
        if f < 7 {
            bb |= FILE_BBS[f + 1];
        }
        t.adjacent_files[f] = bb;
    }
    for sq in Square::ALL {
        let f = sq.file() as usize;
        let files = FILE_BBS[f] | t.adjacent_files[f];
        t.passed_mask[0][sq as usize] = front_span_fast(sq, Color::White) & files;
        t.passed_mask[1][sq as usize] = front_span_fast(sq, Color::Black) & files;
        let mut zone = get_king_moves(sq) | sq.bitboard();
        // Extend the zone one rank further towards the centre so that pieces
        // aiming at the squares in front of the king count as attackers.
        let shifted = match sq.rank() {
            Rank::First | Rank::Second => zone.0 << 8,
            Rank::Seventh | Rank::Eighth => zone.0 >> 8,
            _ => 0,
        };
        zone |= BitBoard(shifted);
        t.king_zone[sq as usize] = zone;
    }
    t
}

const PASSED_MG: [i32; 8] = [0, 4, 8, 18, 34, 60, 100, 0];
const PASSED_EG: [i32; 8] = [0, 12, 22, 38, 62, 100, 150, 0];
const ISOLATED: (i32, i32) = (-12, -16);
const DOUBLED: (i32, i32) = (-10, -22);
const BISHOP_PAIR: (i32, i32) = (28, 48);
const ROOK_OPEN: (i32, i32) = (26, 10);
const ROOK_SEMI: (i32, i32) = (12, 6);
const ROOK_SEVENTH: (i32, i32) = (18, 28);
const KNIGHT_MOB: (i32, i32) = (5, 4);
const BISHOP_MOB: (i32, i32) = (5, 5);
const ROOK_MOB: (i32, i32) = (2, 4);
const QUEEN_MOB: (i32, i32) = (1, 3);
const SHIELD_PAWN: i32 = 10;
const OPEN_FILE_NEAR_KING: i32 = -18;
const ATTACK_WEIGHT: [i32; 6] = [0, 2, 2, 3, 5, 0];

#[rustfmt::skip]
const SAFETY_TABLE: [i32; 100] = [
      0,   0,   1,   2,   3,   5,   7,   9,  11,  13,
     15,  17,  20,  23,  26,  30,  34,  38,  42,  46,
     50,  55,  60,  65,  70,  75,  80,  85,  90,  95,
    100, 105, 110, 115, 120, 125, 130, 135, 140, 145,
    150, 155, 160, 165, 170, 175, 180, 185, 190, 195,
    200, 205, 210, 215, 220, 225, 230, 235, 240, 245,
    250, 255, 260, 265, 270, 275, 280, 285, 290, 295,
    300, 305, 310, 315, 320, 325, 330, 335, 340, 345,
    350, 355, 360, 365, 370, 375, 380, 385, 390, 395,
    400, 405, 410, 415, 420, 425, 430, 435, 440, 445,
];

fn pawn_attacks_of(pawns: BitBoard, color: Color) -> BitBoard {
    let bb = pawns.0;
    let not_a = !File::A.bitboard().0;
    let not_h = !File::H.bitboard().0;
    match color {
        Color::White => BitBoard(((bb & not_a) << 7) | ((bb & not_h) << 9)),
        Color::Black => BitBoard(((bb & not_h) >> 7) | ((bb & not_a) >> 9)),
    }
}

/// Full static evaluation from the side to move's point of view.
pub fn evaluate(t: &Tables, board: &Board) -> i32 {
    let mut mg = [0i32; 2];
    let mut eg = [0i32; 2];
    let mut phase = 0;
    let occ = board.occupied();
    let pawns = board.pieces(Piece::Pawn);
    let pawn_att = [
        pawn_attacks_of(board.colored_pieces(Color::White, Piece::Pawn), Color::White),
        pawn_attacks_of(board.colored_pieces(Color::Black, Piece::Pawn), Color::Black),
    ];
    let kings = [board.king(Color::White), board.king(Color::Black)];
    // Attack accumulation on each king: [attack units, attacker count] indexed by defending colour.
    let mut king_attack = [(0i32, 0i32); 2];

    for color in Color::ALL {
        let ci = color as usize;
        let them = !color;
        let ti = them as usize;
        let us_bb = board.colors(color);
        let our_pawns = us_bb & pawns;
        let their_pawns = board.colors(them) & pawns;
        let mobility_area = !(us_bb & (pawns | board.pieces(Piece::King))) & !pawn_att[ti];
        let enemy_zone = t.king_zone[kings[ti] as usize];

        for sq in us_bb {
            let p = board.piece_on(sq).unwrap();
            let (pi, si) = (p as usize, sq as usize);
            mg[ci] += t.mg[ci][pi][si];
            eg[ci] += t.eg[ci][pi][si];
            phase += PHASE_INC[pi];
            match p {
                Piece::Pawn => {
                    let f = sq.file() as usize;
                    if (t.passed_mask[ci][si] & their_pawns).is_empty() {
                        let rel_rank = sq.rank().relative_to(color) as usize;
                        mg[ci] += PASSED_MG[rel_rank];
                        eg[ci] += PASSED_EG[rel_rank];
                    }
                    if (t.adjacent_files[f] & our_pawns).is_empty() {
                        mg[ci] += ISOLATED.0;
                        eg[ci] += ISOLATED.1;
                    }
                    if (front_span_fast(sq, color) & FILE_BBS[f] & our_pawns).len() > 0 {
                        mg[ci] += DOUBLED.0;
                        eg[ci] += DOUBLED.1;
                    }
                }
                Piece::Knight => {
                    let att = get_knight_moves(sq);
                    let mob = (att & mobility_area).len() as i32 - 4;
                    mg[ci] += mob * KNIGHT_MOB.0;
                    eg[ci] += mob * KNIGHT_MOB.1;
                    let z = (att & enemy_zone).len() as i32;
                    if z > 0 {
                        king_attack[ti].0 += ATTACK_WEIGHT[pi] * z;
                        king_attack[ti].1 += 1;
                    }
                }
                Piece::Bishop => {
                    let att = get_bishop_moves(sq, occ);
                    let mob = (att & mobility_area).len() as i32 - 6;
                    mg[ci] += mob * BISHOP_MOB.0;
                    eg[ci] += mob * BISHOP_MOB.1;
                    let z = (att & enemy_zone).len() as i32;
                    if z > 0 {
                        king_attack[ti].0 += ATTACK_WEIGHT[pi] * z;
                        king_attack[ti].1 += 1;
                    }
                }
                Piece::Rook => {
                    let att = get_rook_moves(sq, occ);
                    let mob = (att & mobility_area).len() as i32 - 7;
                    mg[ci] += mob * ROOK_MOB.0;
                    eg[ci] += mob * ROOK_MOB.1;
                    let z = (att & enemy_zone).len() as i32;
                    if z > 0 {
                        king_attack[ti].0 += ATTACK_WEIGHT[pi] * z;
                        king_attack[ti].1 += 1;
                    }
                    let file_bb = FILE_BBS[sq.file() as usize];
                    if (file_bb & our_pawns).is_empty() {
                        if (file_bb & their_pawns).is_empty() {
                            mg[ci] += ROOK_OPEN.0;
                            eg[ci] += ROOK_OPEN.1;
                        } else {
                            mg[ci] += ROOK_SEMI.0;
                            eg[ci] += ROOK_SEMI.1;
                        }
                    }
                    if sq.rank().relative_to(color) == Rank::Seventh
                        && (kings[ti].rank().relative_to(color) == Rank::Eighth
                            || !(their_pawns & Rank::Seventh.relative_to(color).bitboard()).is_empty())
                    {
                        mg[ci] += ROOK_SEVENTH.0;
                        eg[ci] += ROOK_SEVENTH.1;
                    }
                }
                Piece::Queen => {
                    let att = get_bishop_moves(sq, occ) | get_rook_moves(sq, occ);
                    let mob = (att & mobility_area).len() as i32 - 13;
                    mg[ci] += mob * QUEEN_MOB.0;
                    eg[ci] += mob * QUEEN_MOB.1;
                    let z = (att & enemy_zone).len() as i32;
                    if z > 0 {
                        king_attack[ti].0 += ATTACK_WEIGHT[pi] * z;
                        king_attack[ti].1 += 1;
                    }
                }
                Piece::King => {}
            }
        }

        if (us_bb & board.pieces(Piece::Bishop)).len() >= 2 {
            mg[ci] += BISHOP_PAIR.0;
            eg[ci] += BISHOP_PAIR.1;
        }

        // Pawn shield and open files around our king (middlegame only).
        let ksq = kings[ci];
        let kf = ksq.file() as usize;
        let files = FILE_BBS[kf] | t.adjacent_files[kf];
        let shield_ranks = match color {
            Color::White => Rank::Second.bitboard() | Rank::Third.bitboard(),
            Color::Black => Rank::Seventh.bitboard() | Rank::Sixth.bitboard(),
        };
        if ksq.rank().relative_to(color) as usize <= 1 {
            mg[ci] += (files & shield_ranks & our_pawns).len() as i32 * SHIELD_PAWN;
        }
        for f in kf.saturating_sub(1)..=(kf + 1).min(7) {
            if (FILE_BBS[f] & our_pawns).is_empty() {
                mg[ci] += OPEN_FILE_NEAR_KING;
            }
        }
    }

    // Apply king attack penalties to the defender.
    for ci in 0..2 {
        let (units, count) = king_attack[ci];
        if count >= 2 {
            mg[ci] -= SAFETY_TABLE[(units as usize).min(99)];
        }
    }

    let stm = board.side_to_move() as usize;
    let mg_score = mg[stm] - mg[stm ^ 1];
    let eg_score = eg[stm] - eg[stm ^ 1];
    let mg_phase = phase.min(24);
    let eg_phase = 24 - mg_phase;
    (mg_score * mg_phase + eg_score * eg_phase) / 24 + TEMPO
}

#[inline]
fn front_span_fast(sq: Square, color: Color) -> BitBoard {
    let r = sq.rank() as u32;
    match color {
        Color::White => {
            if r >= 7 {
                BitBoard::EMPTY
            } else {
                BitBoard(!0u64 << (8 * (r + 1)))
            }
        }
        Color::Black => BitBoard((1u64 << (8 * r)) - 1),
    }
}

pub fn piece_value(p: Piece) -> i32 {
    match p {
        Piece::Pawn => 100,
        Piece::Knight => 320,
        Piece::Bishop => 330,
        Piece::Rook => 500,
        Piece::Queen => 950,
        Piece::King => 20000,
    }
}
