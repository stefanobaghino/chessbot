//! Lock-free shared transposition table: 16-byte entries stored as two atomics
//! (key ^ data, data) so torn writes are detected on probe.

use cozy_chess::{Move, Piece, Square};
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Bound {
    None = 0,
    Upper = 1,
    Lower = 2,
    Exact = 3,
}

#[derive(Clone, Copy, Debug)]
pub struct Entry {
    pub mv: u16,
    pub score: i16,
    pub eval: i16,
    pub depth: i8,
    pub bound: u8,
    pub age: u8,
}

impl Entry {
    fn pack(&self) -> u64 {
        (self.mv as u64)
            | ((self.score as u16 as u64) << 16)
            | ((self.eval as u16 as u64) << 32)
            | ((self.depth as u8 as u64) << 48)
            | ((self.bound as u64 & 3) << 56)
            | ((self.age as u64 & 63) << 58)
    }
    fn unpack(d: u64) -> Entry {
        Entry {
            mv: d as u16,
            score: (d >> 16) as u16 as i16,
            eval: (d >> 32) as u16 as i16,
            depth: (d >> 48) as u8 as i8,
            bound: ((d >> 56) & 3) as u8,
            age: ((d >> 58) & 63) as u8,
        }
    }
    pub fn bound(&self) -> Bound {
        match self.bound {
            1 => Bound::Upper,
            2 => Bound::Lower,
            3 => Bound::Exact,
            _ => Bound::None,
        }
    }
    pub fn best_move(&self) -> Option<Move> {
        unpack_move(self.mv)
    }
}

pub fn pack_move(mv: Option<Move>) -> u16 {
    match mv {
        None => 0,
        Some(m) => {
            let promo = match m.promotion {
                None => 0,
                Some(Piece::Knight) => 1,
                Some(Piece::Bishop) => 2,
                Some(Piece::Rook) => 3,
                Some(Piece::Queen) => 4,
                Some(_) => 0,
            };
            1 << 15 | (promo << 12) | ((m.from as u16) << 6) | (m.to as u16)
        }
    }
}

pub fn unpack_move(v: u16) -> Option<Move> {
    if v == 0 {
        return None;
    }
    let promo = match (v >> 12) & 7 {
        1 => Some(Piece::Knight),
        2 => Some(Piece::Bishop),
        3 => Some(Piece::Rook),
        4 => Some(Piece::Queen),
        _ => None,
    };
    Some(Move { from: Square::index(((v >> 6) & 63) as usize), to: Square::index((v & 63) as usize), promotion: promo })
}

struct Slot {
    key: AtomicU64,
    data: AtomicU64,
}

pub struct TranspositionTable {
    table: Vec<Slot>,
    mask: usize,
    age: AtomicU8,
}

impl TranspositionTable {
    pub fn new(mb: usize) -> Self {
        let mut t = TranspositionTable { table: Vec::new(), mask: 0, age: AtomicU8::new(0) };
        t.resize(mb);
        t
    }

    pub fn resize(&mut self, mb: usize) {
        let bytes = mb.max(1) * 1024 * 1024;
        let mut n = bytes / 16;
        n = n.next_power_of_two() / 2;
        n = n.max(1024);
        self.table = (0..n).map(|_| Slot { key: AtomicU64::new(0), data: AtomicU64::new(0) }).collect();
        self.mask = n - 1;
    }

    pub fn clear(&self) {
        for s in self.table.iter() {
            s.key.store(0, Ordering::Relaxed);
            s.data.store(0, Ordering::Relaxed);
        }
        self.age.store(0, Ordering::Relaxed);
    }

    pub fn new_search(&self) {
        self.age.fetch_add(1, Ordering::Relaxed);
    }

    fn age(&self) -> u8 {
        self.age.load(Ordering::Relaxed) & 63
    }

    #[inline]
    fn index(&self, hash: u64) -> usize {
        (hash as usize) & self.mask
    }

    pub fn probe(&self, hash: u64) -> Option<Entry> {
        let s = &self.table[self.index(hash)];
        let k = s.key.load(Ordering::Relaxed);
        let d = s.data.load(Ordering::Relaxed);
        if d != 0 && (k ^ d) == hash {
            Some(Entry::unpack(d))
        } else {
            None
        }
    }

    pub fn store(&self, hash: u64, mv: Option<Move>, score: i32, eval: i32, depth: i32, bound: Bound) {
        let s = &self.table[self.index(hash)];
        let age = self.age();
        let k = s.key.load(Ordering::Relaxed);
        let d = s.data.load(Ordering::Relaxed);
        let existing = if d != 0 && (k ^ d) == hash { Some(Entry::unpack(d)) } else { None };
        if let Some(e) = existing {
            if e.age == age && bound != Bound::Exact && (depth as i8) < e.depth - 2 {
                return;
            }
        }
        let packed = pack_move(mv);
        let mv_to_store = match (packed, existing) {
            (0, Some(e)) => e.mv,
            _ => packed,
        };
        let e = Entry {
            mv: mv_to_store,
            score: score.clamp(-32000, 32000) as i16,
            eval: eval.clamp(-32000, 32000) as i16,
            depth: depth.clamp(-1, 127) as i8,
            bound: bound as u8,
            age,
        };
        let nd = e.pack();
        s.data.store(nd, Ordering::Relaxed);
        s.key.store(hash ^ nd, Ordering::Relaxed);
    }

    pub fn hashfull(&self) -> usize {
        let sample = self.table.len().min(1000);
        let age = self.age();
        let used = self.table[..sample]
            .iter()
            .filter(|s| {
                let d = s.data.load(Ordering::Relaxed);
                d != 0 && Entry::unpack(d).age == age
            })
            .count();
        used * 1000 / sample
    }
}
