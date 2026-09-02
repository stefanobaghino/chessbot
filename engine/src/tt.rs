//! Transposition table with 16-byte entries, replace-by-depth-or-age.

use cozy_chess::{Move, Piece, Square};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Bound {
    None = 0,
    Upper = 1,
    Lower = 2,
    Exact = 3,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct Entry {
    pub key: u32,
    pub mv: u16,
    pub score: i16,
    pub eval: i16,
    pub depth: i8,
    pub bound: u8,
    pub age: u8,
    _pad: u8,
}

impl Entry {
    const EMPTY: Entry = Entry { key: 0, mv: 0, score: 0, eval: 0, depth: -1, bound: 0, age: 0, _pad: 0 };

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

pub struct TranspositionTable {
    table: Vec<Entry>,
    mask: usize,
    pub age: u8,
}

impl TranspositionTable {
    pub fn new(mb: usize) -> Self {
        let mut t = TranspositionTable { table: Vec::new(), mask: 0, age: 0 };
        t.resize(mb);
        t
    }

    pub fn resize(&mut self, mb: usize) {
        let bytes = mb.max(1) * 1024 * 1024;
        let mut n = bytes / std::mem::size_of::<Entry>();
        n = n.next_power_of_two() / 2;
        n = n.max(1024);
        self.table = vec![Entry::EMPTY; n];
        self.mask = n - 1;
    }

    pub fn clear(&mut self) {
        for e in self.table.iter_mut() {
            *e = Entry::EMPTY;
        }
        self.age = 0;
    }

    pub fn new_search(&mut self) {
        self.age = self.age.wrapping_add(1);
    }

    #[inline]
    fn index(&self, hash: u64) -> usize {
        (hash as usize) & self.mask
    }

    pub fn probe(&self, hash: u64) -> Option<Entry> {
        let e = self.table[self.index(hash)];
        if e.key == (hash >> 32) as u32 && e.bound != 0 {
            Some(e)
        } else {
            None
        }
    }

    pub fn store(&mut self, hash: u64, mv: Option<Move>, score: i32, eval: i32, depth: i32, bound: Bound) {
        let idx = self.index(hash);
        let key = (hash >> 32) as u32;
        let e = &mut self.table[idx];
        let same = e.key == key && e.bound != 0;
        // Keep deeper entries from the same search unless we have an exact score.
        if same && e.age == self.age && bound != Bound::Exact && (depth as i8) < e.depth - 2 {
            return;
        }
        let packed = pack_move(mv);
        let mv_to_store = if packed == 0 && same { e.mv } else { packed };
        *e = Entry {
            key,
            mv: mv_to_store,
            score: score.clamp(-32000, 32000) as i16,
            eval: eval.clamp(-32000, 32000) as i16,
            depth: depth.clamp(-1, 127) as i8,
            bound: bound as u8,
            age: self.age,
            _pad: 0,
        };
    }

    pub fn hashfull(&self) -> usize {
        let sample = self.table.len().min(1000);
        let used = self.table[..sample].iter().filter(|e| e.bound != 0 && e.age == self.age).count();
        used * 1000 / sample
    }
}
