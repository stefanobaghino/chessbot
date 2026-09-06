//! Lock-free shared transposition table: 16-byte entries stored as two atomics
//! (key ^ data, data) so torn writes are detected on probe, four to a cache line.

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

impl Slot {
    const fn empty() -> Slot {
        Slot { key: AtomicU64::new(0), data: AtomicU64::new(0) }
    }
}

/// Four entries sharing a cache line (#30): a position may live in any of them, so a
/// collision evicts the shallowest or oldest entry of the bucket instead of the only one.
pub const BUCKET: usize = 4;

#[repr(align(64))]
struct Bucket {
    slots: [Slot; BUCKET],
}

pub struct TranspositionTable {
    table: Vec<Bucket>,
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
        let mut n = bytes / std::mem::size_of::<Bucket>();
        n = n.next_power_of_two() / 2;
        n = n.max(256);
        self.table = (0..n).map(|_| Bucket { slots: [Slot::empty(), Slot::empty(), Slot::empty(), Slot::empty()] }).collect();
        self.mask = n - 1;
    }

    pub fn clear(&self) {
        for b in self.table.iter() {
            for s in b.slots.iter() {
                s.key.store(0, Ordering::Relaxed);
                s.data.store(0, Ordering::Relaxed);
            }
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
    fn bucket(&self, hash: u64) -> &Bucket {
        &self.table[(hash as usize) & self.mask]
    }

    /// Pulls the bucket of `hash` towards the cache; called as soon as a child position's
    /// hash is known so the fetch overlaps with the accumulator update.
    #[inline]
    pub fn prefetch(&self, hash: u64) {
        let p = self.bucket(hash) as *const Bucket;
        #[cfg(target_arch = "aarch64")]
        // SAFETY: prfm is a hint; it never faults and touches no register state.
        unsafe {
            std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) p, options(nomem, nostack, preserves_flags));
        }
        #[cfg(target_arch = "x86_64")]
        // SAFETY: prefetch is a hint that never faults.
        unsafe {
            std::arch::x86_64::_mm_prefetch(p as *const i8, std::arch::x86_64::_MM_HINT_T0);
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        let _ = p;
    }

    pub fn probe(&self, hash: u64) -> Option<Entry> {
        for s in self.bucket(hash).slots.iter() {
            let k = s.key.load(Ordering::Relaxed);
            let d = s.data.load(Ordering::Relaxed);
            if d != 0 && (k ^ d) == hash {
                return Some(Entry::unpack(d));
            }
        }
        None
    }

    pub fn store(&self, hash: u64, mv: Option<Move>, score: i32, eval: i32, depth: i32, bound: Bound) {
        let age = self.age();
        let bucket = self.bucket(hash);
        // The position's own slot if present, else an empty one, else the entry worth the
        // least: shallow and from an old search.
        let mut own: Option<(&Slot, Entry)> = None;
        let mut empty: Option<&Slot> = None;
        let mut worst: Option<(&Slot, i32)> = None;
        for s in bucket.slots.iter() {
            let k = s.key.load(Ordering::Relaxed);
            let d = s.data.load(Ordering::Relaxed);
            if d == 0 {
                empty.get_or_insert(s);
                continue;
            }
            let e = Entry::unpack(d);
            if (k ^ d) == hash {
                own = Some((s, e));
                break;
            }
            let value = e.depth as i32 - 8 * ((age as i32 - e.age as i32) & 63);
            if worst.is_none_or(|w| value < w.1) {
                worst = Some((s, value));
            }
        }
        let (slot, existing) = match (own, empty, worst) {
            (Some((s, e)), _, _) => (s, Some(e)),
            (None, Some(s), _) => (s, None),
            (None, None, Some((s, _))) => (s, None),
            (None, None, None) => return,
        };
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
        slot.data.store(nd, Ordering::Relaxed);
        slot.key.store(hash ^ nd, Ordering::Relaxed);
    }

    pub fn hashfull(&self) -> usize {
        let sample = self.table.len().min(250);
        let age = self.age();
        let used = self.table[..sample]
            .iter()
            .flat_map(|b| b.slots.iter())
            .filter(|s| {
                let d = s.data.load(Ordering::Relaxed);
                d != 0 && Entry::unpack(d).age == age
            })
            .count();
        used * 1000 / (sample * BUCKET)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn move_packing_roundtrips() {
        for from in [Square::A1, Square::E2, Square::H8] {
            for to in [Square::A8, Square::E4, Square::H1] {
                for promo in [None, Some(Piece::Queen), Some(Piece::Knight)] {
                    let mv = Move { from, to, promotion: promo };
                    assert_eq!(unpack_move(pack_move(Some(mv))), Some(mv));
                }
            }
        }
        assert_eq!(unpack_move(pack_move(None)), None);
    }

    #[test]
    fn store_and_probe() {
        let tt = TranspositionTable::new(1);
        let mv = Move { from: Square::E2, to: Square::E4, promotion: None };
        tt.store(0x1234_5678_9abc_def0, Some(mv), 42, 10, 7, Bound::Exact);
        let e = tt.probe(0x1234_5678_9abc_def0).expect("entry");
        assert_eq!(e.best_move(), Some(mv));
        assert_eq!(e.score, 42);
        assert_eq!(e.eval, 10);
        assert_eq!(e.depth, 7);
        assert_eq!(e.bound(), Bound::Exact);
        assert!(tt.probe(0x1234_5678_9abc_def1).is_none());
    }

    #[test]
    fn bucket_keeps_colliding_positions_and_evicts_the_least_valuable() {
        let tt = TranspositionTable::new(1);
        let base = 0x0000_0000_0000_0100u64;
        let stride = (tt.mask as u64 + 1) << 0; // same bucket index, different keys
        let hashes: Vec<u64> = (0..BUCKET as u64 + 1).map(|i| base + i * stride).collect();
        for (i, h) in hashes.iter().enumerate().take(BUCKET) {
            tt.store(*h, None, i as i32, 0, 10 + i as i32, Bound::Exact);
        }
        for (i, h) in hashes.iter().enumerate().take(BUCKET) {
            assert_eq!(tt.probe(*h).expect("kept").score, i as i16);
        }
        // A fifth position evicts the shallowest entry (depth 10, score 0) and only that one.
        tt.store(hashes[BUCKET], None, 99, 0, 5, Bound::Exact);
        assert!(tt.probe(hashes[0]).is_none());
        for (i, h) in hashes.iter().enumerate().take(BUCKET).skip(1) {
            assert_eq!(tt.probe(*h).expect("kept").score, i as i16);
        }
        assert_eq!(tt.probe(hashes[BUCKET]).expect("stored").score, 99);
        // An entry from an older search goes before a shallower one of this search.
        tt.new_search();
        tt.store(base + 7 * stride, None, 7, 0, 3, Bound::Exact);
        assert!(tt.probe(base + 7 * stride).is_some());
    }
}
