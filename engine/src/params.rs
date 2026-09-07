//! Search margins as named parameters (#39). In a normal build every accessor is a
//! constant and inlines away; with the `tune` cargo feature each one reads an atomic that
//! the UCI `setoption` handler writes, so a tuner can move the margins between games.

macro_rules! params {
    ($($name:ident = $default:expr, $min:expr, $max:expr, $step:expr;)*) => {
        #[cfg(feature = "tune")]
        mod cells {
            use std::sync::atomic::AtomicI32;
            $(#[allow(non_upper_case_globals)] pub static $name: AtomicI32 = AtomicI32::new($default);)*
        }
        $(
            #[inline(always)]
            #[allow(dead_code)]
            pub fn $name() -> i32 {
                #[cfg(feature = "tune")]
                { cells::$name.load(std::sync::atomic::Ordering::Relaxed) }
                #[cfg(not(feature = "tune"))]
                { $default }
            }
        )*
        /// Name, default, min, max, step of every parameter, for the UCI option list.
        pub const TABLE: &[(&str, i32, i32, i32, i32)] = &[$((stringify!($name), $default, $min, $max, $step)),*];
        /// Set a parameter by name (case-insensitive), clamped to its range. False if unknown.
        #[cfg(feature = "tune")]
        pub fn set(name: &str, value: i32) -> bool {
            use std::sync::atomic::Ordering;
            $(
                if name.eq_ignore_ascii_case(stringify!($name)) {
                    cells::$name.store(value.clamp($min, $max), Ordering::Relaxed);
                    return true;
                }
            )*
            false
        }
    };
}

params! {
    rfp_margin = 75, 30, 150, 10;
    rfp_improving = 50, 0, 120, 10;
    razor_margin = 250, 100, 500, 25;
    nmp_eval_div = 200, 80, 400, 20;
    probcut_margin = 180, 100, 320, 20;
    probcut_improving = 40, 0, 120, 10;
    futility_base = 100, 0, 300, 20;
    futility_per_depth = 110, 40, 220, 10;
    history_prune = 3000, 1000, 8000, 500;
    see_prune = 60, 20, 160, 10;
    singular_margin = 2, 1, 6, 1;
    aspiration_delta = 18, 8, 48, 4;
    lmr_history_div = 6000, 2000, 14000, 1000;
}
