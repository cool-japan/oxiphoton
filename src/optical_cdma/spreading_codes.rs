//! Optical spreading codes for OCDMA systems.
//!
//! Implements three families of spreading codes used in optical CDMA:
//!
//! * **Optical Orthogonal Codes (OOC)** — unipolar {0,1} codes with
//!   controlled auto/cross-correlation, suitable for incoherent OCDMA.
//! * **OVSF (Orthogonal Variable Spreading Factor)** — Walsh-Hadamard
//!   derived tree structure used in wideband CDMA.
//! * **Gold codes** — m-sequence pairs with bounded cross-correlation,
//!   used in coherent CDMA and ranging.

use std::f64::consts::LOG10_2;

// ---------------------------------------------------------------------------
// Optical Orthogonal Code (OOC)
// ---------------------------------------------------------------------------

/// Optical Orthogonal Code family descriptor: (n, w, λ_a, λ_c).
///
/// An OOC is a set of {0,1} codewords each of length `n` and weight `w`
/// (number of ones) satisfying:
/// * Autocorrelation sidelobe ≤ `lambda_a` for all non-zero shifts.
/// * Cross-correlation ≤ `lambda_c` for all shifts between distinct codewords.
///
/// The theoretical upper bound on codeword count (Johnson bound) for λ=1 is
/// `⌊(n-1) / (w·(w-1))⌋`.
#[derive(Debug, Clone)]
pub struct OpticalOrthogonalCode {
    /// Code length (chips per symbol).
    pub n: usize,
    /// Code weight (number of ones per codeword).
    pub w: usize,
    /// Maximum auto-correlation sidelobe.
    pub lambda_a: usize,
    /// Maximum cross-correlation value.
    pub lambda_c: usize,
}

impl OpticalOrthogonalCode {
    /// Create a new OOC descriptor.
    pub fn new(n: usize, w: usize, lambda_a: usize, lambda_c: usize) -> Self {
        Self {
            n,
            w,
            lambda_a,
            lambda_c,
        }
    }

    /// Johnson upper bound on the number of codewords.
    ///
    /// For λ_a = λ_c = 1: `|Φ| ≤ ⌊(n-1) / (w·(w-1))⌋`.
    /// For w ≤ 1 the bound is n (trivially one chip per codeword).
    pub fn max_codewords(&self) -> usize {
        if self.w <= 1 {
            return self.n;
        }
        (self.n - 1) / (self.w * (self.w - 1))
    }

    /// Cyclic cross-correlation between two binary codewords at shift τ.
    ///
    /// Defined as `Σ_{i=0}^{n-1} c1[i] · c2[(i+τ) mod n]`.
    pub fn crosscorrelation(&self, c1: &[u8], c2: &[u8], tau: usize) -> usize {
        let n = c1.len().min(c2.len());
        (0..n)
            .map(|i| (c1[i] as usize) * (c2[(i + tau) % n] as usize))
            .sum()
    }

    /// Cyclic auto-correlation of a single codeword at shift τ.
    ///
    /// For τ = 0 this equals the code weight w.
    pub fn autocorrelation(&self, code: &[u8], tau: usize) -> usize {
        self.crosscorrelation(code, code, tau)
    }

    /// Verify that two codewords satisfy the OOC constraints.
    ///
    /// Returns `true` when:
    /// * Both codewords have weight `w`.
    /// * Autocorrelation sidelobes (τ ≠ 0) ≤ `lambda_a`.
    /// * All cyclic cross-correlations ≤ `lambda_c`.
    pub fn verify_code(&self, code1: &[u8], code2: &[u8]) -> bool {
        let n = self.n;
        // Weight check
        let w1: usize = code1.iter().map(|&b| b as usize).sum();
        let w2: usize = code2.iter().map(|&b| b as usize).sum();
        if w1 != self.w || w2 != self.w {
            return false;
        }
        // Autocorrelation sidelobe check for code1
        for tau in 1..n {
            if self.autocorrelation(code1, tau) > self.lambda_a {
                return false;
            }
        }
        // Autocorrelation sidelobe check for code2
        for tau in 1..n {
            if self.autocorrelation(code2, tau) > self.lambda_a {
                return false;
            }
        }
        // Cross-correlation check (all shifts)
        for tau in 0..n {
            if self.crosscorrelation(code1, code2, tau) > self.lambda_c {
                return false;
            }
        }
        true
    }

    /// Generate a set of OOC codewords using a greedy cyclic-difference-set
    /// construction.
    ///
    /// Starting from candidate positions the algorithm advances the last `1`-
    /// chip cyclically and accepts each candidate codeword only if it
    /// satisfies the OOC constraints against all already accepted codewords.
    /// The result may contain fewer codewords than the theoretical maximum for
    /// large codes, but is always valid.
    pub fn generate_codes(&self) -> Vec<Vec<u8>> {
        let max = self.max_codewords();
        let mut accepted: Vec<Vec<u8>> = Vec::new();

        // Iterate over all weight-w subsets of {0..n-1} in lexicographic order
        // using a recursive-style iterator encoded as a simple counter over
        // combination indices.
        let mut positions = vec![0usize; self.w];
        // Initialise first candidate: 0, 1, 2, …, w-1
        for (i, p) in positions.iter_mut().enumerate() {
            *p = i;
        }

        'outer: loop {
            // Build candidate codeword
            let mut cand = vec![0u8; self.n];
            for &p in &positions {
                cand[p] = 1;
            }

            // Check autocorrelation sidelobes
            let mut ac_ok = true;
            for tau in 1..self.n {
                if self.autocorrelation(&cand, tau) > self.lambda_a {
                    ac_ok = false;
                    break;
                }
            }

            if ac_ok {
                // Check cross-correlation against all accepted codewords
                let mut cc_ok = true;
                'cc: for prev in &accepted {
                    for tau in 0..self.n {
                        if self.crosscorrelation(&cand, prev, tau) > self.lambda_c {
                            cc_ok = false;
                            break 'cc;
                        }
                    }
                }

                if cc_ok {
                    accepted.push(cand);
                    if accepted.len() >= max {
                        break;
                    }
                }
            }

            // Advance to next combination (combinatorial number system)
            let w = self.w;
            let n = self.n;
            let mut i = w;
            loop {
                if i == 0 {
                    break 'outer; // exhausted all combinations
                }
                i -= 1;
                positions[i] += 1;
                if positions[i] <= n - (w - i) {
                    // Reset all subsequent positions
                    for j in (i + 1)..w {
                        positions[j] = positions[j - 1] + 1;
                    }
                    break;
                }
            }
        }

        accepted
    }
}

// ---------------------------------------------------------------------------
// OVSF (Orthogonal Variable Spreading Factor) tree
// ---------------------------------------------------------------------------

/// OVSF code tree up to a maximum spreading factor.
///
/// OVSF codes are constructed recursively:
/// ```text
/// C_{1,0} = [1]
/// C_{2SF, 2k}   = [C_{SF,k},  C_{SF,k}]
/// C_{2SF, 2k+1} = [C_{SF,k}, -C_{SF,k}]
/// ```
/// Codes at the same spreading factor are mutually orthogonal; a code is
/// *not* orthogonal to its ancestor or descendant nodes in the tree.
#[derive(Debug, Clone)]
pub struct OvsfTree {
    /// Maximum spreading factor (must be a power of two).
    pub max_sf: usize,
}

impl OvsfTree {
    /// Create a new OVSF tree with the given maximum spreading factor.
    ///
    /// # Panics (in debug mode)
    /// Panics if `max_sf` is not a power of two.
    pub fn new(max_sf: usize) -> Self {
        debug_assert!(max_sf.is_power_of_two(), "max_sf must be a power of two");
        Self { max_sf }
    }

    /// Generate OVSF code `C_{sf, idx}` recursively.
    ///
    /// * `sf`  — spreading factor (power of two, 1 ≤ sf ≤ max_sf)
    /// * `idx` — code index within the spreading factor (0 ≤ idx < sf)
    pub fn code(&self, sf: usize, idx: usize) -> Vec<i8> {
        if sf == 1 {
            return vec![1i8];
        }
        let parent_sf = sf / 2;
        let parent_idx = idx / 2;
        let parent = self.code(parent_sf, parent_idx);
        let sign: i8 = if idx % 2 == 0 { 1 } else { -1 };
        let mut out = Vec::with_capacity(sf);
        out.extend_from_slice(&parent);
        out.extend(parent.iter().map(|&v| sign * v));
        out
    }

    /// Check whether two OVSF codes are orthogonal (zero dot product).
    ///
    /// Codes at the **same** spreading factor with different indices are always
    /// orthogonal. A code is *never* orthogonal to its ancestor or descendant.
    pub fn are_orthogonal(&self, sf1: usize, idx1: usize, sf2: usize, idx2: usize) -> bool {
        let c1 = self.code(sf1, idx1);
        let c2 = self.code(sf2, idx2);
        // Compute dot product over the shorter code length by repeating
        // the shorter code (cross-SF case).
        let max_len = c1.len().max(c2.len());
        let dot: i64 = (0..max_len)
            .map(|i| (c1[i % c1.len()] as i64) * (c2[i % c2.len()] as i64))
            .sum();
        dot == 0
    }

    /// Number of codes available at a given spreading factor.
    pub fn capacity_at_sf(&self, sf: usize) -> usize {
        sf
    }
}

// ---------------------------------------------------------------------------
// Gold codes (m-sequences + Gold construction)
// ---------------------------------------------------------------------------

/// Gold-code generator based on two maximal-length shift-register sequences.
///
/// A Gold code family of length N = 2^n − 1 contains N + 2 sequences and
/// exhibits a three-valued cross-correlation set: {−1, −t(n), t(n)−2} where
/// t(n) = 2^⌈(n+2)/2⌉ + 1.
#[derive(Debug, Clone)]
pub struct GoldCode {
    /// Chip-sequence length (2^n_bits - 1).
    pub length: usize,
    /// Shift-register length.
    pub n_bits: usize,
}

impl GoldCode {
    /// Create a new Gold-code generator with shift-register length `n_bits`.
    pub fn new(n_bits: usize) -> Self {
        Self {
            length: (1usize << n_bits).saturating_sub(1),
            n_bits,
        }
    }

    /// Generate a maximal-length (m-) sequence.
    ///
    /// * `seed` — initial shift-register state (must be non-zero).
    /// * `taps` — 1-based tap positions defining the primitive polynomial.
    ///
    /// The output is a `±1` bipolar sequence of length `2^n - 1`.
    /// For n = 7 use taps `&[7, 3]` (x^7 + x^3 + 1).
    pub fn m_sequence(&self, seed: u32, taps: &[usize]) -> Vec<i8> {
        let n = self.n_bits;
        let len = self.length;
        // Clamp seed to valid non-zero n-bit value
        let mask = ((1u32 << n) - 1).max(1);
        let mut state = (seed & mask).max(1) & mask;

        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            // Output LSB as ±1
            let bit = (state & 1) as i8;
            out.push(if bit == 0 { 1i8 } else { -1i8 });

            // XOR tapped bits (1-based positions)
            let feedback: u32 = taps
                .iter()
                .map(|&t| (state >> (t - 1)) & 1)
                .fold(0u32, |acc, b| acc ^ b);
            state = ((state >> 1) | (feedback << (n - 1))) & mask;
        }
        out
    }

    /// Default preferred-pair polynomials for n-bit m-sequences.
    ///
    /// Returns `(taps1, taps2)` for the standard preferred Gold-code pairs.
    fn default_taps(&self) -> (Vec<usize>, Vec<usize>) {
        match self.n_bits {
            3 => (vec![3, 1], vec![3, 2]),
            4 => (vec![4, 1], vec![4, 3]),
            5 => (vec![5, 2], vec![5, 4, 2, 1]),
            6 => (vec![6, 1], vec![6, 5, 2, 1]),
            7 => (vec![7, 3], vec![7, 3, 2, 1]),
            8 => (vec![8, 4, 3, 2], vec![8, 6, 5, 1]),
            9 => (vec![9, 4], vec![9, 6, 4, 3]),
            10 => (vec![10, 3], vec![10, 8, 3, 2]),
            _ => (vec![self.n_bits, 1], vec![self.n_bits, 2]),
        }
    }

    /// Generate a Gold code as the element-wise XOR of two m-sequences.
    ///
    /// * `seed1`, `seed2` — distinct initial states for the two registers.
    ///
    /// The XOR of ±1 sequences is realised as element-wise multiplication.
    pub fn gold_code(&self, seed1: u32, seed2: u32) -> Vec<i8> {
        let (taps1, taps2) = self.default_taps();
        let seq1 = self.m_sequence(seed1, &taps1);
        let seq2 = self.m_sequence(seed2, &taps2);
        seq1.iter().zip(seq2.iter()).map(|(&a, &b)| a * b).collect()
    }

    /// Processing gain in dB.
    ///
    /// `PG = 10 · log10(N_chips / N_bits_per_symbol)`
    pub fn processing_gain_db(&self, bits_per_symbol: usize) -> f64 {
        if bits_per_symbol == 0 {
            return 0.0;
        }
        10.0 * (self.length as f64 / bits_per_symbol as f64).log10()
    }

    /// Three-valued cross-correlation bound t(n).
    ///
    /// t(n) = 2^⌈(n+2)/2⌉ + 1
    pub fn cross_correlation_bound(&self) -> i64 {
        let exp = (self.n_bits + 2).div_ceil(2); // ceiling of (n+2)/2
        (1i64 << exp) + 1
    }
}

/// Bits per octave, used internally for dB calculations.
#[allow(dead_code)]
const LOG10_2_CACHED: f64 = LOG10_2;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ooc_max_codewords() {
        let ooc = OpticalOrthogonalCode {
            n: 13,
            w: 3,
            lambda_a: 1,
            lambda_c: 1,
        };
        // ⌊(13-1)/(3×2)⌋ = ⌊12/6⌋ = 2
        assert_eq!(ooc.max_codewords(), 2);
    }

    #[test]
    fn ooc_autocorrelation_peak() {
        let ooc = OpticalOrthogonalCode {
            n: 7,
            w: 3,
            lambda_a: 1,
            lambda_c: 1,
        };
        // Code 0001011 (positions 3,4,6)
        let code = vec![0, 0, 0, 1, 0, 1, 1];
        assert_eq!(ooc.autocorrelation(&code, 0), 3); // peak = w
    }

    #[test]
    fn ooc_generate_codes_valid() {
        let ooc = OpticalOrthogonalCode {
            n: 13,
            w: 3,
            lambda_a: 1,
            lambda_c: 1,
        };
        let codes = ooc.generate_codes();
        // All generated codes should satisfy the constraints with each other
        assert!(!codes.is_empty());
        for c in &codes {
            assert_eq!(c.iter().map(|&b| b as usize).sum::<usize>(), 3);
        }
        if codes.len() >= 2 {
            assert!(ooc.verify_code(&codes[0], &codes[1]));
        }
    }

    #[test]
    fn ovsf_orthogonal_same_sf() {
        let tree = OvsfTree::new(8);
        // Same SF, different indices → orthogonal
        assert!(tree.are_orthogonal(4, 0, 4, 1));
        assert!(tree.are_orthogonal(4, 0, 4, 2));
        assert!(tree.are_orthogonal(4, 1, 4, 3));
    }

    #[test]
    fn ovsf_parent_child_not_orthogonal() {
        let tree = OvsfTree::new(8);
        // Parent-child relationship: NOT orthogonal
        assert!(!tree.are_orthogonal(4, 0, 2, 0));
        assert!(!tree.are_orthogonal(2, 0, 4, 0));
    }

    #[test]
    fn ovsf_code_length() {
        let tree = OvsfTree::new(16);
        assert_eq!(tree.code(8, 3).len(), 8);
        assert_eq!(tree.code(1, 0), vec![1i8]);
    }

    #[test]
    fn gold_processing_gain() {
        let gc = GoldCode::new(7);
        let pg = gc.processing_gain_db(1);
        // PG = 10 log10(127) ≈ 21.0 dB
        assert!((pg - 21.0).abs() < 1.0, "PG = {} dB (expected ≈21 dB)", pg);
    }

    #[test]
    fn gold_m_sequence_length() {
        let gc = GoldCode::new(5);
        let seq = gc.m_sequence(1, &[5, 2]);
        assert_eq!(seq.len(), 31); // 2^5 - 1
                                   // All elements must be ±1
        assert!(seq.iter().all(|&x| x == 1 || x == -1));
    }
}
