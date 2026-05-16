use crate::kmeans::KMeans;
use std::mem::size_of;

mod kmeans;
#[cfg(test)]
mod tests;

pub struct ProductQuantizer<const M: usize, const D: usize> {
    // M of K of sudim f32s
    codebooks: Vec<f32>,
    k: usize,
    subdims: usize,
    trained: bool,
}

impl<const M: usize, const D: usize> ProductQuantizer<M, D> {
    pub fn new(k: usize) -> Self {
        assert!(k <= 256, "k must fit in an u8");
        assert!(M > 0, "M must be greater than 0");
        assert!(
            D.is_multiple_of(M),
            "number of dimensions must be divisible by number of quantizers"
        );
        assert!(k > 0, "k must be greater than zero");
        Self {
            codebooks: Vec::new(),
            k,
            subdims: D / M,
            trained: false,
        }
    }

    pub fn fit(&mut self, data: &[[f32; D]]) {
        assert!(!self.trained, "ProductQuantizer was already trained");
        assert!(data.len() >= self.k, "not enough vectors");

        let mut codebooks = Vec::with_capacity(M * self.k * self.subdims);
        for m in 0..M {
            let mut d: Vec<Vec<f32>> = Vec::with_capacity(data.len());
            for v in data.iter() {
                d.push(v[self.subdim_range(m)].to_vec());
            }
            let mut quantizer = KMeans::new(d, self.k, self.subdims);
            quantizer.train();
            codebooks.extend(quantizer.centroids);
        }

        self.trained = true;
        self.codebooks = codebooks;
    }

    pub fn encode(&self, query: &[f32; D]) -> [u8; M] {
        assert!(
            self.trained,
            "ProductQuantizer must be trained to encode a query"
        );

        let mut enc = [0; M];
        for m in 0..M {
            let (min, _) = closest_centroid(
                &self.codebooks[self.all_centroids_range(m)],
                self.subdims,
                &query[self.subdim_range(m)],
            );
            enc[m] = min;
        }

        enc
    }

    pub fn decode(&self, code: &[u8; M]) -> [f32; D] {
        assert!(
            self.trained,
            "ProductQuantizer must be trained to decode a code"
        );
        let mut dec = [0_f32; D];
        for m in 0..M {
            assert!(code[m] < self.k as u8, "code index out of bounds");
            let centroid = &self.codebooks[self.one_centroid_range(m, code[m] as usize)];
            dec[self.subdim_range(m)].copy_from_slice(centroid);
        }
        dec
    }

    pub fn adc_table(&self, query: &[f32; D]) -> Vec<f32> {
        assert!(
            self.trained,
            "ProductQuantizer must be trained to build the adc_table"
        );
        let mut table = vec![0_f32; self.k * M];
        for m in 0..M {
            let q = &query[self.subdim_range(m)];
            for id in 0..self.k {
                let dist = l2_squared(q, &self.codebooks[self.one_centroid_range(m, id)]);
                table[(m * self.k) + id] = dist;
            }
        }
        table
    }

    #[inline(always)]
    pub fn adc_distance(&self, table: &[f32], q_code: &[u8; M]) -> f32 {
        assert!(table.len() == self.k * M, "adc table has invalid shape");
        let mut dist = 0.0;
        for m in 0..M {
            assert!(q_code[m] < self.k as u8, "code index out of bounds");
            dist += table[(m * self.k) + q_code[m] as usize];
        }
        dist
    }

    pub fn sdc_table(&self) -> Vec<Vec<Vec<f32>>> {
        assert!(
            self.trained,
            "ProductQuantizer must be trained to build the sdc_table"
        );
        let mut adc_table = vec![vec![vec![0_f32; self.k]; self.k]; M];
        for m in 0..M {
            for i in 0..self.k {
                for j in 0..self.k {
                    let dist = l2_squared(
                        &self.codebooks[self.one_centroid_range(m, i)],
                        &self.codebooks[self.one_centroid_range(m, j)],
                    );
                    adc_table[m][i][j] = dist;
                }
            }
        }
        adc_table
    }

    pub fn heap_usage_bytes(&self) -> usize {
        self.codebooks.capacity() * size_of::<f32>()
    }

    fn subdim_range(&self, m: usize) -> std::ops::Range<usize> {
        let start = m * self.subdims;
        let end = start + self.subdims;
        start..end
    }

    fn all_centroids_range(&self, m: usize) -> std::ops::Range<usize> {
        self.centroid_slice_range(m, 0, self.k)
    }

    fn one_centroid_range(&self, m: usize, id: usize) -> std::ops::Range<usize> {
        self.centroid_slice_range(m, id, 1)
    }

    fn centroid_slice_range(
        &self,
        m: usize,
        starting_from: usize,
        how_many: usize,
    ) -> std::ops::Range<usize> {
        assert!(m < M, "subquantizer index out of bounds");
        assert!(starting_from <= self.k, "centroid index out of bounds");
        assert!(
            starting_from + how_many <= self.k,
            "centroid range out of bounds"
        );
        let start = (m * self.k + starting_from) * self.subdims;
        let end = start + (how_many * self.subdims);
        start..end
    }
}

pub fn sdc_distance<const M: usize>(table: &[Vec<Vec<f32>>], a: &[u8; M], b: &[u8; M]) -> f32 {
    assert_eq!(table.len(), M, "sdc table has wrong number of quantizers");
    let mut dist = 0.0;
    for m in 0..M {
        dist += table[m][a[m] as usize][b[m] as usize];
    }
    dist
}

fn closest_centroid<'a>(centroids: &'a [f32], d: usize, q: &[f32]) -> (u8, &'a [f32]) {
    assert!(!centroids.is_empty(), "not enough centroids");
    assert!(d > 0, "centroid dimensions must be greater than zero");
    assert!(
        centroids.len().is_multiple_of(d),
        "centroids have invalid shape"
    );

    let mut min = 0;
    let mut min_dist = f32::MAX;
    for (i, c) in centroids.chunks(d).enumerate() {
        let dist = l2_squared(q, c);
        if dist < min_dist {
            min_dist = dist;
            min = i;
        }
    }

    let min_start = min * d;
    let min_end = min_start + d;
    (min as u8, &centroids[min_start..min_end])
}

#[inline(always)]
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    assert!(a.len() == b.len(), "mismatched dimensions");
    let d = a.len();

    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    let mut s4 = 0.0f32;
    let mut s5 = 0.0f32;
    let mut s6 = 0.0f32;
    let mut s7 = 0.0f32;

    let mut i = 0;

    while i + 8 <= d {
        let d0 = a[i] - b[i];
        let d1 = a[i + 1] - b[i + 1];
        let d2 = a[i + 2] - b[i + 2];
        let d3 = a[i + 3] - b[i + 3];
        let d4 = a[i + 4] - b[i + 4];
        let d5 = a[i + 5] - b[i + 5];
        let d6 = a[i + 6] - b[i + 6];
        let d7 = a[i + 7] - b[i + 7];

        s0 += d0 * d0;
        s1 += d1 * d1;
        s2 += d2 * d2;
        s3 += d3 * d3;
        s4 += d4 * d4;
        s5 += d5 * d5;
        s6 += d6 * d6;
        s7 += d7 * d7;

        i += 8;
    }

    while i < d {
        let d = a[i] - b[i];
        s0 += d * d;
        i += 1;
    }

    (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7)
}
