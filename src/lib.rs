use crate::kmeans::KMeans;

mod kmeans;
#[cfg(test)]
mod tests;

pub struct ProductQuantizer<const M: usize, const D: usize> {
    codebooks: Vec<Vec<Vec<f32>>>,
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

        let mut codebooks = Vec::with_capacity(M);
        for m in 0..M {
            let mut d: Vec<Vec<f32>> = Vec::with_capacity(data.len());
            for v in data.iter() {
                d.push(v[self.subdim_range(m)].to_vec());
            }
            let mut quantizer = KMeans::new(d, self.k, self.subdims);
            quantizer.train();
            codebooks.push(quantizer.centroids);
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
            let (min, _) = closest_centroid(&self.codebooks[m], &query[self.subdim_range(m)]);
            enc[m] = min;
        }

        enc
    }

    pub fn decode(&self, code: &[u8; M]) -> [f32; D] {
        let mut dec = [0_f32; D];
        for m in 0..M {
            let centroid = &self.codebooks[m][code[m] as usize];
            dec[self.subdim_range(m)].copy_from_slice(centroid);
        }
        dec
    }

    pub fn adc_table(&self, query: &[f32; D]) -> Vec<Vec<f32>> {
        assert!(
            self.trained,
            "ProductQuantizer must be trained to build the adc_table"
        );
        let mut table = vec![vec![0_f32; self.k]; M];
        for m in 0..M {
            let q = &query[self.subdim_range(m)];
            for id in 0..self.k {
                let dist = l2_squared(q, &self.codebooks[m][id]);
                table[m][id] = dist;
            }
        }
        table
    }

    pub fn sdc_table(&self) -> Vec<Vec<Vec<f32>>> {
        let mut adc_table = vec![vec![vec![0_f32; self.k]; self.k]; M];
        for m in 0..M {
            for i in 0..self.k {
                for j in 0..self.k {
                    let dist = l2_squared(&self.codebooks[m][i], &self.codebooks[m][j]);
                    adc_table[m][i][j] = dist;
                }
            }
        }
        adc_table
    }

    fn subdim_range(&self, m: usize) -> std::ops::Range<usize> {
        let start = m * self.subdims;
        let end = start + self.subdims;
        start..end
    }
}

pub fn adc_distance<const M: usize>(table: &[Vec<f32>], q_code: &[u8; M]) -> f32 {
    assert_eq!(table.len(), M, "adc table has wrong number of quantizers");
    let mut dist = 0.0;
    for m in 0..M {
        dist += table[m][q_code[m] as usize];
    }
    dist
}

pub fn sdc_distance<const M: usize>(table: &[Vec<Vec<f32>>], a: &[u8; M], b: &[u8; M]) -> f32 {
    assert_eq!(table.len(), M, "sdc table has wrong number of quantizers");
    let mut dist = 0.0;
    for m in 0..M {
        dist += table[m][a[m] as usize][b[m] as usize];
    }
    dist
}

fn closest_centroid<'a>(centroids: &'a [Vec<f32>], q: &[f32]) -> (u8, &'a [f32]) {
    assert!(!centroids.is_empty(), "not enough centroids");

    let mut min = 0;
    let mut min_dist = f32::MAX;
    for (i, c) in centroids.iter().enumerate() {
        let dist = l2_squared(q, c);
        if dist < min_dist {
            min_dist = dist;
            min = i;
        }
    }

    (min as u8, &centroids[min])
}

fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    assert!(a.len() == b.len(), "mismatched dimensions");
    a.iter().zip(b).map(|(x1, x2)| (x1 - x2).powi(2)).sum()
}
