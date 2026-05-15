use crate::kmeans::KMeans;

mod kmeans;

struct ProductQuantizer<const M: usize, const D: usize> {
    codebooks: Vec<Vec<Vec<f32>>>,
    k: usize,
    subdims: usize,
    trained: bool,
}

impl<const M: usize, const D: usize> ProductQuantizer<M, D> {
    pub fn new(k: usize) -> Self {
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

    pub fn fit(&mut self, data: Vec<[f32; D]>) {
        assert!(!self.trained, "ProductQuantizer was already trained");
        assert!(data.len() >= self.k, "not enough vectors");

        let mut codebooks = Vec::with_capacity(M);
        for i in 0..M {
            let mut d: Vec<Vec<f32>> = Vec::with_capacity(data.len());
            for v in data.iter() {
                let start = i * self.subdims;
                let end = start + self.subdims;
                d.push(v[start..end].to_vec());
            }
            let mut quantizer = KMeans::new(d, self.k, self.subdims);
            quantizer.train();
            codebooks.push(quantizer.centroids);
        }

        self.trained = true;
        self.codebooks = codebooks;
    }

    pub fn encode(&self, query: &[f32; D]) -> [usize; M] {
        assert!(
            self.trained,
            "ProductQuantizer must be trained to encode a query"
        );

        let mut enc = [0; M];
        for (i, centroids) in self.codebooks.iter().enumerate() {
            let start = i * self.subdims;
            let end = start + self.subdims;
            let (min, _) = closest_centroid(centroids, &query[start..end]);
            enc[i] = min;
        }

        enc
    }
}
fn closest_centroid<'a>(centroids: &'a [Vec<f32>], q: &[f32]) -> (usize, &'a [f32]) {
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

    (min, &centroids[min])
}

fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    assert!(a.len() == b.len(), "mismatched dimensions");
    a.iter().zip(b).map(|(x1, x2)| (x1 - x2).powi(2)).sum()
}
