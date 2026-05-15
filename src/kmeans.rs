use crate::{closest_centroid, l2_squared};

pub(crate) struct KMeans {
    pub(crate) centroids: Vec<Vec<f32>>,
    data: Vec<Vec<f32>>,
    cluster_mappings: Vec<usize>,
    k: usize,
    d: usize,
    trained: bool,
}

impl KMeans {
    const MAX_ITERS: usize = 1e3 as usize;
    const MIN_MOVEMENT: f32 = 1e-5;

    pub fn empty(k: usize, d: usize) -> Self {
        assert!(k > 0, "k must be greater than zero");
        Self {
            centroids: Vec::new(),
            data: Vec::new(),
            cluster_mappings: Vec::new(),
            k,
            d,
            trained: false,
        }
    }

    pub fn new(data: Vec<Vec<f32>>, k: usize, d: usize) -> Self {
        assert!(data.len() >= k, "not enough vectors");
        assert!(data[0].len() == d, "mismatched dimensions");

        Self {
            centroids: Self::random_centroids_from_data(&data, k),
            cluster_mappings: vec![0; data.len()],
            data,
            k,
            d,
            trained: false,
        }
    }

    pub fn add_batch(&mut self, data: Vec<Vec<f32>>) {
        assert!(data.len() >= self.k, "not enough vectors");
        assert!(data[0].len() == self.d, "mismatched dimensions");
        assert!(!self.trained, "quantizer already trained");

        self.data.extend(data);
        self.centroids = Self::random_centroids_from_data(&self.data, self.k);
        self.cluster_mappings = vec![0; self.data.len()];
    }

    pub fn train(&mut self) {
        assert!(self.data.len() >= self.k, "not enough vectors");
        assert!(self.data[0].len() == self.d, "mismatched dimensions");
        assert!(!self.trained, "already trained");

        for _ in 0..Self::MAX_ITERS {
            // find closest centroid for each vector
            for (i, v) in self.data.iter().enumerate() {
                let (closest, _) = closest_centroid(&self.centroids, v);
                self.cluster_mappings[i] = closest;
            }

            // recompute centroids
            let mut centroids = vec![vec![0_f32; self.d]; self.k];
            let mut counts = vec![0; self.k];

            for (vec_idx, centroid_idx) in self.cluster_mappings.iter().enumerate() {
                for (dim_idx, scalar) in centroids[*centroid_idx].iter_mut().enumerate() {
                    *scalar += self.data[vec_idx][dim_idx];
                }
                counts[*centroid_idx] += 1;
            }
            for c in 0..self.k {
                // keep old centroid
                if counts[c] == 0 {
                    centroids[c] = self.centroids[c].clone();
                    break;
                }
                for scalar in centroids[c].iter_mut() {
                    // counts[c] can't be zero
                    *scalar /= counts[c] as f32;
                }
            }

            // check if movement is enough
            let mut max_movement = 0_f32;
            for (a, b) in self.centroids.iter().zip(&centroids) {
                let dist = l2_squared(a, b);
                max_movement = max_movement.max(dist);
            }

            self.centroids = centroids;

            if max_movement <= Self::MIN_MOVEMENT {
                break;
            }
        }

        self.trained = true;
        std::mem::take(&mut self.data);
    }

    pub fn encode(&self, q: &[f32]) -> (usize, &[f32]) {
        assert!(q.len() == self.d, "mismatched dimensions");
        assert!(!self.trained, "KMeans must be trained before encoding");
        closest_centroid(&self.centroids, q)
    }

    fn random_centroids_from_data(data: &[Vec<f32>], k: usize) -> Vec<Vec<f32>> {
        let mut rng = rand::rng();
        let rnd_indicies = rand::seq::index::sample(&mut rng, data.len(), k);
        let mut centroids = Vec::with_capacity(k);
        for i in rnd_indicies {
            centroids.push(data[i].clone())
        }
        centroids
    }
}
