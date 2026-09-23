use rand::distr::Distribution;

use crate::{closest_centroid, l2_squared};

pub(crate) struct KMeans {
    pub(crate) centroids: Vec<f32>,
    data: Vec<f32>,
    cluster_mappings: Vec<u8>,
    k: usize,
    dims: usize,
    n_vectors: usize,
    trained: bool,
}

impl KMeans {
    const MAX_ITERS: usize = 50;
    const MIN_MOVEMENT: f32 = 1e-5;

    #[allow(dead_code)]
    pub fn empty(k: usize, dims: usize) -> Self {
        assert!(k <= 256, "k must fit in an u8");
        assert!(k > 0, "k must be greater than zero");
        Self {
            centroids: Vec::new(),
            data: Vec::new(),
            cluster_mappings: Vec::new(),
            k,
            dims,
            n_vectors: 0,
            trained: false,
        }
    }

    pub fn new_flat(data: Vec<f32>, k: usize, dims: usize, n_vectors: usize) -> Self {
        assert!(k <= 256, "k must fit in an u8");
        assert!(n_vectors >= k, "not enough vectors");
        assert!(data.len() / n_vectors == dims, "mismatched dimensions");

        Self {
            centroids: Self::init_centroids(&data, k, dims, n_vectors),
            cluster_mappings: vec![0; n_vectors],
            data,
            k,
            dims,
            n_vectors,
            trained: false,
        }
    }

    #[allow(dead_code)]
    pub fn new(data: Vec<Vec<f32>>, k: usize, dims: usize) -> Self {
        assert!(k <= 256, "k must fit in an u8");
        assert!(data.len() >= k, "not enough vectors");
        assert!(data[0].len() == dims, "mismatched dimensions");

        let n_vectors = data.len();
        let data: Vec<f32> = data.into_iter().flatten().collect();
        Self::new_flat(data, k, dims, n_vectors)
    }

    #[allow(dead_code)]
    pub fn add_batch(&mut self, data: Vec<Vec<f32>>) {
        assert!(self.n_vectors + data.len() >= self.k, "not enough vectors");
        assert!(data[0].len() == self.dims, "mismatched dimensions");
        assert!(!self.trained, "quantizer already trained");

        self.n_vectors += data.len();
        self.data.extend(data.into_iter().flatten());
        self.centroids = Self::init_centroids(&self.data, self.k, self.dims, self.n_vectors);
        self.cluster_mappings = vec![0; self.data.len()];
    }

    pub fn train(&mut self) {
        assert!(self.data.len() >= self.k, "not enough vectors");
        assert!(
            self.data.len() / self.n_vectors == self.dims,
            "mismatched dimensions"
        );
        assert!(!self.trained, "already trained");

        for _ in 0..Self::MAX_ITERS {
            // find closest centroid for each vector
            for (i, v) in self.data.chunks(self.dims).enumerate() {
                let (closest, _) = closest_centroid(&self.centroids, self.dims, v);
                self.cluster_mappings[i] = closest;
            }

            // recompute centroids
            let mut centroids = vec![0_f32; self.k * self.dims];
            let mut counts = vec![0; self.k];

            for (vec_idx, centroid_idx) in self.cluster_mappings.iter().enumerate() {
                let centroid_idx = *centroid_idx as usize;
                for d in 0..self.dims {
                    let idx = (centroid_idx * self.dims) + d;
                    centroids[idx] += self.data[vec_idx * self.dims + d];
                }
                counts[centroid_idx] += 1;
            }
            #[allow(clippy::needless_range_loop)]
            for c in 0..self.k {
                let range = (c * self.dims)..(c * self.dims) + self.dims;
                // keep old centroid
                if counts[c] == 0 {
                    centroids[range.clone()].copy_from_slice(&self.centroids[range]);
                    continue;
                }
                centroids[range]
                    .iter_mut()
                    .for_each(|v| *v /= counts[c] as f32);
            }

            // check if movement is enough
            let mut max_movement = 0_f32;
            for (a, b) in self
                .centroids
                .chunks(self.dims)
                .zip(centroids.chunks(self.dims))
            {
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

    #[allow(dead_code)]
    pub fn encode(&self, q: &[f32]) -> (u8, &[f32]) {
        assert!(q.len() == self.dims, "mismatched dimensions");
        assert!(self.trained, "KMeans must be trained before encoding");
        closest_centroid(&self.centroids, self.dims, q)
    }

    fn init_centroids(data: &[f32], k: usize, d: usize, n_vectors: usize) -> Vec<f32> {
        assert!(n_vectors >= k, "not enough vectors");
        let mut centroids = Vec::with_capacity(k * d);

        if n_vectors == k {
            let mut centroids = Vec::with_capacity(k * d);
            for vec in data.chunks(d) {
                centroids.extend_from_slice(vec);
            }
            return centroids;
        }

        let mut used = std::collections::HashSet::new();
        let start = rand::random_range(0..n_vectors) * d;
        centroids.extend_from_slice(&data[start..start + d]);
        used.insert(start);

        for _ in 1..k {
            // for each vector that has not been used, the distance from it's closest centroid
            let weights: Vec<f32> = data
                .chunks(d)
                .enumerate()
                .map(|(i, vec)| {
                    if used.contains(&i) {
                        return 0f32;
                    }
                    centroids
                        .chunks(d)
                        .map(|c| l2_squared(vec, c))
                        .min_by(|a, b| a.total_cmp(b))
                        .unwrap_or(0f32)
                })
                .collect();
            let idx = sample_from_weights(&weights).unwrap_or_else(|| {
                data.chunks(d)
                    .enumerate()
                    .find_map(|(i, _)| (!used.contains(&i)).then_some(i))
                    .expect("there must be an unused centroid candidate")
            }) * d;
            centroids.extend_from_slice(&data[idx..idx + d]);
            used.insert(idx);
        }
        centroids
    }
}

fn sample_from_weights(weights: &[f32]) -> Option<usize> {
    let mut rng = rand::rng();
    let dist = rand::distr::weighted::WeightedIndex::new(weights).ok()?;
    Some(dist.sample(&mut rng))
}
