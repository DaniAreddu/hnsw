use rand::distr::Distribution;

use crate::{closest_centroid, l2_squared};

pub(crate) struct KMeans {
    pub(crate) centroids: Vec<f32>,
    data: Vec<Vec<f32>>,
    cluster_mappings: Vec<u8>,
    k: usize,
    dims: usize,
    trained: bool,
}

impl KMeans {
    const MAX_ITERS: usize = 1e3 as usize;
    const MIN_MOVEMENT: f32 = 1e-5;

    pub fn empty(k: usize, dims: usize) -> Self {
        assert!(k <= 256, "k must fit in an u8");
        assert!(k > 0, "k must be greater than zero");
        Self {
            centroids: Vec::new(),
            data: Vec::new(),
            cluster_mappings: Vec::new(),
            k,
            dims,
            trained: false,
        }
    }

    pub fn new(data: Vec<Vec<f32>>, k: usize, dims: usize) -> Self {
        assert!(k <= 256, "k must fit in an u8");
        assert!(data.len() >= k, "not enough vectors");
        assert!(data[0].len() == dims, "mismatched dimensions");

        Self {
            centroids: Self::init_centroids(&data, k, dims),
            cluster_mappings: vec![0; data.len()],
            data,
            k,
            dims,
            trained: false,
        }
    }

    pub fn add_batch(&mut self, data: Vec<Vec<f32>>) {
        assert!(data.len() >= self.k, "not enough vectors");
        assert!(data[0].len() == self.dims, "mismatched dimensions");
        assert!(!self.trained, "quantizer already trained");

        self.data.extend(data);
        self.centroids = Self::init_centroids(&self.data, self.k, self.dims);
        self.cluster_mappings = vec![0; self.data.len()];
    }

    pub fn train(&mut self) {
        assert!(self.data.len() >= self.k, "not enough vectors");
        assert!(self.data[0].len() == self.dims, "mismatched dimensions");
        assert!(!self.trained, "already trained");

        for _ in 0..Self::MAX_ITERS {
            // find closest centroid for each vector
            for (i, v) in self.data.iter().enumerate() {
                let (closest, _) = closest_centroid(&self.centroids, self.dims, v);
                self.cluster_mappings[i] = closest;
            }

            // recompute centroids
            let mut centroids = vec![0_f32; self.k * self.dims];
            let mut counts = vec![0; self.k];

            for (vec_idx, centroid_idx) in self.cluster_mappings.iter().enumerate() {
                let centroid_idx = *centroid_idx as usize;
                // for (dim_idx, scalar) in centroids[centroid_idx].iter_mut().enumerate() {
                for d in 0..self.dims {
                    let idx = (centroid_idx * self.dims) + d;
                    centroids[idx] += self.data[vec_idx][d];
                }
                counts[centroid_idx] += 1;
            }
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

    pub fn encode(&self, q: &[f32]) -> (u8, &[f32]) {
        assert!(q.len() == self.dims, "mismatched dimensions");
        assert!(self.trained, "KMeans must be trained before encoding");
        closest_centroid(&self.centroids, self.dims, q)
    }

    fn init_centroids(data: &[Vec<f32>], k: usize, d: usize) -> Vec<f32> {
        assert!(data.len() >= k, "not enough vectors");
        let mut centroids = Vec::with_capacity(k * d);

        if data.len() == k {
            let mut centroids = Vec::with_capacity(k * d);
            for vec in data {
                centroids.extend_from_slice(vec);
            }
            return centroids;
        }

        let mut used = std::collections::HashSet::new();
        let start = rand::random_range(0..data.len());
        centroids.extend_from_slice(&data[start]);
        used.insert(start);

        for _ in 1..k {
            // for each vector that has not been used, the distance from it's closest centroid
            let weights: Vec<f32> = data
                .iter()
                .enumerate()
                .map(|(i, vec)| {
                    if used.contains(&i) {
                        return 0f32;
                    }
                    centroids
                        .chunks(d)
                        .map(|c| l2_squared(vec.as_slice(), c))
                        .min_by(|a, b| a.total_cmp(b))
                        .unwrap_or(0f32)
                })
                .collect();
            let (idx, new_centroid) = sample_from_weights(data, weights);
            centroids.extend_from_slice(new_centroid);
            used.insert(idx);
        }
        centroids
    }
}

fn sample_from_weights(data: &[Vec<f32>], weights: Vec<f32>) -> (usize, &[f32]) {
    let mut rng = rand::rng();
    let dist = rand::distr::weighted::WeightedIndex::new(weights).unwrap();
    let idx = dist.sample(&mut rng);
    (idx, &data[idx])
}

// this is just most distant point from centroids initialization, it suffers from outliers
// fn random_centroids_from_data(data: &[Vec<f32>], k: usize, d: usize) -> Vec<f32> {
//         assert!(data.len() >= k, "not enough vectors");
//         let mut centroids = Vec::with_capacity(k * d);
//
//         if data.len() == k {
//             let mut centroids = Vec::with_capacity(k * d);
//             for vec in data {
//                 centroids.extend_from_slice(vec);
//             }
//             return centroids;
//         }
//
//         let mut used = std::collections::HashSet::new();
//         let start = rand::random_range(0..data.len());
//         centroids.extend_from_slice(&data[start]);
//         used.insert(start);
//
//         for _ in 1..k {
//             // find furthest point from current centroids
//             let mut max_dist = 0f32;
//             let mut max_dist_idx = 0;
//             for (i, vec) in data.iter().enumerate() {
//                 if used.contains(&i) {
//                     continue;
//                 }
//                 let mut curr_dist = 0f32;
//                 for centroid in centroids.chunks(d) {
//                     curr_dist += l2_squared(centroid, vec.as_slice());
//                 }
//                 if curr_dist >= max_dist {
//                     max_dist = curr_dist;
//                     max_dist_idx = i;
//                 }
//             }
//
//             centroids.extend_from_slice(&data[max_dist_idx]);
//             used.insert(max_dist_idx);
//         }
//         centroids
//     }
