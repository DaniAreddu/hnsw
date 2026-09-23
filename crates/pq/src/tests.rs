use crate::{ProductQuantizer, kmeans::KMeans, l2_squared};

#[test]
fn product_quantizer_encodes_and_scores_with_single_centroid() {
    let mut pq = ProductQuantizer::<2, 4>::new(1);
    pq.fit(&[[1.0, 2.0, 10.0, 20.0], [3.0, 4.0, 30.0, 40.0]]);

    let query = [2.0, 3.0, 20.0, 30.0];
    let code = pq.encode(&query);
    let table = pq.adc_table(&query);

    assert_eq!(code, [0, 0]);
    assert_eq!(table.len(), 2);
    assert_eq!(table, vec![0.0, 0.0]);
    assert_eq!(pq.adc_distance(&table, &code), 0.0);
}

#[test]
fn adc_distance_sums_selected_subquantizer_distances() {
    let pq = ProductQuantizer::<3, 3>::new(2);
    let table = vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0];

    assert_eq!(pq.adc_distance(&table, &[1, 0, 1]), 38.0);
}

#[test]
fn l2_squared_returns_sum_of_squared_differences() {
    assert_eq!(l2_squared(&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0]), 8.0);
}

#[test]
fn kmeans_with_one_cluster_converges_to_mean() {
    let mut kmeans = KMeans::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]], 1, 2, &mut rand::rng());

    kmeans.train();
    let (idx, centroid) = kmeans.encode(&[2.0, 3.0]);

    assert_eq!(idx, 0);
    assert_eq!(centroid, &[2.0, 3.0]);
}

#[test]
fn kmeans_separates_two_clear_clusters() {
    let mut kmeans = KMeans::new(
        vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![10.0, 10.0],
            vec![10.0, 11.0],
        ],
        2,
        2,
        &mut rand::rng(),
    );

    kmeans.train();
    let (low_idx, low_centroid) = kmeans.encode(&[0.0, 0.25]);
    let (high_idx, high_centroid) = kmeans.encode(&[10.0, 10.25]);

    assert_ne!(low_idx, high_idx);
    assert!(l2_squared(low_centroid, &[0.0, 0.5]) < 0.01);
    assert!(l2_squared(high_centroid, &[10.0, 10.5]) < 0.01);
}

#[test]
#[should_panic(expected = "ProductQuantizer must be trained to encode a query")]
fn encode_panics_before_training() {
    let pq = ProductQuantizer::<2, 4>::new(1);

    pq.encode(&[0.0; 4]);
}

#[test]
fn kmeans_init_can_pick_every_unused_vector() {
    use rand::{SeedableRng, rngs::StdRng};

    // Two identical vectors and one far away. Whichever vector k-means++
    // starts from, the second centroid must be a vector not yet chosen, and
    // [10, 10] must end up as a centroid.
    let data = [0.0, 0.0, 0.0, 0.0, 10.0, 10.0];
    for seed in 0..64 {
        let centroids = KMeans::init_centroids(&data, 2, 2, 3, &mut StdRng::seed_from_u64(seed));
        assert!(
            centroids.chunks(2).any(|c| c == [10.0, 10.0]),
            "seed {seed}: centroids {centroids:?} miss [10, 10]"
        );
    }
}

fn clustered(n: usize, seed: u64) -> Vec<[f32; 16]> {
    use rand::{RngExt, SeedableRng, rngs::StdRng};
    let mut rng = StdRng::seed_from_u64(seed);
    let centers: Vec<[f32; 16]> = (0..8)
        .map(|_| std::array::from_fn(|_| rng.random_range(-10.0..10.0)))
        .collect();
    (0..n)
        .map(|i| {
            let c = centers[i % centers.len()];
            std::array::from_fn(|d| c[d] + rng.random_range(-1.0..1.0))
        })
        .collect()
}

#[test]
fn seeded_training_is_reproducible_across_thread_counts() {
    let data = clustered(2_000, 1);
    let train = |threads: usize, seed: u64| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            let mut pq = ProductQuantizer::<4, 16>::new(16);
            pq.fit_seeded(&data, seed);
            pq.codebooks().to_vec()
        })
    };
    let one = train(1, 7);
    assert_eq!(one.len(), 4 * 16 * 4);
    assert_eq!(one, train(4, 7));
    assert_eq!(one, train(2, 7));
    assert_ne!(
        one,
        train(4, 8),
        "different seeds should give different codebooks"
    );
}

fn mse(pq: &ProductQuantizer<4, 16>, data: &[[f32; 16]]) -> f32 {
    data.iter()
        .map(|v| l2_squared(v, &pq.decode(&pq.encode(v))))
        .sum::<f32>()
        / data.len() as f32
}

/// Recall@10 of brute-force ADC ranking against exact L2 ranking.
fn adc_recall(pq: &ProductQuantizer<4, 16>, base: &[[f32; 16]], queries: &[[f32; 16]]) -> f64 {
    let codes: Vec<[u8; 4]> = base.iter().map(|v| pq.encode(v)).collect();
    let top10 = |score: &dyn Fn(usize) -> f32| {
        let mut ids: Vec<usize> = (0..base.len()).collect();
        ids.sort_by(|&a, &b| score(a).total_cmp(&score(b)).then(a.cmp(&b)));
        ids.truncate(10);
        ids
    };
    let mut hits = 0;
    for q in queries {
        let table = pq.adc_table(q);
        let exact = top10(&|i| l2_squared(q, &base[i]));
        let approx = top10(&|i| pq.adc_distance(&table, &codes[i]));
        hits += approx.iter().filter(|id| exact.contains(id)).count();
    }
    hits as f64 / (queries.len() * 10) as f64
}

#[test]
fn quantization_quality_against_exact_baseline() {
    let base = clustered(2_000, 2);
    let queries = clustered(50, 3);
    let mut coarse = ProductQuantizer::<4, 16>::new(1);
    coarse.fit_seeded(&base, 1);
    let mut fine = ProductQuantizer::<4, 16>::new(64);
    fine.fit_seeded(&base, 1);
    let (mse_coarse, mse_fine) = (mse(&coarse, &base), mse(&fine, &base));
    // k = 1 can only reconstruct the mean; 64 centroids per 4-D subspace must do
    // far better, and better than knowing only the 8 cluster centers (the
    // within-cluster noise has total variance 16 * 1/3).
    assert!(
        mse_fine < 0.01 * mse_coarse,
        "mse {mse_fine} vs {mse_coarse}"
    );
    assert!(mse_fine < 16.0 / 3.0, "mse {mse_fine}");

    // ADC ranking vs exact L2 ranking; measured 0.46 for k = 64.
    let (recall_coarse, recall_fine) = (
        adc_recall(&coarse, &base, &queries),
        adc_recall(&fine, &base, &queries),
    );
    assert!(recall_coarse <= 0.05, "k=1 recall {recall_coarse}");
    assert!(recall_fine >= 0.35, "k=64 recall {recall_fine}");
}
