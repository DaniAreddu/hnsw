use crate::{ProductQuantizer, adc_distance, kmeans::KMeans, l2_squared};

#[test]
fn product_quantizer_encodes_and_scores_with_single_centroid() {
    let mut pq = ProductQuantizer::<2, 4>::new(1);
    pq.fit(vec![[1.0, 2.0, 10.0, 20.0], [3.0, 4.0, 30.0, 40.0]]);

    let query = [2.0, 3.0, 20.0, 30.0];
    let code = pq.encode(&query);
    let table = pq.adc_table(&query);

    assert_eq!(code, [0, 0]);
    assert_eq!(table.len(), 2);
    assert_eq!(table[0], vec![0.0]);
    assert_eq!(table[1], vec![0.0]);
    assert_eq!(adc_distance(&table, &code), 0.0);
}

#[test]
fn adc_distance_sums_selected_subquantizer_distances() {
    let table = vec![vec![1.0, 2.0], vec![4.0, 8.0], vec![16.0, 32.0]];

    assert_eq!(adc_distance::<3>(&table, &[1, 0, 1]), 38.0);
}

#[test]
fn l2_squared_returns_sum_of_squared_differences() {
    assert_eq!(l2_squared(&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0]), 8.0);
}

#[test]
fn kmeans_with_one_cluster_converges_to_mean() {
    let mut kmeans = KMeans::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]], 1, 2);

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
