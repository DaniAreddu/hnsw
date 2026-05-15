mod kmeans;

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
