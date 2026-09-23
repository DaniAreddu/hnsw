use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Link {
    pub node_index: usize,
    pub distance: f32,
}

/// Links are totally ordered by `distance` (IEEE 754 `totalOrder`, so `-0.0 < 0.0`
/// and NaNs have a fixed position), then by `node_index`. Equality is defined by
/// the same ordering so `Eq`, `Ord` and heap tie-breaking always agree.
impl PartialEq for Link {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}
impl Eq for Link {}
impl PartialOrd for Link {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Link {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| self.node_index.cmp(&other.node_index))
    }
}

#[cfg(test)]
mod tests {
    use super::Link;
    use std::{cmp::Ordering, collections::BinaryHeap};

    fn tricky_links() -> Vec<Link> {
        let distances = [
            0.0,
            -0.0,
            f32::MIN_POSITIVE,
            1.0,
            f32::MAX,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            -f32::NAN,
        ];
        distances
            .iter()
            .flat_map(|&distance| {
                [0, 1, usize::MAX].map(|node_index| Link {
                    node_index,
                    distance,
                })
            })
            .collect()
    }

    #[test]
    fn eq_agrees_with_ord_for_all_values() {
        let links = tricky_links();
        for a in &links {
            for b in &links {
                assert_eq!(
                    a == b,
                    a.cmp(b) == Ordering::Equal,
                    "Eq and Ord disagree for {a:?} vs {b:?}"
                );
                assert_eq!(a.partial_cmp(b), Some(a.cmp(b)));
                assert_eq!(a.cmp(b), b.cmp(a).reverse(), "antisymmetry {a:?} {b:?}");
            }
        }
    }

    #[test]
    fn ord_is_transitive() {
        let links = tricky_links();
        for a in &links {
            for b in &links {
                for c in &links {
                    if a <= b && b <= c {
                        assert!(a <= c, "{a:?} <= {b:?} <= {c:?} but not {a:?} <= {c:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn equal_distances_break_ties_by_node_index() {
        let a = Link {
            node_index: 3,
            distance: 1.0,
        };
        let b = Link {
            node_index: 7,
            distance: 1.0,
        };
        assert_ne!(a, b);
        assert!(a < b);

        let mut heap: BinaryHeap<Link> = [b, a].into_iter().collect();
        assert_eq!(heap.pop().map(|link| link.node_index), Some(7));
        assert_eq!(heap.pop().map(|link| link.node_index), Some(3));
    }
}
