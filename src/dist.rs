use serde::{Deserialize, Serialize};

use crate::HnswError;

/// A distance metric over `D`-dimensional vectors. Smaller means closer.
///
/// # Contract
///
/// For every pair of vectors accepted by [`Distance::validate`], `distance` must
/// return a finite value and should be symmetric. Distances may be negative.
/// The index checks distances computed during `try_insert` and searches and
/// reports [`HnswError::NonFiniteDistance`]; elsewhere (backlink pruning and
/// parallel builds) a non-finite distance is treated as `+inf` so the graph stays
/// structurally valid.
pub trait Distance<const D: usize> {
    fn distance(&self, a: &[f32; D], b: &[f32; D]) -> f32;

    /// Checks that `v` may be inserted or used as a query.
    ///
    /// The default accepts any vector whose components are all finite. Override it
    /// when the metric can overflow or is undefined for some finite inputs.
    fn validate(&self, v: &[f32; D]) -> Result<(), HnswError> {
        check_finite(v)
    }

    /// Identity of the metric recorded in snapshots; loading a snapshot with a
    /// different id fails with `SnapshotError::MetricMismatch`.
    ///
    /// The default is the Rust type name, which is unique but not guaranteed to
    /// stay the same across compiler versions or crate renames (a change makes
    /// old snapshots fail to load, never load as the wrong metric). Override it
    /// with a fixed string for metrics whose snapshots must stay loadable.
    fn metric_id(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
}

/// Returns an error for the first NaN or infinite component of `v`.
pub fn check_finite(v: &[f32]) -> Result<(), HnswError> {
    match v.iter().position(|x| !x.is_finite()) {
        Some(component) => Err(HnswError::NonFiniteComponent {
            component,
            value: v[component],
        }),
        None => Ok(()),
    }
}

/// Squared Euclidean distance.
///
/// [`Distance::validate`] accepts finite components with magnitude at most
/// [`L2Squared::component_limit`], which guarantees that the distance between any
/// two accepted vectors is finite (with a 4x margin below `f32::MAX`).
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
pub struct L2Squared;

impl L2Squared {
    /// Largest accepted absolute component value for dimension `d`:
    /// `sqrt(f32::MAX / d) / 4`, so `d * (2 * limit)^2 <= f32::MAX / 4`.
    pub fn component_limit(d: usize) -> f32 {
        (f32::MAX / d.max(1) as f32).sqrt() / 4.0
    }
}

impl<const D: usize> Distance<D> for L2Squared {
    fn metric_id(&self) -> &'static str {
        "hnsw::L2Squared"
    }

    fn validate(&self, v: &[f32; D]) -> Result<(), HnswError> {
        check_finite(v)?;
        let limit = Self::component_limit(D);
        match v.iter().position(|x| x.abs() > limit) {
            Some(component) => Err(HnswError::ComponentOutOfRange {
                component,
                value: v[component],
                limit,
            }),
            None => Ok(()),
        }
    }

    #[inline(always)]
    fn distance(&self, a: &[f32; D], b: &[f32; D]) -> f32 {
        let mut s0 = 0.0f32;
        let mut s1 = 0.0f32;
        let mut s2 = 0.0f32;
        let mut s3 = 0.0f32;
        let mut s4 = 0.0f32;
        let mut s5 = 0.0f32;
        let mut s6 = 0.0f32;
        let mut s7 = 0.0f32;

        let mut i = 0;

        while i + 8 <= D {
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

        while i < D {
            let d = a[i] - b[i];
            s0 += d * d;
            i += 1;
        }

        (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7)
    }
}
