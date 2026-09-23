use std::fmt;

/// Errors returned by the fallible (`try_*`) index operations.
///
/// The panicking convenience methods (`new`, `insert`, `search`, ...) panic with
/// this error's `Display` text in exactly the cases where the matching `try_*`
/// method returns `Err`.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum HnswError {
    /// A construction or query parameter is outside its supported range.
    InvalidParameter {
        name: &'static str,
        value: usize,
        requirement: &'static str,
    },
    /// A runtime-sized vector does not have the index dimension.
    DimensionMismatch { expected: usize, found: usize },
    /// A vector component is NaN or infinite.
    NonFiniteComponent { component: usize, value: f32 },
    /// A finite vector component is so large that the metric could overflow.
    ComponentOutOfRange {
        component: usize,
        value: f32,
        limit: f32,
    },
    /// The distance metric produced a NaN or infinite distance.
    NonFiniteDistance { distance: f32 },
    /// The operation requires an empty index.
    IndexNotEmpty { len: usize },
    /// The vector at `index` of a batch was rejected; nothing was inserted.
    InvalidBatchVector { index: usize, error: Box<HnswError> },
}

impl fmt::Display for HnswError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter {
                name,
                value,
                requirement,
            } => write!(f, "invalid {name} = {value}: must be {requirement}"),
            Self::DimensionMismatch { expected, found } => write!(
                f,
                "vector has {found} components, but the index dimension is {expected}"
            ),
            Self::NonFiniteComponent { component, value } => write!(
                f,
                "vector component {component} is {value}; all components must be finite"
            ),
            Self::ComponentOutOfRange {
                component,
                value,
                limit,
            } => write!(
                f,
                "vector component {component} is {value}; the metric accepts magnitudes up to {limit} to avoid overflowing the distance"
            ),
            Self::NonFiniteDistance { distance } => write!(
                f,
                "the distance metric returned {distance}; distances must be finite"
            ),
            Self::IndexNotEmpty { len } => write!(
                f,
                "the index already holds {len} vectors; this operation needs an empty index"
            ),
            Self::InvalidBatchVector { index, error } => {
                write!(
                    f,
                    "batch vector {index} rejected, nothing inserted: {error}"
                )
            }
        }
    }
}

impl std::error::Error for HnswError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidBatchVector { error, .. } => Some(error.as_ref()),
            _ => None,
        }
    }
}
