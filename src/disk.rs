use std::{
    collections::HashSet,
    error::Error,
    fs::File,
    io::BufWriter,
    path::Path,
    sync::{
        Mutex, RwLock,
        atomic::{AtomicU8, Ordering},
    },
};

use rand::{
    SeedableRng,
    distr::{Distribution, Open01},
    rngs::StdRng,
};
use serde::{
    Deserialize, Serialize,
    ser::{SerializeSeq, SerializeStruct},
};

use crate::{Hnsw, IdMode, Storage, dist::Distance, node::Node};

struct FlatF32<'a, const D: usize>(&'a [[f32; D]]);
impl<'a, const D: usize> From<&'a [[f32; D]]> for FlatF32<'a, D> {
    fn from(value: &'a [[f32; D]]) -> Self {
        Self(value)
    }
}

#[allow(non_snake_case)]
#[derive(Deserialize)]
struct SerializedHnsw<DS> {
    M: usize,
    M0: usize,
    ef_construction: usize,
    entry_point: usize,
    data: Vec<Vec<f32>>,
    nodes: Vec<Node>,
    dup_next: Vec<u64>,
    ids: Vec<u64>,
    id_mode: u8,
    max_layer: usize,
    ml: f64,
    seed: u64,
    dist: DS,
}

impl<const D: usize, DS> Hnsw<D, DS>
where
    DS: Distance<D> + Serialize,
{
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn Error>> {
        let f = File::create(path)?;
        let w = BufWriter::new(f);
        bincode2::serialize_into(w, self)?;
        Ok(())
    }
}

impl<const D: usize, DS> Hnsw<D, DS>
where
    DS: Distance<D> + for<'de> Deserialize<'de>,
{
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn Error>> {
        let b = std::fs::read(path)?;
        Ok(bincode2::deserialize(&b)?)
    }
}

impl<'a, const D: usize> Serialize for FlatF32<'a, D> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(self.0.len()))?;
        for v in self.0 {
            seq.serialize_element(&v[..])?;
        }
        seq.end()
    }
}

impl<const D: usize, DS> Serialize for Hnsw<D, DS>
where
    DS: Distance<D> + Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let _guard = self.update_lock.write().unwrap();
        let storage = self.storage.read().unwrap();
        let mut state = serializer.serialize_struct("Hnsw", 13)?;
        let (ep, max_layer) = *self.entry.read().unwrap();
        state.serialize_field("M", &(self.M as u64))?;
        state.serialize_field("M0", &(self.M0 as u64))?;
        state.serialize_field("ef_construction", &(self.ef_construction as u64))?;
        state.serialize_field("entry_point", &(ep as u64))?;
        state.serialize_field("data", &FlatF32::from(storage.data.as_slice()))?;
        state.serialize_field("nodes", &storage.nodes)?;
        let dup_next: Vec<u64> = storage.dup_next.iter().map(|&next| next as u64).collect();
        state.serialize_field("dup_next", &dup_next)?;
        let ids: Vec<u64> = storage.ids.iter().map(|&id| id as u64).collect();
        state.serialize_field("ids", &ids)?;
        state.serialize_field("id_mode", &self.id_mode.load(Ordering::Acquire))?;
        state.serialize_field("max_layer", &(max_layer as u64))?;
        state.serialize_field("ml", &self.ml)?;
        state.serialize_field("seed", &self.seed)?;
        state.serialize_field("dist", &self.dist)?;
        state.end()
    }
}

impl<'de, const D: usize, DS> Deserialize<'de> for Hnsw<D, DS>
where
    DS: Distance<D> + Deserialize<'de>,
{
    fn deserialize<DE>(deserializer: DE) -> Result<Self, DE::Error>
    where
        DE: serde::Deserializer<'de>,
    {
        let disk = SerializedHnsw::<DS>::deserialize(deserializer)?;

        let mut data = Vec::with_capacity(disk.data.len());
        for vec in disk.data {
            if vec.len() != D {
                return Err(serde::de::Error::custom(format!(
                    "invalid vector dimensions, expected {D}, got {}",
                    vec.len()
                )));
            }
            data.push(vec.try_into().expect("impossible"));
        }

        if disk.dup_next.len() != data.len() || disk.nodes.len() != data.len() {
            return Err(serde::de::Error::custom(
                "node, duplicate-list and vector counts differ",
            ));
        }
        let mut dup_next = Vec::with_capacity(disk.dup_next.len());
        for next in disk.dup_next {
            match usize::try_from(next) {
                Ok(next) if next < data.len() => dup_next.push(next),
                _ if next == u64::MAX => dup_next.push(crate::NO_DUP),
                _ => return Err(serde::de::Error::custom("duplicate list out of bounds")),
            }
        }

        let id_mode = IdMode::from_u8(disk.id_mode);
        if disk.ids.len() != data.len() || (id_mode.is_none() && !data.is_empty()) {
            return Err(serde::de::Error::custom("invalid id table"));
        }
        let mut ids = Vec::with_capacity(disk.ids.len());
        let mut taken_ids = HashSet::new();
        for (position, id) in disk.ids.into_iter().enumerate() {
            let id = usize::try_from(id)
                .map_err(|_| serde::de::Error::custom("id does not fit in usize"))?;
            let valid = match id_mode {
                Some(IdMode::Positional) => id == position,
                Some(IdMode::Explicit) => taken_ids.insert(id),
                None => false,
            };
            if !valid {
                return Err(serde::de::Error::custom(format!(
                    "invalid or duplicate id {id} at position {position}"
                )));
            }
            ids.push(id);
        }

        // advance rng
        let mut rng = StdRng::seed_from_u64(disk.seed);
        for _ in 0..data.len() {
            let _: f64 = Open01.sample(&mut rng);
        }

        Ok(Self {
            M: disk.M,
            M0: disk.M0,
            ef_construction: disk.ef_construction,
            entry: RwLock::new((disk.entry_point, disk.max_layer)),
            update_lock: RwLock::new(()),
            storage: RwLock::new(Storage {
                data,
                nodes: disk.nodes,
                dup_next,
                ids,
            }),
            ml: disk.ml,
            seed: disk.seed,
            rng: Mutex::new(rng),
            dist: disk.dist,
            id_mode: AtomicU8::new(disk.id_mode),
            taken_ids: Mutex::new(taken_ids),
        })
    }
}
