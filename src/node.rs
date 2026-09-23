use std::sync::RwLock;

use crate::link::Link;

#[derive(Debug)]
pub struct Node {
    pub layers: Vec<RwLock<Vec<Link>>>,
}

pub(crate) fn nodes_heap_usage_bytes(nodes: &Vec<Node>) -> usize {
    let mut bytes = nodes.capacity() * size_of::<Node>();
    for node in nodes {
        bytes += node.layers.capacity() * size_of::<RwLock<Vec<Link>>>();
        for layer in node.layers.iter().map(|lyr| lyr.read().unwrap()) {
            bytes += layer.capacity() * size_of::<Link>();
        }
    }
    bytes
}
