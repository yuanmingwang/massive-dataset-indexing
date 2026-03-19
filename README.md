# Tree Index Project Refactor

This refactor splits the original single-file prototype into modules that are easier to extend for:
- new **local** indexes (KD-Tree, Quad-Tree, packed R-Tree, etc.)
- new **distributed** indexes (PySpark or non-Spark variants)
- new datasets, workloads, and experiment sweeps
- future tests and report-ready benchmark scripts

## Proposed structure

```text
project/
├── README.md
├── requirements.txt
├── pyproject.toml
├── src/
│   └── treeindex/
│       ├── __init__.py
│       ├── core/
│       │   ├── interfaces.py
│       │   └── results.py
│       ├── geometry/
│       │   ├── point.py
│       │   └── rect.py
│       ├── data/
│       │   └── generators.py
│       ├── experiments/
│       │   ├── benchmark.py
│       │   └── scenarios.py
│       ├── indexes/
│       │   ├── local/
│       │   │   ├── bplustree.py
│       │   │   ├── kdtree.py
│       │   │   ├── quadtree.py
│       │   │   └── rtree.py
│       │   └── distributed/
│       │       ├── base.py
│       │       ├── pyspark_bplustree.py
│       │       ├── pyspark_kdtree.py
│       │       ├── pyspark_quadtree.py
│       │       └── pyspark_rtree.py
│       ├── registry/
│       │   └── factories.py
│       ├── utils/
│       │   └── table.py
│       └── cli/
│           ├── run_local_demo.py
│           ├── run_distributed_bplustree.py
│           ├── run_distributed_kdtree.py
│           ├── run_distributed_quadtree.py
│           └── run_distributed_rtree.py
└── tests/
    ├── smoke_test.py
    ├── test_kdtree.py
    ├── test_quadtree.py
    └── test_distributed_rtree.py
```

## Why this layout works well

### 1. Index code is isolated
Each index gets its own module. That means future additions like `kdtree.py` and `quadtree.py` do not force you to touch your benchmark code.

### 2. Distributed code is separated from local code
The PySpark B+ tree depends on Spark-specific logic and job orchestration. Keeping it under `indexes/distributed/` avoids mixing algorithm logic with Spark execution details.

### 3. Dataset generation is reusable
Synthetic data generators are in one place, so every experiment script can reuse them.

### 4. Benchmark logic is shared
The benchmarking code only depends on the common index interfaces. That lets you compare different trees with the same harness.

### 5. CLI scripts stay thin
The CLI scripts only parse arguments and call reusable scenario functions. This makes parameter sweeps and future automation much easier.

## Recommended next additions

1. Add more KD-Tree query variants such as nearest-neighbor search
2. Add more Quadtree variants such as loose quadtrees or MX-quadtrees
3. Add config-driven experiment runners, for example JSON or YAML sweep files
4. Add unit tests for insertion/query correctness before doing more performance work

## How to run

### Local demo
```bash
cd project
python -m treeindex.cli.run_local_demo
```

### Distributed B+ tree demo
```bash
cd project
spark-submit -m treeindex.cli.run_distributed_bplustree
```

### Distributed R-Tree demo
```bash
cd project
spark-submit -m treeindex.cli.run_distributed_rtree
```

### Distributed KD-Tree demo
```bash
cd project
spark-submit -m treeindex.cli.run_distributed_kdtree
```

### Distributed Quadtree demo
```bash
cd project
spark-submit -m treeindex.cli.run_distributed_quadtree
```

If your Spark environment does not support `-m`, run the module file directly:
```bash
spark-submit src/treeindex/cli/run_distributed_bplustree.py
spark-submit src/treeindex/cli/run_distributed_rtree.py
spark-submit src/treeindex/cli/run_distributed_kdtree.py
spark-submit src/treeindex/cli/run_distributed_quadtree.py
```

## Implementation notes

### Centralized local trees

1. Data generation:
   Each local demo creates synthetic data first, then samples queries from that dataset so the workloads are valid. `BPlusTree` uses `generate_key_value_pairs(...)` plus `sample_point_queries(...)` and `sample_range_queries(...)`. `RTree` uses `generate_rectangles(...)` plus `sample_rect_point_queries(...)` and `sample_rect_queries(...)`. `KDTree` and `QuadTree` use `generate_points(...)` plus `sample_exact_point_queries(...)` and `sample_point_rect_queries(...)`.
2. Index/tree construction:
   Each local structure is built entirely in memory. `BPlusTree` is built by repeated insertions with node splits. `RTree` is built by repeated rectangle insertion with subtree selection and MBR updates. `KDTree` is built recursively by alternating split axes and choosing median points. `QuadTree` starts from one bounding box and subdivides into four children when a bucket overflows.
3. Point searching and range query:
   All local trees prune the search space with their structure. `BPlusTree` follows separator keys to one leaf for point search and scans linked leaves for ranges. `RTree` follows only MBRs that contain or intersect the query. `KDTree` prunes by axis-aligned split planes. `QuadTree` prunes by quadrant bounds and descends only into relevant cells.

### Distributed trees

All distributed demos follow the same overall pattern: Spark maps each record to a partition ID, `partitionBy(...)` shuffles records so records with the same partition ID go to the same reducer, each reducer rebuilds one local tree, and the driver collects compact partition metadata to route later queries.

#### Distributed B+ Tree

- Map:
  The mapper computes a partition ID from the key range. After the global minimum and maximum key are found, the key space is divided into contiguous chunks, and each key-value pair is assigned to the chunk containing its key.
- Reduce / build:
  Each reducer receives one key-range partition, sorts the rows by key, rebuilds a local `BPlusTree`, and emits metadata such as `min_key`, `max_key`, item count, and local tree height.
- Query:
  The driver uses the collected partition boundaries as a lightweight directory. A point query routes to exactly one partition. A range query routes to every partition whose key interval overlaps `[low, high]`. Those partitions rebuild/search their local B+ trees, and Spark collects the matching rows.

#### Distributed R-Tree

- Map:
  The mapper computes the global spatial bounds, overlays a 2D grid, and assigns each rectangle to a partition using the rectangle center. This keeps nearby rectangles in the same Spark partition.
- Reduce / build:
  Each reducer rebuilds one local `RTree` from its rectangles and emits metadata including the partition MBR, item count, and local height.
- Query:
  The driver first checks partition MBRs. A point query searches only partitions whose MBR contains the query point. A rectangle query searches only partitions whose MBR intersects the query rectangle. Those candidate partitions rebuild/search their local R-trees, and Spark merges the results.

#### Distributed KD-Tree

- Map:
  The mapper computes global point bounds, divides the space into a 2D grid, and assigns each point to a grid cell partition based on its `(x, y)` coordinates.
- Reduce / build:
  Each reducer rebuilds one local `KDTree` from its assigned points and emits metadata including the partition MBR, item count, and local height.
- Query:
  The driver uses partition MBRs as the routing layer. A point query searches only partitions whose MBR contains the point. A range query searches only partitions whose MBR intersects the query rectangle. Each selected partition runs the local KD-tree query and Spark collects the combined answer.

#### Distributed Quadtree

- Map:
  The mapper computes global point bounds, then assigns each point a partition ID by recursively descending quadtree quadrants until the configured partition depth is reached.
- Reduce / build:
  Each reducer rebuilds one local `QuadTree` for its partition and emits metadata including the partition MBR, item count, and local height.
- Query:
  The driver routes queries using partition MBRs. A point query checks only partitions whose MBR contains the point. A range query checks only partitions whose MBR intersects the query rectangle. Candidate partitions run the local quadtree search, and Spark collects the final matches.

## Editing experiment parameters

The demos now read their default parameters from JSON config files in `configs/`:

- `configs/local_demo.json`
- `configs/distributed_bplustree.json`
- `configs/distributed_kdtree.json`
- `configs/distributed_quadtree.json`
- `configs/distributed_rtree.json`

`configs/local_demo.json` contains one section per local experiment: `local_bplustree`, `local_rtree`, `local_kdtree`, `local_quadtree`, and `local_comparison`. The `local_comparison` section now includes `bplustree`, `rtree`, `kdtree`, and `quadtree` comparison sweeps.

### Config parameter reference

Shared parameters:
- `n_items` / `n_records`: number of generated records before building the index.
- `seed`: base random seed for reproducible data generation and query sampling.
- `n_point_queries`: number of exact point or point-location queries to run.
- `n_queries`: number of spatial range/intersection queries to run.
- `n_range_queries`: number of B+ tree key-range queries to run.
- `n_partitions`: number of Spark partitions used by distributed runs.

B+ Tree parameters:
- `key_min`, `key_max`: minimum and maximum generated key values.
- `allow_duplicates`: whether multiple records may share the same key.
- `tree_order`: B+ tree fanout / node capacity setting.
- `range_low_min`, `range_low_max`: bounds for sampling the lower endpoint of generated range queries.
- `range_max_width`: maximum width of a generated B+ tree range query.
- `orders`: list of B+ tree order values used in the local comparison sweep.

Spatial parameters used by R-tree, KD-tree, and Quadtree:
- `space_width`, `space_height`: width and height of the synthetic 2D data space.
- `max_query_width`, `max_query_height`: maximum size of generated rectangle queries.

KD-tree-specific parameters:
- `leaf_capacity`: maximum number of points stored in a KD-tree leaf bucket before splitting.
- `leaf_capacities`: list of KD-tree leaf capacities used in the local comparison sweep.

R-tree-specific parameters:
- `max_rect_width`, `max_rect_height`: maximum size of generated rectangles in the dataset.
- `max_entries`: maximum number of entries per R-tree node before a split.
- `capacities`: list of `max_entries` values used in the local comparison sweep.

Quadtree-specific parameters:
- `bucket_capacity`: maximum number of points stored in a leaf before subdivision.
- `bucket_capacities`: list of quadtree bucket capacities used in the local comparison sweep.
- `max_depth`: maximum allowed quadtree depth.

For routine changes, edit those files directly instead of changing Python code or passing many CLI flags.

All distributed runners also save the final experiment summary automatically to `results/` as both `.txt` and `.json` files, so the output is preserved even if Spark logs scroll past it. Use `--output-dir <path>` to write those files somewhere else.

Example: increase the distributed dataset size by editing `configs/distributed_bplustree.json`:

```json
{
  "n_records": 500000,
  "n_partitions": 16,
  "tree_order": 64
}
```

You can still override specific distributed settings on the command line when needed:

```bash
spark-submit src/treeindex/cli/run_distributed_bplustree.py \
  --n-records 500000 \
  --n-partitions 16
```

For the distributed R-Tree, edit `configs/distributed_rtree.json` or override selected values:

```bash
spark-submit src/treeindex/cli/run_distributed_rtree.py \
  --n-items 75000 \
  --n-partitions 12 \
  --max-entries 16
```

For the distributed KD-Tree, edit `configs/distributed_kdtree.json` or override selected values:

```bash
spark-submit src/treeindex/cli/run_distributed_kdtree.py \
  --n-items 75000 \
  --n-partitions 12 \
  --n-queries 300
```

For the distributed Quadtree, edit `configs/distributed_quadtree.json` or override selected values:

```bash
spark-submit src/treeindex/cli/run_distributed_quadtree.py \
  --n-items 75000 \
  --n-partitions 12 \
  --bucket-capacity 16
```

Example with an explicit output directory:

```bash
spark-submit src/treeindex/cli/run_distributed_kdtree.py \
  --output-dir Doc/result
```

You can also point either demo at a different config file:

```bash
python -m treeindex.cli.run_local_demo --config configs/local_demo.json
spark-submit src/treeindex/cli/run_distributed_bplustree.py --config configs/distributed_bplustree.json
```
