# partfield: notes

## Role in SimFoundry
Alternative stage-9 segmentation backend (`s9_articulate_objects.method=partfield`,
template `simfoundry/cfg/partfield_template.yaml`, conda env
`articulate-anything-partfield`). `segment_mesh_partfield.py` (fork
`articulate-anything-sf@dec3a618`) runs, in a subprocess:
`partfield_inference.predict(cfg)` with `configs/final/demo.yaml`
(+ `continue_ckpt`, `dataset.data_path`, `feature_output_dir`), loads
`part_feat_<uid>_0_batch.npy`, L2-normalises, builds
`construct_face_adjacency_matrix_facemst(F, V, with_knn=True)` and runs
`AgglomerativeClustering(n_clusters=num_clusters (10), connectivity=adj)`.
The node does exactly this and also returns the raw per-face features.

## Upstream use
- nv-tlabs/PartField, unpinned in SimFoundry; pinned here to `373025d`
  (HEAD at integration). `patches/partfield.patch` verbatim from the fork.
- Input mesh is loaded with `trimesh.load(force="mesh", process=False)`,
  stripped to geometry, written as `<tmp>/data/mesh.glb`, and that same file
  is re-read for clustering so features align 1:1 with faces
  (upstream's loader also uses `process=False`; `preprocess_mesh` is off).
- Upstream writes `exp_results/<result_name>/input_*.ply` relative to the cwd;
  entry.py chdirs into a temp dir for the predict call.
- Lightning `Trainer(devices=-1, strategy=DDP)` runs in-process with one GPU.

## the RTX 5090 host (RTX 5090, sm_120)
- torch 2.7.1 -> 2.8.0 (cu128); torch-scatter built for `{cuda_arch}`.
- tetgen 0.6.5 requires numpy>=2 (conflicts with numpy 1.26.4): pinned 0.6.4.
- `import vtk` (top of partfield/dataloader.py) loads VTK's rendering
  modules, which need libXrender.so.1 (missing on the RTX 5090 host, no X11).
  `patches/lazy-vtk.patch` moves the import into the tetgen remesh branch,
  the only user (not used by this node).
- Visual check: shell, head+antennae, pylon, wheels, legs separated (10 parts).
