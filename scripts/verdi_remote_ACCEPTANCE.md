# verdi_remote.py acceptance evidence (commit 6d274c9; tests run 2026-10-05 from apex, 063 cuda:0)

| check | how | result |
|---|---|---|
| per-run isolation | every call uses ~/verdi_remote/<run_id> on 063 and /tmp/verdi_remote/<run_id> on the relay (run_id = time + 6 hex); all `rm -rf` in the script are of `<run_id>` paths (4 sites, see `grep -n "rm -rf" scripts/verdi_remote.py`) | no wildcard / shared cleanup in code |
| normal run + verified download | foundation_stereo estimate_depth_seq (synthetic fixture), --device auto | rc 0; cuda:0 chosen (31.8 GB free); outputs depth/disparity_px/valid/info + depth/scale.json arrived; remote dir removed only after verification |
| failure keeps results, resume without rerun | same node with --keep-remote, then `--resume <run_id> --out <new dir>` | rc 0 both; resume downloaded the existing result (no node run) |
| missing run | `--resume 20990101-000000-nonexist` | rc 2, kind transport, "remote run dir ... does not exist" |
| restricted delete | `--delete "*"`; `--delete 20990101-000000-abcdef` | refused (exact run id format required); "does not exist" |
| list | `--list` | listed 12 dirs of other calls as `unmarked`, untouched |
| multi-line command transport | remote heredoc + detached launch probe | script written, DONE set, quotes intact (fix: commands sent base64 on stdin; the jump host kept only the first line before) |

Known limits / not fault-injection tested:
* "connection drop does not kill the job" is by construction (setsid nohup on
  063 + polling for DONE); an actual mid-run ssh kill was not injected.
* `--delete` refusal of an ACTIVE run is implemented (marker check) but was
  not exercised against a live run.
* dirs created by the pre-0e79e13 runner have no ACTIVE/DONE marker
  (`unmarked`): `--resume` works for them, `--delete` refuses them.
* the earlier loss of two real runs was caused by a manual wildcard
  `rm -rf ~/verdi_remote/*` by the developer, not by the runner; no such
  cleanup will be done again.
* the provenance does not yet record input content hashes inside verdi
  (SC-03 proposal); real2sim's bridge records them.
