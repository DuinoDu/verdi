# verdi_remote.py acceptance evidence — v2 (R2S-DIAG-20261005 v2-C)

Script under test: `scripts/verdi_remote.py` sha256 `3e873337fbdebcec8316f944693ff85b7cb92076219d5b88ab39e8a34b7baec9`
(unchanged since commit `6d274c9`). Evidence is split into three classes; a row in one
class is never counted for another.

## 1. Real tests already run (real relay / 063, before this batch; no new GPU)
| check | run id / command | observed (stdout as recorded in the dev session) |
|---|---|---|
| normal run + verified download, own dir removed afterwards | `20261005-234820-6bc9e7`: foundation_stereo estimate_depth_seq, synthetic fixture, `--device auto` (063 cuda:0) | `t1 rc=0`; outputs `['depth', 'disparity_px', 'info', 'valid']`; `/tmp/vr_t1: log.txt out result.json stderr.txt`; `depth: 000000.npy 000001.npy scale.json`; run id absent from the next `--list` |
| `--keep-remote` then `--resume` (SUCCESSFUL run kept on purpose — this is NOT a failure-retention test) | `20261005-234927-b6dcc8` | `t2 rc=0`; `--list` showed `... done 0h`; `resume rc=0`, depth path `/tmp/vr_t3/out/depth.npy`; dir gone afterwards |
| resume of a non-existent run | `--resume 20990101-000000-nonexist` | `missing rc=2`, `transport remote run dir ~/verdi_remote/20990101-000000-nonexist does not exist on 063 (deleted?)` |
| delete argument validation | `--delete "*"`; `--delete 20990101-000000-abcdef` | `error: --delete takes one exact run id`; `run ... does not exist` |
| list is read-only | `--list` | listed 12 other calls' dirs as `unmarked`; none modified |

Real FAILED runs that exist (old runner `a016825`, before these fixes) — the incident:
* real2sim runs `ep037_grasp` (063 cuda:7, remote id 20261005-232535-6aaa88) and
  `ep003_place` (cuda:6, 20261005-232536-101395) failed because the developer ran a manual
  wildcard `rm -rf ~/verdi_remote/*` on 063 while they were running. Evidence: 063
  `~/pdebug_home/runs/foundation_stereo/20261005-232547-c5592e` and `-6dfd40`
  (`result.json`: `FileNotFoundError .../verdi_remote/<id>/out/depth/000010.npy`), and the
  consumer's `stereo.result.json` (transport, rsync 23 / unexpected tag 103). This was a
  shared-directory cleanup by a person, not a runner pass/fail; it is recorded, not
  excused. No new-runner real failure run exists.

## 2. CPU state simulation / code inspection (this batch; no model, no network, no real run dir)
`tests/test_verdi_remote.py` (sha256 `6a136b6eb8bd166f1c928598f5d5ef9f4d7d185f0040b2dd2ee1e297bf0eb937`):
relay and 063 simulated by temp dirs (ssh/rsync rewritten to local commands), the remote
`verdi run` is a fake script that counts invocations. Run on 063 at HEAD `b6e25d3`:
`pytest -v tests` -> 17 passed (log `pytest_v2.txt` in the report dir).
| test | asserts |
|---|---|
| test_normal_run_verified_download_then_own_dir_removed | rc 0; output content arrives; node calls = 1; remote root empty afterwards |
| test_failure_keeps_remote_and_resume_downloads_without_rerun | injected download failure (rsync of out/) -> rc 2, kind transport, hint contains `--resume <run_id>`; remote dir kept with DONE, out/x.txt, result.json; node calls = 1; `--resume` -> rc 0, content arrives, node calls still 1; dir removed only after verified resume |
| test_delete_refuses_active_and_unmarked_and_wildcards | `--delete` of an ACTIVE fixture dir and of an unmarked dir -> rc 2, file sha256 of both unchanged; `--delete "*"` / `"20991231-*"` rejected by argument validation; DONE fixture dir deletable; other dirs untouched |
Code inspection: every `rm -rf` in the script targets `<run_id>` paths or the validated
`--delete` id (4 sites, `grep -n "rm -rf" scripts/verdi_remote.py`); integrity check = every
output path listed in result.json must exist locally after download (presence check, no
content hash yet; content hashing is SC-03).

## 3. Not tested
* Connection drop during a real run: the job runs detached (`setsid nohup` on 063, polled
  for DONE) — a code mechanism; no real fault injection was done.
* `--delete` against a live ACTIVE run on 063 (only the CPU fixture above).
* Download failure on the real relay with the new runner (only simulated).
* Content-hash verification of downloads (presence only).
