# Colab guide: the last Phase 7 batch

Runs `configs/queue_phase7_final.yaml` (29 runs) on a Colab A100, one after another. Everything it
needs (data, the 13 earlier Colab runs with their checkpoints, the lab archives) is already on your
Drive. The first Colab batch (13 runs, 2026-10-13) is done and imported; its queue
(`queue_phase7_colab.yaml`) is kept as the record.

| What | Runs | Plan item |
|---|---|---|
| Part B control, no pseudo-labels, seeds 1 and 2 | 2 fine-tunes | 7.11 (control) |
| Square-padded re-evaluation of every checkpoint that exists (`sq_<experiment>_s<seed>`) | 24 evaluations | §0.2 #10 note |
| 7.2 retrained on Colab (`finetune_colab`; the lab did not keep its checkpoints) | 3 fine-tunes | 7.2 |

The re-evaluations cover: probes (3), 7.3 / Part A 10k (3), Part A 1k (1) and 50k (3), DINOv2 (1),
no overlay (1), Part B confidence (3), consistency (3), control (3), and `finetune_colab` (3). The
originals stay as they are. 7.6 (224 px) and 7.7 (no augmentation) have no checkpoint and keep their
original numbers.

**Estimate:** about 6.5 hours, roughly 75-80 compute units at ~12 units per A100 hour:
~25 min of setup and smoke checks, 2 x ~39 min for the control seeds (seed 0 took 39 min),
~7.5 min per re-evaluation (loading the model and 52,818 evaluation queries; measured from the
first batch's run times, ~140 queries/s), 3 x ~45 min for `finetune_colab`. The queue does the control
seeds first, then the re-evaluations, then the retrained 7.2, so a partial session still leaves
complete groups.

---

## 1. Prepare the laptop (2 minutes)

1. Plug in the charger. Keep the lid **open**: closing it puts the Mac to sleep.
2. **System Settings → Battery → Options:** turn on "Prevent automatic sleeping on power adapter
   when the display is off".
3. Open **Terminal** and run the line below; leave the window open. It keeps the Mac awake until
   you press Ctrl+C:
   ```bash
   caffeinate -ims
   ```
   (`-ims` lets the screen turn off while the Mac stays awake.) The GPU work happens on Google's
   machine; the laptop only keeps a browser tab open.

## 2. Open the launcher in Colab

1. Open `notebooks/colab_launcher.ipynb` in Colab (File → Open notebook → GitHub →
   `parsa-hke/veridiagram`, branch `main`). If an older copy is open, close it first: the cells
   changed.
2. **Runtime → Change runtime type → A100 GPU** (turn on High-RAM if it is offered), then Save.
3. The key icon must still list `GH_TOKEN` and `HF_TOKEN` with notebook access on.

## 3. Run the cells from the top, one at a time

| Cell | Check |
|---|---|
| `nvidia-smi` | shows `NVIDIA A100` |
| Drive mount | allow access when asked |
| clone / pull | the last line shows the newest commit (the one that added this guide's queue) |
| `pip install` | no failed install. A red "dependency conflicts" note about Colab's own packages is harmless |
| device info | `cuda`, A100 |
| Data | `data version 1`, then sizes of about 3.1G and 355M (about 2 minutes; nothing is rebuilt) |
| `models.backbone` | last line `facebook/dinov3-vitl16-pretrain-lvd1689m: OK` |
| Phase 6 smoke | last line `GATE 6.6 (smoke): PASS` (about 4 minutes) |
| **Lab checkpoints** | twelve lines, each ending `sha256 OK` (first time about 4 minutes: it unpacks two more 7.3 checkpoints, the probe heads and their configs from `lab_results` on your Drive) |
| **Phase 7 gates** | two PASS lines: `GATE 7 (smoke of the 7.8 and Part B modes): PASS` and `GATE 7 (smoke of the square-padded re-evaluation): PASS` (about 8 minutes) |

If any check differs, stop and send me the cell's output.

## 4. Start the queue

In the **Full runs** cell change `RUN_FULL = False` to `RUN_FULL = True` and run it. It first prints
`0 done, 29 pending of 29`, then one line when each run starts and one when it finishes. A run's own
progress goes to its log on the VM's disk; to watch it, open Colab's **Terminal** (bottom left) and
run `tail -f /content/veridiagram/runs/queue_logs/<run>.log` (Ctrl+C stops watching). A
re-evaluation prints `evaluate: <source run> ... pad square`, then one `eval` line per unit.

Leave the tab open and the laptop awake. You can check progress from your phone in the Colab app or
in `MyDrive/geo-relations/runs/log.csv` (one line per finished run; `status` must be `ok`).

## 5. If Colab disconnects

Nothing is lost. Reconnect (Runtime → Connect, A100), then run all cells from the top again with
`RUN_FULL = True`. Finished runs are skipped, an interrupted fine-tune resumes from its last finished
epoch, an interrupted re-evaluation starts again (a few minutes), and the checkpoint and smoke cells
are quick the second time.

If Colab says you are out of compute units, stop there and tell me which runs finished.

## 6. When the queue has finished

1. The Full runs cell ends with `queue finished: 29 ok, 0 failed`.
2. Run the **Pack the results** cell. It writes `geo-relations/geo_results_colab_final.tar.gz`
   (tens of MB) on your Drive.
3. **Runtime → Disconnect and delete runtime** (stops billing).
4. Press Ctrl+C in the Terminal window with `caffeinate`.
5. Download the archive to `Downloads` and tell me it's done. I'll check it (status ok, two epochs for
   the fine-tunes, the expected backbone, exact prediction counts, re-evaluations aligned with their
   originals), import it into `runs/` and `runs/log.csv`, and regenerate the tables, figures and
   website data.
