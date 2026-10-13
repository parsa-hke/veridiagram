# Colab guide: finishing Phase 7

Runs the 13 remaining Phase 7 runs (`configs/queue_phase7_colab.yaml`) on a Colab A100, one after
another, overnight. Everything else (data, teacher model, results) is already on your Drive.

| What | Runs | Plan item |
|---|---|---|
| Part A, 50k synthetic | 3 seeds | 7.9 |
| Part A, 1k synthetic | 1 seed | 7.9 |
| Part B, confidence filter | 3 seeds | 7.11 (a) |
| Part B, confidence + consistency filter | 3 seeds | 7.11 (b) |
| Part B control, no pseudo-labels | 1 seed | 7.11 (control) |
| DINOv2-L backbone | 1 seed | 7.5 |
| No highlight, query as tokens | 1 seed | 7.8 |

**Estimate (not yet measured on an A100):** about 30-45 minutes per run, so 7-10 hours in total and
roughly 85-110 compute units. The queue does seed 0 of every experiment first, so if it stops early
every row of the results table still has a number.

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
   (`-ims` lets the screen turn off while the Mac stays awake; `-dims` would also keep the screen on.)
   (Running all night on the charger is fine for the Mac: the GPU work happens on Google's machine,
   and the laptop only keeps a browser tab open.)

## 2. Open the launcher in Colab

1. Open `notebooks/colab_launcher.ipynb` in Colab as before (File → Open notebook → GitHub →
   `parsa-hke/veridiagram`, branch `main`). If an older copy is open, close it first.
2. **Runtime → Change runtime type → A100 GPU** (turn on High-RAM if it is offered), then Save.
3. The key icon must still list `GH_TOKEN` and `HF_TOKEN` with notebook access on.

## 3. Run the cells from the top, one at a time

| Cell | Check |
|---|---|
| `nvidia-smi` | shows `NVIDIA A100` |
| Drive mount | allow access when asked |
| clone / pull | the last line shows the newest commit (the one that added this guide) |
| `pip install` | no failed install. A red "dependency conflicts" note about Colab's own packages (e.g. `google-adk` and `opentelemetry`) is harmless |
| device info | `cuda`, A100 |
| Data | `data version 1`, then sizes of about 3.1G and 355M (about 2 minutes; nothing is rebuilt) |
| `models.backbone` | last line `facebook/dinov3-vitl16-pretrain-lvd1689m: OK` |
| Phase 6 smoke | last line `GATE 6.6 (smoke): PASS` (about 4 minutes) |
| **Part B teacher** | `... 606406334 bytes, sha256 OK` (first time about 2 minutes: it unpacks the lab model from `lab_results` on your Drive) |
| **Phase 7 smoke** | last line `GATE 7 (smoke of the 7.8 and Part B modes): PASS` (about 5 minutes) |

If any check differs, stop and send me the cell's output.

## 4. Start the queue

In the **Full runs** cell change `RUN_FULL = False` to `RUN_FULL = True` and run it. It first prints
`0 done, 13 pending of 13`, then one line when each run starts and one when it finishes. A run's own
progress (`LP stage ...`, `epoch 1/2 ...` at the end of each epoch, `eval ...`) goes to its log on the
VM's disk; to watch it, open Colab's **Terminal** (bottom left) and run
`tail -f /content/veridiagram/runs/queue_logs/<run>.log` (Ctrl+C stops watching).

Leave the tab open and the laptop awake. You can check progress from your phone in the Colab app
or by looking at `MyDrive/geo-relations/runs/log.csv` (one line per finished run; `status` must be `ok`:
the queue's "done" only means the process ended).

## 5. If Colab disconnects

Nothing is lost. Reconnect (Runtime → Connect, A100), then run all cells from the top again with
`RUN_FULL = True`. Finished runs are skipped, an interrupted run resumes from its last finished
epoch, and the teacher and smoke cells are quick the second time.

If Colab says you are out of compute units, stop there and tell me: the queue order means the
finished runs are already usable.

## 6. When the queue has finished

1. The Full runs cell ends with `queue finished: 13 ok, 0 failed`.
2. Run the **Pack the results** cell. It writes `geo-relations/geo_results_colab.tar.gz` (tens of MB)
   on your Drive.
3. **Runtime → Disconnect and delete runtime** (stops billing).
4. Press Ctrl+C in the Terminal window with `caffeinate`.
5. Tell me it's done. I'll read the archive from Drive (or download it to `Downloads` if I ask),
   import it next to the lab results and run `python -m analysis.phase7` to wrap up Phase 7.
