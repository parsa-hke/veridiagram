# UOWD AI Lab guide: running Phase 7

**Windows lab machines: see "Windows" at the end** (PowerShell commands for every step). The
sections A-F below are written for Linux.

Every command runs from the repository folder on a Linux machine with NVIDIA GPUs. Copy them
exactly. If anything in a **Check** line looks different, stop there and send the output.

---

## A. Tonight, at home (15 minutes)

**A1. Have two tokens ready** (the same ones Colab uses; keep them in a password manager, not on paper):
- GitHub fine-grained token, read-only, repository `parsa-hke/veridiagram`
- Hugging Face read token (your account has DINOv3 access)

**A2. Make a USB backup of the data** (in case the lab network blocks downloads). Use a stick
formatted **exFAT** with at least 5 GB free. In the Mac Terminal (replace `USB` with the stick's name):

```bash
tar -C /Users/parsa/Documents/veridiagram -cf /Volumes/USB/veridiagram_data.tar data/raw data/processed
```

Check: the file is about 3.5 GB.

**A3. Ask the lab staff, if you can, before you go:**
1. Which GPUs, and how many per machine?
2. Can a job keep running after you log out (overnight or for days)? Is there SSH access from home?
3. Is there a job scheduler (SLURM), and what are its time limits?
4. Do the machines have internet access to GitHub, Hugging Face and PyPI?

---

## B. In the lab: one-time setup (30 to 45 minutes)

**B1. Open a terminal and look at the GPUs**

```bash
nvidia-smi
```

Check: a table with one row per GPU and a driver version at the top. Write down the GPU model,
the memory per GPU and the driver version.

If the machine runs Windows, open "Ubuntu" (WSL2) and continue there; if WSL2 is not installed,
stop and ask the staff.

**B2. Get the code** (paste your GitHub token instead of `GH_TOKEN_HERE`)

```bash
cd ~
git clone https://GH_TOKEN_HERE@github.com/parsa-hke/veridiagram.git
cd veridiagram
git remote set-url origin https://github.com/parsa-hke/veridiagram.git
git log -1 --oneline
```

Check: the last line shows the newest commit. The `set-url` line removes the token from the saved
settings.

**B3. Install everything**

```bash
bash scripts/lab_setup.sh
```

This installs Python 3.11 and the pinned packages into `.venv` (no admin rights needed), picks a
PyTorch build that matches the driver, tests the GPUs and runs the test suite.

Check: it ends with `bf16 matmul ok`, then `147 passed` (or more), then `setup done`.

- If it says **CUDA is NOT available**, run the command it prints (with `TORCH_INDEX=...`).
- If a GPU line says **bf16 NO** (older GPUs such as V100 or T4), add
  `--set finetune.amp_dtype=float16` to every run and queue command below.

**B4. Set the session variables** (paste your Hugging Face token; repeat this block in every new
terminal)

```bash
source .venv/bin/activate
export GEO_OVERLAYS=configs/lab.yaml
export HF_TOKEN=HF_TOKEN_HERE
```

The token lives only in this terminal; nothing is written to disk.

**B5. Get the data. Option 1, the network works** (about 15 to 20 minutes):

```bash
python -m data.prepare --download
```

**Option 2, from the USB stick** (plug it in; replace `/media/USB` with where it appears):

```bash
tar -xf /media/USB/veridiagram_data.tar
```

Check: `du -sh data/raw data/processed` shows about 3.1G and 355M.

**B6. Check DINOv3 access**

```bash
python -m models.backbone
```

Check: the last line is `facebook/dinov3-vitl16-pretrain-lvd1689m: OK` (24 layers, 4 register
tokens, 303.1M parameters). If it says FAILED with 401, `HF_TOKEN` is not set in this terminal.

**B7. Smoke test** (about 5 minutes on a modern GPU)

```bash
python -m experiments.run --config configs/probe.yaml --smoke
python -m experiments.run --config configs/finetune.yaml --smoke
python -m experiments.gate smoke
```

Check: the last line is `GATE 6.6 (smoke): PASS`.

---

## C. Measure, decide, start (20 minutes)

**C1. Benchmark** (about 5 to 10 minutes)

```bash
python -m experiments.benchmark
```

It measures training and evaluation speed on this machine and prints, for every planned run,
the estimated hours, the total, and the wall time for 30, 20, 15 and 10 fine-tune epochs, e.g.

```
wall time by fine-tune epochs: 30 epochs ~XX h, 20 epochs ~XX h, 15 epochs ~XX h, 10 epochs ~XX h
```

**Copy the whole output and send it to me.** If you cannot, decide yourself:

| Time you can leave the machine running | Use |
|---|---|
| more than the "30 epochs" wall time | the plan as written (no extra option) |
| between the 15- and 30-epoch times | `--set finetune.epochs=20` (or 15) |
| less than the 15-epoch time | `--set finetune.epochs=10` and tell me |

**C2. Start the queue inside tmux** (so it keeps running if the terminal closes)

```bash
tmux new -s geo
source .venv/bin/activate && export GEO_OVERLAYS=configs/lab.yaml && export HF_TOKEN=HF_TOKEN_HERE
python -m experiments.queue configs/queue_phase7.yaml
```

Add `--set finetune.epochs=N` at the end if you chose fewer epochs in C1. To use only some GPUs,
add `--gpus 0,1`.

Leave tmux running: press `Ctrl+b`, then `d`. To come back later: `tmux attach -t geo`.
If `tmux` is not installed, use:

```bash
nohup python -m experiments.queue configs/queue_phase7.yaml > runs/queue.out 2>&1 &
```

The queue runs, in order: 3 probes, 3 fine-tunes on real data, 3 fine-tunes with 10k synthetic,
3 ablations (DINOv2, 224 px, no augmentation), Part A at 1k (1 seed) and 50k (3 seeds), one run
per GPU at a time.

---

## D. While it runs

```bash
python -m experiments.queue configs/queue_phase7.yaml --status   # done / pending / failed per run
tail -n 3 runs/log.csv                                           # latest finished runs
tail -f runs/queue_logs/finetune_s0.log                          # one run's progress (Ctrl+C to stop watching)
nvidia-smi                                                       # GPUs busy?
```

- **It is safe to stop and restart.** `Ctrl+C` in the queue's window (or a reboot) interrupts the
  runs; starting the same queue command again skips finished runs and resumes interrupted
  fine-tunes from their last completed epoch.
- **A run failed?** Its log is `runs/queue_logs/<run>.log`. Send me the last 40 lines:
  `tail -n 40 runs/queue_logs/<run>.log`.

---

## E. Before you leave the lab each time

**E1. Pack the results** (small: metrics, predictions, histories, no checkpoints)

```bash
tar --exclude='*/checkpoints' -czf ~/geo_results_$(date +%Y%m%d_%H%M).tar.gz runs reports
```

Copy the `.tar.gz` to the USB stick or upload it to OneDrive or Google Drive, and send it to me.

**E2. When the 7.3 runs are done, also save two checkpoints** (needed for attention maps and Part B):

```bash
tar -czf ~/geo_checkpoints.tar.gz runs/finetune_synth10k_s0/checkpoints/best.pt runs/finetune_s0/checkpoints/best.pt
```

**E3. Last time only: remove your tokens and files from the machine**

```bash
unset HF_TOKEN
rm -f ~/.cache/huggingface/token ~/.git-credentials
```

Delete the `veridiagram` folder only after the results are safely copied.

---

## F. Troubleshooting

| Symptom | What to do |
|---|---|
| `nvidia-smi: command not found` | No NVIDIA driver on this machine: ask the staff for a GPU machine |
| setup says CUDA is NOT available | run the `TORCH_INDEX=... bash scripts/lab_setup.sh` line it prints |
| `401` / `gated repo` | `export HF_TOKEN=...` in this terminal, then rerun |
| `CUDA out of memory` | add `--set finetune.batch_size=8 --set finetune.grad_accum_steps=2` (same effective batch) |
| downloads blocked | use the USB data (B5, option 2); tell me for the DINOv3 weights |
| `No space left on device` | `df -h .`; finished runs keep only `best.pt` (~0.6 GB each); ask for a scratch disk |
| a run keeps failing | `tail -n 40 runs/queue_logs/<run>.log` and send it |

---

## What is not in this queue yet

- 7.8, the no-overlay ablation (needs a model that takes the query as a token index), and
- Part B, self-training (needs the 7.3 model as teacher).

Both come in a later session; the queue skips everything that is already done, so they can be
added to the same queue file and started the same way.

---

## Windows (UOWD AI Lab machines)

All commands in **PowerShell** (Start menu: "Windows PowerShell"), from your home folder.

1. GPUs: `nvidia-smi`
2. Code. With git (`git --version` works):
   `git clone https://GH_TOKEN_HERE@github.com/parsa-hke/veridiagram.git` then `cd veridiagram` and
   `git remote set-url origin https://github.com/parsa-hke/veridiagram.git`.
   Without git:
   `curl.exe -L -H "Authorization: Bearer GH_TOKEN_HERE" -o veridiagram.zip https://api.github.com/repos/parsa-hke/veridiagram/zipball/main`,
   `Expand-Archive veridiagram.zip -DestinationPath .`, `Rename-Item (Get-ChildItem -Directory parsa-hke-veridiagram-*).Name veridiagram`, `cd veridiagram`.
3. Install: `powershell -ExecutionPolicy Bypass -File scripts\lab_setup.ps1` (ends with `setup done`).
4. Every new PowerShell window:
   `Set-ExecutionPolicy -Scope Process Bypass`, `.\.venv\Scripts\Activate.ps1`,
   `$env:GEO_OVERLAYS = "configs/lab.yaml"`, `$env:HF_TOKEN = "HF_TOKEN_HERE"`.
5. Data: `python -m data.prepare --download`.
6. Then the same Python commands as B6-D: `python -m models.backbone`, the smoke runs and
   `python -m experiments.gate smoke`, `python -m experiments.benchmark`,
   `python -m experiments.queue configs/queue_phase7.yaml`.
7. Keep the queue's PowerShell window open. Lock the screen (Win+L) when you leave; do **not** sign
   out or restart (that stops the runs; restarting the queue resumes them).
8. Pack results: `tar --exclude=*/checkpoints -czf $HOME\geo_results.tar.gz runs reports`.
