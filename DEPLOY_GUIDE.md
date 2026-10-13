# Putting the website live (WEBSITE_PLAN.md milestone 6)

The result: **one public web address** (a Hugging Face Space) that serves the whole website and the
live model, on the free CPU tier. The Space shows visitors the website only: the project's code is
fetched at build time from the private GitHub repository with a read-only token, and the trained
checkpoint lives in a private model repository. Nothing here costs money.

You do every step that creates an account, a token, a repository or anything public. Tokens are typed
into your own terminal or into Hugging Face's settings pages, never into a file or a chat.

Time: about 45 minutes, most of it waiting for uploads and the first build.

---

## 0. Before you start

- [ ] The commit you want live is pushed to GitHub (`git push origin main`). The Space builds exactly
      the pushed commit, never local changes.
- [ ] `runs/ablation_dinov2_s0/checkpoints/best.pt` exists on this Mac (609 MB).
- [ ] Check the MAAI 901 / UOWD rules on publishing coursework: the page shows your name and your
      results while the assessment may still be marked. If in doubt, publish after the marks, or keep
      the Space private (step 9) until then.

## 1. A Hugging Face account

1. Go to <https://huggingface.co/join>, sign up, and confirm your e-mail address (the link in the
   welcome e-mail).
2. Note your **username** (shown in the top-right menu). Below it is written `USER`.

## 2. Log in to Hugging Face on this Mac

In Terminal, in the project folder:

```bash
cd ~/Documents/veridiagram
```

```bash
.venv/bin/hf auth login
```

Choose to log in from the browser (or paste a token of type **Write** created at
<https://huggingface.co/settings/tokens>). Then check:

```bash
.venv/bin/hf auth whoami
```

It must print your username. (This also lets `python -m web.export --online` fill in the DINOv3
facts on the page; tell Claude when you have done it.)

## 3. Two tokens for the Space

The Space needs to read two private things: the code on GitHub (while it builds) and the checkpoint
on Hugging Face (when it starts). Make one token for each, each able to read and nothing more.

### 3a. GitHub: read-only access to this repository only

1. <https://github.com/settings/personal-access-tokens/new> (GitHub → Settings → Developer settings →
   Personal access tokens → Fine-grained tokens → Generate new token).
2. **Token name**: `hf-space-veridiagram`. **Expiration**: 90 days (when it expires the running site
   keeps working; only a rebuild needs a new token, see step 10).
3. **Resource owner**: `parsa-hke`. **Repository access**: *Only select repositories* →
   `parsa-hke/veridiagram`.
4. **Permissions** → Repository permissions → **Contents: Read-only** (Metadata: read-only is added by
   itself). Nothing else.
5. **Generate token** and copy it (it starts `github_pat_`). Keep the tab open until step 5.

### 3b. Hugging Face: a read token

1. <https://huggingface.co/settings/tokens> → **Create new token** → token type **Read**.
2. Name it `space-read-model`, create it and copy it (it starts `hf_`).

## 4. Choose the names

- Space: `USER/colour-not-letters` → the website will be at **`https://USER-colour-not-letters.hf.space`**
- Model repository (private): `USER/colour-not-letters-dinov2`

(Any names work; use the same ones below.)

## 5. Publish

Still in Terminal in the project folder. First give the two Space tokens to this terminal session
only. `read -s` keeps them out of the screen and out of the shell history; paste each token when
asked and press Return:

```bash
read -s "SPACE_GITHUB_TOKEN?GitHub token (github_pat_...): " && export SPACE_GITHUB_TOKEN && echo
```

```bash
read -s "SPACE_HF_TOKEN?Hugging Face read token (hf_...): " && export SPACE_HF_TOKEN && echo
```

Then publish (replace `USER` twice):

```bash
.venv/bin/python -m web.deploy publish --space USER/colour-not-letters --model USER/colour-not-letters-dinov2
```

What it does, in order, printing each step:
1. checks that you are logged in and that the current commit is on GitHub;
2. creates the **private** model repository and uploads `best.pt` (609 MB: a few minutes);
3. creates the **public** Docker Space on the free CPU;
4. stores the two tokens as the Space's **secrets** `GITHUB_TOKEN` and `HF_TOKEN`, and sets the
   variables `HF_MODEL_REPO`, `RUN_NAME` and `READ_CAP` (30 questions per whole-figure reading);
5. uploads the Space's only two files (a README card and the Dockerfile, pinned to the current
   commit), which starts the build.

When it finishes, forget the tokens in this terminal:

```bash
unset SPACE_GITHUB_TOKEN SPACE_HF_TOKEN
```

You can check the secrets on the Space's page → **Settings → Variables and secrets** (values are
never shown again, only names).

## 6. Watch the build

1. Open `https://huggingface.co/spaces/USER/colour-not-letters`. The badge says **Building**.
2. Click **Logs** → *Build* to follow it. The first build takes 10-15 minutes (it downloads PyTorch
   and builds the site). The token never appears in the log: the command shows only
   `$(cat /run/secrets/GITHUB_TOKEN)`.
3. When the badge says **Running**, open **`https://USER-colour-not-letters.hf.space`**. Book IV's
   status line says *warming up* for about a minute while the server downloads the checkpoint, then
   *ready (dinov2-l / ablation_dinov2_s0, on cpu)*.

If something goes wrong:

| What you see | Cause | Fix |
|---|---|---|
| Build log: an error about the secret `GITHUB_TOKEN` (missing or required) | the secret is missing or misnamed | Settings → Variables and secrets: add `GITHUB_TOKEN`, then **Factory reboot** |
| Build log: `git fetch` fails (authentication failed, repository not found) | the GitHub token is wrong, expired, or lacks *Contents: read* on this repository | make a new token (3a), replace the secret, **Factory reboot** |
| Build log: `git fetch` cannot find the commit | the pinned commit is not on GitHub | `git push origin main`, then run step 5 again with `--skip-checkpoint` |
| Page loads, status line says *offline*, Logs → *Container* shows `model load failed: ...` | `HF_TOKEN` cannot read the model repository, or `HF_MODEL_REPO` is wrong | check both in Settings, then **Restart** the Space |
| Answers take very long | the free CPU (2 vCPU) | expected: several seconds a question; whole-figure reading takes minutes |

## 7. Check the live site and record its speed

On this Mac, with the checkpoint present (it compares the Space with the model run locally):

```bash
.venv/bin/python -m web.deploy check --url https://USER-colour-not-letters.hf.space
```

It waits until the Space is ready, asks it three validation questions and compares the answers with
the local model (they should match to about 0.001, with no decision different), then times single
questions and a whole-figure reading. It writes `reports/tables/web_deploy_check.csv`. Then:

```bash
.venv/bin/python -m web.export
```

Tell Claude: it commits the new numbers, and you publish once more (step 10) so the colophon shows
the measured speed.

## 8. Try it like a visitor

On your phone and on a laptop (and in a private window):
- [ ] the frontispiece draws itself; the night edition and the pencil cursor work;
- [ ] Book IV: *Construct* → Examples → *A trapezium* → **Point at** two sides → **Ask**: three
      witnesses appear; **Read the whole figure** shows a progress bar;
- [ ] *Bring your own*: upload a clean textbook diagram, click points, ask;
- [ ] *Beat the Transformer* and *Gallery* work (they need no server).

## 9. Who can see it

- The Space is **public**: anyone with the link sees the website. Its *Files* tab shows only the
  README and the Dockerfile; the code stays private on GitHub, the checkpoint stays private on
  Hugging Face.
- To hide it again: Space → Settings → **Change visibility** → Private (only you, logged in, can then
  open it), or **Delete this Space**.
- Free Spaces **sleep** after about 48 hours without visitors. The next visitor wakes it: Book IV
  says *warming up* for a minute or two, and the game and the gallery work meanwhile.

## 10. Updating the site later

After new commits (for example new exported numbers, Phase 8):

```bash
git push origin main
```

Give this terminal the two Space tokens again (step 5, the two `read -s` lines), then:

```bash
.venv/bin/python -m web.deploy publish --space USER/colour-not-letters --model USER/colour-not-letters-dinov2 --skip-checkpoint
```

```bash
unset SPACE_GITHUB_TOKEN SPACE_HF_TOKEN
```

This pins the new commit, refreshes the secrets (useful when the GitHub token has expired) and
rebuilds. `--skip-checkpoint` skips the 609 MB upload when the model has not changed.

## 11. Optional: rehearse the Space build on this Mac

With Docker Desktop running and `SPACE_GITHUB_TOKEN` set as in step 5:

```bash
.venv/bin/python -m web.deploy space
```

```bash
docker build --secret id=GITHUB_TOKEN,env=SPACE_GITHUB_TOKEN -t cnl-space build/space
```

This is the exact image the Space builds; `web/server/Dockerfile` (tested locally) is the same
image built from your working copy.
