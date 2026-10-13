# The website in one container (WEBSITE_PLAN.md §4.3): the static site and the inference API.
#   docker build -f web/server/Dockerfile -t colour-not-letters .
#   docker run -p 7860:7860 -v "$PWD/runs/ablation_dinov2_s0/checkpoints:/ckpt:ro" \
#     -e CHECKPOINT=/ckpt/best.pt colour-not-letters
# On a Hugging Face Space (Docker SDK) set HF_MODEL_REPO (and the HF_TOKEN secret for a private repo)
# instead of mounting the checkpoint. Runs as uid 1000, port 7860, as Spaces expect.

FROM node:24-slim AS site
WORKDIR /site
COPY web/site/package.json web/site/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/site/ ./
RUN npm run build

FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HOME=/home/user HF_HOME=/home/user/.cache/huggingface MPLCONFIGDIR=/tmp/matplotlib
RUN useradd -m -u 1000 user
WORKDIR /home/user/app
COPY web/server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 \
 && pip install --no-cache-dir -r /tmp/requirements.txt
COPY --chown=user configs ./configs
COPY --chown=user data ./data
COPY --chown=user verifier ./verifier
COPY --chown=user models ./models
COPY --chown=user experiments ./experiments
COPY --chown=user analysis ./analysis
COPY --chown=user web/__init__.py ./web/__init__.py
COPY --chown=user web/server ./web/server
COPY --chown=user --from=site /site/dist ./web/site/dist
USER user
# The backbone's architecture (config only, a few KB; the weights come from the checkpoint).
RUN python -c "from transformers import AutoConfig; AutoConfig.from_pretrained('facebook/dinov2-large')"
EXPOSE 7860
CMD ["python", "-m", "web.server.app", "--host", "0.0.0.0", "--port", "7860", "--static", "web/site/dist"]
