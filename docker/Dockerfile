# TMEImmune v2 -- TME scoring and sex-aware immunotherapy response prediction
#
# Everything below runs from this docker/ folder. The package itself comes from PyPI, so nobody needs
# to clone the repository to build the image.
#
# 1. Build for your own machine first and check it works -- minutes, not an hour:
#        docker build -t tmeimmune:test .
#        cd .. && docker run --rm -v $(pwd):/work -w /work tmeimmune:test python docker/docker_test.py
#
# 2. Then build and publish for both architectures. This needs a builder that can do more than the
#    host's own platform, created once:
#        docker buildx create --name tmeimmune --use --bootstrap
#    and then, from docker/ again:
#        docker buildx build --platform linux/amd64,linux/arm64 \
#            -t qiluzhou/tmeimmune:v2.0.0 \
#            -t qiluzhou/tmeimmune:v2 \
#            -t qiluzhou/tmeimmune:latest --push .
#
#    buildx cannot --load a multi-platform build into the local daemon, so --push is required: log in
#    with `docker login` first. On Apple Silicon the linux/amd64 half runs under emulation and pulls
#    the CUDA build of torch, so allow a long build.
#
# 3. Confirm what was published carries both architectures:
#        docker buildx imagetools inspect qiluzhou/tmeimmune:v2.0.0
#
# Running it:
#     docker run --rm -v $(pwd):/work -w /work tmeimmune:v2 python docker/docker_test.py
#     docker run --rm -it -v $(pwd):/work tmeimmune:v2            # just a prompt
#
# The image carries the package and its dependencies, including the pretrained ISAFN model. It does
# not carry the example data, which lives in the repository -- mount it as above.
#
# Size: on linux/amd64 torch from PyPI is the CUDA build, so that half of the image is several GB;
# linux/arm64 has no CUDA wheels and comes out around 2 GB. For a CPU-only amd64 image instead:
#     pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==<version>

FROM python:3.12-slim

LABEL org.opencontainers.image.title="TMEImmune" \
      org.opencontainers.image.version="2.0.0" \
      org.opencontainers.image.description="TME scoring and sex-aware ICI response prediction" \
      org.opencontainers.image.source="https://github.com/ShahriyariLab/TMEImmune" \
      org.opencontainers.image.authors="Qilu Zhou <qiluzhou@umass.edu>"

# figures are written to files, never to a display
ENV MPLBACKEND=Agg \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /work

# build-essential and python3-dev build any dependency without a prebuilt wheel;
# libgomp1 is the OpenMP runtime torch links against and is not in the slim base image
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        g++ \
        python3-dev \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# every version is pinned, so this image is the same next month as it is today
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Fail the build, not somebody's first run, if the image is missing what the package needs.
#
# These checks import torch and unpickle the ISAFN model. Natively that costs seconds. Under qemu
# emulation -- which is what a cross-platform build does -- importing CUDA torch takes over an hour,
# far longer than installing it. So they run only when the target platform is the one doing the
# building, and a cross-built platform is verified afterwards by running docker/docker_test.py
# against the pushed image instead. BuildKit fills these two ARGs in; they are empty for a plain
# non-BuildKit build, which is always native.
#
# Written as one-line python -c rather than a heredoc, which would need the BuildKit dockerfile 1.4+
# frontend and fail on an older parser.
ARG BUILDPLATFORM
ARG TARGETPLATFORM
RUN if [ -z "$TARGETPLATFORM" ] || [ "$BUILDPLATFORM" = "$TARGETPLATFORM" ]; then \
      python -c "import TMEImmune, torch, sklearn, scipy, statsmodels, gseapy, lifelines, inmoose, joblib, matplotlib; \
from TMEImmune import data_processing, gene_id, TME_score, ISAFN, optimal; \
assert TMEImmune.__version__.startswith('2.'), 'expected 2.x, got ' + TMEImmune.__version__; \
assert gene_id.annotation_available(), 'the packaged gene annotation is missing from the wheel'; \
assert hasattr(data_processing, 'set_confirm') and hasattr(optimal, 'decision_curve'), 'this is not the v2 API'; \
m = ISAFN.load_isafn_models(); \
print('TMEImmune', TMEImmune.__version__, '| torch', torch.__version__, '|', len(gene_id.load_annotation()['genes']), 'gene symbols | ISAFN model loaded')"; \
    else \
      echo "cross-building $TARGETPLATFORM on $BUILDPLATFORM: skipping the import and model checks," \
           "which would run under emulation. Verify the pushed image with docker/docker_test.py."; \
    fi

CMD ["python3"]
