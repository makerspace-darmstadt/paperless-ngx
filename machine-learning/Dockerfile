
FROM python:3.11-slim as base

FROM base as requirements

WORKDIR /usr/src/pipenv

COPY Pipfile* ./

RUN set -eux \
  && echo "Installing pipenv" \
    && python3 -m pip install --no-cache-dir --upgrade pipenv==2023.6.12 \
  && echo "Generating requirement.txt" \
    && pipenv requirements > requirements.txt

# Stage: s6-base
# Purpose: Handles setting up s6-overlay
FROM base as s6-base

ENV \
    PS1="$(whoami)@$(hostname):$(pwd)$ " \
    S6_BEHAVIOUR_IF_STAGE2_FAILS=2 \
    S6_CMD_WAIT_FOR_SERVICES_MAXTIME=0 \
    S6_CMD_WAIT_FOR_SERVICES=1 \
    TERM="xterm-256color"

ARG S6_OVERLAY_VERSION=3.1.5.0

WORKDIR /s6

RUN set -eux \
    && echo "Installing s6 build time packages" \
      && apt-get update \
      && apt-get install --yes --quiet --no-install-recommends \
        curl \
        xz-utils \
    && echo "Installing s6-overlay" \
      && curl --fail --silent --show-error --location --output s6-overlay-noarch.tar.xz  "https://github.com/just-containers/s6-overlay/releases/download/v${S6_OVERLAY_VERSION}/s6-overlay-noarch.tar.xz" \
      && curl --fail --silent --show-error --location --output s6-overlay-noarch.tar.xz.sha256 "https://github.com/just-containers/s6-overlay/releases/download/v${S6_OVERLAY_VERSION}/s6-overlay-noarch.tar.xz.sha256" \
      && curl --fail --silent --show-error --location --output s6-overlay-x86_64.tar.xz "https://github.com/just-containers/s6-overlay/releases/download/v${S6_OVERLAY_VERSION}/s6-overlay-x86_64.tar.xz" \
      && curl --fail --silent --show-error --location --output s6-overlay-x86_64.tar.xz.sha256 "https://github.com/just-containers/s6-overlay/releases/download/v${S6_OVERLAY_VERSION}/s6-overlay-x86_64.tar.xz.sha256" \
      && ls -ahl . \
      && echo "Validating s6-archives" \
        && sha256sum -c ./*.sha256 \
      && echo "Unpacking archives" \
        && tar -C / -Jxpf s6-overlay-noarch.tar.xz \
        && tar -C / -Jxpf s6-overlay-x86_64.tar.xz \
      && echo "Removing downloaded archives" \
        && rm ./*.tar.xz \
        && rm ./*.sha256 \
    && echo "Cleaning up image" \
      && apt-get -y purge \
        curl \
        xz-utils \
      && apt-get -y autoremove --purge \
      && apt-get clean --yes \
      && rm -rf /var/lib/apt/lists/* \
      && rm -rf /tmp/* \
      && rm -rf /var/tmp/* \
      && rm -rf /var/cache/apt/archives/* \
      && truncate -s 0 /var/log/*log

#COPY ./rootfs /

FROM s6-base as runtime

WORKDIR /app

# Python dependencies
# Change pretty frequently
COPY --from=requirements /usr/src/pipenv/requirements.txt ./

# hadolint ignore=DL3042
RUN --mount=type=cache,target=/root/.cache/pip/,id=pip-cache \
    set -eux \
    && echo "Installing Python requirements" \
      && python3 -m pip install --default-timeout=1000 --requirement requirements.txt \
    && echo "Cleaning ml user" \
      && addgroup --gid 1000 machine-learner \
      && useradd --uid 1000 --gid machine-learner --home-dir /app machine-learner \
      && chown -R machine-learner:machine-learner /app

COPY ./rootfs /

EXPOSE 8000

ENTRYPOINT ["/init"]
