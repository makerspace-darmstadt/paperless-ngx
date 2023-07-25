
# syntax=docker/dockerfile:1
FROM python:3.11-alpine as base

ENV \
    LANG="C.UTF-8" \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_FIND_LINKS="https://wheels.home-assistant.io/musllinux/ https://wheel-index.linuxserver.io/alpine/" \
    PIP_PREFER_BINARY=1 \
    PYTHONDONTWRITEBYTECODE=1

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

# hadolint ignore=DL3018
RUN set -eux \
    && echo "Installing s6 build time packages" \
      && apk add --no-cache \
        curl \
        outils-sha256 \
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
      apk del --no-cache --purge \
        curl \
        outils-sha256

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
      && addgroup --gid 1000 paperless-ml \
      && adduser --home /app/ --disabled-password --uid 1000 --shell /sbin/nologin --ingroup paperless-ml paperless-ml  \
      && chown -R paperless-ml:paperless-ml  /app

COPY ./rootfs /

EXPOSE 8000

ENTRYPOINT ["/init"]
