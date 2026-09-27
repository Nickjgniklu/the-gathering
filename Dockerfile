# syntax=docker/dockerfile:1

ARG ELIXIR_VERSION=1.20.4
# OTP 29.1 fixes alternate signal stack sizing on Alpine/AMX hosts (OTP-20292).
ARG OTP_VERSION=29.1
ARG ALPINE_VERSION=3.24
ARG NODE_VERSION=26.9.0
ARG AUBE_VERSION=1.21.0

# Hex images pin the OTP patch release as well as Elixir. Keep the builder's
# Alpine minor version aligned with the runner for native release dependencies.
ARG BUILDER_IMAGE=hexpm/elixir:${ELIXIR_VERSION}-erlang-${OTP_VERSION}-alpine-${ALPINE_VERSION}.1
ARG RUNNER_IMAGE=alpine:${ALPINE_VERSION}

FROM node:${NODE_VERSION}-alpine${ALPINE_VERSION} AS node-runtime

FROM ${BUILDER_IMAGE} AS builder

ARG AUBE_VERSION
COPY --from=node-runtime /usr/local /usr/local

RUN apk add --no-cache build-base git curl ca-certificates tar

WORKDIR /app

# aube is the JavaScript package manager used by the repo (pnpm-compatible).
# Pin the release and verify its checksum. Update AUBE_VERSION and both
# checksums together (musl builds, since the builder is Alpine).
ARG TARGETARCH
ARG AUBE_SHA256_AMD64=6761c69514475a87b375d02a3782ebbab1dfdf181a584ee6b6c91814a882cb37
ARG AUBE_SHA256_ARM64=07da9245c5ac2ef540ed59f795aeee6986d8a9ed5127d4d9c97cbfd1cc05c308
RUN set -eu; \
  arch="${TARGETARCH:-$(uname -m)}"; \
  case "$arch" in \
    amd64|x86_64) aube_arch=x86_64; aube_sha="$AUBE_SHA256_AMD64" ;; \
    arm64|aarch64) aube_arch=aarch64; aube_sha="$AUBE_SHA256_ARM64" ;; \
    *) echo "unsupported build arch: ${arch}" >&2; exit 1 ;; \
  esac; \
  curl -fsSL "https://github.com/aubepkg/aube/releases/download/v${AUBE_VERSION}/aube-v${AUBE_VERSION}-${aube_arch}-unknown-linux-musl.tar.gz" -o /tmp/aube.tar.gz; \
  echo "${aube_sha}  /tmp/aube.tar.gz" | sha256sum -c -; \
  tar -xzf /tmp/aube.tar.gz -C /tmp aube; \
  install /tmp/aube /usr/local/bin/aube; \
  rm -f /tmp/aube /tmp/aube.tar.gz; \
  aube --version

RUN mix local.hex --force && mix local.rebar --force

ENV MIX_ENV=prod

COPY mix.exs mix.lock ./
RUN mix deps.get --only $MIX_ENV
RUN mkdir config

COPY config/config.exs config/${MIX_ENV}.exs config/
RUN mix deps.compile

COPY package.json aube-lock.yaml ./
RUN aube install --frozen-lockfile

COPY priv priv
COPY lib lib
COPY vite.config.ts tsconfig.json ./
COPY assets assets

RUN mix compile
RUN mix assets.deploy

COPY config/runtime.exs config/
RUN mix release

FROM ${RUNNER_IMAGE} AS runner

RUN apk upgrade --no-cache \
  && apk add --no-cache libstdc++ openssl ncurses-libs ca-certificates lksctp-tools su-exec rsvg-convert font-dejavu

ENV LANG=C.UTF-8
ENV LANGUAGE=C.UTF-8
ENV LC_ALL=C.UTF-8

WORKDIR /app
RUN addgroup -S app && adduser -S -G app -h /home/app -s /bin/sh app \
  && mkdir -p /data && chown -R app:app /app /data

ENV MIX_ENV=prod
ENV PHX_SERVER=true
ENV PORT=4000
ENV DATA_DIR=/data

COPY --from=builder --chown=app:app /app/_build/prod/rel/the_gathering ./
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod 755 /usr/local/bin/docker-entrypoint.sh

EXPOSE 4000
VOLUME ["/data"]
# BusyBox wget ships with Alpine, so no extra healthcheck binary is needed.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD wget -qO /dev/null "http://127.0.0.1:${PORT}/api/health" || exit 1
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["/app/bin/the_gathering", "start"]
