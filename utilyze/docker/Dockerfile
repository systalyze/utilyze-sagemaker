# syntax=docker/dockerfile:1.7
# A few notes here:
# - builds on rockylinux:8 for glibc 2.28+ compat
# - uses gcc-12 for better C++17 support than rockylinux:8 default gcc-8
# - NVIDIA provides 64-bit desktop ARM support under sbsa not aarch64/arm64

ARG CUDA_VERSION=13.1
ARG CUDA_PKG_SUFFIX=13-1
ARG GCC_TOOLSET=gcc-toolset-12

FROM rockylinux:8 AS builder

ARG TARGETARCH
ARG CUDA_VERSION
ARG CUDA_PKG_SUFFIX
ARG GCC_TOOLSET

ENV PATH=/opt/rh/${GCC_TOOLSET}/root/usr/bin:/usr/local/cuda-${CUDA_VERSION}/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/rh/${GCC_TOOLSET}/root/usr/lib64

RUN set -eux; \
    case "$TARGETARCH" in \
      amd64) nv_arch=x86_64 ;; \
      arm64) nv_arch=sbsa ;; \
      *) echo "unsupported TARGETARCH: $TARGETARCH" >&2; exit 1 ;; \
    esac; \
    dnf -y install 'dnf-command(config-manager)'; \
    dnf config-manager --add-repo "https://developer.download.nvidia.com/compute/cuda/repos/rhel8/${nv_arch}/cuda-rhel8.repo"; \
    dnf -y install \
      "${GCC_TOOLSET}-gcc" \
      "${GCC_TOOLSET}-gcc-c++" \
      "${GCC_TOOLSET}-binutils" \
      gzip \
      make \
      tar \
      "cuda-driver-devel-${CUDA_PKG_SUFFIX}" \
      "cuda-cudart-devel-${CUDA_PKG_SUFFIX}" \
      "cuda-cupti-${CUDA_PKG_SUFFIX}"; \
    dnf clean all; \
    rm -rf /var/cache/dnf

WORKDIR /src
COPY . .

RUN set -eux; \
    make clean; \
    make -j"$(nproc)" \
      CUDA_VERSION="${CUDA_VERSION}" \
      dist-tarball

FROM scratch AS export
COPY --from=builder /src/dist/libutlz_sampler-*.tar.gz /
