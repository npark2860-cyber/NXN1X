#!/usr/bin/env bash
set -eu
export PATH=/clangarm64/bin:/usr/bin:$PATH
cd "$(dirname "$0")/.."
root=$(pwd)
mkdir -p build
prefix="$root/build/ffmpeg-dav1d-arm64"
# The release build is offline once these three pinned source archives are cached.
fetch() {
    local archive="$root/build/$1" url="$2" hash="$3"
    if [[ ! -f "$archive" ]]; then
        curl -fLsS "$url" -o "$archive.part"
        mv "$archive.part" "$archive"
    fi
    echo "$hash  $archive" | sha256sum -c -
}
fetch dav1d-1.5.2.tar.gz https://codeload.github.com/videolan/dav1d/tar.gz/refs/tags/1.5.2 2fc0810b4cdf72784b3c107827ff10b1d83ec709a1ec1fbdbc6a932daf65ead6
fetch ffmpeg-av1-pinned.tar.gz https://codeload.github.com/FFmpeg/FFmpeg/tar.gz/c7b5f1537d9c52efa50fd10d106ca015ddde1818 33cf0059e7fa45251a7834bdcb66d4e409f5be490aa1d91f2d2aa3561f7bf9fd
fetch ffmpeg-amf.tar.gz https://codeload.github.com/GPUOpen-LibrariesAndSDKs/AMF/tar.gz/refs/tags/v1.4.36 240a42033babc7920e5476506d5ac0c5628f67908833168e746406808d0ef146
tar xf build/dav1d-1.5.2.tar.gz -C build
tar xf build/ffmpeg-av1-pinned.tar.gz -C build
mkdir -p "$prefix/include/AMF"
tar xf build/ffmpeg-amf.tar.gz -C "$prefix/include/AMF" --strip-components=4 AMF-1.4.36/amf/public/include
meson setup build/dav1d-arm64 build/dav1d-1.5.2 --buildtype=release --default-library=static \
    --prefix="$(cygpath -m "$prefix")" -Denable_tools=false -Denable_tests=false
ninja -C build/dav1d-arm64 install
export PKG_CONFIG_PATH="$(cygpath -m "$prefix/lib/pkgconfig")"
mkdir -p build/ffmpeg-av1-pinned-objects
cd build/ffmpeg-av1-pinned-objects
src="../FFmpeg-c7b5f1537d9c52efa50fd10d106ca015ddde1818"
vulkan_headers="$root/tools/cpm/vulkanheaders/v1.4.345/include"
"$src/configure" \
    --prefix="$(cygpath -m "$prefix")" \
    --cc=clang --cxx=clang++ --ar=llvm-ar --ranlib=llvm-ranlib \
    --target-os=mingw32 --arch=aarch64 --enable-static --disable-shared \
    --disable-avdevice --disable-avformat --disable-doc --disable-everything \
    --disable-ffmpeg --disable-ffprobe --disable-network --disable-swresample \
    --enable-decoder=h264,vp8,vp9,libdav1d,h264_amf,vp9_amf --enable-libdav1d \
    --enable-filter=yadif,scale --enable-small --enable-pic --enable-vulkan \
    --enable-dxva2 --enable-d3d11va --enable-d3d12va --enable-amf \
    --enable-hwaccel=h264_vulkan,vp9_vulkan,h264_dxva2,vp9_dxva2,h264_d3d11va,h264_d3d11va2,vp9_d3d11va,vp9_d3d11va2,h264_d3d12va,vp9_d3d12va \
    --extra-cflags="-I$(cygpath -m "$vulkan_headers") -I$(cygpath -m "$prefix/include")" \
    > ../ffmpeg-av1-config.log 2>&1
make -j"$(nproc)" install > ../ffmpeg-av1-build.log 2>&1
touch "$prefix/.av1-installed"
