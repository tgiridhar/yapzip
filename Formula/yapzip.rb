# Homebrew formula: this repository is its own tap.
#   brew tap tgiridhar/yapzip https://github.com/tgiridhar/yapzip
#   brew trust tgiridhar/yapzip
#   brew install yapzip            # builds from source (a few minutes), pulls in the model (yapzip-model)
# macOS builds use the Metal GPU; Linux builds use the CPU.
class Yapzip < Formula
  desc "Local text compressor: long articles, chats and threads into terse notes"
  homepage "https://github.com/tgiridhar/yapzip"
  url "https://github.com/tgiridhar/yapzip.git", tag: "v0.2.1"
  license "MIT"
  head "https://github.com/tgiridhar/yapzip.git", branch: "main"

  depends_on "cmake" => :build
  depends_on "tgiridhar/yapzip/yapzip-model"

  # Pinned: Qwen3.5 support and the llama.cpp API calls in native/yapzip.cpp match this release.
  resource "llama.cpp" do
    url "https://github.com/ggml-org/llama.cpp/archive/refs/tags/b11803.tar.gz"
    sha256 "fff546943c8ba3169c1c721a961e52595d4eac0e35b0c5ad768eb33ad846eba8"
  end

  def install
    (buildpath/"native/llama.cpp").install resource("llama.cpp")
    system "cmake", "-S", "native", "-B", "build",
           "-DYAPZIP_DEFAULT_MODEL=#{HOMEBREW_PREFIX}/share/yapzip/yapzip.gguf", *std_cmake_args
    system "cmake", "--build", "build", "--target", "yapzip", "-j"
    bin.install "build/yapzip"
    prefix.install "LICENSE", "NOTICE", "licenses"   # MIT (yapzip), Apache-2.0 (Qwen3.5-0.8B), MIT (llama.cpp)
  end

  test do
    out = pipe_output(bin/"yapzip", "the deploy is NOT going out Friday, it moves to Monday 9am\n")
    assert_match "NOT", out
  end
end
