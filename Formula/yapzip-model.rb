# The yapzip model (fine-tuned Qwen3.5-0.8B, Q8_0 GGUF, ~775 MB), installed as share/yapzip/yapzip.gguf.
# A separate formula so code updates to yapzip do not re-download the model.
# Apache-2.0, like the Qwen3.5-0.8B weights it modifies; NOTICE says what was changed.
class YapzipModel < Formula
  desc "Model weights for yapzip (fine-tuned Qwen3.5-0.8B, Q8_0 GGUF)"
  homepage "https://github.com/tgiridhar/yapzip"
  url "https://github.com/tgiridhar/yapzip/releases/download/model-v4c/yapzip-v4c-q8_0.gguf"
  version "4c"
  sha256 "9c8e1114fe1923cfbaf661add0b0b3ec1b447d75c1188f9c19cf79b6ea557cbd"
  license "Apache-2.0"

  resource "license" do
    url "https://github.com/tgiridhar/yapzip/releases/download/model-v4c/LICENSE"
    sha256 "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a"
  end

  resource "notice" do
    url "https://github.com/tgiridhar/yapzip/releases/download/model-v4c/NOTICE"
    sha256 "f073835470eaec343f3b2a9a0f46035efedbe672dd62718090f76b5411da4054"
  end

  def install
    (share/"yapzip").install Dir["*.gguf"].first => "yapzip.gguf"
    %w[license notice].each do |r|
      resource(r).stage { (share/"yapzip").install Dir["*"].first => r.upcase }
    end
  end

  test do
    assert_path_exists share/"yapzip/yapzip.gguf"
    assert_path_exists share/"yapzip/LICENSE"
  end
end
