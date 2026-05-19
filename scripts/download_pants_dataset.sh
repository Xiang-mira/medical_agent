#!/usr/bin/env bash
set -euo pipefail
# Official PanTS/PanTSMini download helper. Needs ~300GB+ storage.
OUT_DIR=${1:-third_party/PanTS-main/data}
mkdir -p "$OUT_DIR"
cd "$OUT_DIR"
echo "Downloading PanTS/PanTSMini into $(pwd). This is large (~300GB+)."
wget -O metadata.xlsx "https://huggingface.co/datasets/BodyMaps/PanTSMini/resolve/main/metadata.xlsx?download=true"
mkdir -p ImageTr ImageTe
for i in {1..9}; do
  start=$(printf "%08d" $(( (i - 1) * 1000 + 1 )))
  end=$(printf "%08d" $(( i * 1000 )))
  file="PanTSMini_ImageTr_${start}_${end}.tar.gz"
  url="https://huggingface.co/datasets/BodyMaps/PanTSMini/resolve/main/${file}?download=true"
  echo "[$i/9] Downloading $file"
  wget --show-progress -O "$file" "$url"
  tar -xzf "$file" -C ImageTr
  rm "$file"
done
file="PanTSMini_ImageTe_00009001_00009901.tar.gz"
wget --show-progress -O "$file" "https://huggingface.co/datasets/BodyMaps/PanTSMini/resolve/main/$file?download=true"
tar -xzf "$file" -C ImageTe
rm "$file"
wget --show-progress -O PanTSMini_Label.tar.gz "http://www.cs.jhu.edu/~zongwei/dataset/PanTSMini_Label.tar.gz"
mkdir -p LabelAll LabelTr LabelTe
tar -xzf PanTSMini_Label.tar.gz -C LabelAll
rm PanTSMini_Label.tar.gz
find LabelAll -maxdepth 1 -type d \( -name 'PanTS_0000[0-8]*' -o -name 'PanTS_00009000' \) -print0 | xargs -0 -r mv -t LabelTr/
find LabelAll -maxdepth 1 -type d -name 'PanTS_00009*' -print0 | xargs -0 -r mv -t LabelTe/
rmdir LabelAll 2>/dev/null || true
echo "Done. Run: python run_medai_cli.py --json pants-check --pants-root third_party/PanTS-main"
