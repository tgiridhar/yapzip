#!/bin/sh
# Append a GGUF to a yapzip binary (Linux/ELF). Usage: embed.sh BINARY MODEL.gguf
# Layout: [binary][zero padding to 32 bytes][GGUF]["YZGGUF01"][GGUF offset, uint64 little endian]
set -e
bin="$1"; model="$2"
size=$(wc -c < "$bin")
pad=$(( (32 - size % 32) % 32 ))
[ "$pad" -gt 0 ] && head -c "$pad" /dev/zero >> "$bin"
off=$(( size + pad ))
cat "$model" >> "$bin"
printf 'YZGGUF01' >> "$bin"
i=0
while [ $i -lt 8 ]; do
  printf "\\$(printf %03o $(( (off >> (8 * i)) & 255 )))" >> "$bin"
  i=$((i + 1))
done
echo "embedded $(basename "$model") at offset $off in $bin"
