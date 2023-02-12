#!/bin/bash -eux

download() {
  ID=${1?}
  OUT=${2?}

  if [ -e $OUT ]; then
    return
  fi

  CONFIRM=$( \
    wget \
      --quiet \
      --save-cookies /tmp/cookies.txt \
      --keep-session-cookies \
      --no-check-certificate \
      "https://drive.google.com/uc?export=download&id=$ID" \
      -O- | \
    sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1/p')
  wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$CONFIRM&id=$ID" -O $OUT
  rm -f /tmp/cookies.txt
}

download 1Ib96I4bfgT5Bb3TyWnge7XaMhaqjXX4M ./fugumt.zip
unzip fugumt.zip
rm fugumt.zip
mkdir -p ~/.cache/altrans
mv fugumt ~/.cache/altrans/fugumt
