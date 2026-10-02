#!/bin/bash
# Submit submission_NAME.csv files to Kaggle, each with its settings (from submissions_log.tsv) as the description,
# then print the latest scores:   bash ksub.sh s1 x15 ...
cd "$(dirname "$0")"; export PATH=$HOME/.local/bin:$PATH
for n in "$@"; do
  f=submission_$n.csv
  kaggle competitions submit -c testingpj-2 -f "$f" -m "$n | $(awk -F'\t' -v f="$f" '$2 == f {s = $3} END {print s}' submissions_log.tsv 2>/dev/null)"
done
sleep 120; kaggle competitions submissions -c testingpj-2 | head -$(($# + 2))
