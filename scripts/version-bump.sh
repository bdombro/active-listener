#!/usr/bin/env bash
# Bump active-listener semver in Cargo.toml (major | minor | patch), refresh Cargo.lock,
# commit the bump, and create an annotated tag vX.Y.Z (same form as scripts/release.sh).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="${HOME}/.cargo/bin:${PATH}"

KIND="${1:-}"
case "${KIND}" in
  major | minor | patch) ;;
  *)
    echo "Usage: $0 major|minor|patch" >&2
    exit 1
    ;;
esac

CURRENT=$(cargo metadata --no-deps --format-version 1 | python3 -c \
  "import sys, json; d=json.load(sys.stdin); print(d['packages'][0]['version'])")

NEW=$(CURRENT="$CURRENT" KIND="$KIND" python3 -c "
import os
m, mi, p = map(int, os.environ['CURRENT'].split('.'))
k = os.environ['KIND']
if k == 'major':
    print(f'{m + 1}.0.0')
elif k == 'minor':
    print(f'{m}.{mi + 1}.0')
else:
    print(f'{m}.{mi}.{p + 1}')
")

CURRENT="$CURRENT" NEW="$NEW" python3 <<'PY'
from pathlib import Path
import os

current = os.environ["CURRENT"]
new = os.environ["NEW"]
path = Path("Cargo.toml")
lines = path.read_text().splitlines(keepends=True)
old_line = f'version = "{current}"'
new_line = f'version = "{new}"'
replaced = 0
out = []
for line in lines:
    if line.startswith('version = "') and line.strip() == old_line:
        nl = "\n" if line.endswith("\n") else ""
        out.append(new_line + nl)
        replaced += 1
    else:
        out.append(line)
if replaced != 1:
    raise SystemExit(f"expected exactly one line {old_line!r}, matched {replaced}")
path.write_text("".join(out))
PY

cargo check -q

git add Cargo.toml Cargo.lock
git commit -m "chore: bump version to ${NEW}"

echo "Version ${CURRENT} -> ${NEW}"
