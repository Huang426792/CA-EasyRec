$ErrorActionPreference = "Stop"

python -m ca_easyrec.demo `
  --output artifacts/demo `
  --seed 2026 `
  --teacher-epochs 20 `
  --text-epochs 20
