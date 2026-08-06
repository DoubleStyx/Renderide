#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "${repo_root}"
export DOTNET_CLI_HOME="${DOTNET_CLI_HOME:-${repo_root}/.dotnet}"
export DOTNET_CLI_TELEMETRY_OPTOUT="${DOTNET_CLI_TELEMETRY_OPTOUT:-1}"
export DOTNET_NOLOGO="${DOTNET_NOLOGO:-1}"
export DOTNET_SKIP_FIRST_TIME_EXPERIENCE="${DOTNET_SKIP_FIRST_TIME_EXPERIENCE:-1}"
export NUGET_PACKAGES="${NUGET_PACKAGES:-${repo_root}/.nuget/packages}"

require_tool() {
  local tool="$1"
  if ! command -v "${tool}" >/dev/null 2>&1; then
    printf 'error: required tool not found: %s\n' "${tool}" >&2
    exit 127
  fi
}

run() {
  printf '\n==> '
  printf '%q ' "$@"
  printf '\n'
  "$@"
}

is_linux() {
  [[ "$(uname -s)" == Linux* ]]
}

require_tool cargo
require_tool taplo
require_tool dotnet

run cargo fmt --all -- --check
run taplo fmt --check --diff

clippy_features=()
if is_linux; then
  clippy_features=(--all-features)
fi

run cargo clippy --all-targets --locked "${clippy_features[@]}" -- -W clippy::all -D warnings

if ! is_linux; then
  run cargo check --locked -p renderide --features tracy
  run cargo check --locked -p renderide --features tracy-gpu
fi

run cargo build --locked -p renderide -p renderide-test -p bootstrapper
run cargo test --workspace --locked

run dotnet restore Generators.sln --locked-mode

if is_linux; then
  run dotnet format Generators.sln --verify-no-changes --no-restore
fi

run dotnet build generators/SharedTypeGenerator.Tests/SharedTypeGenerator.Tests.csproj --configuration Release --no-restore -warnaserror
run dotnet test generators/SharedTypeGenerator.Tests/SharedTypeGenerator.Tests.csproj --configuration Release --no-build --no-restore
