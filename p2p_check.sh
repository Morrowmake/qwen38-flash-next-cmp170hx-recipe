# Resolve peer access before starting either runtime's server.
# After a pass (or P2P=force) the serving defaults' peer settings apply;
# an explicit VLLM_CUSTOM_ALLREDUCE_PCIE_P2P=0 or NCCL_P2P_LEVEL is preserved.
P2P=${P2P:-auto}
case "$P2P" in
  off|auto|force) ;;
  *) echo 'serve.sh: P2P must be off, auto or force' >&2; exit 2 ;;
esac
if [ "${DRY:-0}" = 1 ]; then
  # A dry run never initializes CUDA and cannot claim a content-check pass.
  if [ "$P2P" = force ] && { [ "$TP" -gt 1 ] || [ "$PP" -gt 1 ]; }; then
    P2P_ENABLED=1
    echo '[p2p] P2P enabled: P2P=force; content check bypassed (dry run)'
  else
    P2P_ENABLED=0
    echo "[p2p] P2P disabled for dry run: mode=$P2P; auto checks before a real multi-GPU start"
  fi
else
  P2P_ENABLED="$("$VENV/bin/python" "$REPO_ROOT/p2p_check.py" \
    --mode "$P2P" --tp "$TP" --pp "$PP")" || {
      P2P_ENABLED=0
      echo '[p2p] P2P disabled: check unavailable' >&2
    }
fi
if [ "$P2P_ENABLED" = 1 ]; then
  export VLLM_CUSTOM_ALLREDUCE_PCIE_P2P=${VLLM_CUSTOM_ALLREDUCE_PCIE_P2P:-$QWEN_DEFAULT_VLLM_CUSTOM_ALLREDUCE_PCIE_P2P}
  unset NCCL_P2P_DISABLE
  export NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-$QWEN_DEFAULT_NCCL_P2P_LEVEL}
else
  export VLLM_CUSTOM_ALLREDUCE_PCIE_P2P=0
  unset NCCL_P2P_LEVEL
  export NCCL_P2P_DISABLE=1
fi
