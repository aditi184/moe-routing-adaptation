# Cluster-specific env: edit for your setup, or set CUDA_MODULE / PYTHON_VENV
# in your shell before invoking.
module load "${CUDA_MODULE:-cuda/12.2}" 2>/dev/null || true
source "${PYTHON_VENV:-../.venv}/bin/activate"
module load httpproxy 2>/dev/null || true

# Check if we're in a SLURM allocation
if [ -z "$SLURM_JOB_ID" ]; then
    echo "Error: Not running in a SLURM allocation."
    echo "Please make sure you're in an active 'salloc' session."
    exit 1
fi

# Get node list - try multiple env vars / methods and clean up.
# On your cluster, SLURM_STEP_NODELIST / SLURM_NODELIST are populated (e.g. tg[10802,10804-10806]).

# Start from the job-level nodelist if present
if [ -z "$SLURM_JOB_NODELIST" ] || [ "$SLURM_JOB_NODELIST" == "(null)" ]; then
    # Prefer step-level nodelist when running under srun
    if [ -n "$SLURM_STEP_NODELIST" ] && [ "$SLURM_STEP_NODELIST" != "(null)" ]; then
        SLURM_JOB_NODELIST="$SLURM_STEP_NODELIST"
    # Fall back to generic SLURM_NODELIST
    elif [ -n "$SLURM_NODELIST" ] && [ "$SLURM_NODELIST" != "(null)" ]; then
        SLURM_JOB_NODELIST="$SLURM_NODELIST"
    else
        # As a last resort, query scontrol / squeue
        RAW_NODELIST=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | grep -i "NodeList" | awk -F'=' '{print $2}' | awk '{print $1}')
        if [ -z "$RAW_NODELIST" ] || [ "$RAW_NODELIST" == "(null)" ]; then
            RAW_NODELIST=$(squeue -j "$SLURM_JOB_ID" -h -o "%N" 2>/dev/null)
        fi
        SLURM_JOB_NODELIST="$RAW_NODELIST"
    fi
fi

# Clean up the node list
SLURM_JOB_NODELIST=$(echo "$SLURM_JOB_NODELIST" | tr '\n' ' ' | sed 's/(null)//g' | sed 's/^[[:space:]]*//' | sed 's/[[:space:]]*$//')
# Keep bracketed ranges like tg[10802,10804-10806] intact; just trim/clean, do not over-parse here.

# Parse node list using scontrol to correctly expand ranges like tg[10802,10804-10806]
HOSTNAMES=()
if [ -n "$SLURM_JOB_NODELIST" ] && [ "$SLURM_JOB_NODELIST" != "(null)" ]; then
    # scontrol show hostnames expands any bracket/range syntax into one hostname per line
    mapfile -t HOSTNAMES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
fi

# Final clean-up of hostnames
if [ ${#HOSTNAMES[@]} -gt 0 ]; then
    CLEAN_HOSTNAMES=()
    for host in "${HOSTNAMES[@]}"; do
        host=$(echo "$host" | sed 's/(null)//g' | sed 's/^[[:space:]]*//' | sed 's/[[:space:]]*$//')
        if [ -n "$host" ] && [ "$host" != "(null)" ]; then
            CLEAN_HOSTNAMES+=("$host")
        fi
    done
    HOSTNAMES=("${CLEAN_HOSTNAMES[@]}")
fi

# Get master node
if [ ${#HOSTNAMES[@]} -gt 0 ]; then
    MASTER_NODE="${HOSTNAMES[0]}"
    MASTER_NODE=$(echo "$MASTER_NODE" | sed 's/(null)//g' | sed 's/^[[:space:]]*//' | sed 's/[[:space:]]*$//')
else
    echo "Error: Could not parse node list from: '$SLURM_JOB_NODELIST'"
    echo "  SLURM_JOB_NODELIST: '$SLURM_JOB_NODELIST'"
    echo "  SLURM_STEP_NODELIST: '$SLURM_STEP_NODELIST'"
    echo "  SLURM_NODELIST: '$SLURM_NODELIST'"
    exit 1
fi
MASTER_ADDR=$MASTER_NODE

# Get node rank
NODE_RANK=0
CURRENT_HOST=$(hostname -s)
for i in "${!HOSTNAMES[@]}"; do
    HOST_SHORT=$(echo ${HOSTNAMES[$i]} | cut -d'.' -f1 | sed 's/(null)//g')
    if [ "$HOST_SHORT" == "$CURRENT_HOST" ]; then
        NODE_RANK=$i
        break
    fi
done

if [ -z "$SLURM_JOB_NUM_NODES" ]; then
    SLURM_JOB_NUM_NODES=${#HOSTNAMES[@]}
fi
echo "=========================================="
echo "Multi-node training setup:"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Raw node list: $SLURM_JOB_NODELIST"
echo "  Parsed hostnames: ${HOSTNAMES[@]}"
echo "  Total nodes: $SLURM_JOB_NUM_NODES"
echo "  Current node: $CURRENT_HOST (rank $NODE_RANK)"
echo "  Master address: $MASTER_ADDR"
echo "  GPUs per node: 4"
echo "=========================================="

if [ -z "$MASTER_ADDR" ] || [ "$MASTER_ADDR" == "(null)" ] || [[ "$MASTER_ADDR" == *"(null)"* ]] || [ "$SLURM_JOB_NUM_NODES" -eq 0 ]; then
    echo "Error: Failed to determine master address or number of nodes"
    echo "  MASTER_ADDR: '$MASTER_ADDR'"
    echo "  SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
    exit 1
fi

# torchrun \
#     --nnodes=$SLURM_JOB_NUM_NODES \
#     --node_rank=$NODE_RANK \
#     --nproc_per_node=4 \
#     --master_addr=$MASTER_ADDR \
#     --master_port=29500 \
#     train.py configs/seven_langs_ukrainian.yml


# torchrun \
#     --nnodes=$SLURM_JOB_NUM_NODES \
#     --node_rank=$NODE_RANK \
#     --nproc_per_node=4 \
#     --master_addr=$MASTER_ADDR \
#     --master_port=29500 \
#     train.py configs/seven_langs_slovak_cont.yml

# torchrun \
#     --nnodes=$SLURM_JOB_NUM_NODES \
#     --node_rank=$NODE_RANK \
#     --nproc_per_node=4 \
#     --master_addr=$MASTER_ADDR \
#     --master_port=29500 \
#     train_unsharded.py configs/seven_langs_catalan_cont.yml


# torchrun \
#     --nnodes=$SLURM_JOB_NUM_NODES \
#     --node_rank=$NODE_RANK \
#     --nproc_per_node=4 \
#     --master_addr=$MASTER_ADDR \
#     --master_port=29500 \
#     train_unsharded.py configs/seven_langs_estonian_cont.yml

torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=4 \
    --master_addr=$MASTER_ADDR \
    --master_port=29500 \
    train_unsharded.py configs/seven_langs_ukrainian_cont.yml

# torchrun \
#     --nnodes=$SLURM_JOB_NUM_NODES \
#     --node_rank=$NODE_RANK \
#     --nproc_per_node=4 \
#     --master_addr=$MASTER_ADDR \
#     --master_port=29500 \
#     train_unsharded.py configs/seven_langs_urdu_cont.yml

#  torchrun \
#      --nnodes=$SLURM_JOB_NUM_NODES \
#      --node_rank=$NODE_RANK \
#      --nproc_per_node=4 \
#      --master_addr=$MASTER_ADDR \
#      --master_port=29500 \
#      train.py configs/seven_langs_dutch.yml



# torchrun \
#     --nnodes=$SLURM_JOB_NUM_NODES \
#     --node_rank=$NODE_RANK \
#     --nproc_per_node=4 \
#     --master_addr=$MASTER_ADDR \
#     --master_port=29500 \
#     train.py configs/seven_langs_slovak.yml

# torchrun \
#     --nnodes=$SLURM_JOB_NUM_NODES \
#     --node_rank=$NODE_RANK \
#     --nproc_per_node=4 \
#     --master_addr=$MASTER_ADDR \
#     --master_port=29500 \
#     train.py configs/seven_langs_urdu.yml




# torchrun \
#     --nnodes=$SLURM_JOB_NUM_NODES \
#     --node_rank=$NODE_RANK \
#     --nproc_per_node=4 \
#     --master_addr=$MASTER_ADDR \
#     --master_port=29500 \
#     train.py configs/seven_langs_catalan.yml

# torchrun \
#     --nnodes=$SLURM_JOB_NUM_NODES \
#     --node_rank=$NODE_RANK \
#     --nproc_per_node=4 \
#     --master_addr=$MASTER_ADDR \
#     --master_port=29500 \
#     train.py configs/seven_langs_estonian.yml
