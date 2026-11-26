#!/bin/bash
# Confronto funzionale: encoder GOD sub-4 (300 esempi) addestrato con le immagini della demo
# o con le proprie (ricostruite dalle 500x500 di Kamitani). Tutto il resto e' identico.
# Griglia: 2 sorgenti immagini x 2 condizioni (baseline, insegnante NSD subj1) x 3 ripetizioni = 12.
# Uso (da /home/g.rinaldi-thesis/projects/b2b_jobs):  sbatch job_sbatch/imgcmp_god_encoder.sh
#SBATCH --job-name=imgcmp
#SBATCH --account=g.rinaldi-thesis
#SBATCH --partition=only-one-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-11%4
#SBATCH --output=job_logs/imgcmp_%A_%a.out

SOURCES=(demo mine)
CONDITIONS=(baseline nsd1)
NREP=3
declare -A NPZ=(
  [demo]=/scratch_share/islab/glauco.rinaldi/demo_data/Processed_data/GOD/images_112.npz
  [mine]=/scratch_share/islab/glauco.rinaldi/processed/GOD/images_112.npz
)

# Indice dell'array -> (sorgente, condizione, ripetizione)
i=$SLURM_ARRAY_TASK_ID
rep=$(( i % NREP + 1 ));          i=$(( i / NREP ))
cond=${CONDITIONS[$(( i % 2 ))]}; i=$(( i / 2 ))
src=${SOURCES[$i]}

# Argomenti di train_encoder.py: gpu studente dati_studente insegnante dati_insegnante ref n_shared n_nonshared
case $cond in
  baseline) ARGS="0 4 0 1 1 1 300 0" ;;   # ref=1: nessun insegnante
  nsd1)     ARGS="0 4 0 1 1 0 300 0" ;;   # insegnante NSD soggetto 1 (come la demo)
esac

source /opt/share/sw/amd/gcc-8.5.0/miniforge3-24.3.0-0/etc/profile.d/conda.sh
conda activate b2b_nvtf
cd /home/g.rinaldi-thesis/projects/Brain-Transformation-with-No-Shared-Data

export B2B_GOD_IMAGES=${NPZ[$src]}
export B2B_RUN_TAG="imgcmp_${src}_${cond}_rep${rep}"
export B2B_SEED=$rep        # stesso seed per demo e mine a parita' di ripetizione

[ -f "$B2B_GOD_IMAGES" ] || { echo "file immagini mancante: $B2B_GOD_IMAGES"; exit 1; }
echo "RUN src=$src cond=$cond rep=$rep seed=$B2B_SEED"
echo "python train_encoder.py $ARGS"
python train_encoder.py $ARGS
