#!/bin/bash
# Figura 6.a del paper: encoder GOD "studente" con e senza insegnante.
# Griglia: 5 soggetti GOD x 4 numeri di esempi x 3 condizioni x 3 ripetizioni = 180 esecuzioni.
# Uso (da /home/g.rinaldi-thesis/projects/b2b_jobs):
#   sbatch job_sbatch/fig6a_god_encoders.sh
#SBATCH --job-name=fig6a
#SBATCH --account=g.rinaldi-thesis
#SBATCH --partition=only-one-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-179%4
#SBATCH --output=job_logs/fig6a_%A_%a.out

SUBJECTS=(1 2 3 4 5)
NEXAMPLES=(300 600 900 1200)
CONDITIONS=(baseline god4 nsd1)
NREP=3

# Indice dell'array -> (soggetto, esempi, condizione, ripetizione)
i=$SLURM_ARRAY_TASK_ID
rep=$(( i % NREP + 1 ));            i=$(( i / NREP ))
cond=${CONDITIONS[$(( i % 3 ))]};   i=$(( i / 3 ))
n=${NEXAMPLES[$(( i % 4 ))]};       i=$(( i / 4 ))
s=${SUBJECTS[$i]}

# Argomenti di train_encoder.py:
#   gpu  studente  dati_studente  insegnante  dati_insegnante  ref  n_shared  n_nonshared
case $cond in
  baseline) ARGS="0 $s 0 1 1 1 $n 0" ;;   # ref=1: nessun insegnante
  god4)     ARGS="0 $s 0 4 0 0 $n 0" ;;   # insegnante GOD soggetto 4
  nsd1)     ARGS="0 $s 0 1 1 0 $n 0" ;;   # insegnante NSD soggetto 1
esac

# Il paper usa GOD sub-4 come insegnante: per lo studente 4 non ha senso, salto.
if [ "$cond" = "god4" ] && [ "$s" = "4" ]; then
  echo "skip: studente 4 con insegnante GOD 4"; exit 0
fi

source /opt/share/sw/amd/gcc-8.5.0/miniforge3-24.3.0-0/etc/profile.d/conda.sh
conda activate b2b_nvtf
cd /home/g.rinaldi-thesis/projects/Brain-Transformation-with-No-Shared-Data

# Il tag distingue condizione e ripetizione nei nomi dei pesi salvati: senza tag la baseline
# dello studente 4 con 1200 esempi sovrascriverebbe l'encoder insegnante GOD sub-4.
export B2B_RUN_TAG="fig6a_${cond}_rep${rep}"
export B2B_SEED=$rep

echo "studente=$s esempi=$n condizione=$cond rep=$rep seed=$B2B_SEED"
echo "python train_encoder.py $ARGS"
python train_encoder.py $ARGS
