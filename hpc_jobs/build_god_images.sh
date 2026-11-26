#!/bin/bash
# Costruisce il proprio images_112.npz dalle immagini GOD 500x500 di Kamitani Lab
# e lo confronta con quello della demo.
# Uso (da /home/g.rinaldi-thesis/projects/b2b_jobs):  sbatch job_sbatch/build_god_images.sh
#SBATCH --job-name=build_god_img
#SBATCH --account=g.rinaldi-thesis
#SBATCH --partition=ulow
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=job_logs/build_god_images_%j.out

OUT_DIR=/scratch_share/islab/glauco.rinaldi/processed/GOD

source /opt/share/sw/amd/gcc-8.5.0/miniforge3-24.3.0-0/etc/profile.d/conda.sh
conda activate crcns
cd /home/g.rinaldi-thesis/projects/Brain-Transformation-with-No-Shared-Data

mkdir -p $OUT_DIR
python preproc/build_god_images.py \
    --images-dir /scratch_share/islab/glauco.rinaldi/datasets/GOD_Ho2022/images \
    --out $OUT_DIR/images_112.npz \
    --ref /scratch_share/islab/glauco.rinaldi/demo_data/Processed_data/GOD/images_112.npz
