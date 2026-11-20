"""
Valuta un encoder gia' addestrato (file di pesi .hdf5) sul test set del suo soggetto.

Ricostruisce l'encoder e carica i dati esattamente come train_encoder.py
(stesse funzioni: prepare_all_params e load_data_and_encoders), poi carica i
pesi indicati e stampa le correlazioni di Pearson per voxel tra fMRI predetta
e fMRI misurata (media, mediana, 75 e 90 percentile), come a fine addestramento.

Uso (dalla cartella del repository):
    python eval_encoder.py <pesi.hdf5> <gpu> <soggetto> <dataset> <insegnante> <dataset_insegnante> <exp_s> <exp_ns>
gli argomenti dopo <pesi.hdf5> sono gli stessi di train_encoder.py (senza 'ref').
Esempio, baseline degli autori per il soggetto 4 di GOD con 300 esempi:
    python eval_encoder.py data/GOD_encoders/encoder_weights_subj4_exp_300_ref.hdf5 "" 4 0 1 1 300 0
"""
import os
import sys
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

if __name__ == '__main__':
    weights_path = sys.argv[1]
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]   # "" = solo CPU
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if not os.path.isfile(weights_path):
        sys.exit('File di pesi non trovato: ' + weights_path)

    from Utils.image_loss import *
    from Utils.params.training_params_encoder import *
    from Utils.train_utils import *

    subject_1 = int(sys.argv[3])               # soggetto dell'encoder da valutare
    subject_student_1_data = int(sys.argv[4])  # 0 = GOD, 1 = NSD
    teacher_subject = int(sys.argv[5])         # servono solo per caricare i dati come in addestramento
    teacher_subject_data = int(sys.argv[6])
    exp_array_s = [int(sys.argv[7])]
    exp_array_ns = [int(sys.argv[8])]
    ref = 1  # con ref=1 load_data_and_encoders non carica pesi: li carichiamo noi sotto

    Voxel_to_choose,num_n_1,num_n_2,l2_reg_dct_1,l2_reg_dct_2,exp_array_s,exp_array_ns,exp_array_s_2,exp_array_ns_2,exp_teacher_s,exp_teacher_ns,\
            scale_1, scale_2,param_loss_weights,extra_save,exp_s,exp_ns,exp_t_s,exp_t_ns,exp_t,exp_s_2,exp_ns_2,l2_reg_1,l2_reg_2,extra_teacher,extra_teacher_1,extra_teacher_2,ablation_text,\
            encoder_weights_1,extra_subj_1,encoder_weights_2,extra_subj_2,transf_weights_1_to_2,\
    transf_weights_2_to_1  = prepare_all_params(encoder_transf_run,share_non_share_param,exp_array_s,exp_array_ns,exp_array_s_2,exp_array_ns_2,\
                                                exp_teacher_s_ns, subject_student_1_data,subject_student_2_data,teacher_subject_data,ref,subject_1,subject_1,teacher_subject)

    print("loading data and building encoder...")
    train_FMRI_1_s,train_FMRI_2_s,NUM_VOXELS_1,NUM_VOXELS_2,encoder_model_1,encoder_model_2,train_images_1, train_FMRI_1, train_images_2,\
    train_FMRI_2,train_images_1_s, train_FMRI_1_s, train_images_2_s, train_FMRI_2_s,test_images_1, test_FMRI_1, test_images_2, test_FMRI_2,\
    test_images_1_s, test_FMRI_1_s, test_images_2_s, test_FMRI_2_s = load_data_and_encoders(encoder_transf_run,subject_student_1_data,subject_student_2_data,teacher_subject_data,\
                          subject_1,subject_1,teacher_subject,get_train_avg,exp_s,exp_s_2,exp_t_s,Voxel_to_choose,scale_1,scale_2,\
                          vgg_loss,encoder_weights_1,encoder_weights_2,ref,exp_t,exp_ns,exp_ns_2)

    encoder_model_1.load_weights(weights_path)
    print('weights: ' + weights_path)
    print('test set: %d immagini, %d voxel' % (test_images_1_s.shape[0], test_FMRI_1_s.shape[1]))
    test_model_and_print(encoder_model_1, test_images_1_s, test_FMRI_1_s, bacth_size=16, num_batches=30)
