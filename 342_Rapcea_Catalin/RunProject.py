import cv2
from numpy.ma.core import shape
from skimage.draw import rectangle

from Parameters import *
from FacialDetector import *
import pdb
from Visualize import *
import time



params: Parameters = Parameters()
params.dim_window = 64
params.dim_hog_cell = 8  # dimensiunea celulei
params.cells_per_block = 4 # 4
params.overlap = 0.03
params.number_positive_examples = 10000  # numarul exemplelor pozitive
params.number_negative_examples = 8000  # numarul exemplelor negative

params.threshold = 0.8 # toate ferestrele cu scorul > threshold si maxime locale devin detectii
params.has_annotations = True

params.use_hard_mining = False  # (optional)antrenare cu exemple puternic negative
params.use_flip_images = True  # adauga imaginile cu fete oglindite

if params.use_flip_images:
    params.number_positive_examples *= 2

facial_detector: FacialDetector = FacialDetector(params)

if False:
    dad_patches, mom_patches, deedee_patches, dexter_patches, unknown_patches, negative_patches_dad, negative_patches_rectangle, negative_patches_deedee\
        = facial_detector.extract_patches(['antrenare/dad', 'antrenare/mom', 'antrenare/deedee', 'antrenare/dexter'], ['antrenare/dad_annotations.txt', 'antrenare/mom_annotations.txt', 'antrenare/deedee_annotations.txt', 'antrenare/dexter_annotations.txt'])

    # _, mom_patches, _, dexter_patches, unknown_patches, _, negative_patches_rectangle, _ \
    #     = facial_detector.extract_patches(['antrenare/dad', 'antrenare/mom', 'antrenare/deedee', 'antrenare/dexter'],
    #                                       ['antrenare/dad_annotations.txt', 'antrenare/mom_annotations.txt',
    #                                        'antrenare/deedee_annotations.txt', 'antrenare/dexter_annotations.txt'])

    rectangle_patches = np.concatenate((mom_patches, dexter_patches, unknown_patches), axis=0)
    if len(negative_patches_dad) > params.number_negative_examples:
        negative_patches_dad = negative_patches_dad[:params.number_negative_examples]
    if len(negative_patches_rectangle) > params.number_negative_examples:
        negative_rectangle_patches = negative_patches_rectangle[:params.number_negative_examples]
    if len(negative_patches_deedee) > params.number_negative_examples:
        negative_deedee_patches = negative_patches_deedee[:params.number_negative_examples]










# Pasii 1+2+3. Incarcam exemplele pozitive (cropate) si exemple negative generate
# verificam daca sunt deja existente
#
# dad_features_path = os.path.join(params.dir_save_files, 'descriptoriExempleDad_' + str(params.dim_hog_cell) + '_' +'.npy')
# if os.path.exists(dad_features_path):
#     dad_features = np.load(dad_features_path)
#     print('Am incarcat descriptorii pentru exemplele dad')
# else:
#     print('Construim descriptorii pentru exemplele dad:')
#     dad_features = facial_detector.get_positive_descriptors(dad_patches)
#     np.save(dad_features_path, dad_features)
#     print('Am salvat descriptorii pentru exemplele pozitive in fisierul %s' % dad_features_path)
#
# negative_features_dad_path = os.path.join(params.dir_save_files, 'descriptoriExempleNegativeDad_' + str(params.dim_hog_cell) + '_' + '.npy')
# if os.path.exists(negative_features_dad_path):
#     negative_features_dad = np.load(negative_features_dad_path)
#     print('Am incarcat descriptorii pentru exemplele negative')
# else:
#     print('Construim descriptorii pentru exemplele negative:')
#     negative_features_dad = facial_detector.get_negative_descriptors(negative_patches_dad)
#     np.save(negative_features_dad_path, negative_features_dad)
#     print('Am salvat descriptorii pentru exemplele negative in fisierul %s' % negative_features_dad_path)
#
# training_examples_dad = np.concatenate((np.squeeze(dad_features), np.squeeze(negative_features_dad)), axis=0)
# train_labels_dad = np.concatenate((np.ones(dad_features.shape[0]), np.zeros(negative_features_dad.shape[0])))
#
# facial_detector.train_classifier(training_examples_dad, train_labels_dad, [], [], [], [])

#
#
# rectangle_features_path = os.path.join(params.dir_save_files, 'descriptoriExempleRectangle_' + str(params.dim_hog_cell) + '_' + '.npy')
# if os.path.exists(rectangle_features_path):
#     rectangle_features = np.load(rectangle_features_path)
#     print('Am incarcat descriptorii pentru exemplele patrate')
# else:
#     print('Construim descriptorii pentru exemplele patrate:')
#     rectangle_features = facial_detector.get_positive_descriptors(rectangle_patches)
#     np.save(rectangle_features_path, rectangle_features)
#     print('Am salvat descriptorii pentru exemplele patrate in fisierul %s' % rectangle_features_path)
#
#
# negative_features_rectangle_path = os.path.join(params.dir_save_files, 'descriptoriExempleNegativeRectangle_' + str(params.dim_hog_cell) + '_' + '.npy')
# if os.path.exists(negative_features_rectangle_path):
#     negative_features_rectangle = np.load(negative_features_rectangle_path)
#     print('Am incarcat descriptorii pentru exemplele negative')
# else:
#     print('Construim descriptorii pentru exemplele negative:')
#     negative_features_rectangle = facial_detector.get_negative_descriptors(negative_patches_rectangle)
#     np.save(negative_features_rectangle_path, negative_features_rectangle)
#     print('Am salvat descriptorii pentru exemplele negative in fisierul %s' % negative_features_rectangle_path)
#
#
# training_examples_rectangle = np.concatenate((np.squeeze(rectangle_features), np.squeeze(negative_features_rectangle[:30000])), axis=0)
# train_labels_rectangle = np.concatenate((np.ones(rectangle_features.shape[0]), np.zeros(negative_features_rectangle[:30000].shape[0])))
#
#
# facial_detector.train_classifier([], [], training_examples_rectangle, train_labels_rectangle, [], [])

#

# deedee_features_path = os.path.join(params.dir_save_files, 'descriptoriExempleDeeDee_' + str(params.dim_hog_cell) + '_' + '.npy')
# if os.path.exists(deedee_features_path):
#     deedee_features = np.load(deedee_features_path)
#     print('Am incarcat descriptorii pentru exemplele deedee')
# else:
#     print('Construim descriptorii pentru exemplele deedee:')
#     deedee_features = facial_detector.get_positive_descriptors(deedee_patches)
#     np.save(deedee_features_path, deedee_features)
#     print('Am salvat descriptorii pentru exemplele deedee in fisierul %s' % deedee_features_path)
#
# negative_features_deedee_path = os.path.join(params.dir_save_files, 'descriptoriExempleNegativeDeedee_' + str(params.dim_hog_cell) + '_' + '.npy')
# if os.path.exists(negative_features_deedee_path):
#     negative_features_deedee = np.load(negative_features_deedee_path)
#     print('Am incarcat descriptorii pentru exemplele negative')
# else:
#     print('Construim descriptorii pentru exemplele negative:')
#     negative_features_deedee = facial_detector.get_negative_descriptors(negative_patches_deedee)
#     np.save(negative_features_deedee_path, negative_features_deedee)
#     print('Am salvat descriptorii pentru exemplele negative in fisierul %s' % negative_features_deedee_path)
#
#
# training_examples_deedee = np.concatenate((np.squeeze(deedee_features), np.squeeze(negative_features_deedee)), axis=0)
# train_labels_deedee = np.concatenate((np.ones(deedee_features.shape[0]), np.zeros(negative_features_deedee.shape[0])))
#
#
# facial_detector.train_classifier([], [], [], [], training_examples_deedee, train_labels_deedee)

# dexter_features_path = os.path.join(params.dir_save_files, 'descriptoriExempleDexter_' + str(params.dim_hog_cell) + '_' +'.npy')
# if os.path.exists(dexter_features_path):
#     dexter_features = np.load(dexter_features_path)
#     print('Am incarcat descriptorii pentru exemplele dexter')
# else:
#     print('Construim descriptorii pentru exemplele dexter:')
#     dexter_features = facial_detector.get_positive_descriptors(dexter_patches)
#     np.save(dexter_features_path, dexter_features)
#     print('Am salvat descriptorii pentru exemplele pozitive in fisierul %s' % dexter_features_path)
#
#
# mom_features_path = os.path.join(params.dir_save_files, 'descriptoriExempleMom_' + str(params.dim_hog_cell) + '_' +'.npy')
# if os.path.exists(mom_features_path):
#     mom_features = np.load(mom_features_path)
#     print('Am incarcat descriptorii pentru exemplele mom')
# else:
#     print('Construim descriptorii pentru exemplele mom:')
#     mom_features = facial_detector.get_positive_descriptors(mom_features)
#     np.save(mom_features_path, mom_features)
#     print('Am salvat descriptorii pentru exemplele pozitive in fisierul %s' % mom_features_path)








# # #
# #
# #
# #
# #
# #
# facial_detector.train_classifier(training_examples_dad, train_labels_dad, training_examples_rectangle, train_labels_rectangle, training_examples_deedee, train_labels_deedee)
facial_detector.train_classifier([], [], [], [], [], [])


#
# svm_dad = train_svm_for_character(dad_patches, 'dad', dad_patches, mom_patches, deedee_patches, dexter_patches)
# svm_mom = train_svm_for_character(mom_patches, 'mom', dad_patches, mom_patches, deedee_patches, dexter_patches)
# svm_deedee = train_svm_for_character(deedee_patches, 'deedee', dad_patches, mom_patches, deedee_patches, dexter_patches)
# svm_dexter = train_svm_for_character(dexter_patches, 'dexter', dad_patches, mom_patches, deedee_patches, dexter_patches)


solution_dir = os.path.join(params.base_dir, "342_Rapcea_Catalin")
os.makedirs(solution_dir, exist_ok=True)

task1_dir = os.path.join(solution_dir, "task1")
os.makedirs(task1_dir, exist_ok=True)

# overlap_values = [i / 10.0 for i in range(1, 7)]  # [0, 0.1, 0.2, ..., 0.6]
# threshold_values = [1 + i * 0.2 for i in range(7)]  # [1, 1.2, 1.4, ..., 3]


# # # #
# overlap_values = [0, 0.5, 0.75, 0.1, 0.2,  0.3, 0.4]  # [0, 0.1, 0.2, ..., 0.6]
# threshold_values = [0.8, 1, 1.1, 1.2]  # [1, 1.2, 1.4, ..., 3]
# for overlap in overlap_values:
#     for threshold in threshold_values:
#         params.overlap = overlap
#         params.threshold = threshold
#
#         print(f"Testing with overlap={overlap} and threshold={threshold}")
#         start_time = time.time()
#
#         # Rulează detectorul cu parametrii curenți
#         detections, scores, file_names = facial_detector.run()
#
#         elapsed_time = time.time() - start_time
#         if params.has_annotations:
#             # Evaluează detections și calculează metrici
#             precision, recall, f1_score = facial_detector.eval_detections(detections, scores, file_names)
#             print(
#                 f"Overlap={overlap}, Threshold={threshold} -> Precision={precision:.4f}, Recall={recall:.4f}, F1={f1_score:.4f}, Time={elapsed_time:.2f} second ")
#
#             # solution_dir = os.path.join(params.base_dir, "342_Rapcea_Catalin")
#             # os.makedirs(solution_dir, exist_ok=True)
#             #
#             # task1_dir = os.path.join(solution_dir, "task1")
#             # os.makedirs(task1_dir, exist_ok=True)
#
#             # Salvează array-urile în format .npy
#             np.save(os.path.join(task1_dir, "detections_all_faces.npy"), detections)
#             np.save(os.path.join(task1_dir, "scores_all_faces.npy"), scores)
#             np.save(os.path.join(task1_dir, "file_names_all_faces.npy"), file_names)
#
#             classified_detections = facial_detector.classify_detections(detections, scores, file_names)
#
#             task2_dir = os.path.join(solution_dir, "task2")
#             os.makedirs(task2_dir, exist_ok=True)
#
#             for character in ['dad', 'mom', 'deedee', 'dexter']:
#                 # Filtrăm detectările pentru caracterul curent
#                 character_detections = [d for d in classified_detections if d['detected_character'] == character]
#
#                 # Extragem coordonatele (bbox), scorurile și numele fișierelor pentru caracterul curent
#                 character_bboxes = [d['detection'] for d in
#                                     character_detections]  # presupunând că 'detection' este [x_min, y_min, x_max, y_max]
#                 character_scores = [d['score'] for d in character_detections]
#                 character_file_names = [d['file_name'] for d in character_detections]
#
#                 # Convertim în numpy arrays
#                 detections_array = np.array(character_bboxes, dtype=object)
#                 scores_array = np.array(character_scores, dtype=float)
#                 file_names_array = np.array(character_file_names, dtype=object)
#
#                 # Salvăm fișierele în format .npy
#                 np.save(f"{task2_dir}/detections_{character}.npy", detections_array)
#                 np.save(f"{task2_dir}/scores_{character}.npy", scores_array)
#                 np.save(f"{task2_dir}/file_names_{character}.npy", file_names_array)
#
#         else:
#             print(f"Overlap={overlap}, Threshold={threshold} -> Detections processed.")
#

detections, scores, file_names = facial_detector.run()



solution_dir = os.path.join(params.base_dir, "342_Rapcea_Catalin")
os.makedirs(solution_dir, exist_ok=True)

task1_dir = os.path.join(solution_dir, "task1")
os.makedirs(task1_dir, exist_ok=True)


# Salvează array-urile în format .npy
np.save(os.path.join(task1_dir, "detections_all_faces.npy"), detections)
np.save(os.path.join(task1_dir, "scores_all_faces.npy"), scores)
np.save(os.path.join(task1_dir, "file_names_all_faces.npy"), file_names)

classified_detections = facial_detector.classify_detections(detections, scores, file_names)


task2_dir = os.path.join(solution_dir, "task2")
os.makedirs(task2_dir, exist_ok=True)


for character in ['dad', 'mom', 'deedee', 'dexter']:
    # Filtrăm detectările pentru caracterul curent
    character_detections = [d for d in classified_detections if d['detected_character'] == character]

    # Extragem coordonatele (bbox), scorurile și numele fișierelor pentru caracterul curent
    character_bboxes = [d['detection'] for d in character_detections]  # presupunând că 'detection' este [x_min, y_min, x_max, y_max]
    character_scores = [d['score'] for d in character_detections]
    character_file_names = [d['file_name'] for d in character_detections]

    # Convertim în numpy arrays
    detections_array = np.array(character_bboxes, dtype=object)
    scores_array = np.array(character_scores, dtype=float)
    file_names_array = np.array(character_file_names, dtype=object)

    # Salvăm fișierele în format .npy
    np.save(f"{task2_dir}/detections_{character}.npy", detections_array)
    np.save(f"{task2_dir}/scores_{character}.npy", scores_array)
    np.save(f"{task2_dir}/file_names_{character}.npy", file_names_array)



#
# if params.has_annotations:
#     facial_detector.eval_detections(detections, scores, file_names)
#     show_detections_with_ground_truth(detections, scores, file_names, params)
# else:
#     show_detections_without_ground_truth(detections, scores, file_names, params)
