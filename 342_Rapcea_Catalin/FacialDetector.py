from numpy.ma.core import shape
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from Parameters import *
import numpy as np
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import glob
import cv2 as cv
import pickle
import ntpath
from copy import deepcopy
import timeit
from skimage.feature import hog



class FacialDetector:
    def __init__(self, params: Parameters):
        self.params = params
        self.best_model_dad = None
        self.best_model_rectangle = None
        self.best_model_deedee = None
        self.clasifier_dad = None
        self.clasifier_mom = None
        self.clasifier_deedee = None
        self.clasifier_dexter = None

    def extract_patches(self, image_folders, annotation_files):
        dad_patches = []
        mom_patches = []
        deedee_patches = []
        dexter_patches = []
        unknown_patches = []

        negative_dad_patches = []
        negative_rectangle_patches = []
        negative_deedee_patches = []

        def generate_deviated_patches(image, x1, y1, x2, y2, max_shift=5, max_scale=0.1):
            height, width = image.shape[:2]
            deviated_patches = []

            for _ in range(3):
                shift_x = np.random.randint(-max_shift, max_shift)
                shift_y = np.random.randint(-max_shift, max_shift)

                scale_factor = 1 + np.random.uniform(-max_scale, max_scale)

                new_x1 = max(0, int(x1 + shift_x))
                new_y1 = max(0, int(y1 + shift_y))
                new_x2 = min(width, int(x2 + shift_x * scale_factor))
                new_y2 = min(height, int(y2 + shift_y * scale_factor))

                deviated_patch = image[new_y1:new_y2, new_x1:new_x2]

                if deviated_patch.shape[0] < 10 or deviated_patch.shape[1] < 10:
                    continue

                deviated_patches.append(deviated_patch)

            return deviated_patches

        for image_folder, annotation_file in zip(image_folders, annotation_files):
            # Read the annotation file
            with open(annotation_file, 'r') as f:
                lines = f.readlines()

            # Load all annotations for the current image
            image_annotations = {}
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 6:
                    continue

                image_name, x1, y1, x2, y2, character = parts
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                if image_name not in image_annotations:
                    image_annotations[image_name] = []
                image_annotations[image_name].append((x1, y1, x2, y2, character))

            # Process each image
            for image_name, annotations in image_annotations.items():
                image_path = os.path.join(image_folder, image_name)
                image = cv.imread(image_path)

                height, width = image.shape[:2]
                face_areas = []

                # Process each annotation
                for x1, y1, x2, y2, character in annotations:
                    # Extract the patch
                    patch = image[y1:y2, x1:x2]
                    patch = cv.cvtColor(patch, cv.COLOR_BGR2GRAY)

                    # Add the patch to the corresponding character's list
                    if character == "dad":
                        patch = cv.resize(patch, (self.params.dim_window, int(self.params.dim_window*1.5)))
                        dad_patches.append(patch)
                        face_width, face_height = self.params.dim_window, int(self.params.dim_window*1.5)
                    elif character == "mom":
                        patch = cv.resize(patch, (self.params.dim_window, self.params.dim_window))
                        mom_patches.append(patch)
                        face_width, face_height = self.params.dim_window, self.params.dim_window
                    elif character == "deedee":
                        patch = cv.resize(patch, (self.params.dim_window*2, self.params.dim_window))
                        deedee_patches.append(patch)
                        face_width, face_height = self.params.dim_window*2, self.params.dim_window
                    elif character == "dexter":
                        patch = cv.resize(patch, (self.params.dim_window, self.params.dim_window))
                        dexter_patches.append(patch)
                        face_width, face_height = self.params.dim_window, self.params.dim_window
                    elif character == "unknown":
                        patch = cv.resize(patch, (self.params.dim_window, self.params.dim_window))
                        unknown_patches.append(patch)
                        face_width, face_height = self.params.dim_window, self.params.dim_window


                    deviated_patches = generate_deviated_patches(image, x1, y1, x2, y2)
                    for deviated_patch in deviated_patches:
                        deviated_patch = cv.cvtColor(deviated_patch, cv.COLOR_BGR2GRAY)
                        deviated_patch = cv.resize(deviated_patch, (face_width, face_height))
                        if character == "dad":
                            dad_patches.append(deviated_patch)
                        elif character == "mom":
                            mom_patches.append(deviated_patch)
                        elif character == "deedee":
                            deedee_patches.append(deviated_patch)
                        elif character == "dexter":
                            dexter_patches.append(deviated_patch)
                        elif character == "unknown":
                            unknown_patches.append(deviated_patch)

                    num_negative_patches = 10 \
                        if character == "deedee" else 5
                    for _ in range(num_negative_patches):
                        for _ in range(50):  # Try up to 50 times to find a valid negative patch
                            neg_x1 = np.random.randint(0, width - face_width)
                            neg_y1 = np.random.randint(0, height - face_height)
                            neg_x2 = neg_x1 + face_width
                            neg_y2 = neg_y1 + face_height

                            # Check overlap with face bounding boxes
                            overlap = False
                            for x1, y1, x2, y2, face_area in face_areas:
                                overlap_x1 = max(x1, neg_x1)
                                overlap_y1 = max(y1, neg_y1)
                                overlap_x2 = min(x2, neg_x2)
                                overlap_y2 = min(y2, neg_y2)

                                overlap_area = max(0, overlap_x2 - overlap_x1) * max(0, overlap_y2 - overlap_y1)
                                if overlap_area > 0.15 * face_area:
                                    overlap = True
                                    break

                            if not overlap:
                                negative_patch = image[neg_y1:neg_y2, neg_x1:neg_x2]
                                negative_patch = cv.cvtColor(negative_patch, cv.COLOR_BGR2GRAY)
                                negative_patch = cv.resize(negative_patch, (face_width, face_height))

                                if character == "dad":
                                    negative_dad_patches.append(negative_patch)
                                elif character == "deedee":
                                    negative_deedee_patches.append(negative_patch)
                                elif character == "mom" or character == "dexter":
                                    negative_rectangle_patches.append(negative_patch)
                                break

                    ratios = [(1, 1), (1, 1.5), (2, 1)]
                    for ratio_width, ratio_height in ratios:
                        for _ in range(2):  # Generate 3 patches for each ratio
                            patch_width = np.random.randint(int(0.2 * face_width), int(0.35 * face_width))
                            patch_height = int(patch_width * ratio_height / ratio_width)

                            if x1 >= x2 - patch_width or y1 >= y2 - patch_height:
                                continue

                            face_x1 = np.random.randint(x1, x2 - patch_width)
                            face_y1 = np.random.randint(y1, y2 - patch_height)
                            face_x2 = face_x1 + patch_width
                            face_y2 = face_y1 + patch_height

                            face_patch = image[face_y1:face_y2, face_x1:face_x2]
                            face_patch = cv.cvtColor(face_patch, cv.COLOR_BGR2GRAY)

                            if ratios == (1, 1.5):
                                face_patch = cv.resize(face_patch,
                                                       (self.params.dim_window, int(self.params.dim_window*1.5)))
                                negative_dad_patches.append(face_patch)
                            elif ratios == (2, 1):
                                face_patch = cv.resize(face_patch,
                                                       (self.params.dim_window*2, self.params.dim_window))
                                negative_deedee_patches.append(face_patch)
                            else:
                                face_patch = cv.resize(face_patch, (self.params.dim_window, self.params.dim_window))
                                negative_rectangle_patches.append(face_patch)
        return (
            dad_patches, mom_patches, deedee_patches, dexter_patches, unknown_patches,
            negative_dad_patches, negative_rectangle_patches, negative_deedee_patches
        )

    def get_positive_descriptors(self, positive_patches):
        # in aceasta functie calculam descriptorii pozitivi
        # vom returna un numpy array de dimensiuni NXD
        # unde N - numar exemplelor pozitive
        # iar D - dimensiunea descriptorului
        # D = (params.dim_window/params.dim_hog_cell - 1) ^ 2 * params.dim_descriptor_cell (fetele sunt patrate)

        num_patches = len(positive_patches)  # numărul de patch-uri pozitive
        positive_descriptors = []
        print('Calculăm descriptorii pentru %d patch-uri pozitive...' % num_patches)

        for i in range(num_patches):
            print('Procesăm patch-ul pozitiv numărul %d...' % i)
            patch = positive_patches[i]  # Patch-ul pozitiv

            # Calculăm descriptorul HOG pentru patch-ul pozitiv
            features = hog(patch, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                           cells_per_block=(self.params.cells_per_block, self.params.cells_per_block),
                           feature_vector=True)

            positive_descriptors.append(features.flatten())

            # Dacă utilizăm imagini oglindite, adăugăm și descriptorul pentru patch-ul oglindit
            if self.params.use_flip_images:
                flipped_patch = np.fliplr(patch)
                flipped_features = hog(flipped_patch,
                                       pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                                       cells_per_block=(
                                       self.params.cells_per_block, self.params.cells_per_block),
                                       feature_vector=True)
                positive_descriptors.append(flipped_features)

        positive_descriptors = np.array(positive_descriptors)
        return positive_descriptors

    def get_negative_descriptors(self, negative_patches):
        # in aceasta functie calculam descriptorii negativi
        # vom returna un numpy array de dimensiuni NXD
        # unde N - numar exemplelor negative
        # iar D - dimensiunea descriptorului
        # avem 274 de imagini negative, vream sa avem self.params.number_negative_examples (setat implicit cu 10000)
        # de exemple negative, din fiecare imagine vom genera aleator self.params.number_negative_examples // 274
        # patch-uri de dimensiune 36x36 pe care le vom considera exemple negative

        num_patches = len(negative_patches)
        negative_descriptors = []
        print('Calculam descriptorii pt %d imagini negative' % num_patches)
        for i in range(num_patches):
            # if i == 8000:
            #     break
            print('Procesam exemplul negativ numarul %d...' % i)
            patch = negative_patches[i]
            features = hog(patch, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                           cells_per_block=(self.params.cells_per_block, self.params.cells_per_block),
                           feature_vector=True)
            negative_descriptors.append(features.flatten())

            if self.params.use_flip_images:
                flipped_patch = np.fliplr(patch)
                flipped_features = hog(flipped_patch,
                                       pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                                       cells_per_block=(
                                       self.params.cells_per_block, self.params.cells_per_block),
                                       feature_vector=True)
                negative_descriptors.append(flipped_features)

        negative_descriptors = np.array(negative_descriptors)
        return negative_descriptors

    def train_classifier(self, training_examples_dad, train_labels_dad, training_examples_rectangle,
                         train_labels_rectangle, training_examples_deedee, train_labels_deedee):
        svm_file_name = os.path.join(self.params.dir_save_files, 'best_model_dad_%d_%d_%d' %
                                     (self.params.dim_hog_cell, self.params.number_negative_examples,
                                      self.params.number_positive_examples))
        if os.path.exists(svm_file_name):
            self.best_model_dad = pickle.load(open(svm_file_name, 'rb'))

        else:
            X_train, X_val, y_train, y_val = train_test_split(training_examples_dad, train_labels_dad,
                                                              test_size=0.1,
                                                              random_state=42)
            best_accuracy = 0
            best_c = 0
            best_model = None
            Cs = [10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 10 ** 0]
            for c in Cs:
                print('Antrenam un clasificator pentru c=%f' % c)
                model = LinearSVC(C=c)
                model.fit(X_train, y_train)  # Antrenează doar pe setul de antrenament
                acc = model.score(X_val, y_val)
                print(acc)
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_c = c
                    best_model = deepcopy(model)

            print('Performanta clasificatorului optim pt c = %f' % best_c)
            # salveaza clasificatorul
            pickle.dump(best_model, open(svm_file_name, 'wb'))
            # scores = best_model.decision_function(X_val)
            self.best_model_dad = best_model

        svm_file_name = os.path.join(self.params.dir_save_files, 'best_model_rectangle_%d_%d_%d' %
                                     (self.params.dim_hog_cell, self.params.number_negative_examples,
                                      self.params.number_positive_examples))
        if os.path.exists(svm_file_name):
            self.best_model_rectangle = pickle.load(open(svm_file_name, 'rb'))
        else:
            X_train, X_val, y_train, y_val = train_test_split(training_examples_rectangle,
                                                              train_labels_rectangle,
                                                              test_size=0.1,
                                                              random_state=42)
            best_accuracy = 0
            best_c = 0
            best_model = None
            Cs = [10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 10 ** 0]
            for c in Cs:
                print('Antrenam un clasificator pentru c=%f' % c)
                model = LinearSVC(C=c)
                model.fit(X_train, y_train)  # Antrenează doar pe setul de antrenament
                acc = model.score(X_val, y_val)
                print(acc)
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_c = c
                    best_model = deepcopy(model)

            print('Performanta clasificatorului optim pt c = %f' % best_c)
            # salveaza clasificatorul
            pickle.dump(best_model, open(svm_file_name, 'wb'))
            # scores = best_model.decision_function(X_val)
            self.best_model_rectangle = best_model

        svm_file_name = os.path.join(self.params.dir_save_files, 'best_model_deedee_%d_%d_%d' %
                                     (self.params.dim_hog_cell, self.params.number_negative_examples,
                                      self.params.number_positive_examples))
        if os.path.exists(svm_file_name):
            self.best_model_deedee = pickle.load(open(svm_file_name, 'rb'))
            return

        X_train, X_val, y_train, y_val = train_test_split(training_examples_deedee, train_labels_deedee,
                                                          test_size=0.1,
                                                          random_state=42)
        best_accuracy = 0
        best_c = 0
        best_model = None
        Cs = [10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 10 ** 0]
        for c in Cs:
            print('Antrenam un clasificator pentru c=%f' % c)
            model = LinearSVC(C=c)
            model.fit(X_train, y_train)  # Antrenează doar pe setul de antrenament
            acc = model.score(X_val, y_val)
            print(acc)
            if acc > best_accuracy:
                best_accuracy = acc
                best_c = c
                best_model = deepcopy(model)

        print('Performanta clasificatorului optim pt c = %f' % best_c)
        # salveaza clasificatorul
        pickle.dump(best_model, open(svm_file_name, 'wb'))
        # scores = best_model.decision_function(X_val)
        self.best_model_deedee = best_model

    def intersection_over_union(self, bbox_a, bbox_b):
        x_a = max(bbox_a[0], bbox_b[0])
        y_a = max(bbox_a[1], bbox_b[1])
        x_b = min(bbox_a[2], bbox_b[2])
        y_b = min(bbox_a[3], bbox_b[3])

        inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

        box_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
        box_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)

        iou = inter_area / float(box_a_area + box_b_area - inter_area)

        return iou

    def non_maximal_suppression(self, image_detections, image_scores, image_size):
        # xmin, ymin, xmax, ymax
        x_out_of_bounds = np.where(image_detections[:, 2] > image_size[1])[0]
        y_out_of_bounds = np.where(image_detections[:, 3] > image_size[0])[0]
        # print(x_out_of_bounds, y_out_of_bounds)
        image_detections[x_out_of_bounds, 2] = image_size[1]
        image_detections[y_out_of_bounds, 3] = image_size[0]
        sorted_indices = np.flipud(np.argsort(image_scores))
        sorted_image_detections = image_detections[sorted_indices]
        sorted_scores = image_scores[sorted_indices]

        is_maximal = np.ones(len(image_detections)).astype(bool)
        iou_threshold = self.params.overlap
        for i in range(len(sorted_image_detections) - 1):
            if is_maximal[
                i] == True:  # don't change to 'is True' because is a numpy True and is not a python True :)
                for j in range(i + 1, len(sorted_image_detections)):
                    if is_maximal[
                        j] == True:  # don't change to 'is True' because is a numpy True and is not a python True :)
                        if self.intersection_over_union(sorted_image_detections[i],
                                                        sorted_image_detections[j]) > iou_threshold:
                            is_maximal[j] = False
                        else:  # verificam daca centrul detectiei este in mijlocul detectiei cu scor mai mare
                            c_x = (sorted_image_detections[j][0] + sorted_image_detections[j][2]) / 2
                            c_y = (sorted_image_detections[j][1] + sorted_image_detections[j][3]) / 2
                            if sorted_image_detections[i][0] <= c_x <= sorted_image_detections[i][2] and \
                                    sorted_image_detections[i][1] <= c_y <= sorted_image_detections[i][3]:
                                is_maximal[j] = False
        return sorted_image_detections[is_maximal], sorted_scores[is_maximal]

    def run(self):
        test_images_path = os.path.join(self.params.dir_test_examples, '*.jpg')
        test_files = sorted(glob.glob(test_images_path))
        detections = []  # Array cu toate detectiile pe care le obtinem
        scores = np.array([])  # Array cu toate scorurile pe care le obtinem
        file_names = np.array([])  # Array cu fisierele asociate detectiilor

        # Modelele si parametrii lor
        models = {
            "dad": {
                "window_size": (self.params.dim_window, int(self.params.dim_window*1.5)),
                "w": self.best_model_dad.coef_.T,
                "bias": self.best_model_dad.intercept_[0]
            },
            "rectangle": {
                "window_size": (self.params.dim_window, self.params.dim_window),
                "w": self.best_model_rectangle.coef_.T,
                "bias": self.best_model_rectangle.intercept_[0]
            },
            "deedee": {
                "window_size": (self.params.dim_window*2, self.params.dim_window),
                "w": self.best_model_deedee.coef_.T,
                "bias": self.best_model_deedee.intercept_[0]
            }
        }

        # test_files = deepcopy(test_files[:20])
        num_test_images = len(test_files)

        scale_factor = 1.05    # Factorul de scalare pentru fereastră
        step_size = 8 # Dimensiunea pasului sliding window-ului

        for i in range(num_test_images):
            start_time = timeit.default_timer()
            # print('Procesam imaginea de testare %d/%d..' % (i + 1, num_test_images))
            img = cv.imread(test_files[i], cv.IMREAD_GRAYSCALE)

            image_scores = []
            image_detections = []

            for model_name, model_data in models.items():
                window_size = model_data["window_size"]
                w = model_data["w"]
                bias = model_data["bias"]

                current_width, current_height = window_size

                # Aplicăm scalarea de mai multe ori
                while current_width <= img.shape[1] and current_height <= img.shape[0]:
                    for y in range(0, img.shape[0] - current_height + 1, step_size):
                        for x in range(0, img.shape[1] - current_width + 1, step_size):
                            patch = img[y:y + current_height, x:x + current_width]
                            patch_resized = cv.resize(patch, window_size)
                            descr = hog(
                                patch_resized,
                                pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                                cells_per_block=(self.params.cells_per_block, self.params.cells_per_block),
                                feature_vector=True
                            )
                            descr = descr.flatten()
                            score = np.dot(descr, w)[0] + bias
                            if model_name == "rectangle":
                                score -= 0.5
                            if model_name == "deedee":
                                score -= 0.4
                            if score > self.params.threshold:
                                x_min, y_min = x, y
                                x_max, y_max = x + current_width, y + current_height
                                image_detections.append([x_min, y_min, x_max, y_max])
                                image_scores.append(score)

                    # Scalăm dimensiunile ferestrei
                    if current_width < 200:
                        current_width = int(current_width * 1.1)
                        current_height = int(current_width * window_size[1] / window_size[0])
                    else:
                        current_width = int(current_width * scale_factor)
                        current_height = int(current_width * window_size[1] / window_size[0])
                    # current_width += 25
                    # current_height += 25

            # Aplica non-maximal suppression dacă există mai multe detectii
            if len(image_scores) > 0:
                suppressed_detections, suppressed_scores = self.non_maximal_suppression(
                    np.array(image_detections),
                    np.array(image_scores),
                    img.shape
                )
                image_detections = suppressed_detections.tolist()  # Conversie la listă
                image_scores = suppressed_scores.tolist()

            if len(image_scores) > 0:
                if len(detections) == 0:
                    detections = image_detections
                else:
                    detections = np.concatenate((detections, image_detections))
                scores = np.append(scores, image_scores)
                short_name = ntpath.basename(test_files[i])
                image_names = [short_name for _ in range(len(image_scores))]
                file_names = np.append(file_names, image_names)

            end_time = timeit.default_timer()
            print('Timpul de procesare al imaginii %d: %f sec.' % (i, end_time - start_time))
        # print(detections, scores, file_names)
        return detections, scores, file_names

    def compute_average_precision(self, rec, prec):
        # functie adaptata din 2010 Pascal VOC development kit
        m_rec = np.concatenate(([0], rec, [1]))
        m_pre = np.concatenate(([0], prec, [0]))
        for i in range(len(m_pre) - 1, -1, 1):
            m_pre[i] = max(m_pre[i], m_pre[i + 1])
        m_rec = np.array(m_rec)
        i = np.where(m_rec[1:] != m_rec[:-1])[0] + 1
        average_precision = np.sum((m_rec[i] - m_rec[i - 1]) * m_pre[i])
        return average_precision

    def eval_detections(self, detections, scores, file_names):
        ground_truth_file = np.loadtxt(self.params.path_annotations, dtype='str')
        ground_truth_file_names = np.array(ground_truth_file[:, 0])
        ground_truth_detections = np.array(ground_truth_file[:, 1:], int)

        num_gt_detections = len(ground_truth_detections)  # numar total de adevarat pozitive
        gt_exists_detection = np.zeros(num_gt_detections)
        # sorteazam detectiile dupa scorul lor
        sorted_indices = np.argsort(scores)[::-1]
        file_names = file_names[sorted_indices]
        scores = scores[sorted_indices]
        detections = detections[sorted_indices]

        num_detections = len(detections)
        true_positive = np.zeros(num_detections)
        false_positive = np.zeros(num_detections)
        duplicated_detections = np.zeros(num_detections)

        for detection_idx in range(num_detections):
            indices_detections_on_image = np.where(ground_truth_file_names == file_names[detection_idx])[0]

            gt_detections_on_image = ground_truth_detections[indices_detections_on_image]
            bbox = detections[detection_idx]
            max_overlap = -1
            index_max_overlap_bbox = -1
            for gt_idx, gt_bbox in enumerate(gt_detections_on_image):
                overlap = self.intersection_over_union(bbox, gt_bbox)
                if overlap > max_overlap:
                    max_overlap = overlap
                    index_max_overlap_bbox = indices_detections_on_image[gt_idx]

            # clasifica o detectie ca fiind adevarat pozitiva / fals pozitiva
            if max_overlap >= 0.3:
                if gt_exists_detection[index_max_overlap_bbox] == 0:
                    true_positive[detection_idx] = 1
                    gt_exists_detection[index_max_overlap_bbox] = 1
                else:
                    false_positive[detection_idx] = 1
                    duplicated_detections[detection_idx] = 1
            else:
                false_positive[detection_idx] = 1

        cum_false_positive = np.cumsum(false_positive)
        cum_true_positive = np.cumsum(true_positive)

        rec = cum_true_positive / num_gt_detections
        prec = cum_true_positive / (cum_true_positive + cum_false_positive)
        average_precision = self.compute_average_precision(rec, prec)
        print(average_precision)
        plt.plot(rec, prec, '-')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Average precision %.3f' % average_precision)
        plt.savefig(os.path.join(self.params.dir_save_files, 'precizie_medie.png'))
        plt.show()

        precision = prec[-1] if len(prec) > 0 else 0
        recall = rec[-1] if len(rec) > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Returnăm valorile calculate
        return precision, recall, f1_score

    def train_svm_for_character(self, patches, character_type, dad_patches, mom_patches, deedee_patches,
                                dexter_patches):
        svm_file_name = os.path.join(self.params.dir_save_files,
                                     f'clasifier_model_{character_type}_{self.params.dim_hog_cell}.pkl')

        if os.path.exists(svm_file_name):
            # Dacă există deja modelul salvat, îl încărcăm
            self.best_model = pickle.load(open(svm_file_name, 'rb'))
            return self.best_model

        patch_size = (self.params.dim_window, self.params.dim_window)

        # Lista de caracteristici și etichete
        features = []
        labels = []

        # Determinăm patch-urile negative
        if character_type == 'dad':
            negative_patches = mom_patches + deedee_patches + dexter_patches
        elif character_type == 'deedee':
            negative_patches = dad_patches + mom_patches + dexter_patches
        elif character_type == 'dexter':
            negative_patches = dad_patches + mom_patches + deedee_patches
        else:
            negative_patches = dad_patches + deedee_patches + dexter_patches

        # Funcție pentru calcularea descriptorilor HOG
        def compute_hog(patch):
            resized_patch = cv.resize(patch, patch_size)
            return hog(
                resized_patch,
                pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                cells_per_block=(self.params.cells_per_block, self.params.cells_per_block),
                block_norm='L2-Hys',
                feature_vector=True
            )

        # Procesăm patch-urile pozitive
        for idx, patch in enumerate(patches):
            print(f"Procesăm patch-ul pozitiv numărul {idx} pentru {character_type}...")
            hog_features = compute_hog(patch)
            features.append(hog_features)
            labels.append(1)

        # Procesăm patch-urile negative
        for idx, patch in enumerate(negative_patches):
            print(f"Procesăm patch-ul negativ numărul {idx} pentru {character_type}...")
            hog_features = compute_hog(patch)
            features.append(hog_features)
            labels.append(0)

        # Standardizăm caracteristicile pentru a îmbunătăți performanța SVM
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        # Împărțim datele în set de antrenament și set de validare
        X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.1, random_state=42)

        best_accuracy = 0
        best_c = 0
        best_model = None
        Cs = [10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 10 ** 0]  # Valorile C pe care le vom testa

        # Testăm mai multe valori pentru C
        for c in Cs:
            print(f'Antrenăm un clasificator pentru C={c}')
            model = LinearSVC(C=c)
            model.fit(X_train, y_train)  # Antrenăm doar pe setul de antrenament
            acc = model.score(X_val, y_val)  # Evaluăm pe setul de validare
            print(f'Acuratețea pentru C={c}: {acc:.4f}')

            if acc > best_accuracy:
                best_accuracy = acc
                best_c = c
                best_model = deepcopy(model)  # Salvăm cel mai bun model

        # Salvăm modelul cel mai bun
        with open(svm_file_name, 'wb') as f:
            pickle.dump(best_model, f)

        print(f"Cel mai bun C pentru {character_type}: {best_c}")
        print(f"Acuratețea obținută: {best_accuracy:.4f}")

        return

    def classify_detections(self, detections, scores, file_names):
        # Dimensiunea fixă a patch-urilor
        patch_size = (self.params.dim_window, self.params.dim_window)

        # Încărcăm modelele SVM pentru fiecare personaj
        models = {}
        for character in ['dad', 'mom', 'deedee', 'dexter']:
            svm_file_name = os.path.join(self.params.dir_save_files,
                                         f'clasifier_model_{character}_{self.params.dim_hog_cell}.pkl')
            if os.path.exists(svm_file_name):
                with open(svm_file_name, 'rb') as f:
                    models[character] = pickle.load(f)
            else:
                print(f"Modelul SVM pentru {character} nu a fost găsit!")
                return

        # Lista pentru rezultatele clasificării
        classified_detections = []

        # Clasificăm fiecare detecție
        for idx, (detection, score, file_name) in enumerate(zip(detections, scores, file_names)):
            print(f"Procesăm detecția {idx} din fișierul {file_name}...")

            x_min, y_min, x_max, y_max = detection

            image = cv.imread(os.path.join(self.params.dir_test_examples, file_name))
            patch = image[y_min:y_max, x_min:x_max]
            patch = cv.cvtColor(patch, cv.COLOR_BGR2GRAY)
            # Redimensionăm detecția la dimensiunea fixă
            resized_detection = cv.resize(patch, patch_size)

            # Extragem descriptorii HOG
            hog_features = hog(
                resized_detection,
                pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                cells_per_block=(self.params.cells_per_block, self.params.cells_per_block),
                block_norm='L2-Hys',
                feature_vector=True
            )

            # Clasificăm utilizând fiecare SVM
            character_scores = {}
            for character, model in models.items():
                character_scores[character] = model.decision_function([hog_features])[0]

            # Găsim personajul cu cel mai mare scor
            best_character = max(character_scores, key=character_scores.get)
            best_score = character_scores[best_character]

            print(f"Detecția {idx}: {best_character} (Scor: {best_score:.4f})")

            # Salvăm rezultatul clasificării
            classified_detections.append({
                "file_name": file_name,
                "detected_character": best_character,
                "score": best_score,
                "original_score": score,
                "character_scores": character_scores,
                "detection": detection
            })

        return classified_detections

