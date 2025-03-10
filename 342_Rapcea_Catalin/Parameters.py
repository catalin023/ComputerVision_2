import os

class Parameters:
    def __init__(self):
        self.base_dir = '/home/catalin/PycharmProjects/cava/342_Rapcea_Catalin' #folderul de baza
        self.dir_test_examples = os.path.join(self.base_dir,'validare')  #folderul cu imaginie de testat
        # self.path_annotations = os.path.join(self.base_dir, 'validare/task1_gt_validare.txt')  # 'exempleTest/groundTruth.txt'
        self.dir_save_files = os.path.join(self.base_dir, 'salveazaFisiere')
        if not os.path.exists(self.dir_save_files):
            os.makedirs(self.dir_save_files)
            print('directory created: {} '.format(self.dir_save_files))
        else:
            print('directory {} exists '.format(self.dir_save_files))

        # set the parameters
        self.dim_window = 100  # exemplele pozitive (fete de oameni cropate) au 36x36 pixeli
        self.dim_hog_cell = 8  # dimensiunea celulei
        self.cells_per_block = 2
        self.overlap = 0.3
        self.number_positive_examples = 6713  # numarul exemplelor pozitive
        self.number_negative_examples = 10000  # numarul exemplelor negative
        self.has_annotations = False
        self.threshold = 0.3
        self.use_flip_images = True
