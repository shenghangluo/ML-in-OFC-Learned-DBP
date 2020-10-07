import numpy as np
from scipy.io import loadmat
N = 16384
M = 8192 

#Train_Data_N = pow(2, 18)
#Train_Data_M = pow(2, 17)

Data_N = pow(2, 15)
Data_M = pow(2, 14)

# pwd="/ubc/ece/home/ll/grads/shenghang/LDBP_20span/Dataset/Dataset_"
# pwd_test = "/ubc/ece/home/ll/grads/shenghang/LDBP_20span/Test_Dataset/Dataset_"
# pwd_x_X_pol_Real="x_X_pol_Real.txt"
# pwd_x_X_pol_Imag="x_X_pol_Imag.txt"
# pwd_y_X_pol_Real="y_X_pol_Real.txt"
# pwd_y_X_pol_Imag="y_X_pol_Imag.txt"

data_train = '/ubc/ece/home/ll/grads/shenghang/LDBP_20span/Dataset/Dataset_train_32_6dbm.mat'
data_test = '/ubc/ece/home/ll/grads/shenghang/LDBP_20span/Test_Dataset/Dataset_test_32_6dbm.mat'

class DatasetMNIST(object):
    def __init__(self, val_size):
        self._val_size = val_size

    def load_data(self):
        """
        :return:
        """
        # Load the Dataset
        A = loadmat(data_train)['Input_Symbol_real_X']
        A = np.array(A)
        print("input A", A.shape)
        x_train_real = A
        print("Input_Symbol_Real", x_train_real.shape)
        A = loadmat(data_train)['Input_Symbol_image_X']
        A = np.array(A)
        x_train_image = A
        A = loadmat(data_train)['Transmit_real_X']
        A = np.array(A)
        y_train_real = A
        A = loadmat(data_train)['Transmit_image_X']
        A = np.array(A)
        y_train_image = A


        # y_train_Complex = y_train_real + 1j*y_train_image
        # y_train = np.c_[y_train_Complex.real, y_train_Complex.imag]

        # Test Power 1
        A = loadmat(data_test)['Input_Symbol_real_X']
        A = np.array(A)
        x_test_real_1 = A
        A = loadmat(data_test)['Input_Symbol_image_X']
        A = np.array(A)
        x_test_image_1 = A

        A = loadmat(data_test)['Transmit_real_X']
        A = np.array(A)
        y_test_real_1 = A
        A = loadmat(data_test)['Transmit_image_X']
        A = np.array(A)
        y_test_image_1 = A

        # y_test_Complex = y_test_real_1 + 1j * y_test_image_1
        # y_test_1 = np.c_[y_test_Complex.real, y_test_Complex.imag]

        ## Test Power 2
        #x_X_pol_Real = np.loadtxt(
        #    fname="C:/MASc/Learned_LDBP/New_Data/2_Dataset_x_X_pol_Real.txt")[:Data_M]
        #x_X_pol_Image = np.loadtxt(
        #    fname="C:/MASc/Learned_LDBP/New_Data/2_Dataset_x_X_pol_Imag.txt")[:Data_M]
        #y_X_pol_Real = np.loadtxt(
        #    fname="C:/MASc/Learned_LDBP/New_Data/2_Dataset_y_X_pol_Real.txt")[:Data_N]
        #y_X_pol_Image = np.loadtxt(
        #    fname="C:/MASc/Learned_LDBP/New_Data/2_Dataset_y_X_pol_Imag.txt")[:Data_N]

        #x_test_real_2 = np.reshape(y_X_pol_Real, (-1, N))  ## Valid and test not yet done
        #x_valid_real_2 = np.reshape(y_X_pol_Real, (-1, N))
        #x_test_image_2 = np.reshape(y_X_pol_Image, (-1, N))  ## Valid and test not yet done
        #x_valid_image_2 = np.reshape(y_X_pol_Image, (-1, N))

        #y_test_real_2 = np.reshape(x_X_pol_Real, (-1, M))  ## Valid and test not yet done
        #y_valid_real_2 = np.reshape(x_X_pol_Real, (-1, M))
        #y_test_image_2 = np.reshape(x_X_pol_Image, (-1, M))  ## Valid and test not yet done
        #y_valid_image_2 = np.reshape(x_X_pol_Image, (-1, M))

        ## Test Power 3
        #x_X_pol_Real = np.loadtxt(
        #    fname="C:/MASc/Learned_LDBP/New_Data/3_Dataset_x_X_pol_Real.txt")[:Data_M]
        #x_X_pol_Image = np.loadtxt(
        #    fname="C:/MASc/Learned_LDBP/New_Data/3_Dataset_x_X_pol_Imag.txt")[:Data_M]
        #y_X_pol_Real = np.loadtxt(
        #    fname="C:/MASc/Learned_LDBP/New_Data/3_Dataset_y_X_pol_Real.txt")[:Data_N]
        #y_X_pol_Image = np.loadtxt(
        #    fname="C:/MASc/Learned_LDBP/New_Data/3_Dataset_y_X_pol_Imag.txt")[:Data_N]

        #x_test_real_3 = np.reshape(y_X_pol_Real, (-1, N))  ## Valid and test not yet done
        #x_valid_real_3 = np.reshape(y_X_pol_Real, (-1, N))
        #x_test_image_3 = np.reshape(y_X_pol_Image, (-1, N))  ## Valid and test not yet done
        #x_valid_image_3 = np.reshape(y_X_pol_Image, (-1, N))

        #y_test_real_3 = np.reshape(x_X_pol_Real, (-1, M))  ## Valid and test not yet done
        #y_valid_real_3 = np.reshape(x_X_pol_Real, (-1, M))
        #y_test_image_3 = np.reshape(x_X_pol_Image, (-1, M))  ## Valid and test not yet done
        #y_valid_image_3 = np.reshape(x_X_pol_Image, (-1, M))

        ## Test Power 4
        #x_X_pol_Real = np.loadtxt(
        #    fname="C:/MASc/Learned_LDBP/New_Data/4_Dataset_x_X_pol_Real.txt")[:Data_M]
        #x_X_pol_Image = np.loadtxt(
        #    fname="C:/MASc/Learned_LDBP/New_Data/4_Dataset_x_X_pol_Imag.txt")[:Data_M]
        #y_X_pol_Real = np.loadtxt(
        #    fname="C:/MASc/Learned_LDBP/New_Data/4_Dataset_y_X_pol_Real.txt")[:Data_N]
        #y_X_pol_Image = np.loadtxt(
        #    fname="C:/MASc/Learned_LDBP/New_Data/4_Dataset_y_X_pol_Imag.txt")[:Data_N]

        #x_test_real_4 = np.reshape(y_X_pol_Real, (-1, N))  ## Valid and test not yet done
        #x_valid_real_4 = np.reshape(y_X_pol_Real, (-1, N))
        #x_test_image_4 = np.reshape(y_X_pol_Image, (-1, N))  ## Valid and test not yet done
        #x_valid_image_4 = np.reshape(y_X_pol_Image, (-1, N))

        #y_test_real_4 = np.reshape(x_X_pol_Real, (-1, M))  ## Valid and test not yet done
        #y_valid_real_4 = np.reshape(x_X_pol_Real, (-1, M))
        #y_test_image_4 = np.reshape(x_X_pol_Image, (-1, M))  ## Valid and test not yet done
        #y_valid_image_4 = np.reshape(x_X_pol_Image, (-1, M))

        ## Test Power 5
        #x_X_pol_Real = np.loadtxt(
        #    fname="C:/MASc/Learned_LDBP/New_Data/5_Dataset_x_X_pol_Real.txt")[:Data_M]
        #x_X_pol_Image = np.loadtxt(
        #    fname="C:/MASc/Learned_LDBP/New_Data/5_Dataset_x_X_pol_Imag.txt")[:Data_M]
        #y_X_pol_Real = np.loadtxt(
        #    fname="C:/MASc/Learned_LDBP/New_Data/5_Dataset_y_X_pol_Real.txt")[:Data_N]
        #y_X_pol_Image = np.loadtxt(
        #    fname="C:/MASc/Learned_LDBP/New_Data/5_Dataset_y_X_pol_Imag.txt")[:Data_N]

        #x_test_real_5 = np.reshape(y_X_pol_Real, (-1, N))  ## Valid and test not yet done
        #x_valid_real_5 = np.reshape(y_X_pol_Real, (-1, N))
        #x_test_image_5 = np.reshape(y_X_pol_Image, (-1, N))  ## Valid and test not yet done
        #x_valid_image_5 = np.reshape(y_X_pol_Image, (-1, N))

        #y_test_real_5 = np.reshape(x_X_pol_Real, (-1, M))  ## Valid and test not yet done
        #y_valid_real_5 = np.reshape(x_X_pol_Real, (-1, M))
        #y_test_image_5 = np.reshape(x_X_pol_Image, (-1, M))  ## Valid and test not yet done
        #y_valid_image_5 = np.reshape(x_X_pol_Image, (-1, M))

        return {
                'X_train_real': x_train_real,
                'X_train_image': x_train_image,
                'y_train_real': y_train_real,
                'y_train_image': y_train_image,

                'X_test_real_1': x_test_real_1,
                'X_test_image_1': x_test_image_1,
                'y_test_real_1': y_test_real_1,
                'y_test_image_1': y_test_image_1
                #'X_valid_real_1': x_valid_real_1,
                #'X_valid_image_1': x_valid_image_1,
                #'y_valid_real_1': y_valid_real_1,
                #'y_valid_image_1': y_valid_image_1,

                #'X_test_real_2': x_test_real_2,
                #'X_test_image_2': x_test_image_2,
                #'y_test_real_2': y_test_real_2,
                #'y_test_image_2': y_test_image_2,
                #'X_valid_real_2': x_valid_real_2,
                #'X_valid_image_2': x_valid_image_2,
                #'y_valid_real_2': y_valid_real_2,
                #'y_valid_image_2': y_valid_image_2,

                #'X_test_real_3': x_test_real_3,
                #'X_test_image_3': x_test_image_3,
                #'y_test_real_3': y_test_real_3,
                #'y_test_image_3': y_test_image_3,
                #'X_valid_real_3': x_valid_real_3,
                #'X_valid_image_3': x_valid_image_3,
                #'y_valid_real_3': y_valid_real_3,
                #'y_valid_image_3': y_valid_image_3,

                #'X_test_real_4': x_test_real_4,
                #'X_test_image_4': x_test_image_4,
                #'y_test_real_4': y_test_real_4,
                #'y_test_image_4': y_test_image_4,
                #'X_valid_real_4': x_valid_real_4,
                #'X_valid_image_4': x_valid_image_4,
                #'y_valid_real_4': y_valid_real_4,
                #'y_valid_image_4': y_valid_image_4,

                #'X_test_real_5': x_test_real_5,
                #'X_test_image_5': x_test_image_5,
                #'y_test_real_5': y_test_real_5,
                #'y_test_image_5': y_test_image_5,
                #'X_valid_real_5': x_valid_real_5,
                #'X_valid_image_5': x_valid_image_5,
                #'y_valid_real_5': y_valid_real_5,
                #'y_valid_image_5': y_valid_image_5
                }

    @staticmethod
    def shuffle_batch(X_real, y_real, X_image, y_image, batch_size):
        rnd_idx = np.random.permutation(len(X_real))
        n_batches = len(X_real) // batch_size
        for batch_idx in np.array_split(rnd_idx, n_batches):
            X_batch_real, y_batch_real = X_real[batch_idx], y_real[batch_idx]
            X_batch_image, y_batch_image = X_image[batch_idx], y_image[batch_idx]
            yield X_batch_real, y_batch_real, X_batch_image, y_batch_image
