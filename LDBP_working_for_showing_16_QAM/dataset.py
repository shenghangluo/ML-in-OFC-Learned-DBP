import numpy as np
from scipy.io import loadmat
N = 512
M = 256
Usesamples = 8
Data_N = pow(2, 14)
Data_M = pow(2, 13)
datapwd = "C:/MASc/Learned_LDBP/Update_code/Dataset/Tensorflowdata_1.mat"

class DatasetMNIST(object):
    def __init__(self, val_size):
        self._val_size = val_size

    def load_data(self):
        """
        :return:
        """
        # Load the Dataset
        x_X_pol_Real = np.loadtxt(
            fname="C:/MASc/Learned_LDBP/New_Data/Dataset_x_X_pol_Real.txt")[:Data_M]
        x_X_pol_Image = np.loadtxt(
            fname="C:/MASc/Learned_LDBP/New_Data/Dataset_x_X_pol_Imag.txt")[:Data_M]
        y_X_pol_Real = np.loadtxt(
            fname="C:/MASc/Learned_LDBP/New_Data/Dataset_y_X_pol_Real.txt")[:Data_N]
        y_X_pol_Image = np.loadtxt(
            fname="C:/MASc/Learned_LDBP/New_Data/Dataset_y_X_pol_Imag.txt")[:Data_N]

        y_train_real = np.reshape(x_X_pol_Real, (-1, M))
        y_train_image = np.reshape(x_X_pol_Image, (-1, M))
        x_train_real = np.reshape(y_X_pol_Real, (-1, N))
        x_train_image = np.reshape(y_X_pol_Image, (-1, N))

        # # Test Power 1
        # x_X_pol_Real = np.loadtxt(
        #     fname="C:/MASc/Learned_LDBP/New_Data/1_Dataset_x_X_pol_Real.txt")[:Data_M]
        # x_X_pol_Image = np.loadtxt(
        #     fname="C:/MASc/Learned_LDBP/New_Data/1_Dataset_x_X_pol_Imag.txt")[:Data_M]
        # y_X_pol_Real = np.loadtxt(
        #     fname="C:/MASc/Learned_LDBP/New_Data/1_Dataset_y_X_pol_Real.txt")[:Data_N]
        # y_X_pol_Image = np.loadtxt(
        #     fname="C:/MASc/Learned_LDBP/New_Data/1_Dataset_y_X_pol_Imag.txt")[:Data_N]
        #
        # x_test_real_1 = np.reshape(y_X_pol_Real, (-1, N))  ## Valid and test not yet done
        # x_valid_real_1 = np.reshape(y_X_pol_Real, (-1, N))
        # x_test_image_1 = np.reshape(y_X_pol_Image, (-1, N))  ## Valid and test not yet done
        # x_valid_image_1 = np.reshape(y_X_pol_Image, (-1, N))
        #
        # y_test_real_1 = np.reshape(x_X_pol_Real, (-1, M))  ## Valid and test not yet done
        # y_valid_real_1 = np.reshape(x_X_pol_Real, (-1, M))
        # y_test_image_1 = np.reshape(x_X_pol_Image, (-1, M))  ## Valid and test not yet done
        # y_valid_image_1 = np.reshape(x_X_pol_Image, (-1, M))
        #
        # # Test Power 2
        # x_X_pol_Real = np.loadtxt(
        #     fname="C:/MASc/Learned_LDBP/New_Data/2_Dataset_x_X_pol_Real.txt")[:Data_M]
        # x_X_pol_Image = np.loadtxt(
        #     fname="C:/MASc/Learned_LDBP/New_Data/2_Dataset_x_X_pol_Imag.txt")[:Data_M]
        # y_X_pol_Real = np.loadtxt(
        #     fname="C:/MASc/Learned_LDBP/New_Data/2_Dataset_y_X_pol_Real.txt")[:Data_N]
        # y_X_pol_Image = np.loadtxt(
        #     fname="C:/MASc/Learned_LDBP/New_Data/2_Dataset_y_X_pol_Imag.txt")[:Data_N]
        #
        # x_test_real_2 = np.reshape(y_X_pol_Real, (-1, N))  ## Valid and test not yet done
        # x_valid_real_2 = np.reshape(y_X_pol_Real, (-1, N))
        # x_test_image_2 = np.reshape(y_X_pol_Image, (-1, N))  ## Valid and test not yet done
        # x_valid_image_2 = np.reshape(y_X_pol_Image, (-1, N))
        #
        # y_test_real_2 = np.reshape(x_X_pol_Real, (-1, M))  ## Valid and test not yet done
        # y_valid_real_2 = np.reshape(x_X_pol_Real, (-1, M))
        # y_test_image_2 = np.reshape(x_X_pol_Image, (-1, M))  ## Valid and test not yet done
        # y_valid_image_2 = np.reshape(x_X_pol_Image, (-1, M))
        #
        # # Test Power 3
        # x_X_pol_Real = np.loadtxt(
        #     fname="C:/MASc/Learned_LDBP/New_Data/3_Dataset_x_X_pol_Real.txt")[:Data_M]
        # x_X_pol_Image = np.loadtxt(
        #     fname="C:/MASc/Learned_LDBP/New_Data/3_Dataset_x_X_pol_Imag.txt")[:Data_M]
        # y_X_pol_Real = np.loadtxt(
        #     fname="C:/MASc/Learned_LDBP/New_Data/3_Dataset_y_X_pol_Real.txt")[:Data_N]
        # y_X_pol_Image = np.loadtxt(
        #     fname="C:/MASc/Learned_LDBP/New_Data/3_Dataset_y_X_pol_Imag.txt")[:Data_N]
        #
        # x_test_real_3 = np.reshape(y_X_pol_Real, (-1, N))  ## Valid and test not yet done
        # x_valid_real_3 = np.reshape(y_X_pol_Real, (-1, N))
        # x_test_image_3 = np.reshape(y_X_pol_Image, (-1, N))  ## Valid and test not yet done
        # x_valid_image_3 = np.reshape(y_X_pol_Image, (-1, N))
        #
        # y_test_real_3 = np.reshape(x_X_pol_Real, (-1, M))  ## Valid and test not yet done
        # y_valid_real_3 = np.reshape(x_X_pol_Real, (-1, M))
        # y_test_image_3 = np.reshape(x_X_pol_Image, (-1, M))  ## Valid and test not yet done
        # y_valid_image_3 = np.reshape(x_X_pol_Image, (-1, M))
        #
        # # Test Power 4
        # x_X_pol_Real = np.loadtxt(
        #     fname="C:/MASc/Learned_LDBP/New_Data/4_Dataset_x_X_pol_Real.txt")[:Data_M]
        # x_X_pol_Image = np.loadtxt(
        #     fname="C:/MASc/Learned_LDBP/New_Data/4_Dataset_x_X_pol_Imag.txt")[:Data_M]
        # y_X_pol_Real = np.loadtxt(
        #     fname="C:/MASc/Learned_LDBP/New_Data/4_Dataset_y_X_pol_Real.txt")[:Data_N]
        # y_X_pol_Image = np.loadtxt(
        #     fname="C:/MASc/Learned_LDBP/New_Data/4_Dataset_y_X_pol_Imag.txt")[:Data_N]
        #
        # x_test_real_4 = np.reshape(y_X_pol_Real, (-1, N))  ## Valid and test not yet done
        # x_valid_real_4 = np.reshape(y_X_pol_Real, (-1, N))
        # x_test_image_4 = np.reshape(y_X_pol_Image, (-1, N))  ## Valid and test not yet done
        # x_valid_image_4 = np.reshape(y_X_pol_Image, (-1, N))
        #
        # y_test_real_4 = np.reshape(x_X_pol_Real, (-1, M))  ## Valid and test not yet done
        # y_valid_real_4 = np.reshape(x_X_pol_Real, (-1, M))
        # y_test_image_4 = np.reshape(x_X_pol_Image, (-1, M))  ## Valid and test not yet done
        # y_valid_image_4 = np.reshape(x_X_pol_Image, (-1, M))
        #
        # # Test Power 5
        # x_X_pol_Real = np.loadtxt(
        #     fname="C:/MASc/Learned_LDBP/New_Data/5_Dataset_x_X_pol_Real.txt")[:Data_M]
        # x_X_pol_Image = np.loadtxt(
        #     fname="C:/MASc/Learned_LDBP/New_Data/5_Dataset_x_X_pol_Imag.txt")[:Data_M]
        # y_X_pol_Real = np.loadtxt(
        #     fname="C:/MASc/Learned_LDBP/New_Data/5_Dataset_y_X_pol_Real.txt")[:Data_N]
        # y_X_pol_Image = np.loadtxt(
        #     fname="C:/MASc/Learned_LDBP/New_Data/5_Dataset_y_X_pol_Imag.txt")[:Data_N]
        #
        # x_test_real_5 = np.reshape(y_X_pol_Real, (-1, N))  ## Valid and test not yet done
        # x_valid_real_5 = np.reshape(y_X_pol_Real, (-1, N))
        # x_test_image_5 = np.reshape(y_X_pol_Image, (-1, N))  ## Valid and test not yet done
        # x_valid_image_5 = np.reshape(y_X_pol_Image, (-1, N))
        #
        # y_test_real_5 = np.reshape(x_X_pol_Real, (-1, M))  ## Valid and test not yet done
        # y_valid_real_5 = np.reshape(x_X_pol_Real, (-1, M))
        # y_test_image_5 = np.reshape(x_X_pol_Image, (-1, M))  ## Valid and test not yet done
        # y_valid_image_5 = np.reshape(x_X_pol_Image, (-1, M))

        return {
                'X_train_real': x_train_real,
                'X_train_image': x_train_image,
                'y_train_real': y_train_real,
                'y_train_image': y_train_image,

                # 'X_test_real_1': x_test_real_1,
                # 'X_test_image_1': x_test_image_1,
                # 'y_test_real_1': y_test_real_1,
                # 'y_test_image_1': y_test_image_1,
                # 'X_valid_real_1': x_valid_real_1,
                # 'X_valid_image_1': x_valid_image_1,
                # 'y_valid_real_1': y_valid_real_1,
                # 'y_valid_image_1': y_valid_image_1,
                #
                # 'X_test_real_2': x_test_real_2,
                # 'X_test_image_2': x_test_image_2,
                # 'y_test_real_2': y_test_real_2,
                # 'y_test_image_2': y_test_image_2,
                # 'X_valid_real_2': x_valid_real_2,
                # 'X_valid_image_2': x_valid_image_2,
                # 'y_valid_real_2': y_valid_real_2,
                # 'y_valid_image_2': y_valid_image_2,
                #
                # 'X_test_real_3': x_test_real_3,
                # 'X_test_image_3': x_test_image_3,
                # 'y_test_real_3': y_test_real_3,
                # 'y_test_image_3': y_test_image_3,
                # 'X_valid_real_3': x_valid_real_3,
                # 'X_valid_image_3': x_valid_image_3,
                # 'y_valid_real_3': y_valid_real_3,
                # 'y_valid_image_3': y_valid_image_3,
                #
                # 'X_test_real_4': x_test_real_4,
                # 'X_test_image_4': x_test_image_4,
                # 'y_test_real_4': y_test_real_4,
                # 'y_test_image_4': y_test_image_4,
                # 'X_valid_real_4': x_valid_real_4,
                # 'X_valid_image_4': x_valid_image_4,
                # 'y_valid_real_4': y_valid_real_4,
                # 'y_valid_image_4': y_valid_image_4,
                #
                # 'X_test_real_5': x_test_real_5,
                # 'X_test_image_5': x_test_image_5,
                # 'y_test_real_5': y_test_real_5,
                # 'y_test_image_5': y_test_image_5,
                # 'X_valid_real_5': x_valid_real_5,
                # 'X_valid_image_5': x_valid_image_5,
                # 'y_valid_real_5': y_valid_real_5,
                # 'y_valid_image_5': y_valid_image_5
                }

    @staticmethod
    def shuffle_batch(X_real, y_real, X_image, y_image, batch_size):
        rnd_idx = np.random.permutation(np.arange(0, len(X_real), batch_size))
        #print("rnd_idx", rnd_idx)
        for batch_idx in rnd_idx:
            #print("batch_idx", batch_idx)
            X_batch_real, y_batch_real = X_real[batch_idx:batch_idx + batch_size], y_real[(batch_idx * Usesamples / batch_size).astype('int32'):(batch_idx * Usesamples / batch_size + Usesamples).astype('int32')]
            #print("X_batch_real shape", X_batch_real.shape)
            #print("y_batch_real shape", y_batch_real.shape)
            X_batch_image, y_batch_image = X_image[batch_idx:batch_idx + batch_size], y_image[(batch_idx * Usesamples / batch_size).astype('int32'):(batch_idx * Usesamples / batch_size + Usesamples).astype('int32')]
            yield X_batch_real, y_batch_real, X_batch_image, y_batch_image
