
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
# matplotlib.use('Agg')

def plot_3Ddata(data):
    #plot one mat data depended on values on time  and locations
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = Axes3D(fig)
    X = [X for X in range(data.shape[0])]
    Y = [Y for Y in range(data.shape[1])]
    X, Y = np.meshgrid(X, Y)

    Z = data[X, Y, :]
    ax.plot_surface(X, Y, Z,cmap = 'rainbow')
    plt.axis('off')
    plt.savefig('defect/output/N1_3D.png')


def plot_curve(mat_path):
    '''
    :param mat_path: path to read a mat file
    :return: draw Temperature Change Curve for some points that you listed in this function
    '''
    data_struct = sio.loadmat(mat_path)  #if 195t.mat
    data = data_struct['data']
    x = [x for x in range(data.shape[2])]
    y1 = data[197, 151, x]
    y2 = data[209, 149, x]
    y3 = data[212, 151, x]
    y4 = data[251, 149, x]
    y5 = data[251, 180, x]
    plt.figure()
    plt.plot(x, y1, color='dodgerblue', label='Line1')
    plt.plot(x, y2, color='orangered', label='Line2')
    plt.plot(x, y3, color='orange', label='Line3')
    plt.plot(x, y4, color='mediumorchid', label='Line4')
    plt.plot(x, y5, color='limegreen', label='Line5')

    plt.ylim((24, 36))
    plt.xlim((0, 200))
    y_ticks = np.linspace(24, 36, num=6)
    x_ticks = np.linspace(0, 200, num=5)
    plt.yticks(y_ticks)
    plt.xticks(x_ticks)
    plt.legend()
    plt.xlabel('Frame')
    plt.ylabel('Centigrade Temperature(℃)')
    plt.title('Temperature Change Curve')
    plt.show()
    plt.savefig('./Line.png')
    return

def plot_curve(mat_path):
    '''
    :param mat_path: path to read a mat file
    :return: draw Temperature Change Curve for some points that you listed in this function
    '''
    data_struct = sio.loadmat(mat_path)  #if 195t.mat
    data = data_struct['data']
    x = [x for x in range(data.shape[2])]
    y1 = data[140, 51, x] / 10
    # y2 = data[209, 149, x]
    # y3 = data[212, 151, x]
    # y4 = data[251, 149, x]
    y5 = data[145, 57, x] / 10
    plt.figure()
    plt.plot(x, y1, color='dodgerblue', label='defect edge point')
    plt.plot(x, y5, color='purple', label='thermal diffusion point')

    # plt.plot(x, y1, color='dodgerblue', label='Line1')
    # plt.plot(x, y2, color='orangered', label='Line2')
    # plt.plot(x, y3, color='orange', label='Line3')
    # plt.plot(x, y4, color='mediumorchid', label='Line4')
    # plt.plot(x, y5, color='limegreen', label='Line5')

    # plt.ylim((24, 36))
    plt.xlim((0, 180))
    # y_ticks = np.linspace(24, 36, num=6)
    # x_ticks = np.linspace(0, 200, num=5)
    # plt.yticks(y_ticks)
    # plt.xticks(x_ticks)
    plt.legend()
    plt.xlabel('Frame')
    plt.ylabel('Centigrade Temperature(℃)')
    # plt.title('Temperature Change Curve')
    # plt.show()
    plt.savefig('defect/output/Line_2.png')
    return


# plot_curve('defect/data/mat/temperature/N1.mat')
file_path = 'defect/data/mat/temperature/N1.mat'
plot_curve(file_path)
# data_struct = sio.loadmat(file_path)
# data = data_struct['data']
# t_len = data.shape[2]
# print(t_len)
# plt.imshow(data[:, :, 70], cmap='gray')
# plt.show()
# for i in range(t_len):
#     img = data[:, :, i]
#     plt.imshow(img, cmap='gray')
#     plt.axis('off')
#     plt.savefig('defect/output/N1/N1_{}.png'.format(i))

