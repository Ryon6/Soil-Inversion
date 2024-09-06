"""
data目录树如下，请根据该目录树自行修改文件名：
data
    cultivated land
        Imagedata.hdr
        Imagedata.img
        化验数据及对应光谱数据.xlsx
        说明文档.txt
    mining region
        GF-5_image.tif
        soil_samples.xlsx
        土壤赛道-矿区数据集说明.txt
"""
import rasterio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_false_color(img_array, nir=100, r=50, g=30):
    # 选择三个波段（例如，近红外、红、绿）
    band_nir = img_array[nir, :, :]
    band_red = img_array[r, :, :]
    band_green = img_array[g, :, :]

    # 归一化
    band_nir = (band_nir - band_nir.min()) / (band_nir.max() - band_nir.min())
    band_red = (band_red - band_red.min()) / (band_red.max() - band_red.min())
    band_green = (band_green - band_green.min()) / (band_green.max() - band_green.min())

    # 合并波段
    rgb = np.stack([band_nir, band_red, band_green], axis=2)

    # 显示假彩色图像
    plt.imshow(rgb)
    plt.title('False Color Image')
    plt.show()


def load_tif_data(plot=False):
    file_name = 'data/mining region/'
    # 读取高光谱影像
    with rasterio.open(file_name + 'GF-5_image.tif') as src:
        img_array = src.read()  # 读取所有波段的数据
        img_meta = src.meta  # 获取影像的元数据

    if plot:
        plot_false_color(img_array)

    # 读取土壤样本数据
    df = pd.read_excel(file_name + 'soil_samples.xlsx')

    # 获取土壤样本在影像中的位置
    row_indices = np.int8(df.iloc[3, 1:])
    col_indices = np.int8(df.iloc[4, 1:])
    zn_content = np.array(df.iloc[1, 1:])
    som_content = np.array(df.iloc[2, 1:])

    # 提取土壤样本对应的光谱数据
    # 假设每个像素对应一个样本，且波段顺序与Excel中的波段顺序一致
    samples_spectral = img_array[:, row_indices, col_indices]
    # print(img_array.min(), img_array.max())
    return img_array, samples_spectral, zn_content, som_content


def load_img_data(plot=False):
    file_name = 'data/cultivated land/'
    # 读取高光谱影像
    with rasterio.open(file_name + 'Imagedata.img') as src:
        img_array = src.read()  # 读取所有波段的数据
        img_meta = src.meta  # 获取影像的元数据

    if plot:
        plot_false_color(img_array)

    # 读取Excel数据
    df = pd.read_excel(file_name + '化验数据及对应光谱数据.xlsx')
    salt_content = np.array(df.iloc[:, 2])
    # print(salt_content)
    som_content = np.array(df.iloc[:, 1])
    # print(som_content)
    samples_spectral = np.array(df.iloc[:, 3:])
    # print(samples_spectral)

    # print(img_array.min(), img_array.max())
    return img_array, samples_spectral, salt_content, som_content


if __name__ == '__main__':
    load_tif_data()
