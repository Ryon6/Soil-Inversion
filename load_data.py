"""
TODO: int8 -> int32
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
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import spectral


def display_and_save_bands(img_array, output_folder="output"):
    """
    显示并保存高光谱图像的每个波段灰度图

    Args:
        img_array: 高光谱图像数据，形状为(波段数, 高, 宽)
        output_folder: 输出文件夹路径
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i in range(img_array.shape[0]):
        band_img = img_array[i, :, :]
        plt.imshow(band_img, cmap='gray')
        plt.title(f"band{i + 1}")
        plt.axis('off')
        plt.savefig(os.path.join(output_folder, f"band_{i + 1}.png"), dpi=300)
        plt.close()


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


def delete_all_zero(img_array, samples_spectral, wavelengths):
    # 检查全为0的波段，并剔除
    zero_bands = []
    for i in range(img_array.shape[0]):
        if np.all(img_array[i] == 0):
            zero_bands.append(i)
    img_array = np.delete(img_array, zero_bands, axis=0)

    samples_spectral = np.delete(samples_spectral, zero_bands, axis=0)
    wavelengths = np.delete(wavelengths, zero_bands)
    return img_array, samples_spectral, wavelengths


def load_mining_region_data(plot=False, need_wavelengths=False):
    file_name = 'data/mining region/'
    # 读取高光谱影像
    with rasterio.open(file_name + 'GF-5_image.tif') as src:
        img_array = src.read()  # 读取所有波段的数据

    if plot:
        plot_false_color(img_array)

    # 读取土壤样本数据
    df = pd.read_excel(file_name + 'soil_samples.xlsx')

    # 获取土壤样本在影像中的位置
    row_indices = np.int32(df.iloc[3, 1:])
    col_indices = np.int32(df.iloc[4, 1:])
    zn_content = np.array(df.iloc[1, 1:])
    som_content = np.array(df.iloc[2, 1:])

    # 提取土壤样本对应的光谱数据
    samples_spectral = img_array[:, row_indices, col_indices]

    # 读取波长信息
    wavelengths = []
    with open(file_name + '土壤赛道-矿区数据集说明.txt', 'r') as f:
        # 跳过前12行
        for _ in range(12):
            next(f)
        # 逐行读取，提取波长值
        for line in f:
            # 去除行尾的换行符
            line = line.strip()
            # 分割行，获取波段号和波长值
            band, wavelength = line.split(',')
            # 将波长值转换为浮点数，并添加到列表中
            wavelengths.append(float(wavelength))
    wavelengths = np.array(wavelengths)

    img_array = img_array.astype(np.float64)
    samples_spectral = samples_spectral.astype(np.float64)
    # 检查全为0的波段，并剔除
    img_array, samples_spectral, wavelengths = delete_all_zero(img_array, samples_spectral, wavelengths)
    if need_wavelengths:
        return img_array, samples_spectral, zn_content, som_content, wavelengths
    else:
        return img_array, samples_spectral, zn_content, som_content


def load_cultivated_land_data(plot=False, need_wavelengths=False):
    file_name = 'data/cultivated land/'
    # 读取高光谱影像
    img = spectral.envi.open(file_name + 'Imagedata.hdr', file_name + 'Imagedata.img')
    img_array = np.array(img.load())
    wavelengths = np.array(img.bands.centers)

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

    img_array = img_array.astype(np.float64)
    samples_spectral = samples_spectral.astype(np.float64)
    # 检查全为0的波段，并剔除
    img_array, samples_spectral, wavelengths = delete_all_zero(img_array, samples_spectral, wavelengths)
    if need_wavelengths:
        return img_array, samples_spectral, salt_content, som_content, wavelengths
    else:
        return img_array, samples_spectral, salt_content, som_content


def main():
    # 加载数据
    img_array, samples_spectral, salt_content, som_content, wavelengths = load_mining_region_data(
        need_wavelengths=True)

    # 进行转置操作
    img_array = np.transpose(img_array, (2, 0, 1))
    print(img_array.shape)
    # 显示并保存
    display_and_save_bands(img_array)


if __name__ == '__main__':
    main()
