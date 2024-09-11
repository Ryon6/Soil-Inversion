# TODO: 提高效率
import numpy as np
import pywt
from load_data import load_mining_region_data
from feature_engineering import delete_all_zero, first_order_differential
import matplotlib.pyplot as plt


def wavelet_denoising(img, wavelet='db4', level=3):
    """
    使用小波变换对高光谱图像进行降噪

    Args:
        img: 高光谱图像数据，形状为(波段数, 高, 宽)
        wavelet: 小波基函数，默认为'db4'
        level: 分解层数

    Returns:
        降噪后的高光谱图像
    """

    # 获取图像尺寸
    bands, height, width = img.shape

    # 初始化降噪后的图像
    denoised_img = np.zeros_like(img)

    # 对每个像素点对应的光谱进行降噪
    for i in range(height):
        for j in range(width):
            # 提取当前像素点的光谱曲线
            coeffs = pywt.wavedec(img[:, i, j], wavelet, level=level)

            # 设置阈值（这里使用软阈值，你可以根据实际情况调整）
            threshold = np.median(np.abs(coeffs[-level])) / 0.6745

            # 对细节系数进行阈值处理
            for k in range(1, len(coeffs)):
                coeffs[k] = pywt.threshold(coeffs[k], threshold, mode='soft')

            # 重构信号
            denoised_signal = pywt.waverec(coeffs, wavelet)

            # 将降噪后的信号保存到输出图像中
            denoised_img[:, i, j] = denoised_signal

    return denoised_img


def main():
    img_array, samples_spectral, zn_content, som_content, wavelengths = load_mining_region_data(need_wavelengths=True)
    samples_spectral = samples_spectral.T
    y = zn_content

    # 异常值去除
    img_array, samples_spectral, wavelengths = delete_all_zero(img_array, samples_spectral, wavelengths)

    # 光谱微分变换
    samples_spectral = first_order_differential(samples_spectral, wavelengths, axis=1)
    img_array = first_order_differential(img_array, wavelengths, axis=0)

    reconstructed_image = wavelet_denoising(img_array, wavelet='db4', level=4)

    # 提取原始和重构的光谱曲线
    original_spectra = img_array[:, 1, 1]
    reconstructed_spectra = reconstructed_image[:, 1, 1]
    plt.plot(original_spectra)
    plt.plot(reconstructed_spectra)
    plt.show()

    # # 绘制光谱曲线图
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.plot(original_spectra)
    # # plt.imshow(original_spectra, cmap='gray')
    # plt.title('Original Spectra')
    # plt.axis('off')
    #
    # plt.subplot(1, 2, 2)
    # plt.plot(reconstructed_spectra)
    # # plt.imshow(reconstructed_spectra, cmap='gray')
    # plt.title('Reconstructed Spectra')
    # plt.axis('off')
    #
    # plt.tight_layout()
    # plt.show()



if __name__ == '__main__':
    main()