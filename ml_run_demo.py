from load_data import load_cultivated_land_data, load_mining_region_data
from model.machine_learning import ml_model_test


if __name__ == '__main__':
    img_array, samples_spectral, zn_content, som_content = load_mining_region_data()
    samples_spectral = samples_spectral.T
    ml_model_test(samples_spectral, som_content, plot=True)
