import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# from torchgeo.models import dofa_base_patch16_224
# from torchgeo.models.dofa import DOFABase16_Weights
from load_data import load_mining_region_data


def main():
    img_array, samples_spectral, zn_content, som_content, wavelengths = load_mining_region_data(need_wavelengths=True)
    X = samples_spectral.T
    y = np.float32(som_content)

    # # 光谱微分变换
    # X = first_order_differential(X, wavelengths, axis=1)
    # img_array = first_order_differential(img_array, wavelengths, axis=0)
    #
    # # 离散小波变换
    # X = wavelet_denoising(X.T, 'db4', 5).T
    # img_array = wavelet_denoising(img_array, 'db4', 5)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    band = X.shape[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.tensor(X, dtype=torch.float32, device=device)
    y = torch.tensor(y, dtype=torch.float32, device=device)

    # 将y reshape为(batch, 1)
    y = y.reshape(-1, 1)

    # 将数据分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 将数据调整为ViT的输入形状
    X_train = X_train.reshape(-1, band, 1)
    X_test = X_test.reshape(-1, band, 1)

    # model = ViT(
    #     image_size=1,
    #     near_band=1,
    #     num_patches=band,
    #     num_classes=1,
    #     dim=64,
    #     depth=1,
    #     heads=4,
    #     mlp_dim=16,
    #     dropout=0.1,
    #     emb_dropout=0.1,
    #     mode='CAF'
    # )
    # 实例化模型
    hidden_size = 64
    num_layers = 2
    output_size = 1
    model = BiLSTMRegressor(input_size=band, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)

    model = model.to(device)

    # 训练模型
    num_epochs = 1000
    batch_size = X_train.shape[0]  # 设置batch_size为训练集样本总数

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs // 100, gamma=0.9)

    for epoch in range(num_epochs):
        scheduler.step()

        model.train()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 每10个epoch打印一次评估指标
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                outputs = model(X_test)
                mse = mean_squared_error(y_test.cpu().numpy(), outputs.cpu().numpy())
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test.cpu().numpy(), outputs.cpu().numpy())
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R^2: {r2:.4f}')

    # 在测试集上进行测试
    model.eval()
    with torch.no_grad():
        outputs = model(X_test).cpu()
        y_test = y_test.cpu()
        print(f'Test MSE: {mean_squared_error(y_test.numpy(), outputs.numpy()):.4f}')
        print(f'Test RMSE: {np.sqrt(mean_squared_error(y_test.numpy(), outputs.numpy())):.4f}')
        print(f'Test R^2: {r2_score(y_test.numpy(), outputs.numpy()):.4f}')


if __name__ == '__main__':
    main()


def Test():
    pass
    # SpectalGPT
    # check_point = torch.load('pretrain_weights/SpectralGPT+.pth')
    # model = vit_base_patch16()
    # checkpoint = model.load_state_dict(check_point, strict=False)
    # checkpoint_model = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    # # model.load_state_dict(torch.load(weights_path, map_location=device)['model'])
    # model.load_state_dict(checkpoint_model, strict=False)
    # model.cuda()
    # msg = model.load_state_dict(checkpoint_model, strict=False)
    # print(msg)
    #
    # input1 = torch.rand(2, 12, 128, 128).cuda()
    # input2 = torch.rand(2, 12, 128, 128).cuda()
    # output = model(input1, input2)
    # print((output.shape))

    # missing_keys=['fc_norm.weight', 'fc_norm.bias', 'head.weight', 'head.bias'], unexpected_keys=['mask_token', 'norm.weight', 'norm.bias', 'projector.weight', 'projector.bias']

    # DOFA
    # check_point = torch.load('pretrain_weights/DOFA_ViT_base_e120_full_weight.pth')
    # vit_model = vit_base_patch16()
    # msg = vit_model.load_state_dict(check_point, strict=False)
    # # missing_keys=['fc_norm.weight', 'fc_norm.bias', 'head.weight', 'head.bias'], unexpected_keys=['mask_token', 'norm.weight', 'norm.bias', 'projector.weight', 'projector.bias']
    # vit_model = vit_model.cuda()
    # x = torch.rand([1, 2, 16, 16]).cuda()
    # wavelengths = [3.75, 3.75]
    # output = vit_model.forward_features(x, wavelengths)
    # output = vit_model.forward(x, wavelengths)
    # print(output.shape)
