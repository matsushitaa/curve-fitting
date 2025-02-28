import math
import sys

import glob
import math
from PIL import Image
import cv2
from scipy.ndimage import zoom
import openpnm as op
import porespy as ps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('./pages/FVM/0.2.8/')
from solver.conduct import Conduct, trim_nonpercolating_paths_tool
from solver.optstr import calc_grad, top_n_true, arrays_similarity


def main():
    st.header('Prediction Demo', anchor=False)
    if 'state' not in st.session_state:
        st.session_state.state = 'stop'
    if 'opt_results' not in st.session_state:
        st.session_state.opt_results = {}

    # モデルのロード
    st.subheader('Load Model', anchor=False, divider=True)
    model, device = load_model_gui()

    # 構造作成
    st.subheader('Create Structure', anchor=False, divider=True)
    data, wall = create_structure_gui()

    # 計算
    st.subheader('Calculation', anchor=False, divider=True)
    mode = st.radio('mode', ('感度推論', '形状最適化'), horizontal=True)
    if mode=='感度推論':
        view_flux_gui(data, wall, device, model)
    elif mode=='形状最適化':
        opt_gui(data, wall, device, model)


def load_model_gui():
    # モデルの作成
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    architecture = st.selectbox('architecture', ('U-Net','Flexible U-Net','Dynamic U-Net','Dynamic U-Net2'), index=1)
    if architecture=='U-Net':
        model = UNet().to(device)
        file_paths = [fp for fp in glob.glob('./pages/flux/best_unet*.pth') if not ('_f' in fp or '_d' in fp)]
    elif architecture=='Flexible U-Net':
        model = FlexibleUNet().to(device)
        file_paths = glob.glob('./pages/flux/best_unet_f*.pth')
    elif architecture=='Dynamic U-Net':
        model = DynamicUNet().to(device)
        file_paths = glob.glob('./pages/flux/best_unet_d*.pth')
    elif architecture=='Dynamic U-Net2':
        model = DynamicUNet().to(device)
        file_paths = glob.glob('./pages/flux/best_unet_d*.pth')

    file_path = st.selectbox('file_path', file_paths, index=len(file_paths)-1)
    model.load_state_dict(torch.load(file_path, map_location=device, weights_only=True))
    model.eval()  # 推論モードに切り替え
    return model, device


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # エンコーダー部分
        self.enc1 = self.conv_block(1, 64)     # (N, 1, 32, 32) -> (N, 64, 32, 32)
        self.pool1 = nn.MaxPool2d(2)           # (N, 64, 32, 32) -> (N, 64, 16, 16)
        self.enc2 = self.conv_block(64, 128)   # (N, 64, 16, 16) -> (N, 128, 16, 16)
        self.pool2 = nn.MaxPool2d(2)           # (N, 128, 16, 16) -> (N, 128, 8, 8)
        self.enc3 = self.conv_block(128, 256)  # **追加** (N, 128, 8, 8) -> (N, 256, 8, 8)
        self.pool3 = nn.MaxPool2d(2)           # (N, 256, 8, 8) -> (N, 256, 4, 4)

        # ボトム部分
        self.bottleneck = self.conv_block(256, 512)  # (N, 256, 4, 4) -> (N, 512, 4, 4)

        # デコーダー部分
        self.upconv3 = self.upconv_block(512, 256)  # (N, 512, 4, 4) -> (N, 256, 8, 8)
        self.dec3 = self.conv_block(512, 256)       # (N, 512, 8, 8) -> (N, 256, 8, 8)
        self.upconv2 = self.upconv_block(256, 128)  # (N, 256, 8, 8) -> (N, 128, 16, 16)
        self.dec2 = self.conv_block(256, 128)       # (N, 256, 16, 16) -> (N, 128, 16, 16)
        self.upconv1 = self.upconv_block(128, 64)   # (N, 128, 16, 16) -> (N, 64, 32, 32)
        self.dec1 = self.conv_block(128, 64)        # (N, 128, 32, 32) -> (N, 64, 32, 32)

        # 出力層
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)  # (N, 64, 32, 32) -> (N, 1, 32, 32)

    def conv_block(self, in_channels, out_channels):
        """畳み込み + ReLU + バッチ正規化"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def upconv_block(self, in_channels, out_channels):
        """アップサンプリング + 畳み込み"""
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # エンコーダー
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))  # **追加**
        # ボトム
        bottleneck = self.bottleneck(self.pool3(enc3))  # **追加**
        # デコーダー
        dec3 = self.upconv3(bottleneck)  
        dec3 = self.dec3(torch.cat([dec3, enc3], dim=1))  # **追加**
        dec2 = self.upconv2(dec3)
        dec2 = self.dec2(torch.cat([dec2, enc2], dim=1))  # スキップコネクション
        dec1 = self.upconv1(dec2)
        dec1 = self.dec1(torch.cat([dec1, enc1], dim=1))  # スキップコネクション
        # 出力層（非負にする）
        out = self.final_conv(dec1)
        out = torch.relu(out)  # **非負にする**
        return out
    

class FlexibleUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_filters=64):
        super(FlexibleUNet, self).__init__()

        # エンコーダー
        self.enc1 = self.conv_block(in_channels, base_filters)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = self.conv_block(base_filters, base_filters * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = self.conv_block(base_filters * 2, base_filters * 4)
        self.pool3 = nn.MaxPool2d(2)

        # ボトム（ボトルネック）
        self.bottleneck = self.conv_block(base_filters * 4, base_filters * 8)

        # デコーダー
        self.up3 = self.upconv_block(base_filters * 8, base_filters * 4)
        self.dec3 = self.conv_block(base_filters * 8, base_filters * 4)

        self.up2 = self.upconv_block(base_filters * 4, base_filters * 2)
        self.dec2 = self.conv_block(base_filters * 4, base_filters * 2)

        self.up1 = self.upconv_block(base_filters * 2, base_filters)
        self.dec1 = self.conv_block(base_filters * 2, base_filters)

        # 出力層
        self.final_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        """パディングを考慮してサイズを保持する畳み込みブロック"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )

    def upconv_block(self, in_channels, out_channels):
        """アップサンプリング + 畳み込み"""
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # エンコーダー
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))

        # ボトム（ボトルネック）
        bottleneck = self.bottleneck(self.pool3(enc3))

        # デコーダー
        dec3 = self.up3(bottleneck)
        dec3 = F.interpolate(dec3, size=enc3.shape[2:], mode="bilinear", align_corners=False)
        dec3 = self.dec3(torch.cat([dec3, enc3], dim=1))

        dec2 = self.up2(dec3)
        dec2 = F.interpolate(dec2, size=enc2.shape[2:], mode="bilinear", align_corners=False)
        dec2 = self.dec2(torch.cat([dec2, enc2], dim=1))

        dec1 = self.up1(dec2)
        dec1 = F.interpolate(dec1, size=enc1.shape[2:], mode="bilinear", align_corners=False)
        dec1 = self.dec1(torch.cat([dec1, enc1], dim=1))

        # 出力層（ReLU で非負に）
        out = self.final_conv(dec1)
        out = torch.relu(out)

        return out
    

class DynamicUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_filters=64, input_size=64):
        super(DynamicUNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_filters = base_filters
        self.build_network(input_size)  # 初期化時に構築

    def build_network(self, input_size):
        """入力サイズに基づいてエンコーダー・デコーダーを構築"""
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        # エンコーダーの構築
        current_size = input_size
        filters = self.base_filters
        while current_size > 4:
            self.encoders.append(self.conv_block(self.in_channels, filters))
            self.pools.append(nn.MaxPool2d(2))
            self.in_channels = filters
            filters *= 2
            current_size = math.ceil(current_size / 2)

        # ボトルネック
        self.bottleneck = self.conv_block(self.in_channels, filters)

        # デコーダーの構築（エンコーダーの逆順）
        for encoder in reversed(self.encoders):
            filters //= 2
            self.upsamples.append(nn.ConvTranspose2d(filters * 2, filters, kernel_size=2, stride=2))
            self.decoders.append(self.conv_block(filters * 2, filters))

        # 出力層
        self.final_conv = nn.Conv2d(self.base_filters, self.out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        # ネットワーク構築（初回のみ）
        if not hasattr(self, 'encoders'):
            self.build_network(x.shape[2])  # 入力サイズを渡す

        skip_connections = []

        # エンコーダー
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skip_connections.append(x)
            x = pool(x)

        # ボトルネック
        x = self.bottleneck(x)

        # デコーダー
        for upsample, decoder, skip in zip(self.upsamples, self.decoders, reversed(skip_connections)):
            x = upsample(x)
            # サイズをスキップ接続に合わせる
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = decoder(x)

        # 出力層
        out = self.final_conv(x)
        return torch.relu(out)
    

class DynamicUNet2(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_filters=64, bottleneck_size=4):
        super(DynamicUNet2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_filters = base_filters
        self.bottleneck_size = bottleneck_size
        self.layers_built = False

    def build_layers(self, input_size):
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.ups = nn.ModuleList()

        # 計算: 必要な層数
        h, w = input_size
        depth = int(math.log2(min(h, w) / self.bottleneck_size))

        filters = self.base_filters
        for _ in range(depth):
            self.encoders.append(self.conv_block(self.in_channels, filters))
            self.pools.append(nn.MaxPool2d(2))
            self.in_channels = filters
            filters *= 2

        # ボトルネック
        self.bottleneck = self.conv_block(filters // 2, filters)

        # デコーダー
        for _ in range(depth):
            filters //= 2
            self.ups.append(self.upconv_block(filters * 2, filters))
            self.decoders.append(self.conv_block(filters * 2, filters))

        self.final_conv = nn.Conv2d(filters, self.out_channels, kernel_size=1)
        self.layers_built = True

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        if not self.layers_built:
            self.build_layers(x.shape[2:])

        enc_features = []
        out = x

        # エンコーダー
        for enc, pool in zip(self.encoders, self.pools):
            out = enc(out)
            enc_features.append(out)
            out = pool(out)

        # ボトルネック
        out = self.bottleneck(out)

        # デコーダー
        for up, dec, enc_feat in zip(self.ups, self.decoders, reversed(enc_features)):
            out = up(out)
            out = F.interpolate(out, size=enc_feat.shape[2:], mode='bilinear', align_corners=False)
            out = torch.cat([out, enc_feat], dim=1)
            out = dec(out)

        out = self.final_conv(out)
        return out
    

def create_structure_gui():
    """構造作成GUI"""
    
    struct_mode = st.radio('struct mode', ('random', 'update', 'manual'), horizontal=True)

    cols = st.columns(2)
    with cols[0]:
        size = st.number_input('size', value=64, min_value=16, step=32)
        shape = (size , size )
        wall_type = st.radio('Wall Type', ('no wall', 'wall', 'random free space'), index=1, horizontal=True)
        data, wall = create_porespy_data(shape=shape)
        if struct_mode!='random':
            # stroke_color = st.color_picker('color', value='#000000')
            stroke_color = st.radio('color', ('pen', 'eraser'), horizontal=True)
            if stroke_color=='pen':
                stroke_color = '#000000'
            else:
                stroke_color = '#ffffff'

    with cols[1]:
        if struct_mode=='random':
            if wall_type=='no wall':
                wall = np.zeros_like(data, dtype=bool)
            elif wall_type=='wall':
                wall = np.where(data==0,True, False)
            fig = create_structure_fig(data, wall)
            st.pyplot(fig)
        elif struct_mode=='update':
            data = canvas(stroke_color, data, shape=shape)
            wall = np.where((data==0) & wall, True, False)
            if wall_type=='no wall':
                wall = np.zeros_like(data, dtype=bool)
            elif wall_type=='wall':
                wall = np.where(data==0,True, False)
        else:
            wall = data
            data = canvas(stroke_color, shape=shape)
            wall = np.where((data==0) & wall, True, False)
            if wall_type=='no wall':
                wall = np.zeros_like(data, dtype=bool)
            elif wall_type=='wall':
                wall = np.where(data==0,True, False)

    if not is_continuous_path(data):
        with cols[0]:
            st.warning(f'Warning: The channels are not connected vertically.')
    return data, wall


def create_porespy_data(shape = (32, 32)):
    """Porespyで屈曲構想作成"""
    seed = st.number_input('random seed', value=1, min_value=0)
    cols = st.columns(2)
    porosity = cols[0].number_input('porosity', value=0.6, step=0.1)
    blob_size = cols[1].number_input('blob_size', value=1.0, step=0.5)
    # shape = (32, 32)  # 画像サイズ
    data = ps.generators.blobs(shape=shape, porosity=porosity, blobiness=blob_size, seed=seed)
    wall = ps.generators.blobs(shape=shape, porosity=porosity, blobiness=blob_size, seed=seed+1)
    wall = (data==0) & wall
    return data, wall


def canvas(stroke_color, data=None, shape = (32, 32)):
    """手書きで構造作成"""
    height = 320
    width = 320
    if data is not None:
        bg_image = (1- data.astype(np.uint8)) * 255
        bg_image = Image.fromarray(bg_image, mode="L").resize((height, width), Image.NEAREST) 
        bg_color = '#ffffff'
    else:
        bg_image = None
        bg_color = '#ffffff'

    canvas_result = st_canvas(
        stroke_width=50,
        stroke_color=stroke_color,
        update_streamlit=True,
        background_color=bg_color,
        background_image=bg_image,
        height=height,
        width=width,
        drawing_mode='freedraw',
        point_display_radius=50,
        key='canvas',
    )
    if canvas_result.image_data is not None:
        # st.write(canvas_result.__dict__)
        im = np.array(canvas_result.image_data)
        # im[:,:,-1] = 1
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = np.where(im==0, 255 ,0)

        if np.max(im)!=0:
            # new_shape = (32, 32)
            # data = downsample_image(im, shape)
            data = zoom(im, (shape[0] / 320, shape[1] / 320), order=0)==255
        else:
            data = np.zeros(shape)
    else:
        data = np.zeros(shape)

    # st.write(data.shape, data.dtype)
    # st.write(data1.shape, data1.dtype)
    # st.write(data1)
    # st.write(np.array_equal(data, data1))
    return data


# def downsample_image(image, new_shape):
#     # 元の画像の形状
#     original_shape = image.shape
#     # 新しい画像の形状
#     new_height, new_width = new_shape
#     orig_height, orig_width = original_shape
#     # 高さと幅の倍率を計算
#     height_factor = orig_height // new_height
#     width_factor = orig_width // new_width
#     # ダウンサンプリング後の画像を作成
#     downsampled_image = np.zeros(new_shape)
#     for i in range(new_height):
#         for j in range(new_width):
#             # ブロックのインデックスを計算
#             block = image[i*height_factor:(i+1)*height_factor, j*width_factor:(j+1)*width_factor]
#             # ブロックの平均値を計算
#             downsampled_image[i, j] = np.mean(block)
#     # 0~1に正規化
#     downsampled_image /= 255
#     # int8に変換
#     downsampled_image = downsampled_image.astype(np.int8)
#     return downsampled_image


def is_continuous_path(data):
    """上下に連続経路があればTrue"""
    con_data = trim_nonpercolating_paths_tool(data, axis=0)
    judge = np.sum(con_data)!=0
    return judge


def flux_simulation(data):
    """FVMで流束計算"""
    data = np.expand_dims(data, axis=-1) 
    c = Conduct(data, log=False)
    if np.sum(c.solver.data)==0:
        return np.zeros_like(data), 0
    c.solve()
    flux = c.solver.js[6]
    return flux[:,:,0], c.eff


def flux_prediction(data, device, model):
    """流束を推論"""
    input_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():  # 勾配計算を無効化
        output = model(input_tensor)
    
    flux = output.squeeze(0).squeeze(0).cpu().numpy()  # バッチ次元とチャネル次元を削除
    eff = np.mean(flux[-1])
    return flux, eff


def view_flux_gui(data, wall, device, model):
    flux_sim, eff_sim = flux_simulation(data)
    flux_pred, eff_pred = flux_prediction(data, device, model)
    vmax = np.max(np.concatenate([flux_sim, flux_pred]))

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    im1 = axes[0].imshow(flux_sim/data, cmap="jet", vmin=0, vmax=vmax)
    axes[0].set_title("CAE")
    fig.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(flux_pred/data, cmap="jet", vmin=0, vmax=vmax)
    axes[1].set_title("AI")
    fig.colorbar(im2, ax=axes[1])

    for ax in axes:
        ax.axis("off")
    
    plt.tight_layout()
    st.pyplot(fig)

    st.dataframe(pd.DataFrame([[eff_sim, eff_pred]], columns=['CAE', 'AI'], index=['Effective Value']), 
                #  hide_index=True,
                 use_container_width=True)


# --------------------------------------------------------------

def opt_gui(data, wall, device, model):
    """形状最適化デモ"""
    with st.form('run_opt', border=False):
        cols = st.columns(3)
        with cols[0]:
            # addition = st.checkbox('Allow addition', value=True)
            reduction = st.number_input('Reduction rate', value=0.5, step=0.05)
            step = st.number_input('Step Number', value=10)
            opt_with = st.radio('Optimize with', ('CAE', 'AI'), horizontal=True)
            run_button = st.empty()

        with cols[1]:
            placeholder1 = st.empty()
        with cols[2]:
            placeholder2 = st.empty()

    constraint=~wall
    # st.write(constraint)

    datas = []
    effs = []
    n = np.sum(data==1)
    target_n = int(n*(1-reduction)) # 材料半減
    delta_n = int((n-target_n)/step) # 刻み
    next_n = n
    datas.append(data)
    if opt_with=='AI':
        flux, eff = flux_prediction(data, device, model) 
    else:
        flux, eff = flux_simulation(data) # 2d
    effs.append(eff)

    if run_button.form_submit_button('実行'):
        st.session_state["state"] = "running"
    if st.session_state["state"] != "running":
        st.stop()

    for _ in range(50):
        # 構造更新
        if opt_with=='AI':
            flux, eff = flux_prediction(data, device, model) # 2d
        else:
            flux, eff = flux_simulation(data) # 2d

        if eff==0:
            break

        grad = calc_grad(flux.copy(), constraint=~wall) # 2d
        next_n = max(target_n, next_n-delta_n)
        new_data = top_n_true(grad, next_n, constraint=~wall)

        # データバックアップ
        datas.append(new_data)
        effs.append(eff)

        # 進捗可視化
        fig = create_structure_fig(new_data, wall)  # 画像を生成
        placeholder1.pyplot(fig)  # プレースホルダーに描画
        fig = create_history_fig(datas, effs)  # 画像を生成
        placeholder2.pyplot(fig)  # プレースホルダーに描画

        # 収束判定
        if arrays_similarity(data, new_data,  threshold=0.99) and next_n==target_n:
            if opt_with=='AI':
                flux, eff = flux_simulation(data) # 2d
                effs.append(eff)
            break
        data = new_data

    # サマリー
    datas = [data+wall*2 for data in datas]
    fig = visualize_grids(datas, auto_range=True)
    if opt_with=='AI':
        st.session_state.opt_results['AI'] = {'fig':fig, 'effs':effs}
    if opt_with=='CAE':
        st.session_state.opt_results['CAE'] = {'fig':fig, 'effs':effs}
    tabs = st.tabs(['CAE','AI'])
    for _type, tab in zip(['CAE','AI'], tabs): 
        with tab:
            fig = st.session_state.opt_results[_type]['fig']
            eff = st.session_state.opt_results[_type]['effs'][-1]
            st.pyplot(fig)
            st.write(f'{eff:.3f}')
    st.session_state["state"] = "stop"


def create_structure_fig(data, wall):
    data = data + wall * 2

    fig, axes = plt.subplots(1, 1, figsize=(3, 3))
    axes.imshow(data, interpolation='none', cmap=cmap, vmin=0, vmax=2)
    axes.axis('off')  # 軸を非表示にする
    plt.tight_layout()
    return fig


def create_history_fig(datas, effs):
    ns = [np.sum(data==1) for data in datas]
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.plot(ns, effs, marker='o', linestyle='-', color='#084594')

    ax.set_xlabel('Number of elements')
    ax.set_ylabel('Effective values')
    ax.set_xlim([0, None])
    ax.set_ylim([0, None])
    ax.grid(True)
    # ax.legend()
    return fig



def visualize_grids(datas, auto_range=True):
    # nとmを決定する（n*m = num_plots で、n≒mに調整）
    num_plots = len(datas)
    n = int(math.sqrt(num_plots))
    m = math.ceil(num_plots / n)

    # グリッドサイズに基づいてプロット作成
    fig, axes = plt.subplots(n, m, figsize=(m*3, n*3))
    axes = axes.flatten()  # axesをフラットにしてループで使いやすくする

    # 各データをグリッドに表示
    for i, data in enumerate(datas):
        axes[i].imshow(data, interpolation='none', cmap=cmap, vmin=0, vmax=2)
        axes[i].axis('off')  # 軸を非表示にする
    
    # 余分なサブプロットを非表示にする
    for j in range(i+1, n*m):
        axes[j].axis('off')

    plt.tight_layout()
    return fig
    


def get_min_max(datas):
    results = []
    mins = [np.min(arr) for arr in datas]
    maxs = [np.max(arr) for arr in datas]
    min_value = np.min(mins)
    max_value = np.max(maxs)
    return min_value, max_value


if __name__=='__main__':
    cmap = ListedColormap([(0.968627, 0.984314, 1.0),
                        # (0.337, 0.611, 0.839, 1.0),
                        '#084594',
                        (0.7,0.7,0.7,1)])
    main()
