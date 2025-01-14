import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

def save_nii_to_png_3views(nii_path, output_dir):
    # 读取nii.gz文件
    img = nib.load(nii_path)
    data = img.get_fdata()
    
    # 创建输出文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取文件名（不包含扩展名）
    base_name = os.path.splitext(os.path.splitext(os.path.basename(nii_path))[0])[0]
    
    # 创建三个子文件夹
    axial_dir = os.path.join(output_dir, 'axial')
    sagittal_dir = os.path.join(output_dir, 'sagittal')
    coronal_dir = os.path.join(output_dir, 'coronal')
    
    for d in [axial_dir, sagittal_dir, coronal_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
    
    # 保存轴向切片 (Axial)
    for i in range(data.shape[2]):
        plt.figure(figsize=(10, 10))
        plt.imshow(data[:, :, i], cmap='gray')
        plt.axis('off')
        output_path = os.path.join(axial_dir, f'{base_name}_axial_{i:03d}.png')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    
    # 保存矢状面切片 (Sagittal)
    for i in range(data.shape[0]):
        plt.figure(figsize=(10, 10))
        plt.imshow(data[i, :, :], cmap='gray')
        plt.axis('off')
        output_path = os.path.join(sagittal_dir, f'{base_name}_sagittal_{i:03d}.png')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    
    # 保存冠状面切片 (Coronal)
    for i in range(data.shape[1]):
        plt.figure(figsize=(10, 10))
        plt.imshow(data[:, i, :], cmap='gray')
        plt.axis('off')
        output_path = os.path.join(coronal_dir, f'{base_name}_coronal_{i:03d}.png')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

def process_folder(input_folder, output_folder):
    # 遍历文件夹中的所有nii.gz文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.nii.gz'):
            nii_path = os.path.join(input_folder, filename)
            # 为每个nii文件创建一个子文件夹
            sub_output_dir = os.path.join(output_folder, filename[:-7])
            save_nii_to_png_3views(nii_path, sub_output_dir)
            print(f"Processed: {filename}")

# 使用示例
input_folder = "path/to/your/input/folder"  # 替换为你的输入文件夹路径
output_folder = "path/to/your/output/folder"  # 替换为你想保存图片的文件夹路径

process_folder(input_folder, output_folder)