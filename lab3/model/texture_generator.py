from PIL import Image
import numpy as np


def create_texture_map(tag_image_path, grid_size=(4, 3)):
    # 读取tag图像
    tag = Image.open(tag_image_path)
    # 在paste之前添加重采样
    tag = tag.resize((100, 100), Image.NEAREST)  # 使用最近邻插值

    # 创建一个灰色背景的大图像
    # 假设每个面都是正方形，所以使用tag的宽度作为单位大小
    unit_size = tag.size[0]
    texture_width = grid_size[0] * unit_size
    texture_height = grid_size[1] * unit_size

    # 创建灰色背景 (RGB值为128的中灰色)
    texture = Image.new("RGB", (texture_width, texture_height), (128, 128, 128))

    # 将tag图像粘贴到正面位置（假设是左上角第一个格子）
    texture.paste(tag, (1 * unit_size, 1 * unit_size))

    for i, j in [(0, 1), (1, 0), (1, 2), (2, 1), (3, 1)]:  # 先列后行
        texture.paste(Image.new("RGB", (unit_size, unit_size), (255, 255, 255)), (i * unit_size, j * unit_size))

    return texture

texture = create_texture_map("meshes/tag36_11_00000.png")
texture.save("meshes/tag_texture.png")
