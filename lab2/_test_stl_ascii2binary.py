import trimesh
import argparse
import os

# 设置命令行参数
parser = argparse.ArgumentParser(description="将ASCII STL文件转换为二进制STL文件")
parser.add_argument("input", help="输入的ASCII STL文件路径")
parser.add_argument("--simplify", type=float, help="简化网格的目标面数比例(0-1之间)", default=0.01)
args = parser.parse_args()

# 读取 ASCII STL
mesh = trimesh.load_mesh(args.input)

# # 如果指定了简化比例，则进行简化
# if args.simplify < 1.0:
#     # 计算目标面数
#     target_faces = int(mesh.faces.shape[0] * args.simplify)
#     # 简化网格
#     mesh = mesh.simplify_quadric_decimation(args.simplify)
#     print(f"网格已简化至 {args.simplify*100}% 的面数")

# 随机采样点云
points = mesh.sample(10000)  # 采样 1000 个点
# 从点云重建 STL
new_mesh = trimesh.Trimesh(vertices=points)


# 生成输出文件路径 - 在原文件名后添加 _binary
output_path = args.input.rsplit(".", 1)[0] + "_binary.stl"

# 保存为二进制 STL
# mesh.export(output_path, file_type="stl")
new_mesh.export(output_path, file_type="stl")
print(f"转换完成，二进制文件已保存为：{output_path}")
print(f"面数：{new_mesh.faces.shape[0]}，顶点数：{new_mesh.vertices.shape[0]}")
print(f"文件大小：{os.path.getsize(output_path)} 字节")
