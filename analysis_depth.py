import cv2
import numpy as np

def visualize_depth_with_box(depth_image, region):
    # 读取深度图像
    depth = cv2.imread(depth_image, cv2.IMREAD_UNCHANGED)
    
    print(f"Depth shape: {depth.shape}")
    print(depth )
    # 确定感兴趣的区域
    x, y, w, h = region
    roi = depth[y:y+h, x:x+w]
    
    # 计算感兴趣区域的平均深度值
    mean_depth = np.mean(roi)/255
    
    # 在原深度图上绘制矩形框和深度值
    print(f"Mean depth: {mean_depth:.2f} units")
    
    # 显示原深度图
    print(roi)
    
    
    # depth_with_box = depth.copy()
    # cv2.rectangle(depth_with_box, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # cv2.putText(depth_with_box, f"Depth: {mean_depth:.2f} units", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 显示带有矩形框和深度值的深度图
    # cv2.imshow("Depth with Box", depth_with_box)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# 示例用法
depth_image = '/home/wudi/code/YOLOdepth/datasets/51sim_BDD_style/depth_label/train/000000.png'
# 设置感兴趣区域的矩形框坐标 (x, y, width, height)
region_of_interest = (500, 500, 50, 50)

visualize_depth_with_box(depth_image, region_of_interest)
