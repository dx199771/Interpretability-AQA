
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
import os 
import cv2
import pandas as pd

def self_map_vis(graph_attn):
    self_map = graph_attn[0][1]
    soft_map = F.softmax(torch.div(torch.matmul(self_map.transpose(0,1), self_map.transpose(0,1).transpose(-1,-2)), self_map.size(-1)),dim=1).cpu().detach().numpy()
    plt.cla()
    plt.clf()
    plt.imshow(soft_map[0], cmap='hot', interpolation='nearest')
        
    plt.colorbar()
    plt.savefig("atmap.png")


def inter_vis(means, weight, name="logo",dataset="logo"):
    
    from scipy.ndimage import gaussian_filter1d
    from scipy.interpolate import make_interp_spline
    import cv2
    
    plt.cla()
    plt.clf()
    plt.figure(figsize=(21, 6))
    quality = means.squeeze(-1)[0].detach().cpu().numpy()
    difficulty = weight.squeeze(-1)[0].detach().cpu().numpy()
    
    sample_rate =1
    quality = ((quality - quality.min()) / (quality.max() - quality.min()))[::sample_rate]
    difficulty = ((difficulty - difficulty.min()) / (difficulty.max() - difficulty.min()))[::sample_rate]
    if dataset == "logo":
        clip_num = 48
        folder_base = os.path.join("/mnt/welles/scratch/datasets/condor/backup/logo/Video_result",name[0][0],str(name[1][0].item()))
    elif dataset == "gym":
        clip_num = 68
        folder_base = os.path.join("/mnt/welles/scratch/datasets/condor/backup/data/images",name[0])
    elif dataset == "fisv":
        clip_num = 136
        folder_base = os.path.join("/mnt/welles/scratch/datasets/condor/backup/data/images",name[0])
    x = np.arange(clip_num)
    y = difficulty
    X_Y_Spline_qual = make_interp_spline(x[::sample_rate], quality)
    X_Y_Spline_diff = make_interp_spline(x[::sample_rate], difficulty)

    # Returns evenly spaced numbers
    # over a specified interval.
    X_ = np.linspace(x.min(), x.max(), 450)
    y_smooth_qual = X_Y_Spline_qual(X_)
    y_smooth_diff = X_Y_Spline_diff(X_)
    # y_smooth = (y_smooth - y_smooth.min()) / (y_smooth.max() - y_smooth.min())
    
    # plt.plot(np.arange(48)[::sample_rate],  quality[::sample_rate], marker='o', color='b', linestyle='-', linewidth=2, markersize=8)
    # plt.plot(np.arange(48)[::sample_rate],  difficulty[::sample_rate], marker='o', color='r', linestyle='-', linewidth=2, markersize=8)
    plt.plot(X_,  y_smooth_qual, marker=None, color='g', linestyle='-', linewidth=2)
    plt.plot(X_,  y_smooth_diff, marker=None, color='b', linestyle='-', linewidth=2)
    plt.plot(X_,  y_smooth_qual * y_smooth_diff, marker=None, color='r', linestyle='-', linewidth=2)
    
    x_min, x_max = int(min(X_)), int(max(X_))
    xticks = np.arange(x_min, x_max + 1, 2)  # 每 5 个单位一个刻度
    plt.xticks(xticks)  # 应用 x 轴刻度
    plt.grid(axis='x', linestyle='--', alpha=0.5)  # 仅在 x 轴加网格线
    if dataset == "logo":
        plt.title(f"{dataset}: Video: {name[0][0]}, Seq: {name[1][0].item()}",fontsize=16)
    else:
        plt.title(f"{dataset}: Video: {name[0]}",fontsize=16)

    # 添加 x/y 轴标签
    plt.xlabel("Clip",fontsize=16)
    plt.ylabel("Score",fontsize=16)
       
        
    image_folder_path = sorted(os.listdir(folder_base))
    num_image = len(image_folder_path)//clip_num
    selected_images = image_folder_path[::clip_num]
    concat_list = [os.path.join(folder_base,i) for i in selected_images]
        # img_list = [cv2.imread(i) for i in concat_list]
        # big_image = concat_images(img_list)
        # import pdb; pdb.set_trace()
        # cv2.imwrite(big_image,"big.png")
        # image_gray = cv2.imread(, cv2.IMREAD_GRAYSCALE)
        # import pdb; pdb.set_trace()
    if dataset == "logo":
        plt.savefig(f"inter_results/{dataset}: Video: {name[0][0]}, Seq: {name[1][0].item()}, GAP:{num_image}.png")
    else:
        plt.savefig(f"inter_results_rg/{dataset}: Video: {name[0]}, GAP:{num_image}.png")
    
def concat_images(image_list, images_per_row=6, padding=10, background_color=(255, 255, 255)):
    """
    将一组图像按每行 `images_per_row` 个排列，生成一张大图。

    参数:
    - image_list: List of images (cv2 读取的 numpy 数组)
    - images_per_row: 每行放多少张图片
    - padding: 图片间距
    - background_color: 背景颜色 (默认白色)

    返回:
    - 拼接后的大图
    """
    
    # 确保所有图像大小一致（这里取第一张图的大小）
    img_h, img_w = image_list[0].shape[:2]

    # 计算行数
    num_images = len(image_list)
    num_rows = (num_images + images_per_row - 1) // images_per_row  # 向上取整

    # 计算大图尺寸
    big_img_h = num_rows * img_h + (num_rows + 1) * padding
    big_img_w = images_per_row * img_w + (images_per_row + 1) * padding

    # 创建大图画布
    big_image = np.full((big_img_h, big_img_w, 3), background_color, dtype=np.uint8)

    # 将每张图像放入大图
    for idx, img in enumerate(image_list):
        row, col = divmod(idx, images_per_row)
        y = row * (img_h + padding) + padding
        x = col * (img_w + padding) + padding
        big_image[y:y+img_h, x:x+img_w] = img

    return big_image


def user_study(means,weight,name,dataset):
    clip_num = 48
    folder_base = os.path.join("/mnt/welles/scratch/datasets/condor/backup/logo/Video_result",name[0][0],str(name[1][0].item()))
    
    quality = means.squeeze(-1)[0].detach().cpu().numpy()
    difficulty = weight.squeeze(-1)[0].detach().cpu().numpy()
    sample_rate = 1
    # quality = ((quality - quality.min()) / (quality.max() - quality.min()))[::sample_rate]
    # difficulty = ((difficulty - difficulty.min()) / (difficulty.max() - difficulty.min()))[::sample_rate]
    
    image_folder_path = sorted(os.listdir(folder_base))
    num_image = len(image_folder_path)//clip_num  
    
    top3_high = sorted(enumerate(quality), key=lambda x: x[1], reverse=True)[:1]
    top3_low = sorted(enumerate(quality), key=lambda x: x[1], reverse=False)[:1]
    top3_high_difficulty = sorted(enumerate(difficulty), key=lambda x: x[1], reverse=True)[:1]
    top3_low_difficulty = sorted(enumerate(difficulty), key=lambda x: x[1], reverse=False)[:1]
    # import pdb ; pdb.set_trace()

    top3_high_indices, top3_high_values = zip(*top3_high)
    top3_low_indices, top3_low_values = zip(*top3_low)
    top3_high_indices_difficulty, top3_high_values_difficulty = zip(*top3_high_difficulty)
    top3_low_indices_difficulty, top3_low_values_difficulty = zip(*top3_low_difficulty)
    
    values_quality = top3_high_values + top3_low_values
    values_difficulty = top3_high_values_difficulty + top3_low_values_difficulty
    
    values = values_quality + values_difficulty
    
    top1_images = image_folder_path[top3_high_indices[0]*num_image-40:top3_high_indices[0]*num_image+40]
    # top2_images = image_folder_path[top3_high_indices[1]*num_image-40:top3_high_indices[1]*num_image+40]
    # top3_images = image_folder_path[top3_high_indices[2]*num_image-40:top3_high_indices[2]*num_image+40]
    
    top1_low_images = image_folder_path[top3_low_indices[0]*num_image-40:top3_low_indices[0]*num_image+40]
    # top2_low_images = image_folder_path[top3_low_indices[1]*num_image-40:top3_low_indices[1]*num_image+40]
    # top3_low_images = image_folder_path[top3_low_indices[2]*num_image-40:top3_low_indices[2]*num_image+40]
    
    top1_images_difficulty = image_folder_path[top3_high_indices_difficulty[0]*num_image-40:top3_high_indices_difficulty[0]*num_image+40]
    # top2_images_difficulty = image_folder_path[top3_high_indices_difficulty[1]*num_image-40:top3_high_indices_difficulty[1]*num_image+40]
    # top3_images_difficulty = image_folder_path[top3_high_indices_difficulty[2]*num_image-40:top3_high_indices_difficulty[2]*num_image+40]
    
    top1_low_images_difficulty = image_folder_path[top3_low_indices_difficulty[0]*num_image-40:top3_low_indices_difficulty[0]*num_image+40]
    # top2_low_images_difficulty = image_folder_path[top3_low_indices_difficulty[1]*num_image-40:top3_low_indices_difficulty[1]*num_image+40]
    # top3_low_images_difficulty = image_folder_path[top3_low_indices_difficulty[2]*num_image-40:top3_low_indices_difficulty[2]*num_image+40]
    
    
    top_low = ["quality_top"]*1 + ["quality_low"]*1 + ["diff_top"]*1 + ["diff_low"]*1
    # image_list = [top1_images,top2_images,top3_images,top1_low_images,top2_low_images,top3_low_images,
    #               top1_images_difficulty,top2_images_difficulty,top3_images_difficulty,
    #               top1_low_images_difficulty,top2_low_images_difficulty,top3_low_images_difficulty]
    image_list = [top1_images,top1_low_images,top1_images_difficulty,top1_low_images_difficulty]
    # import pdb; pdb.set_trace()
    
    
    


    for idx, top_image in enumerate(image_list):
        if top_image == []:
            continue
        print({name[0][0]}-{name[1][0].item()})
        print(top_image)
        #output_video = f"user_study/{name[0][0]}-{name[1][0].item()}_{top_low[idx]}_{idx}.mp4"
        output_video = f"user_study/{name[0][0]}-{name[1][0].item()}_{idx}.mp4"
        
        first_frame = cv2.imread(os.path.join(folder_base, top_image[0]))
        height, width, layers = first_frame.shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = 25  # 设置帧率
        video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        
        text1 = f"dataset:{name[0][0]}-{name[1][0].item()}_{idx}"
        #text2 = f"quality/diff:{top_low[idx]}_{values[idx]}"
        
        df = pd.DataFrame({
            "Label": [f"{name[0][0]}-{name[1][0].item()}_{idx}"],
            "Score": [f"{values[idx]}"],
            "Diff": [f"{values[idx]}"],
            "Score / Diff": [f"{top_low[idx]}"]
            })
        df.to_csv("lookup_userstudy.csv", mode='a', index=False, header=False)

        for img_idx,image in enumerate(top_image):
            
            img_path = os.path.join(folder_base, image)
            frame = cv2.imread(img_path)
            
            
            font = cv2.FONT_HERSHEY_SIMPLEX  # 字体
            font_scale = 1  # 字体大小
            font_color = (255, 255, 255)  # 文字颜色（白色）
            thickness = 2  # 文字粗细
            line_type = cv2.LINE_AA  # 线条平滑

            # 在图片上绘制文字
            cv2.putText(frame, text1, (10, 30), font, font_scale, font_color, thickness, line_type)
            #cv2.putText(frame, text2, (10, 60), font, font_scale, font_color, thickness, line_type)
            cv2.putText(frame, f"frame: {img_idx}", (10, 90), font, font_scale, font_color, thickness, line_type)
            
            video.write(frame)
        video.release()
        cv2.destroyAllWindows()

    # import pdb; pdb.set_trace()
    # image_folder_path[]
    
    
    # print("top3 quality", top3_values)
    # print("top3 quality indicies", top3_indices)
    