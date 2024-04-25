import os
import cv2
import numpy as np
import random
import time
import yaml

######################### Hyper Parameters ##########################
back_ground_images_folder = "BG-20k"
augumet_images_folers = ["GreenLights", "RedLights"]
test_data_ratio = 0.25
labels = {
    0 : 'green lights',
    1 : 'red lights'
}
mean_width = 150
max_num_object = 3
scale_bias = 0.2
scale_factor = 1.5
#####################################################################

def images_in_folder(folder_path):
    # 이미지 파일 확장자
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif']
    file_names = [f for f in os.listdir(folder_path) if any(f.lower().endswith(ext) for ext in image_extensions)]

    return len(file_names), file_names

def create_label_file(bg, obj, x_offset, y_offset, class_id, obj_number, FLAG_train):
    # 중심 좌표와 너비 및 높이의 비율 계산
    x_center = x_offset + obj.shape[1]//2
    y_center = y_offset + obj.shape[0]//2
    x = x_center / bg.shape[1]
    y = y_center / bg.shape[0]
    w = obj.shape[1] / bg.shape[1]
    h = obj.shape[0] / bg.shape[0]

    # 라벨 파일에 저장할 줄 생성
    label_line = f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"

    if(FLAG_train):
        output_path = current_folder_path + f"/train/labels/{obj_number}.txt"
    else:
        output_path = current_folder_path + f"/test/labels/{obj_number}.txt"

    # 라벨 파일 쓰기
    with open(output_path, 'a') as file:
        file.write(label_line)

def data_augmentation(bg, obj, mask, class_id, obj_number, FLAG_train):
    height_bg, width_bg = bg.shape[:2]
    height_obj, width_obj = obj.shape[:2]

    mask_blurred = cv2.GaussianBlur(mask, (15, 15), 0)
    mask_blurred = mask_blurred.astype(float) / 255 

    x_offset = random.randint(0, width_bg - width_obj)
    y_offset = random.randint(0, height_bg - height_obj)

    y1, y2 = y_offset, y_offset + height_obj
    x1, x2 = x_offset, x_offset + width_obj

    obj_blended = (obj * mask_blurred[..., None]).astype(np.uint8)
    roi = bg[y1:y2, x1:x2]
    roi_blended = (roi * (1- mask_blurred[..., None])).astype(np.uint8)
    dst = cv2.add(obj_blended, roi_blended)

    bg[y1:y2, x1:x2] = dst

    if(FLAG_train):
        output_path = current_folder_path + f"/train/images/{obj_number}.jpg"
    else:
        output_path = current_folder_path + f"/test/images/{obj_number}.jpg"

    #cv2.imshow("Ouput", bg)
    #cv2.waitKey(0)
    create_label_file(bg, obj, x_offset, y_offset, class_id, obj_number, FLAG_train)
    cv2.imwrite(output_path, bg)
    return

def img_transform(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation_scale = random.random() + 0.5
    value_scale = random.random() + 0.5
    img_hsv[:,:,1] = cv2.multiply(img_hsv[:,:,1], np.array([saturation_scale]))
    img_hsv[:,:,2] = cv2.multiply(img_hsv[:,:,2], np.array([value_scale]))
    img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    (h, w) = img.shape[:2]

    mask = np.ones((h,w), dtype='uint8') * 255
    center = (w // 2, h // 2)

    angle = random.randint(-180, 180)
    scale = random.random() * scale_factor + scale_bias

    M = cv2.getRotationMatrix2D(center, angle, scale)

    H = h * scale
    W = w * scale
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    W = int((h*sin) + (w*cos))
    H = int((h*cos) + (w *sin))

    M[0, 2] += (W / 2) - center[0]
    M[1, 2] += (H / 2) - center[1]

    rotated = cv2.warpAffine(img, M, (W, H))
    rotated_mask = cv2.warpAffine(mask, M, (W, H))

    return rotated, rotated_mask

if __name__ == "__main__":
    img_counts = []
    img_names = []
    
    current_folder_path = f"{os.getcwd()}/Datasets"

    folder_path = f"{current_folder_path}/{back_ground_images_folder}"
    back_ground_counts, back_ground_names = images_in_folder(folder_path)

    for folder_names in augumet_images_folers:
        folder_path = f"{current_folder_path}/{folder_names}"
        img_counts_temp, img_names_temp = images_in_folder(folder_path)
        img_counts.append(img_counts_temp)
        img_names.append(img_names_temp)

    print(f"Background Images num: {back_ground_counts}")
    for i in range(len(augumet_images_folers)):
        print(f"{augumet_images_folers[i]} Images num: {img_counts[i]}")

    folder_path = current_folder_path + '/train'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    folder_path = current_folder_path + '/train/images'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    folder_path = current_folder_path + '/train/labels'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    folder_path = current_folder_path + '/test'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    folder_path = current_folder_path + '/test/images'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    folder_path = current_folder_path + '/test/labels'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    data = {
        'train' : current_folder_path + '/train/images',
        'val' : current_folder_path + '/test/images',
        'test' : current_folder_path + '/test/images',
        'names' : list(labels.values()),
        'nc' : len(labels)
    }

    with open(current_folder_path + '/datasets.yaml', 'w') as f:
        yaml.dump(data, f)

    for i in range(back_ground_counts):
        folder_path = f"{current_folder_path}/{back_ground_images_folder}"
        file_name = f"{folder_path}/{back_ground_names[i]}"
        bg = cv2.imread(file_name)

        if i % round(1/test_data_ratio) == 0:
            FLAG_train = False
        else:
            FLAG_train = True

        object_num = random.randint(0,max_num_object)
        for j in range(object_num):
            n = random.randint(0,len(augumet_images_folers)-1)
            class_id = n
            folder_path = f"{current_folder_path}/{augumet_images_folers[n]}"

            img_names_temp = img_names[n]
            img_names_idx = random.randint(0, img_counts[n]-1)
            file_name = f"{folder_path}/{img_names_temp[img_names_idx]}"
            obj = cv2.imread(file_name)

            original_ratio = obj.shape[0]/obj.shape[1]
            obj = cv2.resize(obj, (mean_width, int(mean_width * original_ratio)), interpolation=cv2.INTER_LINEAR)
            obj, mask = img_transform(obj)
            try:
                data_augmentation(bg, obj, mask, class_id, i, FLAG_train)
            except Exception as e:
                pass

        if i % 100 == 0:
            print(f"Maked Images num: {i}, Reamining Images num: {back_ground_counts-i}, Progress Ratio: {i/back_ground_counts*100}%")