import os

# 替换为你的实际数据集路径
root = "../lfw-align-128"
out_file = "train_data_list.txt"

with open(out_file, "w") as f:
    for label, person in enumerate(sorted(os.listdir(root))):
        person_dir = os.path.join(root, person)
        if not os.path.isdir(person_dir):
            continue
        for img in os.listdir(person_dir):
            if img.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(person, img)  # 相对路径
                f.write(f"{img_path} {label}\n")

print("Train list saved to:", out_file)
