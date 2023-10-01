import os

folder_path = './dataset/Duck/'

file_list = os.listdir(folder_path)

num_files_to_rename = 400

for i, file_name in enumerate(file_list[:num_files_to_rename]):
    new_name = f'Tuan_{i + 1}.png'
    
    old_path = os.path.join(folder_path, file_name)
    new_path = os.path.join(folder_path, new_name)
    
    
    os.rename(old_path, new_path)
    print(f'Thay đổi tên tệp {file_name} thành {new_name}')
