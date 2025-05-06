import os
import xml.etree.ElementTree as ET
import shutil
from tqdm import tqdm
from multiprocessing import Pool
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='将ImageNet验证集图片按类别组织到子目录')
    parser.add_argument('--xml_dir', type=str, default='/fs/scratch/PAS2473/MM2025/neurpis2025/dataset/ILSVRC/Annotations/CLS-LOC/val',
                        help='XML标注文件目录')
    parser.add_argument('--img_dir', type=str, default='/fs/scratch/PAS2473/MM2025/neurpis2025/dataset/ILSVRC/Data/CLS-LOC/val',
                        help='原始验证集图片目录')
    parser.add_argument('--workers', type=int, default=8, help='并行处理的工作线程数')
    return parser.parse_args()

def extract_info_from_xml(xml_file):
    """从XML文件中提取图片名称和类别ID"""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    filename = root.find('filename').text
    # 有些文件名可能不包含扩展名，需要添加
    if not filename.endswith('.JPEG'):
        filename = filename + '.JPEG'
    
    # 获取类别ID
    class_id = root.find('.//object/name').text
    
    return filename, class_id

def process_xml(xml_file, xml_dir, img_dir):
    """处理单个XML文件并移动对应的图片"""
    try:
        relative_path = os.path.relpath(xml_file, xml_dir)
        filename, class_id = extract_info_from_xml(xml_file)
        
        # 创建类别目录
        dest_dir = os.path.join(img_dir, class_id)
        os.makedirs(dest_dir, exist_ok=True)
        
        # 源图片和目标路径
        src_img = os.path.join(img_dir, filename)
        dst_img = os.path.join(dest_dir, filename)
        
        # 移动图片
        if os.path.exists(src_img):
            shutil.move(src_img, dst_img)
            return True
        else:
            print(f"警告：找不到图片 {src_img}")
            return False
    except Exception as e:
        print(f"处理 {xml_file} 时出错: {e}")
        return False

def main():
    args = parse_args()
    
    # 确保目标目录存在
    os.makedirs(args.img_dir, exist_ok=True)
    
    # 获取所有XML文件
    xml_files = [os.path.join(args.xml_dir, f) for f in os.listdir(args.xml_dir) 
                if f.endswith('.xml')]
    print(f"找到 {len(xml_files)} 个XML标注文件")
    
    # 首先提取所有类别并创建目录
    print("创建类别目录...")
    class_ids = set()
    for xml_file in tqdm(xml_files):
        try:
            _, class_id = extract_info_from_xml(xml_file)
            class_ids.add(class_id)
        except:
            pass
    
    for class_id in class_ids:
        os.makedirs(os.path.join(args.img_dir, class_id), exist_ok=True)
    
    print(f"创建了 {len(class_ids)} 个类别目录")
    
    # 使用多进程移动图片
    print("开始移动图片...")
    with Pool(args.workers) as p:
        results = list(tqdm(
            p.starmap(
                process_xml, 
                [(xml_file, args.xml_dir, args.img_dir) for xml_file in xml_files]
            ),
            total=len(xml_files)
        ))
    
    success_count = sum(results)
    print(f"成功移动 {success_count}/{len(xml_files)} 个图片")
    
    # 检查是否有残留的图片
    remaining_imgs = [f for f in os.listdir(args.img_dir) 
                     if os.path.isfile(os.path.join(args.img_dir, f)) and f.endswith('.JPEG')]
    if remaining_imgs:
        print(f"警告：还有 {len(remaining_imgs)} 个图片未被移动")
        if len(remaining_imgs) < 10:
            print("未移动的图片:", remaining_imgs)

if __name__ == '__main__':
    main()