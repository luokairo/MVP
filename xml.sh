#!/bin/bash

# 设置路径
XML_DIR="/fs/scratch/PAS2473/MM2025/neurpis2025/dataset/ILSVRC/Annotations/CLS-LOC/val"
IMG_DIR="/fs/scratch/PAS2473/MM2025/neurpis2025/dataset/ILSVRC/Data/CLS-LOC/val"

# 记录脚本开始时间
echo "开始重组验证集图片..."
start_time=$(date +%s)

# 创建一个临时文件来存储图片与类别的映射
MAPPING_FILE=$(mktemp)

# 从XML文件中提取信息
echo "解析XML文件..."
for xml_file in "$XML_DIR"/*.xml; do
    filename=$(grep -oP '<filename>\K[^<]+' "$xml_file" | head -1)
    class_id=$(grep -oP '<name>\K(n\d+)(?=</name>)' "$xml_file" | head -1)
    
    # 如果文件名没有扩展名，添加.JPEG
    if [[ ! $filename =~ \.JPEG$ ]]; then
        filename="${filename}.JPEG"
    fi
    
    # 将文件名和类别ID写入映射文件
    if [[ -n "$filename" && -n "$class_id" ]]; then
        echo "$filename $class_id" >> "$MAPPING_FILE"
    fi
done

# 读取映射文件并创建类别目录，然后移动文件
echo "创建类别目录和移动图片..."
mkdir -p "$IMG_DIR/temp_backup"

# 首先创建所有目录
cat "$MAPPING_FILE" | cut -d' ' -f2 | sort | uniq | while read class_id; do
    mkdir -p "$IMG_DIR/$class_id"
done

# 然后移动文件
total_files=$(wc -l < "$MAPPING_FILE")
current=0

cat "$MAPPING_FILE" | while read line; do
    filename=$(echo "$line" | cut -d' ' -f1)
    class_id=$(echo "$line" | cut -d' ' -f2)
    
    # 移动文件
    if [ -f "$IMG_DIR/$filename" ]; then
        mv "$IMG_DIR/$filename" "$IMG_DIR/$class_id/"
        
        # 显示进度
        current=$((current + 1))
        if [ $((current % 100)) -eq 0 ]; then
            echo "已处理 $current / $total_files 个文件"
        fi
    else
        echo "警告：找不到图片 $filename"
    fi
done

# 检查是否有未移动的图片
remaining=$(find "$IMG_DIR" -maxdepth 1 -name "*.JPEG" | wc -l)
if [ "$remaining" -gt 0 ]; then
    echo "警告：还有 $remaining 个图片未被移动"
    find "$IMG_DIR" -maxdepth 1 -name "*.JPEG" | head -5
fi

# 清理临时文件
rm "$MAPPING_FILE"

# 记录结束时间和总用时
end_time=$(date +%s)
duration=$((end_time - start_time))
echo "任务完成！总用时: $duration 秒"