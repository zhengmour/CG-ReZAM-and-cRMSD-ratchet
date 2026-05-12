
#!/bin/bash

# LAMMPS轨迹文件时间处理脚本
# 功能：根据init_time筛选文件，按ctime排序，分成锯齿段，重新分配时间

set -e  # 遇到错误立即退出

# 检查必要文件是否存在
if [[ ! -f "params.json" ]]; then
    echo "错误: params.json 文件不存在"
    exit 1
fi

# 从params.json提取init_time
init_time=$(grep '"init_time"' params.json | sed -E 's/.*:\s*([0-9.]+).*/\1/')
if [[ -z "$init_time" || "$init_time" == "null" ]]; then
    echo "错误: 无法从params.json中提取init_time"
    exit 1
fi

echo "初始时间: $init_time"

# 查找所有lmp_*.lammpstrj文件
lmp_files=($(ls lmp_*.lammpstrj 2>/dev/null || true))
if [[ ${#lmp_files[@]} -eq 0 ]]; then
    echo "错误: 未找到任何lmp_*.lammpstrj文件"
    exit 1
fi

echo "找到 ${#lmp_files[@]} 个lmp文件"

# 创建临时文件存储文件信息
temp_file=$(mktemp)
trap "rm -f $temp_file" EXIT

# 提取每个文件的时间戳和创建时间
for file in "${lmp_files[@]}"; do
    if [[ -f "$file" ]]; then
        # 从文件名提取时间戳 (lmp_TIMESTAMP.lammpstrj)
        timestamp=$(echo "$file" | sed -E 's/^lmp_([0-9.]+)\.lammpstrj$/\1/')
        
        # 检查时间戳是否有效
        if [[ "$timestamp" =~ ^[0-9.]+$ ]]; then
            # 筛选时间戳 > init_time 的文件
            if awk "BEGIN {exit !($timestamp > $init_time)}"; then
                # 获取文件创建时间
                if [[ "$(uname)" == "Darwin" ]]; then
                    # macOS
                    ctime=$(stat -f "%B" "$file")
                else
                    # Linux
                    ctime=$(stat -c "%Z" "$file")
                fi
                
                echo "$ctime $timestamp $file" >> "$temp_file"
            fi
        fi
    fi
done

# 检查是否有符合条件的文件
if [[ ! -s "$temp_file" ]]; then
    echo "错误: 没有找到时间戳大于 $init_time 的文件"
    exit 1
fi

# 按文件创建时间(ctime)升序排序
sort -n "$temp_file" > "${temp_file}.sorted"

echo "符合条件的文件（按ctime排序）:"
cat "${temp_file}.sorted"

# 读取排序后的文件信息
declare -a file_info
while IFS=' ' read -r ctime timestamp filename; do
    file_info+=("$ctime:$timestamp:$filename")
done < "${temp_file}.sorted"

# 分析锯齿段
echo "分析锯齿段..."
declare -a segments
current_segment=()
prev_timestamp=""

for info in "${file_info[@]}"; do
    IFS=':' read -r ctime timestamp filename <<< "$info"
    
    # 检查是否需要开始新的锯齿段
    if [[ -n "$prev_timestamp" ]] && awk "BEGIN {exit !($timestamp < $prev_timestamp)}"; then
        # 时间回到比前一个文件小，开始新锯齿段
        if [[ ${#current_segment[@]} -gt 0 ]]; then
            segments+=("$(IFS='|'; echo "${current_segment[*]}")")
        fi
        current_segment=()
        echo "检测到新锯齿段，时间从 $prev_timestamp 回到 $timestamp"
    fi
    
    current_segment+=("$info")
    prev_timestamp="$timestamp"
done

# 添加最后一个段
if [[ ${#current_segment[@]} -gt 0 ]]; then
    segments+=("$(IFS='|'; echo "${current_segment[*]}")")
fi

echo "总共分成 ${#segments[@]} 个锯齿段"

# 处理每个锯齿段
output_file="processed_timeline.txt"
echo "# 处理后的时间线" > "$output_file"
echo "# 格式: 新时间戳 原时间戳 文件名" >> "$output_file"

current_time="$init_time"

for i in "${!segments[@]}"; do
    segment="${segments[$i]}"
    IFS='|' read -ra segment_files <<< "$segment"
    
    echo "处理锯齿段 $((i+1))..."
    
    # 计算段内时间信息（清空数组避免累积）
    seg_timestamps=()
    seg_files=()

    # 计算段内时间信息
    declare -a seg_timestamps
    declare -a seg_files
    
    for file_info in "${segment_files[@]}"; do
        IFS=':' read -r ctime timestamp filename <<< "$file_info"
        seg_timestamps+=("$timestamp")
        seg_files+=("$filename")
    done
    
    echo ${segment_files[@]}

    # 计算段的相对时间跨度
    min_time="${seg_timestamps[0]}"
    max_time="${seg_timestamps[0]}"
    
    for ts in "${seg_timestamps[@]}"; do
        if awk "BEGIN {exit !($ts < $min_time)}"; then
            min_time="$ts"
        fi
        if awk "BEGIN {exit !($ts > $max_time)}"; then
            max_time="$ts"
        fi
    done
    
    time_span=$(awk "BEGIN {printf \"%.6f\", $max_time - $init_time}")
    
    # # 更新当前时间（段开始时间）
    # if [[ "$i" -gt 0 ]]; then
    #     current_time=$(awk "BEGIN {printf \"%.6f\", $current_time + $time_span}")
    # fi
    
    segment_start_time="$current_time"
    
    echo "  段 $((i+1)): 原时间范围 [$min_time, $max_time], 跨度: $time_span"
    echo "  段 $((i+1)): 新开始时间: $segment_start_time"
    
    # 重新分配段内文件时间并立即输出当前段结果
    echo "  段 $((i+1)) 处理结果:"
    for j in "${!seg_files[@]}"; do
        original_ts="${seg_timestamps[$j]}"
        relative_offset=$(awk "BEGIN {printf \"%.4f\", $original_ts - $init_time}")
        new_timestamp=$(awk "BEGIN {printf \"%.4f\", $segment_start_time + $relative_offset}")
        
        # 立即写入文件和输出当前段结果
        echo "$new_timestamp $original_ts ${seg_files[$j]}" >> "$output_file"
        echo "    ${seg_files[$j]}: $original_ts -> $new_timestamp"
    done
    
    # 更新当前时间到段结束
    current_time=$(awk "BEGIN {printf \"%.4f\", $segment_start_time + $time_span}")
    
    echo ""
done

echo "处理完成！结果已保存到 $output_file"

# 生成重命名脚本
rename_script="rename_files.sh"
echo "#!/bin/bash" > "$rename_script"
echo "# 文件重命名脚本" >> "$rename_script"
echo "# 根据新的时间戳重命名文件" >> "$rename_script"
echo "" >> "$rename_script"

while IFS=' ' read -r new_ts orig_ts filename; do
    if [[ "$new_ts" =~ ^[0-9]+\.?[0-9]*$ ]]; then
        new_filename="lmp_${new_ts}.lammpstrj"
        if [[ "$filename" != "$new_filename" ]]; then
            echo "mv '$filename' '$new_filename'" >> "$rename_script"
        fi
    fi
done < <(grep -v "^#" "$output_file")

chmod +x "$rename_script"
echo "重命名脚本已生成: $rename_script"
echo "执行 ./$rename_script 来重命名文件"

# 清理临时文件
rm -f "${temp_file}.sorted"