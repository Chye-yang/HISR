import ast
import ipaddress
import csv
import json

def bytes_to_hex_string(b_data):
    """将bytes数据转换为十六进制字符串"""
    return b_data.hex()

def try_decode_as_string(b_data, encoding='utf-8', errors='replace'):
    """尝试将bytes数据解码为字符串，如果失败则替换不可解码字符"""
    try:
        return b_data.decode(encoding, errors=errors)
    except Exception:
        return f"[无法解码: {bytes_to_hex_string(b_data)}]"

def try_parse_ip_from_prefix(b_data):
    """尝试将bytes数据的前4个字节解析为IPv4地址"""
    if len(b_data) >= 4:
        try:
            ip_addr = ipaddress.IPv4Address(b_data[:4])
            return str(ip_addr)
        except ipaddress.AddressValueError:
            pass
    return "N/A" # 表示无法解析为IP地址

# --- 主要处理逻辑 ---
file_path = 'ground_truth.txt' # 您的数据文件路径
parsed_data = []

print(f"--- 正在读取和解析文件: {file_path} ---")
with open(file_path, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if not line: # 跳过空行
            continue

        parts = line.split('\t')
        if len(parts) == 2:
            try:
                # 使用 ast.literal_eval 安全地解析 Python 字节字符串字面量
                byte_str_literal = parts[0]
                byte_data = ast.literal_eval(byte_str_literal)
                count = int(parts[1])

                # 收集原始解析的数据
                parsed_data.append({
                    'original_bytes': byte_data,
                    'count': count
                })

            except (ValueError, SyntaxError) as e:
                print(f"警告: 第 {line_num} 行解析失败 '{line}' - 错误: {e}")
        else:
            print(f"警告: 第 {line_num} 行格式不正确 '{line}'")

print(f"--- 解析完成，共发现 {len(parsed_data)} 条有效记录 ---")

# --- 1. 打印转换后的数据（转换为十六进制） ---
print("\n--- 转换结果（Bytes -> 十六进制字符串）---")
for item in parsed_data:
    hex_representation = bytes_to_hex_string(item['original_bytes'])
    print(f"二进制 (Hex): {hex_representation}, 计数: {item['count']}")

# --- 2. 尝试根据猜测的结构进行解析 (例如，IP地址) ---
print("\n--- 尝试根据前4字节解析为IPv4地址（假设）---")
# 注意：这只是一个示例。如果实际数据结构不同，需要对应调整解析逻辑。
# 例如，如果IP地址不在开头，或者有其他字段，都需要精确定义。
for item in parsed_data:
    hex_representation = bytes_to_hex_string(item['original_bytes'])
    possible_ip = try_parse_ip_from_prefix(item['original_bytes'])
    print(f"二进制 (Hex): {hex_representation}, 可能的IP: {possible_ip}, 计数: {item['count']}")

# --- 3. 保存到 CSV 文件 ---
csv_file_path = 'network_flow_output.csv'
print(f"\n--- 保存解析结果到 CSV 文件: {csv_file_path} ---")
with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)
    # 写入 CSV 表头
    csv_writer.writerow(['Bytes_Hex', 'Possible_IP', 'Count'])
    for item in parsed_data:
        hex_representation = bytes_to_hex_string(item['original_bytes'])
        possible_ip = try_parse_ip_from_prefix(item['original_bytes'])
        csv_writer.writerow([hex_representation, possible_ip, item['count']])
print("CSV 文件保存成功。")

# --- 4. 保存到 JSON 文件 ---
json_file_path = 'network_flow_output.json'
print(f"\n--- 保存解析结果到 JSON 文件: {json_file_path} ---")
json_output_data = []
for item in parsed_data:
    hex_representation = bytes_to_hex_string(item['original_bytes'])
    possible_ip = try_parse_ip_from_prefix(item['original_bytes'])
    json_output_data.append({
        'bytes_hex': hex_representation,
        'possible_ip': possible_ip,
        'count': item['count']
    })

with open(json_file_path, 'w', encoding='utf-8') as jsonfile:
    json.dump(json_output_data, jsonfile, indent=4, ensure_ascii=False) # indent用于美化输出，ensure_ascii=False支持中文
print("JSON 文件保存成功。")

