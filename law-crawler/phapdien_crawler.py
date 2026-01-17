import json
import os
import re
from utils import *
from bs4 import BeautifulSoup
import uuid
from tqdm import tqdm
import logging

logging.basicConfig(
    filename='phapdien_crawler.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

input_file = "BoPhapDienDienTu/jsonData.js"
output_dir = "phap_dien"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
logging.info(f"Reading {input_file}...")

with open(input_file, 'r', encoding='utf-8-sig') as f:
    lines = f.readlines()
    
for line in lines:
    line = line.strip()

    if not line:
        logging.warning(f"Empty line: {line}")
        continue
        
    # Match "var VariableName = [...]"
    match = re.match(r'var\s+(\w+)\s*=\s*(.*)', line)
    if match:
        var_name = match.group(1)
        json_str = match.group(2)
        
        # Remove trailing semicolon if present
        if json_str.endswith(';'):
            json_str = json_str[:-1]
            
        logging.info(f"Exporting {var_name}...")
        
        try:
            data = json.loads(json_str)
            output_file = os.path.join(output_dir, f"{var_name}.json")
            with open(output_file, 'w', encoding='utf-8') as out:
                json.dump(data, out, ensure_ascii=False, indent=4)
            logging.info(f"Saved to {output_file}")
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON for {var_name}: {e}")

with open("phap_dien/jdAllTree.json", "r") as file:
    tree_nodes = json.load(file)

demuc_dir = "phap_dien/demuc"

chuong_data = []
dieu_data = []
lienquan_data = []
# file_data = []

demuc_files = os.listdir(demuc_dir)
demuc_files.sort()

# Load Checkpoints if they exist
processed_demuc_ids = set()
if os.path.exists(output_dir + "/Chuong.json"):
    try:
        with open(output_dir + "/Chuong.json", 'r', encoding='utf-8') as f:
            chuong_data = json.load(f)
            processed_demuc_ids = {c.get("DeMucID") for c in chuong_data if c.get("DeMucID")}
            logging.info(f"Loaded {len(chuong_data)} chapters from checkpoint. Found {len(processed_demuc_ids)} processed DeMucs.")
    except Exception as e:
        logging.error(f"Error loading Chuong.json checkpoint: {e}")

if os.path.exists(output_dir + "/Dieu.json"):
    try:
        with open(output_dir + "/Dieu.json", 'r', encoding='utf-8') as f:
            dieu_data = json.load(f)
            logging.info(f"Loaded {len(dieu_data)} articles from checkpoint.")
    except Exception as e:
        logging.error(f"Error loading Dieu.json checkpoint: {e}")

if os.path.exists(output_dir + "/LienQuan.json"):
    try:
        with open(output_dir + "/LienQuan.json", 'r', encoding='utf-8') as f:
            lienquan_data = json.load(f)
            logging.info(f"Loaded {len(lienquan_data)} related items from checkpoint.")
    except Exception as e:
        logging.error(f"Error loading LienQuan.json checkpoint: {e}")

for file in tqdm(demuc_files, desc="Processing files"):
    demuc_id = file.split(".")[0]
    
    if demuc_id in processed_demuc_ids:
        continue

    with open(demuc_dir + '/' + file, "r") as demuc_file:
        demuc_html = demuc_file.read()

    # with open("phap_dien/demuc/fa959814-d73c-4253-8857-795185e44d16.html", "r") as demuc_file:
    #     demuc_html = demuc_file.read()

    demuc_html = BeautifulSoup(demuc_html, "html.parser")
    demuc_id = file.split(".")[0]
    demuc_nodes = [node for node in tree_nodes if node["DeMucID"] == demuc_id]

    if len(demuc_nodes) == 0:
        logging.warning("Không tìm thấy node cho đề mục: " + demuc_dir + '/' + file)
        continue

    chude_id = demuc_nodes[0].get('ChuDeID')
    demuc_chuongs = [node for node in demuc_nodes if node["TEN"].startswith("Chương ")]
    demuc_dieus = [node for node in demuc_nodes if node not in demuc_chuongs]

    for chuong in demuc_chuongs:
        chuong.update({"STT": convert_roman_to_num(chuong.get("ChiMuc"))})

    if len(demuc_chuongs) == 0:
        chuong = {"MAPC": str(uuid.uuid4()), "TEN": "", "STT": 0, "DeMucID": demuc_id}
        demuc_chuongs.append(chuong)

    chuong_data.extend(demuc_chuongs)

    stt = 0
    for dieu in tqdm(demuc_dieus, desc=f"Articles in {file}", leave=False):
        mapc = dieu["MAPC"]
        # mapc = "250120000000000040000380000000000000000000802592301130000700"

        if len(demuc_chuongs) == 1:
            dieu["ChuongMAPC"] = demuc_chuongs[0].get("MAPC")
        else:
            for chuong in demuc_chuongs:
                if mapc.startswith(chuong.get("MAPC")):
                    dieu["ChuongMAPC"] = chuong.get("MAPC")
                    break

        if "ChuongMAPC" not in dieu:
            logging.error(f"Error ChuongMAPC not found: {mapc} in {file}")
            pass
        
        try:
            dieu_html_list = demuc_html.select(f'a[name="{mapc}"]')
            if not dieu_html_list:
                logging.warning(f"Warning: No anchor for {mapc} in {file}")
            dieu_html = dieu_html_list[0]
        except Exception as e:
            logging.error(f"Error processing {mapc} in {file}: {e}")

        # ten = ""
        # try:
        #     ten = str(dieu_html.next_sibling).strip()
        # except Exception as e:
        #     print(f"Lỗi khi extract tên điều {mapc} in {file}: {e}")

        noidung_html = dieu_html.parent.find_next("p", {"class": "pNoiDung"})  
        noidung = ""
        if noidung_html:
            for content in noidung_html.contents:
                if content.name == "table":
                    table_md = table_to_md(content)
                    noidung += table_md + "\n"
                else:
                    noidung += str(content.get_text().strip()) + "\n"

        stt += 1

        references = extract_vbqppl(dieu_html)
        dieu.update({
            "STT": stt,
            "NoiDung": noidung, 
            "VBQPPL": references
        })
        
        # files = extract_files(dieu_html, mapc)
        related_dieus, related_vbqppls = extract_lienquan(dieu_html)
        
        lienquan_data.extend(related_dieus)
        lienquan_data.extend(related_vbqppls)
        # file_data.extend(files)

    dieu_data.extend(demuc_dieus)
    
    # Checkpoint saving every 10 files
    if (demuc_files.index(file) + 1) % 10 == 0:
        logging.info(f"Checkpoint save at file {file}")
        with open(output_dir + "/Chuong.json", 'w', encoding='utf-8') as out:
            json.dump(chuong_data, out, ensure_ascii=False, indent=4)
        with open(output_dir + "/Dieu.json", 'w', encoding='utf-8') as out:
            json.dump(dieu_data, out, ensure_ascii=False, indent=4)
        with open(output_dir + "/LienQuan.json", 'w', encoding='utf-8') as out:
            json.dump(lienquan_data, out, ensure_ascii=False, indent=4)

with open(output_dir + "/Chuong.json", 'w', encoding='utf-8') as out:
    json.dump(chuong_data, out, ensure_ascii=False, indent=4)

with open(output_dir + "/Dieu.json", 'w', encoding='utf-8') as out:
    json.dump(dieu_data, out, ensure_ascii=False, indent=4)

# with open(output_dir + "/File.json", 'w', encoding='utf-8') as out:
#     json.dump(file_data, out, ensure_ascii=False, indent=4)

with open(output_dir + "/LienQuan.json", 'w', encoding='utf-8') as out:
    json.dump(lienquan_data, out, ensure_ascii=False, indent=4)