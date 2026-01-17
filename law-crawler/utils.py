import re

def convert_roman_to_num(roman_num):
    roman_num = roman_num.upper()
    roman_to_num = {'I': 10, 'V': 50, 'X': 100, 'L': 500, 'C': 1000, 'D': 5000, 'M': 10000}
    alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    num = 0
    for i in range(len(roman_num)):
        romain_char = roman_num[i]
        if romain_char not in roman_to_num.keys():
            num += alphabet.index(romain_char) + 1
            continue
        if i > 0 and roman_to_num[romain_char] > roman_to_num[roman_num[i - 1]]:
            num += roman_to_num[romain_char] - 2 * roman_to_num[roman_num[i - 1]]
        else:
            num += roman_to_num[romain_char]
    return num

def table_to_md(table) -> str:
    rows = []
    for row in table.find_all("tr"):
        cells = [cell.get_text(strip=True) for cell in row.find_all(["th", "td"])]
        if cells: # Avoid empty rows
            rows.append(cells)
            
    if not rows:
        return ""
        
    header = rows[0]
    body = rows[1:]
    md = "| " + " | ".join(header) + " |\n"
    md += "| " + " | ".join(["---"] * len(header)) + " |\n"
    for r in body:
        md += "| " + " | ".join(r) + " |\n"
    return md

def extract_vbqppl(dieu_anchor):
    """
    Extracts legal document references (VBQPPL) following an article anchor.
    dieu_anchor is the <a> tag (article marker).
    """
    references = []
    if not dieu_anchor or not (dieu_anchor.parent):
        return references

    curr = dieu_anchor.parent.find_next_sibling()
    
    # Collect all paragraphs that are notes (pGhiChu)
    while curr and curr.name == 'p':
        classes = curr.get('class', [])
        if 'pGhiChu' in classes:
            links = curr.select("a")
            if not links:
                # Keep text even if no link
                references.append({"name": curr.get_text(strip=True), "link": None})
            else:
                for a in links:
                    href = a.get('href')
                    if href and href != '#':
                        references.append({"name": curr.get_text(strip=True), "link": href})
            curr = curr.find_next_sibling()
        else:
            break
            
    return references

# def extract_files(dieu_anchor, mapc):
#     """
#     Extracts file/attachment links following the article content.
#     Files are <a> tags that appear after pNoiDung and before pDieu.
#     """
#     files = []
#     if not dieu_anchor or not dieu_anchor.parent:
#         return files
    
#     # Start looking after pDieu
#     curr = dieu_anchor.parent.find_next_sibling()
    
#     # Skip pGhiChu and pNoiDung to find <a> tags
#     while curr:
#         classes = curr.get('class', [])
#         if 'pGhiChu' in classes or 'pNoiDung' in classes:
#             # If it's pNoiDung, files might be immediately after it
#             if 'pNoiDung' in classes:
#                 # Search siblings of this pNoiDung
#                 sibling = curr.find_next_sibling()
#                 while sibling and sibling.name == "a":
#                     link = sibling.get("href")
#                     if link:
#                         files.append({
#                             "dieu_id": mapc,
#                             "link": link,
#                             "path": ""
#                         })
#                     sibling = sibling.find_next_sibling()
#             curr = curr.find_next_sibling()
#         elif curr.name == 'a':
#             # This handles case where <a> is a top-level sibling of pDieu/pNoiDung
#             link = curr.get("href")
#             if link:
#                 files.append({
#                     "DieuMAPC": mapc,
#                     "link": link,
#                     "path": ""
#                 })
#             curr = curr.find_next_sibling()
#         elif 'pDieu' in classes:
#             # Stop if we hit the next article
#             break
#         else:
#             curr = curr.find_next_sibling()
            
#     return files

def extract_lienquan(dieu_anchor):
    """
    Extracts related data (lien quan) from pChiDan paragraphs.
    Returns two lists: (related_dieus, related_vbqppls)
    """
    related_dieus = []
    related_vbqppls = []
    
    if not dieu_anchor or not dieu_anchor.parent:
        return related_dieus, related_vbqppls

    curr = dieu_anchor.parent.find_next_sibling()
    mapc_source = dieu_anchor.get('name')

    while curr:
        classes = curr.get('class', [])
        if 'pChiDan' in classes:
            links = curr.select("a")
            for a in links:
                onclick = a.get('onclick', '')
                if 'ViewNoiDungPhapDien' in onclick:
                    # Extract MAPC from ViewNoiDungPhapDien('MAPC')
                    match = re.search(r"ViewNoiDungPhapDien\('(\w+)'\)", onclick)
                    if match:
                        related_dieus.append({
                            "source_MAPC": mapc_source,
                            "target_MAPC": match.group(1)
                        })
                else:
                    href = a.get('href')
                    if href and href != '#':
                        related_vbqppls.append({
                            "DieuMAPC": mapc_source,
                            "name": a.get_text(strip=True),
                            "link": href
                        })
        elif 'pDieu' in classes:
            break
        curr = curr.find_next_sibling()
            
    return related_dieus, related_vbqppls