import requests
from bs4 import BeautifulSoup
import time
import random
import pandas as pd
import re


BASE_URL = "https://luatvietnam.vn"
START_URL = "https://luatvietnam.vn/luat-su-tu-van.html"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
}

def get_soup(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        if response.status_code == 200:
            return BeautifulSoup(response.content, 'html.parser')
    except Exception as e:
        print(f"Lỗi kết nối {url}: {e}")
    return None

def get_total_pages(soup):
    max_page = 1
    try:
        pagination = soup.find('div', class_='pag-right')
        if pagination:
            links = pagination.find_all('a', class_='page-numbers')
            for link in links:
                try:
                    text = link.get_text(strip=True)
                    if text.isdigit():
                        page_num = int(text)
                        if page_num > max_page:
                            max_page = page_num
                except:
                    continue
            
    except Exception as e:
        print(f"Không xác định được tổng số trang: {e}")
    
    return max_page

def get_links_from_soup(soup):
    links_data = []
    try:
        table = soup.find('table', class_='table-hoi-dap')
        if table:
            rows = table.select('tbody tr')
            for row in rows:
                h3_tag = row.find('h3', class_='article-hoi-dap')
                if h3_tag:
                    a_tag = h3_tag.find('a')
                    if a_tag:
                        title = a_tag.get_text(strip=True)
                        href = a_tag.get('href')
                        full_link = BASE_URL + href if href.startswith('/') else href
                        links_data.append({'title': title, 'url': full_link})
    except Exception as e:
        print(f"Lỗi bóc tách: {e}")
    return links_data


def main():
    
    first_page_soup = get_soup(START_URL)
    if not first_page_soup:
        print("Không thể truy cập trang chủ để đếm số trang.")
        return

    total_pages = get_total_pages(first_page_soup)
    print(f"Tìm thấy tổng cộng: {total_pages} trang.")

    all_links = []
    
    for page in range(1, total_pages + 1):
        print(f"Dang crawl trang {page}/{total_pages} ...", end=" ")
        
        if page == 1:
            url = START_URL
        else:
            url = f"{START_URL}?page={page}"
        
        soup = get_soup(url)
        if soup:
            new_links = get_links_from_soup(soup)
            all_links.extend(new_links)
            print(f"lay duoc {len(new_links)} link. Tong: {len(all_links)}")
        else:
            print("Lỗi tải trang.")

        time.sleep(random.uniform(1, 3))

    if all_links:
        df = pd.DataFrame(all_links)
        filename = 'all_question_links.csv'
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"\Đã lưu {len(all_links)} link vào file '{filename}'")
    else:
        print("Không lấy được link nào.")

if __name__ == "__main__":
    main()