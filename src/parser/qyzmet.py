import asyncio
import csv
import urllib.parse
import os
from playwright.async_api import async_playwright

async def scrape_qyzmet_deep(keywords, max_pages=20):
    os.makedirs('data', exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False) 
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
        page = await context.new_page()

        for keyword in keywords:
            
            filename = f'data/qyzmet_{keyword}.csv'
            safe_keyword = urllib.parse.quote_plus(keyword)
            
            with open(filename, mode='w+', encoding='utf-8-sig', newline="") as file:
                writer = csv.writer(file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(['title', 'salary', 'city', 'job', 'publish_date', 'requirements', 'responsibilities', 'url', 'source'])
                
                for p_num in range(1, max_pages + 1):
                    url = f"https://qyzmet.kz/vacansii?q={safe_keyword}&page={p_num}"
                    print(f"Page {p_num}: {url}")
                    
                    try:
                        await page.goto(url, timeout=15000)
                        await asyncio.sleep(2)
                    except Exception:
                        print("Problem with Page")
                        continue
                    
                    links_locators = await page.locator('article.job a.job-title').all()
                    links = []
                    for loc in links_locators:
                        href = await loc.get_attribute('href')
                        if href: 
                            if not href.startswith('http'):
                                href = 'https://qyzmet.kz' + href
                            links.append(href)
                    
                    if not links:
                        print("Have not got any links ")
                        break
                        
                    print(f" Find {len(links)} links")
                    
                    for url_vacancy in links:
                        try:
                            await page.goto(url_vacancy, timeout=20000, wait_until='domcontentloaded')
                            await asyncio.sleep(2.5) # Даем время на редирект
                            
                            title_loc = page.locator('h1')
                            title = await title_loc.first.inner_text() if await title_loc.count() > 0 else ""
                            
                            if not title:
                                title = await page.title()

                            description = ""
                            for selector in ['main', 'article', '[class*="vacancy"]', '[class*="description"]', 'body']:
                                loc = page.locator(selector)
                                if await loc.count() > 0:
                                    text = await loc.first.inner_text()
                                    if text and len(text) > 150:
                                        description = text
                                        break
                            
                            if title and description:
                                writer.writerow([
                                    title.strip().replace('\n', ' '), 
                                    '', # salary 
                                    '', # city
                                    '', # company
                                    '', # publish_date
                                    '', # requirements
                                    description.strip().replace('\n', ' '), 
                                    url_vacancy, 
                                    'qyzmet_deep_parser'
                                ])
                            else:
                                print(f" Can't take information from {page.url}")
                                
                        except Exception as e:
                            print("Time Out")
            
            print(f"Dat was saved{filename}")
        
        await browser.close()
        print("\nParsing ended successfully")

def main():
    it_professions = [
        "Frontend",
        "Backend",
        "Full stack",
        "DevOps",
        "QA Engineer",
        "Data Scientist",
        "Data Analyst",
        "Data Engineer",
        "Mobile Developer",
        "Cybersecurity"
    ]
    
    asyncio.run(scrape_qyzmet_deep(it_professions, max_pages=20))

if __name__ == '__main__':
    main()