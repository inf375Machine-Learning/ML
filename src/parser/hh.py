import time
import csv
import urllib.parse
import re
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException


def setup_driver():
    options = Options()
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
    driver = webdriver.Chrome(options=options)
    driver.execute_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )
    return driver



def get_links(driver, keyword, country_code, max_pages):
    links = []
    safe_keyword = urllib.parse.quote_plus(keyword)
    base_domain = "hh.kz" if country_code == 40 else "hh.ru"

    for page in range(max_pages):
        url = f"https://{base_domain}/search/vacancy?text={safe_keyword}&area={country_code}&page={page}&items_on_page=50"
        driver.get(url)
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, '[data-qa="vacancy-serp__vacancy"]')
                )
            )
        except TimeoutException:
            break

        time.sleep(2)
        cards = driver.find_elements(By.CSS_SELECTOR, '[data-qa="serp-item__title"]')
        if not cards:
            break

        for card in cards:
            href = card.get_attribute("href")
            if href:
                links.append(href.split("?")[0])

    return list(set(links))


def parse_vacancy(driver, url):
    data = {
        k: ""
        for k in [
            "title",
            "salary",
            "city",
            "job",
            "publish_date",
            "requirements",
            "responsibilities",
            "schedule",
            "experience",
            "employment",
        ]
    }
    data["url"] = url

    try:
        driver.get(url)
        WebDriverWait(driver, 5).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, '[data-qa="vacancy-title"]')
            )
        )
    except TimeoutException:
        return data

    try:
        data["title"] = driver.find_element(
            By.CSS_SELECTOR, '[data-qa="vacancy-title"]'
        ).text
    except NoSuchElementException:
        pass

    try:
        data["salary"] = (
            driver.find_element(By.CSS_SELECTOR, '[data-qa="vacancy-salary"]')
            .text.replace("до вычета налогов", "")
            .replace("на руки", "")
            .strip()
        )
    except NoSuchElementException:
        pass

    try:
        data["job"] = driver.find_element(
            By.CSS_SELECTOR, '[data-qa="vacancy-company-name"]'
        ).text
    except NoSuchElementException:
        pass

    try:
        loc = driver.find_elements(By.CSS_SELECTOR, '[data-qa="vacancy-view-location"]')
        if loc:
            data["city"] = loc[0].text.split(",")[0]
        else:
            data["city"] = driver.find_element(
                By.CSS_SELECTOR, '[data-qa="vacancy-view-raw-address"]'
            ).text.split(",")[0]
    except NoSuchElementException:
        pass

    try:
        data["experience"] = driver.find_element(
            By.CSS_SELECTOR, '[data-qa="vacancy-experience"]'
        ).text
    except NoSuchElementException:
        pass

    try:
        emp_mode = driver.find_element(
            By.CSS_SELECTOR, '[data-qa="vacancy-view-employment-mode"]'
        ).text
        parts = [p.strip() for p in emp_mode.split(",")]
        data["employment"] = parts[0] if len(parts) > 0 else ""
        data["schedule"] = parts[1] if len(parts) > 1 else ""
    except NoSuchElementException:
        pass

    try:
        full_desc = driver.find_element(
            By.CSS_SELECTOR, '[data-qa="vacancy-description"]'
        ).text.replace("\n", " ")
        resp_match = re.search(
            r"(Обязанности:|Чем предстоит заниматься:|Задачи:)(.*?)(Требования:|Условия:|Что мы предлагаем:|$)",
            full_desc,
            re.IGNORECASE,
        )
        req_match = re.search(
            r"(Требования:|Что мы ждем:|Ожидания:)(.*?)(Обязанности:|Условия:|Что мы предлагаем:|$)",
            full_desc,
            re.IGNORECASE,
        )

        if resp_match:
            data["responsibilities"] = resp_match.group(2).strip()
        if req_match:
            data["requirements"] = req_match.group(2).strip()
        if not data["requirements"] and not data["responsibilities"]:
            data["requirements"] = full_desc.strip()
    except NoSuchElementException:
        pass

    return data


def scrape_hh_selenium(keyword, country_code=40, max_pages=3):
    driver = setup_driver()
    filename = f"../data/hh/{keyword}_full.csv"

    print(f"Сбор ссылок для '{keyword}'...")
    links = get_links(driver, keyword, country_code, max_pages)

    if not links:
        driver.quit()
        return

    try:
        with open(filename, mode="w+", encoding="utf-8-sig", newline="") as file:
            writer = csv.writer(file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
            writer.writerow(
                [
                    "title",
                    "salary",
                    "city",
                    "job",
                    "requirements",
                    "responsibilities",
                    "experience",
                    "employment",
                    "url",
                ]
            )

            for i, url in enumerate(links):
                print(f"Обработка {i + 1}/{len(links)}")
                details = parse_vacancy(driver, url)

                writer.writerow(
                    [
                        details["title"],
                        details["salary"],
                        details["city"],
                        details["job"],
                        details["requirements"],
                        details["responsibilities"],
                        details["experience"],
                        details["employment"],
                        details["url"],
                    ]
                )
                time.sleep(1.5)

    finally:
        driver.quit()


def main():
    country_codes = {
        "UA": 5,
        "AZ": 9,
        "BY": 16,
        "GE": 28,
        "KZ": 40,
        "KG": 48,
        "UZ": 97,
        "RU": 113,
    }
    country = input("Initial of country: ").upper()
    if country not in country_codes:
        return

    country_code = country_codes[country]
    keyword = input("Keywords like Data Science: ")
    scrape_hh_selenium(keyword, country_code=country_code, max_pages=4)


if __name__ == "__main__":
    main()
