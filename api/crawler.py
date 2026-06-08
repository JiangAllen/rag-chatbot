from flask import Flask, request, jsonify
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import requests
import psycopg2

app = Flask(__name__)
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "zh-TW,zh;q=0.5",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Referer": "",
}
output = {"subject": "", "body": ""}

def verify_token(token, api_id):
    connection = psycopg2.connect(
        database="DigitimesDB",
        user="dotvb",
        password="internaluse",
        host="digitimes-postgresql.ctvkso74w6yb.ap-northeast-1.rds.amazonaws.com"
    )
    import hashlib
    token = hashlib.md5(token.encode()).hexdigest()
    code = 0
    msg = ''
    api_seq = 0
    sql = """SELECT a.apiid, a.apisecret, b.seq  FROM deng_authlog b LEFT JOIN deng_authuser a ON a.apiid = b.apiid WHERE b.requesttoken = %s AND b.expiry > CURRENT_TIMESTAMP  AND a.apiid = %s"""
    with connection.cursor() as cur:
        cur.execute(sql, (token, api_id))
        get_rs = cur.fetchone()
    print(get_rs)
    if get_rs is None:
        code = 403
        msg = "Access denied"
        result = False
    else:
        api_secret = get_rs[1]
        api_seq = get_rs[2]
        rule_token = hashlib.md5(hashlib.sha256((api_id + ':' + api_secret + ':' + str(api_seq)).encode()).hexdigest().encode()).hexdigest()
        if token != rule_token:
            code = 403
            msg = "Access denied"
            result = False
        else:
            result = True
    print({"code": code, "msg": msg, "api_seq": api_seq})
    return result

def call_diffbot(url):
    response = requests.get("https://api.diffbot.com/v3/article?token=af78e68f2016813fe9c659d98abc2e47&url={}".format(url), headers={"accept": "application/json"})
    output["subject"] = response.json().get("objects", [{}])[0].get("title", "subject not found")
    output["body"] = response.json().get("objects", [{}])[0].get("text", "body not found")

def reuters(soup):
    output["subject"] = (soup.find("h1") or {"get_text": lambda: "subject not found"}).get_text(strip=True, separator="\n").strip()
    output["body"] = "".join(p.text.encode().decode("unicode_escape") for p in paragraphs) if (paragraphs := soup.find_all("div", {"data-testid": lambda x: x and x.startswith("paragraph-")})) else "body not found"

def cnbc(soup):
    output["subject"] = (soup.find("h1") or {"get_text": lambda: "subject not found"}).get_text(strip=True, separator="\n").strip()
    output["body"] = "".join([p.get_text(strip=True, separator="\n").strip() for p in soup.select("div.group p")]) if soup.select("div.group p") else "body not found"

def tomshardware(soup):
    output["subject"] = (soup.find("h1") or {"get_text": lambda: "subject not found"}).get_text(strip=True, separator="\n").strip()
    output["body"] = "".join([p.get_text(strip=True, separator="\n").strip() for p in (soup.find("div", {"id": "article-body"}) or {}).find_all("p")]) if soup.find("div", {"id": "article-body"}) else "body not found"

def techcrunch(soup):
    output["subject"] = (soup.find("h1") or {"get_text": lambda: "subject not found"}).get_text(strip=True, separator="\n").strip()
    output["body"] = "".join([p.get_text(strip=True, separator="\n").strip() for p in soup.find_all("p", class_="wp-block-paragraph")]) if soup.find_all("p", class_="wp-block-paragraph") else "body not found"

def theverge(soup):
    output["subject"] = (soup.find("h1") or {"get_text": lambda: "subject not found"}).get_text(strip=True, separator="\n").strip()
    output["body"] = "".join([p.get_text(strip=True, separator="\n").strip() for div in soup.find_all("div", class_="duet--article--article-body-component") for p in div.find_all("p")]) or "body not found"

def theregister(soup):
    output["subject"] = (soup.find("h1") or {"get_text": lambda: "subject not found"}).get_text(strip=True, separator="\n").strip()
    output["body"] = "".join([p.get_text(strip=True, separator="\n").strip() for p in soup.find("div", id="body").find_all("p")])

def electrek_co(soup):
    output["subject"] = (soup.find("h1") or {"get_text": lambda: "subject not found"}).get_text(strip=True, separator="\n").strip()
    output["body"] = "".join([p.get_text(strip=True, separator="\n").strip() for p in soup.find_all("p")]) if soup.find_all("p") else "body not found"

def ninetofivemac(soup):
    output["subject"] = (soup.find("h1") or {"get_text": lambda: "subject not found"}).get_text(strip=True, separator="\n").strip()
    output["body"] = "".join([p.get_text(strip=True, separator="\n").strip() for p in soup.find_all("p")]) if soup.find("div", class_="container med post-content") else "body not found"

def ninetofivegoogle(soup):
    output["subject"] = (soup.find("h1") or {"get_text": lambda: "subject not found"}).get_text(strip=True, separator="\n").strip()
    output["body"] = "".join([p.get_text(strip=True, separator="\n").strip() for p in soup.find_all("p")]) if soup.find("div", class_="container med post-content") else "body not found"

def appleinsider(soup):
    output["subject"] = (soup.find("h1") or {"get_text": lambda: "subject not found"}).get_text(strip=True, separator="\n").strip()
    output["body"] = "".join([para.get_text(strip=True, separator="\n").strip() for para in soup.find_all("p", class_="col-sm-12") if para.find_parent("div", class_="article-body")][0]) if [para.get_text(strip=True, separator="\n").strip() for para in soup.find_all("p", class_="col-sm-12") if para.find_parent("div", class_="article-body")] else "body not found"

def macrumors(soup):
    output["subject"] = (soup.find("h1") or {"get_text": lambda: "subject not found"}).get_text(strip=True, separator="\n").strip()
    output["body"] = "".join([p.get_text(strip=True, separator="\n").strip() for p in soup.select(".content--2u3grYDr.js-content p") if p.get_text(strip=True, separator="\n").strip()]) or "body not found"

def yahoo_jp(soup): # 有上下頁
    pass

def newswitch_jp(soup): # 抓不到標題
    output["subject"] = soup.find("h1").get_text(strip=True, separator="\n").strip() if soup.find("h1") else "subject not found"
    output["body"] = "".join(p.get_text(strip=True, separator="\n").strip() for p in soup.select("div.article-detail-body p"))

def monoist_itmedia(soup): # 有上下頁
    output["subject"] = soup.find("h1").get_text(strip=True, separator="\n").strip() if soup.find("h1") else "subject not found"
    output["body"] = "".join(p.get_text(strip=True, separator="\n").strip() for p in soup.find_all("p")) or "body not found"

def sankei(soup):
    output["subject"] = soup.find("h1").get_text(strip=True, separator="\n").strip() if soup.find("h1") else "subject not found"
    output["body"] = "".join([para.get_text(strip=True, separator="\n").strip() for para in soup.find_all("p", class_="article-text")]) or "body not found"

def zdnet_jp(soup): # 有上下頁
    output["subject"] = soup.find("h1").get_text(strip=True, separator="\n").strip() if soup.find("h1") else "subject not found"
    output["body"] = "".join([p.get_text(strip=True, separator="\n").strip() for p in soup.find("div", class_="article-contents").find_all('p')]) or "body not found"

def ddaily_kr(soup):
    output["subject"] = soup.find("p", class_="title").get_text(strip=True, separator="\n").strip() if soup.find("p", class_="title") else "subject not found"
    output["body"] = "".join([p.get_text(strip=True, separator="\n").strip() for p in soup.select(".article_content p")]) if any(soup.select(".article_content p")) else "body not found"

def sedaily(soup):
    output["subject"] = soup.find("h1", class_="art_tit").text if soup.find("h1", class_="art_tit") else "subject not found"
    output["body"] = "".join([br.next_sibling.string.strip() for br in soup.find_all("br") if br.next_sibling and br.next_sibling.string and br.next_sibling.string.strip()]) or "body not found"

def hankyung(soup): # 沒有規則的內文
    output["subject"] = soup.find("h1", class_="headline").get_text(strip=True, separator="\n").strip() if soup.find("h1", class_="headline") else "subject not found"
    output["body"] = "".join([a.get_text(strip=True, separator="\n").strip() for a in soup.find("div", {"id": "articletxt"})] or "body not found")

def etnews(soup):
    output["subject"] = soup.find("h2", {"id": "article_title_h2"}).text
    output["body"] = "".join([p.get_text(strip=True, separator="\n").strip() for p in soup.find("div", {"class": "article_body", "id": "articleBody"}).find_all("p")]).strip() or "body not found"

def biz_chosunbiz(soup):
    pass

def get_domain_function(url):
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower()
    if "www.reuters.com" in domain:
        headers["Referer"] = "https://www.reuters.com/technology/"
        return reuters
    elif "www.cnbc.com" in domain:
        headers["Referer"] = "https://www.cnbc.com/technology/"
        return cnbc
    elif "www.tomshardware.com" in domain:
        headers["Referer"] = "https://www.tomshardware.com/news/"
        return tomshardware
    elif "techcrunch" in domain:
        headers["Referer"] = "https://techcrunch.com/latest/"
        return techcrunch
    elif "www.theverge.com" in domain:
        headers["Referer"] = "https://www.theverge.com/tech/"
        return theverge
    elif "www.theregister.com" in domain:
        headers["Referer"] = "https://www.theregister.com/"
        return theregister
    elif "electrek.co" in domain:
        headers["Referer"] = "https://electrek.co/feature/exclusives/"
        return electrek_co
    elif "9to5mac.com" in domain:
        headers["Referer"] = "https://9to5mac.com/feature/exclusive/"
        return ninetofivemac
    elif "9to5google.com" in domain:
        headers["Referer"] = "https://9to5google.com/"
        return ninetofivegoogle
    elif "appleinsider.com" in domain:
        headers["Referer"] = "https://appleinsider.com/"
        return appleinsider
    elif "www.macrumors.com" in domain:
        headers["Referer"] = "https://www.macrumors.com/"
        return macrumors
    elif "yahoo.co.jp" in domain:
        headers["Referer"] = "https://news.yahoo.co.jp/categories/business"
        # return yahoo_jp
        return call_diffbot
    elif "newswitch.jp" in domain:
        headers["Referer"] = "https://newswitch.jp/articles/latest"
        return newswitch_jp
    elif "monoist.itmedia" in domain:
        headers["Referer"] = "https://monoist.itmedia.co.jp/mn/subtop/embedded/"
        return monoist_itmedia
    elif "www.sankei.com" in domain:
        headers["Referer"] = "https://www.sankei.com/economy/business/"
        return sankei
    elif "japan.zdnet" in domain:
        headers["Referer"] = "https://japan.zdnet.com/manufacturing_it/"
        return zdnet_jp
    elif "ddaily.co.kr" in domain:
        headers["Referer"] = "https://www.ddaily.co.kr/industry"
        return ddaily_kr
    elif "www.sedaily.com" in domain:
        headers["Referer"] = "https://www.sedaily.com/v/NewsMain/GD"
        return sedaily
    elif "www.hankyung.com" in domain:
        headers["Referer"] = "https://www.hankyung.com/industry"
        return hankyung
    elif "www.etnews.com" in domain:
        headers["Referer"] = "https://www.etnews.com/news/section.html?id1=06"
        return etnews
    elif "biz.chosun" in domain:
        headers["Referer"] = "https://biz.chosun.com/industry/"
        # return biz_chosunbiz
        return call_diffbot
    else:
        return None

@app.route("/webcrawler", methods=["POST"])
def webcrawler():
    input = request.get_json()
    domain = get_domain_function(input["url"])
    if domain:
        response = requests.get(input["url"], headers=headers)
        if response.status_code == 200:
            if domain == call_diffbot:
                domain(input["url"])
            else:
                domain(BeautifulSoup(response.content, "html.parser"))
            return jsonify(output)
        else:
            return jsonify({"message": "Failed to retrieve the webpage, status code: {}".format(response.status_code)})
    else:
        return jsonify({"message": "Domain not supported"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False, port="4455")