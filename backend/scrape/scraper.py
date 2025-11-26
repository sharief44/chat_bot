"""
IMPROVED Web Scraper - Gets More Content
Scrapes deeper and extracts better text content
Supports HTTP(S) and local file paths (file:/// or absolute path).
Uses Playwright to render JS-heavy pages when requested or when a
simple heuristic decides the page is JS-driven.

CONFIGURED FOR: ScriptBees.com
"""

import os
import json
import time
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import argparse

# Playwright import (installed separately)
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except Exception:
    PLAYWRIGHT_AVAILABLE = False

# Default configuration values
DEFAULT_MAX_PAGES = 100  # Increased for ScriptBees
DEFAULT_OUTPUT_DIR = "content"
DELAY = 1.5  # Slightly increased to be respectful
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

def clean_text(text):
    """Clean and normalize text content"""
    return ' '.join(text.split())

def _read_local_file(path):
    """Return HTML content from a local file path (path may be file:/// or absolute)."""
    if path.startswith("file://"):
        file_path = path[len("file://"):]
    else:
        file_path = path
    if not os.path.isabs(file_path):
        file_path = os.path.abspath(file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Local file not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def render_page_with_playwright(url, timeout=30000, scroll_times=3):
    """
    Use Playwright to render the page and return the final HTML.
    Blocks until network idle (or timeout).
    """
    if not PLAYWRIGHT_AVAILABLE:
        print("  ✗ Playwright is not installed. Install with: pip install playwright && playwright install")
        return None

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(user_agent=USER_AGENT)
            # goto with networkidle to wait for JS requests to finish
            page.goto(url, wait_until="networkidle", timeout=timeout)
            # Basic scroll to trigger lazy loading
            for _ in range(scroll_times):
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(500)
            html = page.content()
            browser.close()
            return html
    except Exception as e:
        print(f"  ✗ Playwright render error: {e}")
        return None

def extract_content(soup, url):
    """Extract title and main content from HTML"""
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    elif soup.find('h1'):
        title = soup.find('h1').get_text().strip()
    else:
        title = url.split('/')[-1] or "Page"

    # Remove non-content elements
    for element in soup(['script', 'style', 'nav', 'footer', 'header',
                         'aside', 'form', 'button', 'input', 'iframe']):
        element.decompose()

    # Try to find main content area
    content_area = None
    for selector in ['main', 'article', '[role="main"]', '.content',
                     '#content', '.main-content', '#main', '.container']:
        content_area = soup.select_one(selector)
        if content_area:
            break

    if not content_area:
        content_area = soup.body if soup.body else soup

    text_parts = []
    for tag in content_area.find_all(['h1','h2','h3','h4','h5','h6','p','li','div','span']):
        text = tag.get_text(strip=True)
        if len(text) > 20:
            text_parts.append(text)

    full_text = ' '.join(text_parts)
    full_text = clean_text(full_text)

    # Fallback if text is too short
    if len(full_text) < 100:
        full_text = content_area.get_text(separator=' ', strip=True)
        full_text = clean_text(full_text)

    return {'title': title[:200], 'text': full_text[:10000]}

def scrape_page(url, use_js_render=False):
    """
    Scrape a single page. Supports:
     - http(s) URL -> requests.get
     - file:///path or absolute filesystem path -> open local file
    If use_js_render=True, attempts Playwright rendering.
    """
    try:
        parsed = urlparse(url)

        # Local file
        if parsed.scheme == 'file' or (parsed.scheme == '' and os.path.exists(url)):
            html = _read_local_file(url)
        else:
            headers = {'User-Agent': USER_AGENT}
            resp = requests.get(url, headers=headers, timeout=20)
            resp.raise_for_status()
            content_type = resp.headers.get('Content-Type', '')
            if 'text/html' not in content_type:
                return None
            html = resp.text

            # Heuristic: if HTML is small or looks like an SPA shell, use Playwright
            lower = html.lower()
            if use_js_render or len(html) < 3000 or '<div id="root"' in lower or '<div id="app"' in lower or 'loading' in lower[:1500]:
                print(f"  → Using Playwright for {url}")
                rendered = render_page_with_playwright(url)
                if rendered:
                    html = rendered

        soup = BeautifulSoup(html, 'html.parser')
        content = extract_content(soup, url)

        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            absolute_url = urljoin(url, href)
            links.append(absolute_url)

        return {'content': content, 'links': links}

    except FileNotFoundError as e:
        print(f"  ✗ Local file error: {e}")
        return None
    except requests.RequestException as e:
        print(f"  ✗ Request error: {str(e)[:120]}")
        return None
    except Exception as e:
        print(f"  ✗ Parse error: {str(e)[:120]}")
        return None

def is_valid_url(url, domain):
    """Check if URL should be scraped"""
    try:
        parsed = urlparse(url)
        # allow local files even if no domain
        if parsed.scheme == 'file':
            return True
        if parsed.netloc != domain:
            return False

        skip_ext = ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.zip',
                    '.exe', '.doc', '.docx', '.xls', '.xlsx', '.ppt',
                    '.pptx', '.mp4', '.mp3', '.avi', '.css', '.js',
                    '.svg', '.ico', '.woff', '.woff2', '.ttf']
        path_lower = parsed.path.lower()
        if any(path_lower.endswith(ext) for ext in skip_ext):
            return False

        skip_paths = ['/wp-admin', '/admin', '/login', '/signup',
                      '/cart', '/checkout', '/account', '/wp-content',
                      '/wp-includes', '/feed', '/tag/', '/category/',
                      '/author/', '/page/', '/search']
        if any(skip in path_lower for skip in skip_paths):
            return False

        return True
    except:
        return False

def scrape_website(start_url, max_pages=DEFAULT_MAX_PAGES, delay=DELAY, use_js_render=False):
    """Main scraping function"""
    print(f"\n{'='*60}")
    print("🕷️  SCRIPTBEES WEB SCRAPER")
    print(f"{'='*60}")
    print(f"Target: {start_url}")
    print(f"Max pages: {max_pages}")
    print(f"Delay: {delay}s\n")

    parsed = urlparse(start_url)
    domain = parsed.netloc if parsed.scheme != 'file' else None
    visited = set()
    to_visit = [start_url]
    pages = []

    pbar = tqdm(total=max_pages, desc="Scraping ScriptBees", unit="page")

    while to_visit and len(pages) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)

        result = scrape_page(url, use_js_render=use_js_render)
        if result and result['content']['text']:
            content = result['content']
            if len(content['text']) > 100:
                pages.append({
                    'id': len(pages),
                    'url': url,
                    'title': content['title'],
                    'text': content['text']
                })
                pbar.update(1)
                pbar.set_postfix({
                    'chars': len(content['text']),
                    'queue': len(to_visit),
                    'title': content['title'][:30]
                })

            for link in result['links']:
                if (is_valid_url(link, domain) and 
                    link not in visited and 
                    link not in to_visit and 
                    len(to_visit) < 300):
                    to_visit.append(link)
        
        time.sleep(delay)

    pbar.close()
    print(f"\n✓ Scraped {len(pages)} pages from ScriptBees.com")
    print(f"✓ Total content: {sum(len(p['text']) for p in pages):,} characters")
    
    if pages:
        print(f"\n📄 Sample pages scraped:")
        for p in pages[:5]:
            print(f"   • {p['title'][:50]}")
    
    return pages

def save_pages(pages, output_dir=DEFAULT_OUTPUT_DIR):
    """Save scraped pages to JSON file"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'pages.json')

    if not pages:
        print("\n⚠️  No pages scraped! Using fallback data.")
        pages = [{
            'id': 0,
            'url': 'https://scriptbees.com',
            'title': 'ScriptBees - No Content',
            'text': 'No content could be extracted from ScriptBees.com. Please check your internet connection and try again.'
        }]

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(pages, f, indent=2, ensure_ascii=False)

    print(f"💾 Saved to {output_path}\n")
    sample = pages[:5]
    return output_path, sample

def run_scrape(start_url, max_pages=DEFAULT_MAX_PAGES, output_dir=DEFAULT_OUTPUT_DIR, delay=DELAY, use_js_render=False):
    """Main entry point for scraping"""
    # If user passed absolute local path, normalize to file://
    if os.path.exists(start_url) and not start_url.startswith('file://'):
        start_url = 'file://' + os.path.abspath(start_url)
    
    pages = scrape_website(start_url, max_pages=max_pages, delay=delay, use_js_render=use_js_render)
    output_path, sample = save_pages(pages, output_dir=output_dir)
    return pages, output_path, sample

def _cli():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="ScriptBees Web Scraper - Extract content for RAG chatbot"
    )
    parser.add_argument(
        '--start-url',
        default='https://scriptbees.com',
        help='Start URL to crawl (default: https://scriptbees.com)'
    )
    parser.add_argument('--max-pages', type=int, default=DEFAULT_MAX_PAGES)
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--delay', type=float, default=DELAY, help='Delay between requests in seconds')
    parser.add_argument('--use-js-render', action='store_true', help='Force JS rendering with Playwright')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("🐝 SCRIPTBEES RAG CHATBOT - WEB SCRAPER")
    print("="*60)
    print(f"Target: {args.start_url}")
    print(f"Max pages: {args.max_pages}")
    print(f"Output: {args.output_dir}")
    print("="*60 + "\n")

    pages, output_path, sample = run_scrape(
        args.start_url,
        args.max_pages,
        args.output_dir,
        args.delay,
        args.use_js_render
    )
    
    print(f"\n✅ SCRAPING COMPLETE!")
    print(f"📁 Saved to: {output_path}")
    print(f"📊 Total pages: {len(pages)}")
    print(f"\n📄 Sample page titles:")
    for p in sample:
        print(f"   • {p.get('title')}")
    
    print(f"\n🔜 Next step: Run embeddings")
    print(f"   python embeddings/embedder.py")

if __name__ == '__main__':
    _cli()