import os
import io
import base64
import streamlit as st
from PIL import Image
from zhipuai import ZhipuAI

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

st.set_page_config(page_title="智能文字辨識", page_icon="📝", layout="wide")

VISION_MODEL = "glm-4v-flash"
TEXT_MODEL   = "glm-4-flash"
MAX_PAGES_PER_RUN = 30   # 單次最多辨識頁數，避免跑太久

PROMPTS = {
    "標準模式": "請辨識圖片中的所有文字，按原本的排版與順序輸出。只輸出辨識到的文字內容，不要加任何說明或評論。",
    "表格模式": "請辨識圖片中的表格內容，用 Markdown 表格格式輸出。只輸出表格本身，不要加說明。",
    "手寫模式": "圖片中可能包含手寫文字，請仔細辨識所有文字（印刷＋手寫），按原本排版輸出。只輸出辨識到的文字，不要加說明。",
}

def get_api_key():
    try:
        return st.secrets.get("ZHIPU_API_KEY", "")
    except Exception:
        return os.environ.get("ZHIPU_API_KEY", "")

@st.cache_resource
def get_client():
    key = get_api_key()
    return ZhipuAI(api_key=key) if key else None

def image_to_b64(image: Image.Image) -> str:
    if image.mode != "RGB":
        image = image.convert("RGB")
    w, h = image.size
    if max(w, h) > 3000:
        ratio = 3000 / max(w, h)
        image = image.resize((int(w * ratio), int(h * ratio)))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()

def call_vision(client, b64, mode):
    resp = client.chat.completions.create(
        model=VISION_MODEL, temperature=0.1,
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": b64}},
            {"type": "text", "text": PROMPTS[mode]},
        ]}],
    )
    return resp.choices[0].message.content

def call_proofread(client, text):
    resp = client.chat.completions.create(
        model=TEXT_MODEL, temperature=0.1,
        messages=[{"role": "user", "content":
            "以下是一段 OCR 辨識結果，可能有形近字誤判（例如 0/O、1/l/I、rn/m）或標點錯誤。\n"
            "請依上下文校正明顯的辨識錯誤：只改錯字，不改寫語句、不增刪內容、保持原本格式，直接輸出校正後全文。\n\n" + text
        }],
    )
    return resp.choices[0].message.content

def pdf_page_count(file_bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    n = doc.page_count
    doc.close()
    return n

def render_pdf_pages(file_bytes, start, end):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    images = []
    for i in range(start - 1, end):
        pix = doc[i].get_pixmap(matrix=fitz.Matrix(2, 2))  # 約144dpi，OCR夠用
        images.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
    doc.close()
    return images

def parse_page_range(s, total):
    s = (s or "").strip()
    if not s or s.lower() in ("all", "全部"):
        return 1, min(total, MAX_PAGES_PER_RUN)
    try:
        if "-" in s:
            a, b = (int(x) for x in s.split("-", 1))
            a, b = max(1, a), min(total, b)
            if a > b:
                a, b = b, a
            return a, min(b, a + MAX_PAGES_PER_RUN - 1)
        a = int(s)
        return a, a
    except ValueError:
        return 1, min(total, MAX_PAGES_PER_RUN)

st.title("📝 智能文字辨識")
st.caption("GLM-4V-Flash 讀圖辨字 → GLM-4-Flash 二次校正（雙模型皆免費）｜支援圖片與 PDF")

client = get_client()
left, right = st.columns(2)

with left:
    up = st.file_uploader("上傳圖片或 PDF", type=["png", "jpg", "jpeg", "webp", "bmp", "pdf"])
    cam = st.camera_input("或直接拍照")
    source = cam or up
    mode = st.radio("辨識模式", ["標準模式", "表格模式", "手寫模式"], horizontal=True)
    fix = st.checkbox("啟用二次校正", value=True)

    is_pdf, total_pages, page_range = False, 0, ""
    if source is not None and source.name.lower().endswith(".pdf"):
        if fitz is None:
            st.error("PDF 功能未載入成功：請確認 requirements.txt 已加入 PyMuPDF 後等待重新部署")
        else:
            try:
                total_pages = pdf_page_count(source.getvalue())
                is_pdf = True
                st.info(f"偵測到 PDF，共 {total_pages} 頁")
                page_range = st.text_input(
                    f"辨識頁碼範圍（每次最多 {MAX_PAGES_PER_RUN} 頁，例如 3-8）",
                    value=f"1-{min(total_pages, MAX_PAGES_PER_RUN)}",
                )
            except Exception:
                st.error("無法讀取此 PDF（可能已加密或檔案損毀）")

    run = st.button("🚀 開始辨識", type="primary", use_container_width=True)

with right:
    if run:
        if client is None:
            st.error("尚未設定 API Key：請到 Manage app → Settings → Secrets 加入 ZHIPU_API_KEY")
        elif source is None:
            st.warning("請先上傳圖片 / PDF，或拍一張照片")
        else:
            pages, labels, ok = None, None, True
            try:
                if is_pdf:
                    start, end = parse_page_range(page_range, total_pages)
                    pages = render_pdf_pages(source.getvalue(), start, end)
                    labels = [f"第 {i} 頁" for i in range(start, end + 1)]
                else:
                    pages = [Image.open(source)]
                    labels = ["圖片"]
            except Exception as e:
                st.error(f"讀取檔案失敗：{e}")
                ok = False

            if ok and pages:
                raw_parts, fixed_parts = [], []
                bar = st.progress(0.0, text="準備辨識…")
                for idx, (img, label) in enumerate(zip(pages, labels)):
                    try:
                        raw = call_vision(client, image_to_b64(img), mode)
                    except Exception as e:
                        st.error(f"{label} 辨識失敗：{e}")
                        ok = False
                        break
                    raw_parts.append(f"【{label}】\n{raw}")
                    if fix:
                        try:
                            fixed = call_proofread(client, raw)
                        except Exception:
                            fixed = f"（校正失敗，保留初步結果）\n{raw}"
                        fixed_parts.append(f"【{label}】\n{fixed}")
                    else:
                        fixed_parts.append(f"【{label}】\n（未啟用二次校正）")
                    bar.progress((idx + 1) / len(pages), text=f"已完成 {idx + 1} / {len(pages)}")
                bar.empty()

                if ok:
                    st.markdown("#### ① 初步辨識結果")
                    st.text_area("初步結果", "\n\n".join(raw_parts), height=300, key="raw")
                    st.markdown("#### ② 校正後結果（可複製）")
                    fixed_all = "\n\n".join(fixed_parts)
                    st.text_area("校正結果", fixed_all, height=300, key="fixed")
                    st.download_button("⬇️ 下載結果", fixed_all, file_name="ocr_result.txt")
