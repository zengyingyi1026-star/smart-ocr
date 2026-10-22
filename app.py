import os
import io
import base64
import streamlit as st
from PIL import Image
from zhipuai import ZhipuAI

st.set_page_config(page_title="智能文字辨識", page_icon="📝", layout="wide")

VISION_MODEL = "glm-4v-flash"   # 免費視覺模型：讀圖辨字
TEXT_MODEL   = "glm-4-flash"    # 免費文字模型：二次校正

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
        model=VISION_MODEL,
        temperature=0.1,
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": b64}},
            {"type": "text", "text": PROMPTS[mode]},
        ]}],
    )
    return resp.choices[0].message.content

def call_proofread(client, text):
    resp = client.chat.completions.create(
        model=TEXT_MODEL,
        temperature=0.1,
        messages=[{"role": "user", "content":
            "以下是一段 OCR 辨識結果，可能有形近字誤判（例如 0/O、1/l/I、rn/m）或標點錯誤。\n"
            "請依上下文校正明顯的辨識錯誤：只改錯字，不改寫語句、不增刪內容、保持原本格式，直接輸出校正後全文。\n\n" + text
        }],
    )
    return resp.choices[0].message.content

st.title("📝 智能文字辨識")
st.caption("GLM-4V-Flash 讀圖辨字 → GLM-4-Flash 二次校正（雙模型皆免費）")

client = get_client()

left, right = st.columns(2)

with left:
    up = st.file_uploader("上傳圖片", type=["png", "jpg", "jpeg", "webp", "bmp"])
    cam = st.camera_input("或直接拍照")
    source = cam or up
    mode = st.radio("辨識模式", ["標準模式", "表格模式", "手寫模式"], horizontal=True)
    fix = st.checkbox("啟用二次校正", value=True)
    run = st.button("🚀 開始辨識", type="primary", use_container_width=True)

with right:
    if run:
        if client is None:
            st.error("尚未設定 API Key：請到 App 的 Settings → Secrets 加入 ZHIPU_API_KEY")
        elif source is None:
            st.warning("請先上傳或拍一張圖片")
        else:
            image = Image.open(source)
            raw = None
            with st.spinner("GLM-4V-Flash 辨識中…"):
                try:
                    b64 = image_to_b64(image)
                    raw = call_vision(client, b64, mode)
                except Exception as e:
                    st.error(f"辨識失敗：{e}")
            if raw:
                st.markdown("#### ① 初步辨識結果")
                st.text_area("初步結果", raw, height=260, key="raw")
                if fix:
                    with st.spinner("GLM-4-Flash 校正中…"):
                        try:
                            fixed = call_proofread(client, raw)
                        except Exception as e:
                            fixed = f"（校正失敗：{e}，保留初步結果）\n{raw}"
                    st.markdown("#### ② 校正後結果（可複製）")
                    st.text_area("校正結果", fixed, height=260, key="fixed")
                    st.download_button("⬇️ 下載結果", fixed, file_name="ocr_result.txt")
