"""
문서 스캐너 웹앱
- 웹캠 실시간 미리보기
- OpenCV 윤곽선 검출 + 투시 변환으로 문서 자동 보정
- 보정된 이미지를 PDF로 저장
"""

import io
import os

import cv2
import img2pdf
import numpy as np
import streamlit as st
from PIL import Image

# A4 용지 비율 (세로: 가로 = 1 : 1.414)
A4_RATIO = 1.414
OUTPUT_WIDTH = 1000
OUTPUT_HEIGHT = int(OUTPUT_WIDTH * A4_RATIO)


# ---------------------------------------------------------------------------
# 전처리 · 윤곽선 검출 · 투시 변환
# ---------------------------------------------------------------------------

def order_points(pts: np.ndarray) -> np.ndarray:
    """네 꼭짓점을 [좌상, 우상, 우하, 좌하] 순서로 정렬한다."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _build_edge_maps(gray: np.ndarray) -> list[np.ndarray]:
    """여러 전처리 전략으로 테두리 맵을 생성한다."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)

    # 1) Canny + 팽창: 끊어진 테두리를 연결
    canny = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    canny = cv2.dilate(canny, kernel, iterations=2)
    canny = cv2.erode(canny, kernel, iterations=1)

    # 2) 적응형 이진화: 밝은 종이 vs 어두운 책상 대비 강화
    adaptive = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2,
    )
    adaptive = cv2.morphologyEx(
        adaptive, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
    )

    # 3) 형태학적 그래디언트: 문서 외곽 에지 추출
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    gradient = cv2.morphologyEx(bilateral, cv2.MORPH_GRADIENT, morph_kernel)
    _, gradient = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return [canny, adaptive, gradient]


def _is_valid_document_quad(approx: np.ndarray, image_area: float) -> bool:
    """감지된 사각형이 실제 문서일 가능성이 있는지 검증한다."""
    area = cv2.contourArea(approx)
    if area < image_area * 0.08 or area > image_area * 0.97:
        return False
    if not cv2.isContourConvex(approx):
        return False

    pts = approx.reshape(4, 2).astype("float32")
    rect = order_points(pts)

    width = max(
        np.linalg.norm(rect[1] - rect[0]),
        np.linalg.norm(rect[2] - rect[3]),
    )
    height = max(
        np.linalg.norm(rect[3] - rect[0]),
        np.linalg.norm(rect[2] - rect[1]),
    )
    if width < 1 or height < 1:
        return False

    aspect = max(width, height) / min(width, height)
    if aspect > 2.5:
        return False

    return True


def find_document_contour(image: np.ndarray) -> np.ndarray | None:
    """
    findContours로 이미지에서 가장 큰 사각형 외곽선(4개 점)을 찾는다.
    여러 전처리 전략을 시도하여 감지 성공률을 높인다.
    """
    h, w = image.shape[:2]
    image_area = h * w

    # 큰 이미지는 축소 후 감지 → 원본 좌표로 복원
    scale = min(1.0, 600.0 / max(h, w))
    if scale < 1.0:
        small = cv2.resize(image, None, fx=scale, fy=scale)
    else:
        small = image

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edge_maps = _build_edge_maps(gray)

    best_quad: np.ndarray | None = None
    best_area = 0.0
    small_area = small.shape[0] * small.shape[1]

    for edge_map in edge_maps:
        contours, _ = cv2.findContours(
            edge_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours[:20]:
            area = cv2.contourArea(contour)
            if area < small_area * 0.05:
                continue

            peri = cv2.arcLength(contour, True)
            for epsilon in (0.01, 0.02, 0.03, 0.04, 0.05):
                approx = cv2.approxPolyDP(contour, epsilon * peri, True)
                if len(approx) != 4:
                    continue
                if not _is_valid_document_quad(approx, small_area):
                    continue
                if area > best_area:
                    best_area = area
                    best_quad = approx.reshape(4, 2).astype("float32")
                break

    if best_quad is None:
        return None

    return best_quad / scale


def perspective_transform_a4(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    감지된 4개 점을 [좌상, 우상, 우하, 좌하]로 정렬한 뒤
    cv2.getPerspectiveTransform + cv2.warpPerspective로
    A4 비율(1:1.414)의 정면 사각형으로 펴준다.
    """
    rect = order_points(pts)

    width_top = np.linalg.norm(rect[1] - rect[0])
    width_bottom = np.linalg.norm(rect[2] - rect[3])
    height_left = np.linalg.norm(rect[3] - rect[0])
    height_right = np.linalg.norm(rect[2] - rect[1])

    doc_width = max(width_top, width_bottom)
    doc_height = max(height_left, height_right)

    # 문서 방향에 맞춰 A4 출력 크기 결정
    if doc_width >= doc_height:
        out_w = int(OUTPUT_WIDTH * A4_RATIO)
        out_h = OUTPUT_WIDTH
    else:
        out_w = OUTPUT_WIDTH
        out_h = OUTPUT_HEIGHT

    dst = np.array(
        [
            [0, 0],
            [out_w - 1, 0],
            [out_w - 1, out_h - 1],
            [0, out_h - 1],
        ],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (out_w, out_h))


def smart_center_crop(image: np.ndarray, crop_ratio: float = 0.82) -> np.ndarray:
    """
    사각형 감지 실패 시 스마트폰 스캔 앱처럼
    이미지 중심부를 A4 비율로 크롭한다.
    """
    h, w = image.shape[:2]

    crop_h = int(h * crop_ratio)
    crop_w = int(w * crop_ratio)

    # A4 세로 비율에 맞게 크롭 영역 조정
    current_ratio = crop_w / crop_h
    target_ratio = 1.0 / A4_RATIO

    if current_ratio > target_ratio:
        crop_w = int(crop_h * target_ratio)
    else:
        crop_h = int(crop_w / target_ratio)

    x1 = max(0, (w - crop_w) // 2)
    y1 = max(0, (h - crop_h) // 2)
    cropped = image[y1 : y1 + crop_h, x1 : x1 + crop_w]

    return cv2.resize(
        cropped, (OUTPUT_WIDTH, OUTPUT_HEIGHT),
        interpolation=cv2.INTER_CUBIC,
    )


def apply_scan_enhancement(image_bgr: np.ndarray) -> np.ndarray:
    """
    어댑티브 스레시홀딩으로 스캔본처럼 보정한다.
    배경은 하얗게, 글씨는 검게 처리한다.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    enhanced = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15,
        C=8,
    )
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def scan_document(image: np.ndarray) -> tuple[np.ndarray, str]:
    """
    문서를 자동 보정한다.
    반환: (보정 이미지, 처리 방식)
      - "perspective": 투시 변환 성공
      - "crop": 중심부 A4 크롭 (감지 실패 대체)
    """
    contour = find_document_contour(image)
    if contour is not None:
        return perspective_transform_a4(image, contour), "perspective"
    return smart_center_crop(image), "crop"


def save_as_pdf(image_bgr: np.ndarray, output_path: str) -> None:
    """BGR 이미지를 PDF 파일로 저장한다."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=95)
    buffer.seek(0)

    with open(output_path, "wb") as f:
        f.write(img2pdf.convert(buffer.getvalue()))


# ---------------------------------------------------------------------------
# 웹캠 초기화
# ---------------------------------------------------------------------------

def init_camera() -> cv2.VideoCapture:
    """웹캠을 열고, 실패 시 에러를 표시한다."""
    if "cap" not in st.session_state:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("웹캠을 열 수 없습니다. 카메라 연결을 확인해 주세요.")
            st.stop()
        st.session_state.cap = cap
        st.session_state.current_frame = None
    return st.session_state.cap


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="문서 스캐너", page_icon="📄", layout="centered")

st.title("📄 문서 스캐너")
st.markdown(
    "교실 책상·교무실 탁자 위의 종이를 촬영하면 "
    "**테두리를 자동 감지**하여 **A4 비율로 펴서 PDF**로 저장합니다."
)

scan_mode = st.toggle(
    "📑 스캔 화질 보정 (어댑티브 스레시홀딩)",
    value=False,
    help="켜면 배경은 하얗게, 글씨는 검게 보정하여 스캔본처럼 선명하게 만듭니다.",
)

cap = init_camera()


@st.fragment(run_every=0.1)
def live_camera_feed():
    """0.1초마다 웹캠 프레임을 갱신하여 실시간 미리보기를 표시한다."""
    ret, frame = cap.read()
    if ret:
        st.session_state.current_frame = frame.copy()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(rgb_frame, channels="RGB", use_container_width=True)
    else:
        st.warning("카메라에서 영상을 읽을 수 없습니다.")


live_camera_feed()

st.divider()

if st.button("📸 문서 캡처 및 PDF 저장", type="primary", use_container_width=True):
    frame = st.session_state.get("current_frame")

    if frame is None:
        st.error("캡처할 영상이 없습니다. 카메라가 정상 작동하는지 확인해 주세요.")
    else:
        with st.spinner("문서 테두리를 감지하고 보정하는 중..."):
            scanned, method = scan_document(frame)

            if scan_mode:
                scanned = apply_scan_enhancement(scanned)

            output_path = os.path.join(os.getcwd(), "scanned_output.pdf")
            save_as_pdf(scanned, output_path)

        if method == "perspective":
            st.success(
                f"✅ 문서 테두리를 감지하여 A4 비율로 펴서 저장했습니다!\n\n"
                f"PDF 경로: `{output_path}`"
            )
        else:
            st.info(
                f"ℹ️ 테두리 자동 감지에 실패하여 중심부를 A4 비율로 크롭했습니다.\n\n"
                f"종이 네 모서리가 화면에 보이도록 카메라 각도를 조절하면 "
                f"더 정확한 보정이 가능합니다.\n\n"
                f"PDF 경로: `{output_path}`"
            )

        st.subheader("보정된 이미지 미리보기")
        preview_rgb = cv2.cvtColor(scanned, cv2.COLOR_BGR2RGB)
        st.image(preview_rgb, channels="RGB", use_container_width=True)

st.divider()
st.caption(
    "💡 **촬영 팁**: 종이를 어두운 책상 위에 놓고, "
    "네 모서리가 화면 안에 들어오도록 카메라를 비스듬히 맞추면 "
    "자동 테두리 감지가 훨씬 잘 됩니다."
)
