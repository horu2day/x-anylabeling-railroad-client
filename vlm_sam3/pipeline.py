"""
VLM + SAM3 철도 객체 검출 파이프라인
====================================
VLM(Vision Language Model)이 이미지를 분석하여 철도 객체를 식별하고,
SAM3가 해당 객체의 정밀한 segmentation mask를 생성하는 2-Stage 파이프라인.

기존 서버 기반 구조와 달리, SAM3를 직접 로딩하여 독립 실행합니다.
RTX 3060 12GB 환경에서 VLM→SAM3 순차 로딩 전략을 사용합니다.

사용법:
    # Florence-2 + SAM3 (기본, 순차 로딩)
    python vlm_sam3/pipeline.py

    # Florence-2 + SAM3 (동시 로딩, 16GB+ GPU)
    python vlm_sam3/pipeline.py --mode concurrent

    # Claude Vision API + SAM3 (API VLM)
    python vlm_sam3/pipeline.py --vlm claude --mode api_vlm

    # 입출력 경로 지정
    python vlm_sam3/pipeline.py --input 노선영상/extracted_frames --output vlm_sam3/output

    # 설정 파일 지정
    python vlm_sam3/pipeline.py --config vlm_sam3/config.yaml
"""

import gc
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import click
import cv2
import numpy as np
import torch
import yaml
from loguru import logger
from PIL import Image
from tqdm import tqdm

# ============================================================
# 데이터 구조
# ============================================================

@dataclass
class Detection:
    """VLM이 식별한 객체 하나."""
    class_name: str           # 영문 클래스명
    bbox: list[float]         # [x_min, y_min, x_max, y_max] pixel coords
    confidence: float = 0.5
    description: str = ""
    bbox_normalized: list[float] = field(default_factory=list)  # [0-1] ratio


@dataclass
class SegmentationResult:
    """SAM3가 생성한 마스크 결과."""
    class_name: str
    mask: np.ndarray          # (H, W) binary mask
    bbox: list[float]         # [x_min, y_min, x_max, y_max]
    score: float
    polygon: list[list[float]] = field(default_factory=list)  # [[x,y], ...]


# ============================================================
# VLM Backends
# ============================================================

class VLMBackend:
    """VLM 백엔드 인터페이스."""

    def load(self):
        raise NotImplementedError

    def detect(self, image: Image.Image, target_objects: list[dict]) -> list[Detection]:
        raise NotImplementedError

    def unload(self):
        raise NotImplementedError


class Florence2Backend(VLMBackend):
    """Microsoft Florence-2: 경량 (~1.5GB) 시각 그라운딩 모델."""

    def __init__(self, model_id: str = "microsoft/Florence-2-large",
                 device: str = "cuda:0", dtype: str = "float16"):
        self.model_id = model_id
        self.device = device
        self.dtype = getattr(torch, dtype, torch.float16)
        self.model = None
        self.processor = None

    def load(self):
        from transformers import AutoModelForCausalLM, AutoProcessor

        logger.info(f"Florence-2 로딩: {self.model_id}")
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        ).to(self.device).eval()
        logger.info(f"Florence-2 로드 완료 (VRAM ~1.5GB)")

    def detect(self, image: Image.Image, target_objects: list[dict]) -> list[Detection]:
        """Florence-2로 철도 객체 검출.

        <CAPTION_TO_PHRASE_GROUNDING> 태스크: 텍스트에 언급된 객체의 bbox를 찾음.
        """
        if self.model is None:
            raise RuntimeError("Florence-2 모델이 로드되지 않았습니다. load() 먼저 호출하세요.")

        w, h = image.size
        all_detections = []

        # 대상 객체 이름들을 결합하여 프롬프트 생성
        object_names = [obj["name"] for obj in target_objects]
        text_prompt = ". ".join(object_names) + "."

        # 1) CAPTION_TO_PHRASE_GROUNDING: 텍스트 기반 bbox 검출
        task = "<CAPTION_TO_PHRASE_GROUNDING>"
        inputs = self.processor(
            text=task, images=image, return_tensors="pt"
        )
        # text_input 추가 (grounding 태스크에 필요)
        text_inputs = self.processor.tokenizer(
            text_prompt, return_tensors="pt"
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
            )

        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        result = self.processor.post_process_generation(
            generated_text, task=task, image_size=(w, h)
        )

        if task in result:
            bboxes = result[task].get("bboxes", [])
            labels = result[task].get("labels", [])
            for bbox, label in zip(bboxes, labels):
                # Florence-2 bbox: [x1, y1, x2, y2] in pixel coords
                matched_class = self._match_class(label, target_objects)
                all_detections.append(Detection(
                    class_name=matched_class or label,
                    bbox=[float(bbox[0]), float(bbox[1]),
                          float(bbox[2]), float(bbox[3])],
                    confidence=0.7,  # Florence-2는 confidence 미제공
                    description=label,
                    bbox_normalized=[
                        bbox[0] / w, bbox[1] / h,
                        bbox[2] / w, bbox[3] / h,
                    ],
                ))

        # 2) 추가: DENSE_REGION_CAPTION으로 누락된 객체 보완
        task2 = "<DENSE_REGION_CAPTION>"
        inputs2 = self.processor(
            text=task2, images=image, return_tensors="pt"
        )
        inputs2 = {k: v.to(self.device) for k, v in inputs2.items()}

        with torch.inference_mode():
            gen_ids2 = self.model.generate(
                input_ids=inputs2["input_ids"],
                pixel_values=inputs2["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
            )

        gen_text2 = self.processor.batch_decode(gen_ids2, skip_special_tokens=False)[0]
        result2 = self.processor.post_process_generation(
            gen_text2, task=task2, image_size=(w, h)
        )

        if task2 in result2:
            for bbox, label in zip(
                result2[task2].get("bboxes", []),
                result2[task2].get("labels", []),
            ):
                matched = self._match_class(label, target_objects)
                if matched and not self._is_duplicate(bbox, all_detections):
                    all_detections.append(Detection(
                        class_name=matched,
                        bbox=[float(bbox[0]), float(bbox[1]),
                              float(bbox[2]), float(bbox[3])],
                        confidence=0.5,
                        description=f"(dense caption) {label}",
                        bbox_normalized=[
                            bbox[0] / w, bbox[1] / h,
                            bbox[2] / w, bbox[3] / h,
                        ],
                    ))

        return all_detections

    def _match_class(self, label: str, target_objects: list[dict]) -> Optional[str]:
        """VLM 출력 라벨을 대상 클래스에 매칭."""
        label_lower = label.lower()
        for obj in target_objects:
            name = obj["name"].lower()
            if name in label_lower or label_lower in name:
                return obj["name"]
            for alias in obj.get("aliases", []):
                if alias.lower() in label_lower or label_lower in alias.lower():
                    return obj["name"]
        return None

    def _is_duplicate(self, bbox: list, detections: list[Detection],
                      iou_threshold: float = 0.5) -> bool:
        """IoU 기반 중복 검출 제거."""
        for det in detections:
            iou = self._compute_iou(bbox, det.bbox)
            if iou > iou_threshold:
                return True
        return False

    @staticmethod
    def _compute_iou(box1: list, box2: list) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0

    def unload(self):
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("Florence-2 언로드 완료")


class Qwen25VLBackend(VLMBackend):
    """Qwen2.5-VL: 강력한 VLM, 장면 이해 + bbox 출력 가능."""

    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
                 device: str = "cuda:0", dtype: str = "bfloat16"):
        self.model_id = model_id
        self.device = device
        self.dtype = getattr(torch, dtype, torch.bfloat16)
        self.model = None
        self.processor = None

    def load(self):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        logger.info(f"Qwen2.5-VL 로딩: {self.model_id}")
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            device_map=self.device,
        ).eval()
        logger.info("Qwen2.5-VL 로드 완료")

    def detect(self, image: Image.Image, target_objects: list[dict]) -> list[Detection]:
        if self.model is None:
            raise RuntimeError("Qwen2.5-VL 모델이 로드되지 않았습니다.")

        w, h = image.size
        object_names = ", ".join(obj["name"] for obj in target_objects)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": (
                        f"Detect all instances of: {object_names}.\n"
                        "For each object found, output a JSON array where each element has:\n"
                        '{"class": "class_name", "bbox": [x_min, y_min, x_max, y_max], '
                        '"confidence": 0.0-1.0, "description": "brief note"}\n'
                        "Coordinates should be pixel values. "
                        f"Image size is {w}x{h}. Output ONLY the JSON array."
                    )},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=2048)

        output_ids_trimmed = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)
        ]
        response = self.processor.batch_decode(
            output_ids_trimmed, skip_special_tokens=True
        )[0]

        return self._parse_json_response(response, target_objects, w, h)

    def _parse_json_response(self, response: str, target_objects: list[dict],
                             w: int, h: int) -> list[Detection]:
        """JSON 응답 파싱 → Detection 리스트."""
        detections = []
        try:
            # JSON 배열 추출
            start = response.find("[")
            end = response.rfind("]") + 1
            if start >= 0 and end > start:
                items = json.loads(response[start:end])
                for item in items:
                    bbox = item.get("bbox", [0, 0, 0, 0])
                    detections.append(Detection(
                        class_name=item.get("class", "unknown"),
                        bbox=[float(b) for b in bbox],
                        confidence=float(item.get("confidence", 0.5)),
                        description=item.get("description", ""),
                        bbox_normalized=[
                            bbox[0] / w, bbox[1] / h,
                            bbox[2] / w, bbox[3] / h,
                        ],
                    ))
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Qwen2.5-VL 응답 파싱 실패: {e}")
            logger.debug(f"Raw response: {response[:500]}")
        return detections

    def unload(self):
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("Qwen2.5-VL 언로드 완료")


class ClaudeVLMBackend(VLMBackend):
    """Claude Vision API: 가장 강력한 장면 이해, VRAM 사용 없음."""

    def __init__(self, model: str = "claude-sonnet-4-20250514",
                 max_tokens: int = 2048):
        self.model_name = model
        self.max_tokens = max_tokens
        self.client = None

    def load(self):
        import anthropic
        self.client = anthropic.Anthropic()
        logger.info(f"Claude Vision API 초기화: {self.model_name}")

    def detect(self, image: Image.Image, target_objects: list[dict]) -> list[Detection]:
        import base64
        import io

        if self.client is None:
            raise RuntimeError("Claude API 클라이언트 초기화 필요")

        w, h = image.size
        object_names = ", ".join(obj["name"] for obj in target_objects)

        # 이미지를 base64로 인코딩 (리사이즈)
        max_size = 1568
        if max(w, h) > max_size:
            ratio = max_size / max(w, h)
            image = image.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=90)
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": img_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                f"Detect all instances of these railway objects: {object_names}.\n"
                                "For each object, output a JSON array where each element has:\n"
                                '{"class": "class_name", "bbox": [x_min, y_min, x_max, y_max], '
                                '"confidence": 0.0-1.0, "description": "brief note"}\n'
                                f"Coordinates as ratio [0.0-1.0] of image size ({w}x{h}).\n"
                                "Output ONLY the JSON array, no other text."
                            ),
                        },
                    ],
                }
            ],
        )

        response_text = message.content[0].text
        return self._parse_response(response_text, w, h)

    def _parse_response(self, response: str, w: int, h: int) -> list[Detection]:
        detections = []
        try:
            start = response.find("[")
            end = response.rfind("]") + 1
            if start >= 0 and end > start:
                items = json.loads(response[start:end])
                for item in items:
                    bbox_ratio = item.get("bbox", [0, 0, 0, 0])
                    # ratio → pixel coords
                    bbox_pixel = [
                        bbox_ratio[0] * w, bbox_ratio[1] * h,
                        bbox_ratio[2] * w, bbox_ratio[3] * h,
                    ]
                    detections.append(Detection(
                        class_name=item.get("class", "unknown"),
                        bbox=bbox_pixel,
                        confidence=float(item.get("confidence", 0.7)),
                        description=item.get("description", ""),
                        bbox_normalized=bbox_ratio,
                    ))
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Claude 응답 파싱 실패: {e}")
        return detections

    def unload(self):
        self.client = None
        logger.info("Claude API 클라이언트 해제")


class OpenAIVLMBackend(VLMBackend):
    """OpenAI GPT-4V API: 강력한 장면 이해, VRAM 사용 없음."""

    def __init__(self, model: str = "gpt-4o", max_tokens: int = 2048):
        self.model_name = model
        self.max_tokens = max_tokens
        self.client = None

    def load(self):
        import openai
        self.client = openai.OpenAI()
        logger.info(f"OpenAI Vision API 초기화: {self.model_name}")

    def detect(self, image: Image.Image, target_objects: list[dict]) -> list[Detection]:
        import base64
        import io

        if self.client is None:
            raise RuntimeError("OpenAI API 클라이언트 초기화 필요")

        w, h = image.size
        object_names = ", ".join(obj["name"] for obj in target_objects)

        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=90)
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        response = self.client.chat.completions.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_b64}",
                                "detail": "high",
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                f"Detect all instances of: {object_names}.\n"
                                "Output a JSON array: "
                                '[{"class": "name", "bbox": [x_min, y_min, x_max, y_max], '
                                '"confidence": 0.0-1.0, "description": "..."}]\n'
                                f"bbox as ratio [0.0-1.0] of image ({w}x{h}). ONLY JSON array."
                            ),
                        },
                    ],
                }
            ],
        )

        resp_text = response.choices[0].message.content
        return ClaudeVLMBackend._parse_response(None, resp_text, w, h)

    def unload(self):
        self.client = None


# ============================================================
# SAM3 Direct Loader
# ============================================================

class SAM3Segmenter:
    """SAM3 직접 로딩 (서버 없이)."""

    def __init__(self, model_path: str, device: str = "cuda:0",
                 conf_threshold: float = 0.25, mask_threshold: float = 0.5):
        self.model_path = model_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.mask_threshold = mask_threshold
        self.model = None
        self.processor = None

    def load(self):
        """SAM3 모델 로드."""
        # sam3 패키지는 X-AnyLabeling-Server 하위에 있으므로 경로 추가
        sam3_parent = str(Path(self.model_path).parent)
        if sam3_parent not in sys.path:
            sys.path.insert(0, sam3_parent)

        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        logger.info(f"SAM3 로딩: {self.model_path}")
        self.model = build_sam3_image_model(
            device=self.device,
            checkpoint_path=self.model_path,
            eval_mode=True,
        )
        self.processor = Sam3Processor(
            model=self.model,
            device=self.device,
            confidence_threshold=self.conf_threshold,
        )
        logger.info("SAM3 로드 완료 (VRAM ~6-8GB)")

    def segment_with_text(self, image: Image.Image,
                          text_prompt: str) -> list[SegmentationResult]:
        """텍스트 프롬프트로 concept segmentation."""
        state = self.processor.set_image(image)
        state = self.processor.set_text_prompt(text_prompt, state)
        return self._extract_results(state, text_prompt, image.size)

    def segment_with_boxes(self, image: Image.Image,
                           detections: list[Detection]) -> list[SegmentationResult]:
        """VLM bbox를 SAM3 프롬프트로 전달하여 마스크 생성."""
        state = self.processor.set_image(image)
        w, h = image.size
        results = []

        for det in detections:
            self.processor.reset_all_prompts(state)
            # SAM3 box format: [center_x, center_y, width, height] normalized [0-1]
            x1, y1, x2, y2 = det.bbox
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            state = self.processor.add_geometric_prompt(
                box=[cx, cy, bw, bh],
                label=True,
                state=state,
            )

            masks = state.get("masks")
            scores = state.get("scores")
            boxes = state.get("boxes")

            if masks is not None and len(masks) > 0:
                # 최고 점수 마스크 선택
                best_idx = scores.argmax().item() if scores is not None else 0
                mask_np = masks[best_idx].squeeze().cpu().numpy()
                mask_binary = (mask_np > self.mask_threshold).astype(np.uint8)
                score = float(scores[best_idx]) if scores is not None else det.confidence

                polygon = self._mask_to_polygon(mask_binary)
                results.append(SegmentationResult(
                    class_name=det.class_name,
                    mask=mask_binary,
                    bbox=det.bbox,
                    score=score,
                    polygon=polygon,
                ))

        return results

    def segment_with_text_and_boxes(self, image: Image.Image,
                                    detections: list[Detection]) -> list[SegmentationResult]:
        """텍스트 concept prompt + bbox 결합 (최고 정확도).

        먼저 텍스트로 전체 concept을 검출하고,
        VLM bbox와 매칭하여 누락 없이 보완.
        """
        w, h = image.size
        results = []

        # 1) 클래스별 텍스트 프롬프트로 concept segmentation
        class_groups = {}
        for det in detections:
            class_groups.setdefault(det.class_name, []).append(det)

        text_results = {}
        for class_name in class_groups:
            state = self.processor.set_image(image)
            state = self.processor.set_text_prompt(class_name, state)
            text_results[class_name] = self._extract_results(state, class_name, image.size)

        # 2) VLM bbox로 보완 (텍스트 검출에서 누락된 것)
        for class_name, dets in class_groups.items():
            text_masks = text_results.get(class_name, [])

            for det in dets:
                # 이미 텍스트로 검출된 것과 IoU 체크
                is_covered = False
                for tm in text_masks:
                    iou = Florence2Backend._compute_iou(det.bbox, tm.bbox)
                    if iou > 0.3:
                        is_covered = True
                        break

                if is_covered:
                    continue

                # 텍스트로 놓친 객체 → bbox prompt로 보완
                box_results = self.segment_with_boxes(image, [det])
                results.extend(box_results)

        # 텍스트 결과 추가
        for class_name, masks in text_results.items():
            results.extend(masks)

        return results

    def _extract_results(self, state: dict, class_name: str,
                         image_size: tuple) -> list[SegmentationResult]:
        """SAM3 state에서 결과 추출."""
        results = []
        masks = state.get("masks")
        scores = state.get("scores")
        boxes = state.get("boxes")

        if masks is None or len(masks) == 0:
            return results

        w, h = image_size
        for i in range(len(masks)):
            mask_np = masks[i].squeeze().cpu().numpy()
            mask_binary = (mask_np > self.mask_threshold).astype(np.uint8)

            score = float(scores[i]) if scores is not None else 0.5
            if score < self.conf_threshold:
                continue

            bbox = [0.0, 0.0, 0.0, 0.0]
            if boxes is not None and i < len(boxes):
                bbox = boxes[i].cpu().tolist()

            polygon = self._mask_to_polygon(mask_binary)
            results.append(SegmentationResult(
                class_name=class_name,
                mask=mask_binary,
                bbox=bbox,
                score=score,
                polygon=polygon,
            ))

        return results

    @staticmethod
    def _mask_to_polygon(mask: np.ndarray, epsilon_factor: float = 0.002) -> list[list[float]]:
        """바이너리 마스크 → 폴리곤 좌표."""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return []

        # 가장 큰 컨투어 선택
        contour = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon_factor * peri, True)
        return approx.squeeze().tolist()

    def unload(self):
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("SAM3 언로드 완료")


# ============================================================
# Visualization
# ============================================================

# 클래스별 색상 (BGR)
CLASS_COLORS = {
    "railway track":     (0, 180, 255),
    "railway signal":    (0, 0, 255),
    "catenary pole":     (0, 200, 255),
    "catenary wire":     (255, 200, 0),
    "junction box":      (255, 100, 0),
    "railroad switch":   (0, 255, 255),
    "sleeper":           (180, 180, 0),
    "fence":             (0, 255, 100),
    "platform":          (255, 0, 200),
}


def get_color(class_name: str) -> tuple:
    """클래스 이름에 해당하는 색상 반환."""
    if class_name in CLASS_COLORS:
        return CLASS_COLORS[class_name]
    # 해시 기반 색상 생성
    h = hash(class_name) % 360
    # HSV → BGR
    hsv = np.array([[[h // 2, 200, 230]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return tuple(int(c) for c in bgr)


def draw_vlm_detections(image_bgr: np.ndarray,
                        detections: list[Detection]) -> np.ndarray:
    """VLM 검출 결과 (bbox) 시각화."""
    vis = image_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        color = get_color(det.class_name)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{det.class_name} ({det.confidence:.2f})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def draw_segmentation(image_bgr: np.ndarray,
                      results: list[SegmentationResult],
                      alpha: float = 0.4) -> np.ndarray:
    """SAM3 마스크 + 라벨 시각화."""
    vis = image_bgr.copy()
    overlay = image_bgr.copy()

    for r in results:
        color = get_color(r.class_name)
        mask_bool = r.mask.astype(bool)

        # 마스크 채우기
        overlay[mask_bool] = color

        # 컨투어 그리기
        contours, _ = cv2.findContours(
            r.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis, contours, -1, color, 2)

        # 라벨
        if contours:
            M = cv2.moments(contours[0])
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                label = f"{r.class_name} {r.score:.2f}"
                cv2.putText(vis, label, (cx - 40, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(vis, label, (cx - 40, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            color, 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)

    # 범례
    y = 25
    seen = set()
    for r in results:
        if r.class_name in seen:
            continue
        seen.add(r.class_name)
        color = get_color(r.class_name)
        cv2.rectangle(vis, (10, y - 12), (24, y + 2), color, -1)
        cv2.putText(vis, r.class_name, (28, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 22

    return vis


# ============================================================
# Output Formatters
# ============================================================

def to_xanylabeling_json(image_path: Path, results: list[SegmentationResult],
                         image_shape: tuple) -> dict:
    """X-AnyLabeling 호환 JSON 형식 생성."""
    h, w = image_shape[:2]
    shapes = []
    for r in results:
        if r.polygon and len(r.polygon) >= 3:
            points = r.polygon if isinstance(r.polygon[0], list) else [r.polygon]
            shapes.append({
                "label": r.class_name,
                "points": [[float(p[0]), float(p[1])] for p in points]
                          if isinstance(points[0], list) else points,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {},
                "score": round(r.score, 4),
            })

    return {
        "version": "3.3.9",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w,
    }


def to_coco_annotation(results: list[SegmentationResult],
                       image_id: int, ann_id_start: int,
                       image_shape: tuple) -> list[dict]:
    """COCO 형식 annotation 생성."""
    annotations = []
    for i, r in enumerate(results):
        if not r.polygon or len(r.polygon) < 3:
            continue
        flat_polygon = []
        for p in r.polygon:
            if isinstance(p, (list, tuple)):
                flat_polygon.extend([float(p[0]), float(p[1])])
            else:
                flat_polygon.append(float(p))

        x_coords = flat_polygon[0::2]
        y_coords = flat_polygon[1::2]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        annotations.append({
            "id": ann_id_start + i,
            "image_id": image_id,
            "category_name": r.class_name,
            "segmentation": [flat_polygon],
            "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
            "area": cv2.contourArea(np.array(r.polygon, dtype=np.float32)),
            "iscrowd": 0,
            "score": round(r.score, 4),
        })

    return annotations


# ============================================================
# Main Pipeline
# ============================================================

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_vlm_backend(config: dict) -> VLMBackend:
    """설정에 따라 VLM 백엔드 생성."""
    vlm_cfg = config.get("vlm", {})
    backend_name = vlm_cfg.get("backend", "florence2")

    if backend_name == "florence2":
        cfg = vlm_cfg.get("florence2", {})
        return Florence2Backend(
            model_id=cfg.get("model_id", "microsoft/Florence-2-large"),
            device=cfg.get("device", "cuda:0"),
            dtype=cfg.get("dtype", "float16"),
        )
    elif backend_name == "qwen2.5vl":
        cfg = vlm_cfg.get("qwen2_5vl", {})
        return Qwen25VLBackend(
            model_id=cfg.get("model_id", "Qwen/Qwen2.5-VL-3B-Instruct"),
            device=cfg.get("device", "cuda:0"),
            dtype=cfg.get("dtype", "bfloat16"),
        )
    elif backend_name == "claude":
        cfg = vlm_cfg.get("claude", {})
        return ClaudeVLMBackend(
            model=cfg.get("model", "claude-sonnet-4-20250514"),
            max_tokens=cfg.get("max_tokens", 2048),
        )
    elif backend_name == "openai":
        cfg = vlm_cfg.get("openai", {})
        return OpenAIVLMBackend(
            model=cfg.get("model", "gpt-4o"),
            max_tokens=cfg.get("max_tokens", 2048),
        )
    else:
        raise ValueError(f"알 수 없는 VLM 백엔드: {backend_name}")


def collect_images(input_dir: str, extensions: list[str]) -> list[Path]:
    """입력 디렉토리에서 이미지 수집."""
    input_path = Path(input_dir)
    if input_path.is_file():
        return [input_path]

    images = []
    for ext in extensions:
        images.extend(input_path.glob(f"*{ext}"))
    return sorted(images)


def resize_if_needed(image: Image.Image, max_size: int) -> Image.Image:
    """긴 변이 max_size를 초과하면 리사이즈."""
    if max_size <= 0:
        return image
    w, h = image.size
    if max(w, h) <= max_size:
        return image
    ratio = max_size / max(w, h)
    new_w, new_h = int(w * ratio), int(h * ratio)
    return image.resize((new_w, new_h), Image.LANCZOS)


@click.command()
@click.option("--config", "config_path", default="vlm_sam3/config.yaml",
              help="설정 파일 경로")
@click.option("--input", "input_dir", default=None,
              help="입력 이미지 경로 (설정 파일 오버라이드)")
@click.option("--output", "output_dir", default=None,
              help="출력 경로 (설정 파일 오버라이드)")
@click.option("--vlm", "vlm_backend", default=None,
              type=click.Choice(["florence2", "qwen2.5vl", "claude", "openai"]),
              help="VLM 백엔드 선택 (설정 파일 오버라이드)")
@click.option("--mode", "pipeline_mode", default=None,
              type=click.Choice(["sequential", "concurrent", "api_vlm"]),
              help="파이프라인 모드")
@click.option("--prompt-mode", default=None,
              type=click.Choice(["text", "bbox", "both"]),
              help="SAM3 프롬프트 모드")
@click.option("--conf", type=float, default=None,
              help="SAM3 confidence threshold")
def main(config_path, input_dir, output_dir, vlm_backend,
         pipeline_mode, prompt_mode, conf):
    """VLM + SAM3 철도 객체 검출 파이프라인."""

    # 설정 로드
    config_file = Path(config_path)
    if config_file.exists():
        config = load_config(str(config_file))
        logger.info(f"설정 로드: {config_file}")
    else:
        logger.warning(f"설정 파일 없음: {config_file}, 기본값 사용")
        config = {}

    # CLI 오버라이드 적용
    if vlm_backend:
        config.setdefault("vlm", {})["backend"] = vlm_backend
    if pipeline_mode:
        config["pipeline_mode"] = pipeline_mode
    if prompt_mode:
        config.setdefault("sam3", {})["prompt_mode"] = prompt_mode
    if conf is not None:
        config.setdefault("sam3", {})["conf_threshold"] = conf

    io_cfg = config.get("io", {})
    input_path = input_dir or io_cfg.get("input_dir", "노선영상/extracted_frames")
    output_path = output_dir or io_cfg.get("output_dir", "vlm_sam3/output")
    mode = config.get("pipeline_mode", "sequential")
    sam3_cfg = config.get("sam3", {})
    proc_cfg = config.get("processing", {})
    railway_cfg = config.get("railway", {})

    # 출력 디렉토리 생성
    out_base = Path(output_path)
    out_masks = out_base / "masks"
    out_overlays = out_base / "overlays"
    out_vlm_vis = out_base / "vlm_detections"
    out_annotations = out_base / "annotations"

    for d in [out_base, out_masks, out_overlays, out_vlm_vis, out_annotations]:
        d.mkdir(parents=True, exist_ok=True)

    # 이미지 수집
    extensions = proc_cfg.get("image_extensions",
                              [".jpg", ".jpeg", ".png", ".tif", ".tiff"])
    images = collect_images(input_path, extensions)
    if not images:
        logger.error(f"이미지 없음: {input_path}")
        return

    target_objects = railway_cfg.get("target_objects", [
        {"name": "railway track", "aliases": ["rail"]},
        {"name": "railway signal", "aliases": ["signal"]},
        {"name": "catenary pole", "aliases": ["electric pole"]},
    ])
    max_size = proc_cfg.get("max_image_size", 2048)

    logger.info(f"파이프라인 모드: {mode}")
    logger.info(f"이미지: {len(images)}장")
    logger.info(f"대상 객체: {[obj['name'] for obj in target_objects]}")

    # ── Stage 1: VLM 검출 ──────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Stage 1: VLM 객체 검출")
    logger.info("=" * 60)

    vlm = create_vlm_backend(config)
    vlm.load()

    all_detections: dict[str, list[Detection]] = {}
    t0 = time.time()

    for img_path in tqdm(images, desc="VLM 분석"):
        pil_image = Image.open(str(img_path)).convert("RGB")
        pil_image = resize_if_needed(pil_image, max_size)

        detections = vlm.detect(pil_image, target_objects)
        all_detections[img_path.name] = detections

        # VLM bbox 시각화 저장
        if io_cfg.get("save_overlays", True):
            img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            vis = draw_vlm_detections(img_bgr, detections)
            cv2.imwrite(str(out_vlm_vis / f"{img_path.stem}_vlm.jpg"), vis)

        n = len(detections)
        classes = set(d.class_name for d in detections)
        logger.debug(f"  {img_path.name}: {n}개 검출 → {classes}")

    vlm_time = time.time() - t0
    total_det = sum(len(d) for d in all_detections.values())
    logger.info(f"VLM 완료: {total_det}개 검출, {vlm_time:.1f}초")

    # 순차 로딩: VLM 언로드
    if mode == "sequential":
        vlm.unload()

    # ── Stage 2: SAM3 세그멘테이션 ─────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Stage 2: SAM3 정밀 세그멘테이션")
    logger.info("=" * 60)

    sam3 = SAM3Segmenter(
        model_path=sam3_cfg.get("model_path",
                                "X-AnyLabeling-Server/sam3.pt"),
        device=sam3_cfg.get("device", "cuda:0"),
        conf_threshold=sam3_cfg.get("conf_threshold", 0.25),
        mask_threshold=sam3_cfg.get("mask_threshold", 0.5),
    )
    sam3.load()

    prompt_mode_cfg = sam3_cfg.get("prompt_mode", "both")
    all_results: dict[str, list[SegmentationResult]] = {}
    coco_annotations = []
    ann_id = 1
    t1 = time.time()

    for idx, img_path in enumerate(tqdm(images, desc="SAM3 세그멘테이션")):
        detections = all_detections.get(img_path.name, [])
        if not detections:
            logger.debug(f"  {img_path.name}: VLM 검출 없음 → 스킵")
            continue

        pil_image = Image.open(str(img_path)).convert("RGB")
        pil_image_resized = resize_if_needed(pil_image, max_size)

        # SAM3 프롬프트 모드에 따라 분기
        if prompt_mode_cfg == "text":
            # 클래스별 텍스트 프롬프트
            results = []
            seen_classes = set(d.class_name for d in detections)
            for cls in seen_classes:
                r = sam3.segment_with_text(pil_image_resized, cls)
                results.extend(r)
        elif prompt_mode_cfg == "bbox":
            results = sam3.segment_with_boxes(pil_image_resized, detections)
        else:  # "both"
            results = sam3.segment_with_text_and_boxes(
                pil_image_resized, detections
            )

        all_results[img_path.name] = results

        # 원본 이미지로 시각화
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            buf = np.fromfile(str(img_path), dtype=np.uint8)
            img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)

        if img_bgr is not None:
            h_img, w_img = img_bgr.shape[:2]

            # 마스크 오버레이 시각화
            if io_cfg.get("save_overlays", True):
                vis = draw_segmentation(
                    img_bgr, results, io_cfg.get("overlay_alpha", 0.4)
                )
                cv2.imwrite(str(out_overlays / f"{img_path.stem}_seg.jpg"), vis)

            # 바이너리 마스크 저장
            if io_cfg.get("save_masks", True):
                for i, r in enumerate(results):
                    mask_path = out_masks / f"{img_path.stem}_{r.class_name}_{i}.png"
                    mask_full = r.mask
                    if mask_full.shape[:2] != (h_img, w_img):
                        mask_full = cv2.resize(
                            mask_full, (w_img, h_img),
                            interpolation=cv2.INTER_NEAREST
                        )
                    cv2.imwrite(str(mask_path), mask_full * 255)

            # X-AnyLabeling JSON
            if io_cfg.get("save_annotations", True):
                ann = to_xanylabeling_json(img_path, results, img_bgr.shape)
                json_path = out_annotations / f"{img_path.stem}.json"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(ann, f, ensure_ascii=False, indent=2)

            # COCO format
            if io_cfg.get("save_coco", True):
                coco_anns = to_coco_annotation(
                    results, image_id=idx, ann_id_start=ann_id,
                    image_shape=img_bgr.shape,
                )
                coco_annotations.extend(coco_anns)
                ann_id += len(coco_anns)

        n_masks = len(results)
        classes = set(r.class_name for r in results)
        logger.debug(f"  {img_path.name}: {n_masks}개 마스크 → {classes}")

    sam3_time = time.time() - t1

    # COCO 전체 저장
    if io_cfg.get("save_coco", True) and coco_annotations:
        coco_output = {
            "images": [
                {"id": i, "file_name": img.name}
                for i, img in enumerate(images)
            ],
            "annotations": coco_annotations,
            "categories": [
                {"id": i, "name": obj["name"]}
                for i, obj in enumerate(target_objects)
            ],
        }
        with open(out_base / "coco_annotations.json", "w", encoding="utf-8") as f:
            json.dump(coco_output, f, ensure_ascii=False, indent=2)

    sam3.unload()
    if mode != "sequential":
        vlm.unload()

    # ── 결과 요약 ─────────────────────────────────────────────────────
    total_masks = sum(len(r) for r in all_results.values())
    logger.info("=" * 60)
    logger.info("파이프라인 완료")
    logger.info("=" * 60)
    logger.info(f"처리 이미지: {len(images)}장")
    logger.info(f"VLM 검출:   {total_det}개 ({vlm_time:.1f}초)")
    logger.info(f"SAM3 마스크: {total_masks}개 ({sam3_time:.1f}초)")
    logger.info(f"총 소요시간: {vlm_time + sam3_time:.1f}초")

    # 클래스별 집계
    class_stats: dict[str, int] = {}
    for results in all_results.values():
        for r in results:
            class_stats[r.class_name] = class_stats.get(r.class_name, 0) + 1

    if class_stats:
        logger.info("\n클래스별 마스크 수:")
        for cls, cnt in sorted(class_stats.items(), key=lambda x: -x[1]):
            avg = cnt / max(len(all_results), 1)
            bar = "#" * min(cnt, 30)
            logger.info(f"  {cls:20s}: {cnt:4d}개  평균 {avg:.1f}/장  {bar}")

    logger.info(f"\n출력 경로: {out_base.resolve()}")
    logger.info(f"  마스크:       {out_masks}")
    logger.info(f"  시각화:       {out_overlays}")
    logger.info(f"  VLM 검출:     {out_vlm_vis}")
    logger.info(f"  Annotations:  {out_annotations}")


if __name__ == "__main__":
    main()
