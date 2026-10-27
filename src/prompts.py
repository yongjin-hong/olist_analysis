"""라벨 정의, 공통 프롬프트, 모델 응답 파싱.

- 라벨 기준: label_guide.md v1.1
- 프롬프트를 수정하면 PROMPT_VERSION을 올려서 결과 파일에 어떤 버전으로 돌렸는지 남긴다.
"""

import json
import re

# ── 버전 정보 (결과 파일에 함께 기록) ──
LABEL_GUIDE_VERSION = "v1.1"
PROMPT_VERSION = "v1"

# ── 라벨 코드 ↔ 한글 매핑 (모델은 코드로 답하고, 평가할 때 한글로 되돌림) ──
LABEL_CODE_TO_KO = {
    "delivery_delay": "배송 지연",
    "missing_item": "누락",
    "wrong_item": "제품 오배송",
    "defective": "제품 불량",
    "counterfeit": "위조·비정품",
    "dissatisfied": "제품 불만족",
    "other": "기타",
}
LABEL_KO_TO_CODE = {label_ko: label_code for label_code, label_ko in LABEL_CODE_TO_KO.items()}

# ── 공통 기본 프롬프트 (zero-shot은 이것만, few-shot은 뒤에 예시 대화 턴을 붙임) ──
BASE_PROMPT = """You classify negative customer reviews (in Brazilian Portuguese) from a Brazilian e-commerce marketplace by the main cause of the complaint. Use only the review text.

Labels:
- delivery_delay: The product arrived late or has not arrived. Includes lost packages, marked as delivered but not received, delivered to a wrong address or city, and delays in exchange or re-shipment.
- wrong_item: A different product was sent, or an objective specification differs from the listing (model, brand, color, size, capacity, material, voltage, pattern).
- defective: The received product has an objective defect: does not work, broken, dented, scratched, looks used, poor finish or assembly, expired or spoiled.
- missing_item: Part of the ordered quantity or a component did not arrive (e.g., bought 2 and received 1; missing parts or filter; a box the listing said was included).
- counterfeit: Sold as a genuine brand product, but a fake, replica, generic, or unbranded product arrived.
- dissatisfied: No objective defect, but the customer is subjectively unhappy with the product: quality, material, smell, performance, battery life, "not what I expected", unclear product description, close to expiration date, arrived without a box or poorly packed.
- other: None of the above, or the cause is unclear: vague complaints that do not mention the product ("bad", "don't recommend"), payment, cancellation, stock or customer-service issues, price or shipping cost, having to pick up at the post office, only the outer packaging damaged while the product is fine, praise despite a low score.

Rules:
1. If several causes appear, use this priority: wrong_item / defective / missing_item / counterfeit > dissatisfied > delivery_delay > other. Among the first four, choose the most specifically described; if equally specific, the one mentioned first.
2. Exception: if the main complaint is that an exchange or re-shipment for a product problem is taking too long, choose delivery_delay. This does not apply to refunds for products the customer simply did not like.
3. If only part of the order arrived, choose delivery_delay only when the text gives evidence that the rest is on the way (in transit, at the post office, arrived late, the promised date has passed, the seller said it was shipped separately). Otherwise choose missing_item, even if the customer says they are still waiting or hopes it is coming.
4. Compatibility: if the seller stated compatibility with the customer's device or car and it does not fit, choose wrong_item. If it is unclear whose mistake it is, choose other. Damage caused by the customer's misuse is other.
5. "Different from the photo/ad": choose wrong_item only if the review says what differs (color, size, model...) or says the wrong product was sent; otherwise dissatisfied. Calling it "a different product" only because of poor quality, "false advertising" without a specific difference, or cheap-knockoff slang ("paraguaio") without saying it was sold as genuine are all dissatisfied.
6. If the title and body contradict each other, follow the body.
7. Short delivery mentions ("still waiting", "where is my product?", "delivery problem") are delivery_delay. Waiting for something unspecified (e.g., "waiting for a reply") is other.

Answer with one line of JSON only: {"label": "<label>"}"""

# ── 응답 파싱 ──
# 모델이 ```json 코드블록이나 설명 문장을 붙이는 경우를 대비해 첫 { ... } 만 꺼냄
JSON_OBJECT_PATTERN = re.compile(r"\{.*?\}", re.DOTALL)


def parse_label_code(raw_text):
    """모델 응답에서 라벨 코드를 꺼낸다. 형식이 틀리거나 7개 코드가 아니면 None."""
    json_match = JSON_OBJECT_PATTERN.search(raw_text)
    if json_match is None:
        return None
    try:
        label_code = json.loads(json_match.group()).get("label")
    except json.JSONDecodeError:
        return None
    if not isinstance(label_code, str):
        return None
    # 대소문자·공백 차이만 정리
    label_code = label_code.strip().lower()
    return label_code if label_code in LABEL_CODE_TO_KO else None


# ── 동작 확인: python src/prompts.py 로 실행 (④ CI 단위 테스트의 바탕) ──
if __name__ == "__main__":
    parse_cases = [
        ('{"label": "wrong_item"}', "wrong_item"),                  # 정상
        ('```json\n{"label": "other"}\n```', "other"),              # 코드블록으로 감쌈
        ('{"label": " Delivery_Delay "}', "delivery_delay"),        # 대소문자·공백
        ('The answer is wrong_item', None),                         # JSON 아님
        ('{"label": "배송 지연"}', None),                            # 코드가 아닌 한글
        ('{"label": "late_delivery"}', None),                       # 없는 코드
        ('{"category": "other"}', None),                            # label 키 없음
    ]
    for raw_text, expected_code in parse_cases:
        parsed_code = parse_label_code(raw_text)
        assert parsed_code == expected_code, f"{raw_text!r} → {parsed_code} (기대값 {expected_code})"
    print(f"parse_label_code 테스트 {len(parse_cases)}건 통과")
