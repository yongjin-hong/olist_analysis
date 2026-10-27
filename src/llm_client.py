"""Ollama 모델 호출: 타임아웃, 재시도, 시간 분해 기록.

- 일시적 오류(타임아웃, 연결 끊김, 서버 5xx)는 재시도하고, 끝내 실패하면 error에 기록한다.
- 설정 오류(모델 이름 틀림 등 4xx)는 재시도해도 소용없으므로 바로 예외를 올려 실험을 멈춘다.
"""

import time

import httpx
import ollama

from prompts import LABEL_CODE_TO_KO, parse_label_code

NANOSEC_PER_SEC = 1_000_000_000


# ── 나노초 → 초 (Ollama가 값을 안 주면 None) ──
def nanosec_to_sec(duration_ns):
    return None if duration_ns is None else round(duration_ns / NANOSEC_PER_SEC, 3)


# ── 재시도할 만한 오류인지 판단 ──
def is_retryable(error):
    if isinstance(error, (httpx.TimeoutException, ConnectionError, httpx.TransportError)):
        return True
    # 서버 내부 오류(5xx)는 일시적일 수 있어 재시도, 4xx는 설정 문제라 재시도 안 함
    return isinstance(error, ollama.ResponseError) and error.status_code >= 500


class OllamaClassifier:
    """모델 하나를 담당하는 분류기. 실험 루프에서 classify(messages)만 부르면 된다."""

    # ── 초기화: 모델 이름·옵션, 타임아웃·재시도 설정 ──
    def __init__(self, model_name, model_options=None, host=None,
                 timeout_sec=120, max_retries=2, retry_wait_sec=5):
        self.model_name = model_name
        self.model_options = model_options or {}   # 예: {"think": False}
        self.max_retries = max_retries
        self.retry_wait_sec = retry_wait_sec
        self.client = ollama.Client(host=host, timeout=timeout_sec)

    # ── 실제 요청 1회 ──
    def _chat_once(self, messages):
        return self.client.chat(
            model=self.model_name,
            messages=messages,
            options={"temperature": 0},
            **self.model_options,
        )

    # ── 워밍업: 모델을 메모리에 올려서 첫 측정에 로딩 시간이 섞이지 않게 함 ──
    def warmup(self):
        start_time = time.perf_counter()
        self._chat_once([{"role": "user", "content": "ok"}])
        return round(time.perf_counter() - start_time, 2)

    # ── 분류: 재시도 포함, 결과를 dict 하나로 반환 ──
    def classify(self, messages):
        result = {
            "model": self.model_name,
            "pred_code": None, "pred_label": None, "parse_ok": False,
            "raw_output": None, "error": None, "attempts": 0,
            "elapsed_sec": None, "load_sec": None, "prompt_sec": None, "gen_sec": None,
            "input_tokens": None, "output_tokens": None,
        }
        for attempt in range(1, self.max_retries + 2):   # 첫 시도 + 재시도 횟수
            result["attempts"] = attempt
            start_time = time.perf_counter()
            try:
                response = self._chat_once(messages)
            except Exception as error:
                if not is_retryable(error):
                    raise   # 모델 이름 오류 등: 실험을 멈추고 설정을 고쳐야 함
                result["error"] = f"{type(error).__name__}: {error}"[:300]
                if attempt <= self.max_retries:
                    time.sleep(self.retry_wait_sec)
                continue

            # 성공: 응답 파싱 + 시간·토큰 기록
            raw_output = response.message.content
            pred_code = parse_label_code(raw_output)
            result.update(
                pred_code=pred_code,
                pred_label=LABEL_CODE_TO_KO.get(pred_code),
                parse_ok=pred_code is not None,
                raw_output=raw_output,
                error=None,
                elapsed_sec=round(time.perf_counter() - start_time, 3),
                load_sec=nanosec_to_sec(response.load_duration),          # 모델 로딩
                prompt_sec=nanosec_to_sec(response.prompt_eval_duration), # 입력 처리
                gen_sec=nanosec_to_sec(response.eval_duration),           # 출력 생성
                input_tokens=response.prompt_eval_count,
                output_tokens=response.eval_count,
            )
            return result
        return result   # 재시도까지 모두 실패: error 채워진 상태로 반환


# ── 동작 확인: python3 src/llm_client.py qwen3.5:4b (Ollama 실행 중이어야 함) ──
if __name__ == "__main__":
    import sys

    from prompts import BASE_PROMPT

    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3.5:4b"
    model_options = {"think": False} if model_name.startswith("qwen") else {}
    classifier = OllamaClassifier(model_name, model_options)

    print(f"워밍업 {classifier.warmup()}초")
    test_messages = [
        {"role": "system", "content": BASE_PROMPT},
        {"role": "user", "content": "Ainda não recebi o produto"},  # 아직 상품을 못 받았어요
    ]
    test_result = classifier.classify(test_messages)
    for key, value in test_result.items():
        print(f"  {key:13s} {value}")
