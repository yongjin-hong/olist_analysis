"""② 실험 실행: 설정 파일의 모델 × 프롬프트 조합으로 eval을 분류하고 결과를 JSONL로 저장.

실행 (프로젝트 루트에서):
    python3 src/run_experiment.py --config configs/experiment.toml
    python3 src/run_experiment.py --config configs/experiment.toml --models qwen3.5:4b
    python3 src/run_experiment.py --config configs/experiment.toml --prompt-types zero_shot rag --limit 5

- 결과 파일: {results_dir}/{모델}__{프롬프트}.jsonl (한 줄 = 리뷰 1건)
- 다시 실행하면 이미 성공한 리뷰는 건너뛰고, 에러로 끝난 리뷰만 다시 시도한다.
"""

import argparse
import json
import statistics
import sys
import time
import tomllib
from datetime import datetime
from pathlib import Path

from examples import PROMPT_TYPES, ExampleSelector
from llm_client import OllamaClassifier
from prompts import LABEL_GUIDE_VERSION, PROMPT_VERSION


# ── 명령줄 옵션 ──
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="② 프롬프트 × 모델 비교 실험")
    parser.add_argument("--config", default="configs/experiment.toml", help="설정 파일 경로")
    parser.add_argument("--models", nargs="+", help="실행할 모델 이름 (생략하면 설정 파일의 전체)")
    parser.add_argument("--prompt-types", nargs="+", choices=PROMPT_TYPES,
                        help="실행할 프롬프트 종류 (생략하면 설정 파일의 전체)")
    parser.add_argument("--limit", type=int, help="eval 앞에서부터 N건만 실행 (동작 확인용)")
    return parser.parse_args(argv)


# ── 결과 파일 경로: 모델 이름의 ':'와 '/'는 파일 이름에 쓰기 애매해서 '_'로 바꿈 ──
def result_path_for(results_dir, model_name, prompt_type):
    safe_model_name = model_name.replace(":", "_").replace("/", "_")
    return Path(results_dir) / f"{safe_model_name}__{prompt_type}.jsonl"


# ── 기존 결과 정리: 성공한 줄만 남기고(같은 리뷰는 마지막 기록), 파일을 다시 씀 ──
# 에러로 끝난 줄과 중간에 끊겨 깨진 줄은 지워서, 이번 실행에서 다시 시도되게 함
def load_completed_records(result_path):
    if not result_path.exists():
        return {}
    completed_records = {}
    with open(result_path, encoding="utf-8") as result_file:
        for line in result_file:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue   # 강제 종료로 마지막 줄이 깨진 경우
            if record.get("error") is None:
                completed_records[record["review_id"]] = record
    with open(result_path, "w", encoding="utf-8") as result_file:
        for record in completed_records.values():
            result_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return completed_records


# ── 진행 상황 한 줄 표시 (같은 줄을 덮어씀) ──
def print_progress(label, done_count, total_count, recent_elapsed):
    if recent_elapsed:
        remaining_sec = statistics.mean(recent_elapsed) * (total_count - done_count)
        eta_text = f"남은 시간 약 {remaining_sec / 60:.0f}분"
    else:
        eta_text = "남은 시간 계산 중"
    print(f"\r{label} {done_count}/{total_count} · {eta_text}   ", end="", flush=True)


# ── 조합 요약: 정확도, 파싱 실패율, 에러 수, p50 시간 ──
def summarize(records):
    finished = [record for record in records if record.get("error") is None]
    if not finished:
        return "완료된 건 없음"
    accuracy = statistics.mean(record["pred_label"] == record["gold_label"] for record in finished)
    parse_fail_rate = statistics.mean(not record["parse_ok"] for record in finished)
    p50_sec = statistics.median(record["elapsed_sec"] for record in finished)
    error_count = len(records) - len(finished)
    return (f"{len(finished)}건 · 정확도 {accuracy:.3f} · 파싱 실패율 {parse_fail_rate:.3f}"
            f" · p50 {p50_sec:.2f}초 · 에러 {error_count}건")


# ── 조합 하나 실행: 모델 1개 × 프롬프트 1종 ──
def run_combination(classifier, selector, prompt_type, k_examples, results_dir, eval_positions):
    result_path = result_path_for(results_dir, classifier.model_name, prompt_type)
    completed_records = load_completed_records(result_path)
    eval_review_ids = selector.eval_df["review_id"].to_list()
    gold_labels = selector.eval_df["label"].to_list()

    # 이번에 할 리뷰: 범위 안에서 아직 성공하지 못한 것
    todo_positions = [position for position in eval_positions
                      if eval_review_ids[position] not in completed_records]
    progress_label = f"[{classifier.model_name} | {prompt_type}]"
    print(f"{progress_label} 대상 {len(eval_positions)}건 · 완료 {len(eval_positions) - len(todo_positions)}건"
          f" · 이번에 {len(todo_positions)}건")

    session_records = []
    recent_elapsed = []   # 남은 시간 추정용 (최근 20건)
    with open(result_path, "a", encoding="utf-8") as result_file:
        for done_index, eval_position in enumerate(todo_positions, start=1):
            example_review_ids, messages = selector.prepare(prompt_type, eval_position)
            classify_result = classifier.classify(messages)

            # 분류 결과 + 실험 조건을 한 줄로 저장
            record = {
                "review_id": eval_review_ids[eval_position],
                "eval_position": eval_position,
                "gold_label": gold_labels[eval_position],
                "prompt_type": prompt_type,
                "k": 0 if prompt_type == "zero_shot" else k_examples,
                "example_review_ids": example_review_ids,
                "prompt_version": PROMPT_VERSION,
                "label_guide_version": LABEL_GUIDE_VERSION,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                **classify_result,
            }
            result_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            result_file.flush()   # 중간에 끊겨도 여기까지는 파일에 남음
            session_records.append(record)

            if classify_result["elapsed_sec"] is not None:
                recent_elapsed = (recent_elapsed + [classify_result["elapsed_sec"]])[-20:]
            print_progress(progress_label, done_index, len(todo_positions), recent_elapsed)

    if todo_positions:
        print()   # 진행 표시 줄바꿈
    in_range_ids = {eval_review_ids[position] for position in eval_positions}
    all_records = [record for record in completed_records.values()
                   if record["review_id"] in in_range_ids] + session_records
    print(f"{progress_label} 요약: {summarize(all_records)}")


# ── 전체 실행 ──
def main(argv=None):
    args = parse_args(argv)
    with open(args.config, "rb") as config_file:
        config = tomllib.load(config_file)

    # 실행할 모델·프롬프트 고르기
    model_configs = config["models"]
    if args.models:
        known_names = [model_config["name"] for model_config in model_configs]
        unknown_names = [name for name in args.models if name not in known_names]
        if unknown_names:
            sys.exit(f"설정 파일에 없는 모델: {unknown_names} (가능: {known_names})")
        model_configs = [model_config for model_config in model_configs if model_config["name"] in args.models]
    prompt_types = args.prompt_types or config["run"]["prompt_types"]

    # 예시 선택기 준비
    paths = config["paths"]
    example_config = config["examples"]
    selector = ExampleSelector(
        pool_path=paths["pool"],
        eval_path=paths["eval"],
        embedding_path=paths["embeddings"],
        k_examples=example_config["k"],
        fixed_shuffle_seed=example_config["fixed_shuffle_seed"],
        random_base_seed=example_config["random_base_seed"],
    )
    Path(paths["results_dir"]).mkdir(parents=True, exist_ok=True)
    eval_count = selector.eval_df.height
    eval_positions = list(range(min(args.limit, eval_count) if args.limit else eval_count))

    # 모델마다 워밍업 → 프롬프트 종류를 차례로 (8GB 메모리라 모델을 번갈아 부르지 않음)
    start_time = time.perf_counter()
    try:
        for model_config in model_configs:
            classifier = OllamaClassifier(
                model_name=model_config["name"],
                model_options=model_config.get("options", {}),
                timeout_sec=model_config.get("timeout_sec", 120),
                max_retries=model_config.get("max_retries", 2),
            )
            print(f"\n=== {classifier.model_name} 워밍업 {classifier.warmup()}초 ===")
            for prompt_type in prompt_types:
                run_combination(classifier, selector, prompt_type, example_config["k"],
                                paths["results_dir"], eval_positions)
    except KeyboardInterrupt:
        print("\n중단됨. 같은 명령으로 다시 실행하면 끝난 건은 건너뛰고 이어서 진행합니다.")
        return
    print(f"\n전체 소요 {(time.perf_counter() - start_time) / 60:.1f}분")


if __name__ == "__main__":
    main()
