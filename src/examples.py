"""few-shot 예시 선택과 메시지 조립.

- 프롬프트 종류: zero_shot / fixed / random / rag
- pool(예시 풀)에서만 예시를 고른다. eval 라벨은 여기서 쓰지 않는다.
"""

import json

import numpy as np
import polars as pl

from prompts import BASE_PROMPT, LABEL_CODE_TO_KO, LABEL_KO_TO_CODE

PROMPT_TYPES = ("zero_shot", "fixed", "random", "rag")


# ── CSV 읽기: utf-8-sig로 저장돼 첫 컬럼명에 붙는 BOM 제거 ──
def read_labeled_csv(csv_path):
    labeled_df = pl.read_csv(csv_path)
    return labeled_df.rename(lambda col_name: col_name.lstrip("﻿"))


class ExampleSelector:
    """pool·eval 데이터와 임베딩을 들고, 리뷰마다 예시를 골라 메시지를 만든다."""

    # ── 초기화: 데이터·임베딩 로드, review_id 순서 검증, 고정 예시 계산 ──
    def __init__(self, pool_path, eval_path, embedding_path,
                 k_examples=7, fixed_shuffle_seed=2026, random_base_seed=2026):
        self.k_examples = k_examples
        self.random_base_seed = random_base_seed

        self.pool_df = read_labeled_csv(pool_path)
        self.eval_df = read_labeled_csv(eval_path)

        # 임베딩 파일의 행 순서가 CSV 행 순서와 같은지 확인 (다르면 엉뚱한 예시가 뽑힘)
        embedding_file = np.load(embedding_path)
        pool_ids = np.array(self.pool_df["review_id"].to_list(), dtype=str)
        eval_ids = np.array(self.eval_df["review_id"].to_list(), dtype=str)
        if not np.array_equal(embedding_file["pool_review_ids"], pool_ids):
            raise ValueError("pool 임베딩의 review_id 순서가 pool CSV와 다릅니다.")
        if not np.array_equal(embedding_file["eval_review_ids"], eval_ids):
            raise ValueError("eval 임베딩의 review_id 순서가 eval CSV와 다릅니다.")
        self.pool_embeddings = embedding_file["pool_embeddings"]
        self.eval_embeddings = embedding_file["eval_embeddings"]

        # 메시지 조립에 쓸 pool 텍스트·라벨
        self.pool_texts = self.pool_df["review_text"].to_list()
        self.pool_labels = np.array(self.pool_df["label"].to_list())
        self.pool_codes = [LABEL_KO_TO_CODE[label_ko] for label_ko in self.pool_labels]

        self.fixed_example_indices = self._select_fixed_examples(fixed_shuffle_seed)

    # ── 고정 예시: 라벨별 임베딩 평균(중심)에 가장 가까운 리뷰 1개씩, 순서는 시드로 섞어 고정 ──
    def _select_fixed_examples(self, fixed_shuffle_seed):
        fixed_indices = []
        for label_ko in LABEL_CODE_TO_KO.values():
            label_indices = np.where(self.pool_labels == label_ko)[0]
            centroid = self.pool_embeddings[label_indices].mean(axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            closest_position = np.argmax(self.pool_embeddings[label_indices] @ centroid)
            fixed_indices.append(int(label_indices[closest_position]))
        shuffle_rng = np.random.default_rng(fixed_shuffle_seed)
        return [int(pool_index) for pool_index in shuffle_rng.permutation(fixed_indices)]

    # ── 랜덤 예시: 리뷰마다 k개, (기본 시드 + eval 행 번호)로 재현 가능 ──
    def _select_random_examples(self, eval_position):
        example_rng = np.random.default_rng(self.random_base_seed + eval_position)
        chosen = example_rng.choice(len(self.pool_texts), size=self.k_examples, replace=False)
        return [int(pool_index) for pool_index in chosen]

    # ── RAG 예시: 유사도 상위 k개, 가장 비슷한 예시가 마지막(리뷰 바로 앞) ──
    def _select_rag_examples(self, eval_position):
        similarities = self.pool_embeddings @ self.eval_embeddings[eval_position]
        top_indices = np.argsort(-similarities)[: self.k_examples]
        return [int(pool_index) for pool_index in top_indices[::-1]]

    # ── 프롬프트 종류별 예시 인덱스 ──
    def select_examples(self, prompt_type, eval_position):
        if prompt_type == "zero_shot":
            return []
        if prompt_type == "fixed":
            return list(self.fixed_example_indices)
        if prompt_type == "random":
            return self._select_random_examples(eval_position)
        if prompt_type == "rag":
            return self._select_rag_examples(eval_position)
        raise ValueError(f"알 수 없는 prompt_type: {prompt_type} (가능: {PROMPT_TYPES})")

    # ── 메시지 조립: 시스템 프롬프트 → 예시 대화 턴(user 리뷰 / assistant JSON) → 실제 리뷰 ──
    def build_messages(self, example_indices, review_text):
        messages = [{"role": "system", "content": BASE_PROMPT}]
        for example_index in example_indices:
            messages.append({"role": "user", "content": self.pool_texts[example_index]})
            messages.append({"role": "assistant",
                             "content": json.dumps({"label": self.pool_codes[example_index]})})
        messages.append({"role": "user", "content": review_text})
        return messages

    # ── 실험 루프용: eval 리뷰 1건에 대해 (예시 review_id 목록, 메시지) 반환 ──
    def prepare(self, prompt_type, eval_position):
        example_indices = self.select_examples(prompt_type, eval_position)
        review_text = self.eval_df["review_text"][eval_position]
        example_review_ids = [self.pool_df["review_id"][pool_index] for pool_index in example_indices]
        return example_review_ids, self.build_messages(example_indices, review_text)


# ── 동작 확인: python src/examples.py (프로젝트 루트에서, 기본 경로 사용) ──
if __name__ == "__main__":
    selector = ExampleSelector(
        pool_path="data/processed/pool.csv",
        eval_path="data/processed/eval.csv",
        embedding_path="data/processed/embeddings_qwen3-0.6b.npz",
    )
    # 고정 예시 7개 (노트북 점검 1과 같아야 함)
    print("[고정 예시]")
    for pool_index in selector.fixed_example_indices:
        print(f"  {selector.pool_labels[pool_index]:8s} | {selector.pool_texts[pool_index][:60]}")

    # 프롬프트 종류별로 eval 첫 리뷰의 예시 수와 메시지 역할 순서 확인
    print("[메시지 구조: eval 0번]")
    for prompt_type in PROMPT_TYPES:
        example_review_ids, messages = selector.prepare(prompt_type, 0)
        role_sequence = "".join(message["role"][0] for message in messages)  # s/u/a 첫 글자
        print(f"  {prompt_type:9s} | 예시 {len(example_review_ids)}개 | {role_sequence}")
