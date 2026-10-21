# 올리스트 불만 리뷰 원인 분류 — 단계별 수행 계획

**데이터:** Kaggle `olistbr/brazilian-ecommerce` (2016~2018)
**분석 대상:** 1~2점이면서 코멘트가 있는 리뷰
**목표:** 리뷰 텍스트로 불만 원인을 자동 분류하고, 불만이 몰리는 판매자·카테고리·지역을 찾는다.

**전체 흐름:** ①에서 정답셋과 예시 풀을 만들고 → ②에서 6개 조합(프롬프트 3종 × 모델 2종)을 비교하고 → ③④에서 고른 조합을 서비스로 묶는다.

---

## 사용 데이터

| 파일 | 용도 |
|---|---|
| `olist_order_reviews_dataset.csv` | 분석 대상 (점수, 제목, 코멘트) |
| `olist_orders_dataset.csv` | 주문 상태, 배송일, 예정일 → 배송 지연 검증용 |
| `olist_order_items_dataset.csv` | 주문 → 상품·판매자 연결 |
| `olist_products_dataset.csv` + `product_category_name_translation.csv` | 카테고리 (영문 번역) |
| `olist_sellers_dataset.csv` | 판매자 지역 |
| `olist_customers_dataset.csv` | 고객 지역 |

- `geolocation`은 처음엔 제외한다. 100만 행이 넘고 우편번호가 중복돼 있으며, 주(state) 단위 집계면 충분하다.
- marketing-funnel 데이터셋은 쓰지 않는다. 판매자 업종 같은 속성이 필요해지면 그때 검토한다.

**조인 경로**

```
reviews ─order_id→ orders ─customer_id→ customers(customer_state)
                     └─order_id→ order_items ─product_id→ products → translation
                                             └─seller_id→ sellers(seller_state)
```

**주의사항**

- 리뷰 텍스트는 포르투갈어이고, 코멘트가 비어 있는 리뷰가 절반이 넘는다.
- `review_id`는 유일하지 않다. `(review_id, order_id)` 기준으로 중복을 처리한다.
- 한 주문에 판매자가 여러 명일 수 있다. 판매자 귀속 규칙이 필요하다.
- 배송 지연은 `order_delivered_customer_date > order_estimated_delivery_date` 조건으로 라벨 검증에 활용할 수 있다.
- 평가셋을 비교 대상과 같은 계열의 LLM으로 선라벨링하면 평가가 편향된다. 사람이 검수한 라벨을 기준으로 삼는다.

---

## ① 데이터 정제 + 평가셋 라벨링

### 정제
- [ ] 6개 테이블 조인
- [ ] 대상 필터링: 1~2점이면서 코멘트 있음, `(review_id, order_id)` 기준 중복 제거
- [ ] 판매자 귀속 규칙 결정 (단일 판매자 주문만 사용 / 관련 판매자 전원 귀속)
- [ ] 텍스트 정리: 제목과 본문 합치기, 공백 정리. 원문은 포르투갈어 그대로 두고, 번역은 사람이 읽을 때만 사용

### 라벨 체계 정의 (가장 중요)
- [ ] 카테고리 확정. 예: 배송지연 / 미수령 / 오배송·누락 / 품질·파손 / 설명과 다름 / 응대·환불 / 기타
- [ ] 카테고리별 정의, 경계 규칙, 애매한 사례를 문서화 (예: "늦게 왔는데 파손"은 무엇으로 볼지)
- [ ] 단일 라벨(주 원인 하나)과 다중 라벨 중 결정. 처음엔 단일 주 원인을 권장

### 라벨링
- [ ] 50건으로 파일럿 라벨링 후 가이드라인 수정
- [ ] **평가셋** 300~500건 라벨링 (층화 샘플, 성능 측정 전용)
- [ ] **예시 풀** 200~500건 라벨링 (few-shot 예시와 RAG 검색 대상). **평가셋과 절대 겹치지 않게 한다.**
- [ ] 일관성 점검: 일부를 며칠 뒤 다시 라벨링해 일치율(kappa) 확인
- [ ] 배송지연 라벨을 실제 배송일 데이터와 대조

**산출물:** 정제 parquet, 라벨 가이드 문서, `eval.csv`, `pool.csv`

---

## ② 프롬프트 3종 × 모델 2종 비교

### 준비
- [ ] 상용 API 모델 하나 선정
- [ ] 오픈 LLM 선정: 포르투갈어가 되는 소형 instruct 모델
- [ ] 오픈 LLM을 vLLM/Ollama 등으로 Docker에서 서빙. OpenAI 호환 엔드포인트를 쓰면 두 모델을 같은 코드로 호출할 수 있다.
- [ ] 다국어 임베딩 모델로 예시 풀을 벡터화해 pgvector에 적재

### 실험 설계

세 프롬프트 모두 예시 수 k, 출력 형식(JSON), temperature=0을 동일하게 맞춘다.

| 프롬프트 | 내용 |
|---|---|
| zero-shot | 카테고리 정의만 제공 |
| 고정 few-shot | 풀에서 고른 대표 예시 k개를 고정 |
| RAG few-shot | 리뷰마다 유사도 top-k 예시를 검색 |
| (권장) 랜덤 few-shot | 리뷰마다 풀에서 랜덤 k개 — 대조군 |

**효과 분리**
- 예시 효과 = 고정 few-shot − zero-shot
- 검색 효과 = RAG − 고정 few-shot
- RAG와 고정 few-shot은 "리뷰마다 예시가 바뀐다"는 점과 "비슷한 예시다"라는 점이 동시에 다르다. 랜덤 few-shot 대조군과 비교해야 순수한 유사도 효과를 분리할 수 있다.

### 지표
- **정확도:** macro-F1, 클래스별 F1, 혼동행렬, JSON 파싱 실패율
- **비용:** API는 토큰 × 단가, 로컬은 GPU 시간 환산
- **속도:** 지연시간 p50/p95, 처리량
- 평가셋이 작으므로 부트스트랩 신뢰구간을 함께 보고한다.

### 마무리
- [ ] 설정 파일로 돌아가는 실험 스크립트 작성, 결과표 CSV 저장
- [ ] 최종 조합을 골라 대상 전체(약 1만 건) 분류

**산출물:** 비교 결과표, 선택 근거, 전체 예측 결과

---

## ③ API + 비동기 워커로 서비스화

- [ ] FastAPI 엔드포인트
  - `POST /classify`: 작업을 큐에 넣고 job_id 반환
  - `GET /jobs/{id}`: 결과 조회
- [ ] Redis: 작업 큐(RQ / arq / Celery 중 택1)와 결과 캐시(같은 텍스트면 재사용)
- [ ] 워커: pgvector 검색 → 모델 호출 → DB 저장. 재시도, 타임아웃, API rate limit, 파싱 실패 처리 포함
- [ ] PostgreSQL 스키마: `reviews`, `gold_labels`, `examples`(vector), `predictions`
  - `predictions`에는 모델, 프롬프트, 라벨, 지연시간, 토큰, 비용을 기록
- [ ] docker-compose: api / worker / redis / postgres(pgvector) / llm 서버

**완료 기준:** `docker compose up` 한 번으로 요청부터 결과 저장까지 동작

---

## ④ 통합, CI, 대시보드

### GitHub Actions
- [ ] PR에서 프롬프트나 코드가 바뀌면 평가셋으로 자동 평가
- [ ] macro-F1이 기준보다 떨어지면 실패 처리
- [ ] 기본 러너에는 GPU가 없으므로 CI에서는 API 모델(또는 소수 샘플)만 실행하고, 로컬 모델 평가는 수동으로 실행
- [ ] API 키는 GitHub Secrets로 관리
- [ ] 단위 테스트 추가 (출력 파싱 등)

### 불만 집중 분석
- [ ] 판매자, 카테고리, 지역별 불만 원인 분포 집계
- [ ] 건수가 아니라 **주문 수 대비 비율**로 비교. 최소 표본 수 기준이나 스무딩을 적용해 소규모 판매자의 과대평가를 막는다.
- [ ] 지역은 고객 주와 판매자 주를 구분한다. 배송지연은 경로의 영향이 크다.

### 대시보드 / 문서
- [ ] Streamlit이나 Metabase로 Postgres에 연결
  - 모델 비교 결과
  - 불만 핫스팟
- [ ] README: 아키텍처, 결과 요약, 실행 방법
