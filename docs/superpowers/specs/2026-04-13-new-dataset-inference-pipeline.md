# New Dataset Inference Pipeline & 속도 최적화 분석

**Date**: 2026-04-13
**Context**: 다음 주(2026-04-20 주간) 실제 3D 카메라 촬영 데이터로 live demo 예정. 본 문서는 new_dataset(ISS+FPFH/SHOT+LG) 파이프라인의 end-to-end 추론 흐름, per-stage 비용, 속도 최적화 옵션과 tradeoff를 정리한다.

---

## 1. End-to-End 추론 흐름

**입력**: 두 장의 zmap (16-bit PNG, camera-frame mm 좌표)
**출력**: 매칭 keypoint 쌍 + 4×4 rigid transform T (master → input)

### 단일 view 준비
```
zmap (cv2)                                                        [step 1]
  ↓ ISS 검출 (u,v,depth_scaled 공간)                                [step 2]
keypoints (u, v) × 512
  ↓ Camera-frame PCD 빌드 (x=u·0.056, y=v·0.056, z=raw·0.0085 mm)   [step 3]
  ↓ voxel_down_sample(5 mm) → ~1000 points
  ↓ estimate_normals(radius=25 mm, max_nn=30)
  ↓ compute_fpfh(radius=50 mm, max_nn=100)  [OR load SHOT .bin]    [step 4]
voxel PCD (~1000, 3) + descriptors (~1000, 33 or 352)
  ↓ KDTree 최근접 lookup: keypoint XYZ → nearest voxel point        [step 5]
keypoint descriptors (512, 33 or 352)
```

### Two-view + matcher + registration
```
view0, view1 each run step 1-5
  ↓ pack dict {view0, view1} → model forward (LightGlue)            [step 6]
pred["matches0"] (512,), pred["matches_scores"]  → valid 매칭 쌍
  ↓ backproject: kp(u,v) → camera-frame mm → world (R_cam, t_cam)  [step 7]
  ↓ RANSAC + SVD (1000 iter, inlier_th=5 mm)                       [step 8]
T_est (4×4 rigid transform)
  ↓ visualize: 4-color match overlay + 3D cloud alignment           [step 9]
```

## 2. Per-stage 비용 분해 (새 파라미터: voxel=5 / normal_r=25 / fpfh_r=50)

| 단계 | 연산 | 측정 비용 (원본 zmap ~2413×2000) | 이미지 크기 의존 |
|---|---|---|---|
| 1 | zmap 로드 (cv2) | ~0.1s | 약함 |
| 2 | ISS 검출 | ~1-2s | **강함** (pixel 수 비례) |
| 3 | PCD 빌드 + voxel 5mm | ~0.5s | 강함 |
| 4a | estimate_normals | ~0.3s | 약함 (voxel 이후 크기에 의존) |
| 4b | compute_fpfh | ~0.8s | 약함 |
| 4' | SHOT `.bin` 로드 + filter NaN | ~0.1s | 거의 없음 |
| 5 | KDTree lookup 512 kp | <0.05s | 없음 |
| — | **view 1장 합계** | **~3-4s (FPFH) / ~2-3s (SHOT)** | — |
| 6 | LightGlue matcher forward (GPU) | ~50ms | 없음 (kp 수만 의존) |
| 7 | backproject + world transform | <10ms | 없음 |
| 8 | RANSAC (1000 iter) + SVD | ~100-500ms | 없음 (매칭 수만 의존) |
| 9 | matplotlib 시각화 | ~300ms | 중간 |
| — | **Pair 전체 추론** | **~6-8s (CPU bound)** | — |

**병목**: CPU 쪽 ISS 검출 + PCD/FPFH. GPU (LG)는 거의 무시 가능.

## 3. 속도 최적화 옵션 — 효과 vs Risk

### 3.1 Zmap 해상도 다운스케일 (예: 0.5×)

| 단계 | 비용 변화 |
|---|---|
| ISS 검출 | 2.4M → 0.6M pixels → **~4× 빠름** |
| PCD 빌드 + voxel | **~3× 빠름** |
| estimate_normals + FPFH | 거의 변화 없음 (voxel=5mm 그대로) |
| **Per view 총** | **~3.8s → ~1.5s (2.5× 전체)** |

**주의점:**
- **INTER_NEAREST 필수** — depth에 bilinear 적용 시 경계 false 포인트 생성 (이미 적용 중)
- **Train-inference mismatch**: 현재 precompute는 원본 zmap 기준. 추론만 0.5× 다운스케일하면 voxel 내부 포인트 수 감소 → normal/FPFH noise 증가 → descriptor 분포 shift → LG 성능 저하 가능.
- **안전 적용**: 학습부터 0.5× zmap으로 precompute 재실행 → 학습 → 추론 모두 동일 scale. 일관성 확보.
- **사전 검증**: `scripts/diagnose_fpfh_discriminability.py`에 0.5× zmap 옵션 추가 → pos<neg가 현재 0.628에서 크게 떨어지지 않는지 확인 후 결정.

### 3.2 Voxel 크기 증가 (5mm → 10mm)

| 장점 | 위험 |
|---|---|
| FPFH ~2× 빠름 | PCD 포인트 ~1000 → ~250, FPFH neighbor 수 감소 → noise |
| estimate_normals 빠름 | SHOT은 이미 v5가 5mm voxel이므로 재추출 필요 |
| | 구분력 재검증 필수 (diagnostic 재실행) |

### 3.3 max_num_keypoints 축소 (512 → 256)

| 장점 | 위험 |
|---|---|
| KDTree lookup 2× 빠름 (무시 수준) | 매칭 가능 쌍 수 감소 |
| LG forward 약간 빠름 | Recall 타격 가능 |

효과 작음. 우선순위 낮음.

### 3.4 ISS 파라미터 완화

| 옵션 | 효과 | 위험 |
|---|---|---|
| salient_r 증가 | non-max suppression 반경 커짐 → 후보 kp 수 감소 → **ISS 빠름** | kp 품질 저하 가능 |
| gamma_21, gamma_32 증가 | 강한 saliency만 → 후보 감소 | 매칭 candidates 줄어듦 |
| min_neighbors 증가 | noise 필터링 강화, 연산 약간 감소 | boundary kp 손실 |

ISS 검출이 가장 큰 병목이므로 효과는 큼. 하지만 학습 시 사용한 값과 달라지면 descriptor 일관성 깨짐. **학습과 동일 값 유지 권장**.

### 3.5 CPU 멀티프로세싱 (view 병렬)

| 장점 | 위험 |
|---|---|
| Two view를 multiprocessing으로 병렬 → ~**50% 단축** | 파이프라인 코드 변경 필요 |
| Open3D는 멀티스레드 안전(대부분) | 디버깅 복잡도 증가 |

**효과/위험 균형 가장 좋음**. 모델 무관, 학습 재실행 불필요.

### 3.6 `.npz` 저장 생략 (추론 시)

| 장점 | 위험 |
|---|---|
| 파일 I/O ~50ms 절약 | 없음 |

추론 전용 코드에서는 당연히 적용.

### 3.7 FPFH → SPFH로 교체

| 장점 | 위험 |
|---|---|
| Neighbor 합산 없어 훨씬 빠름 | 구분력 FPFH보다 낮음 (diagnostic 재실행 필수) |
| | 학습 재실행 필요 |

데모 시점까지 시간 부족. 우선순위 낮음.

## 4. Demo 준비 단계 (Action Items)

### Phase A — 학습 완료 (현재 진행)
- [x] FPFH/SHOT v5 precompute (641 scenes 각각 ~30-40분, 병렬)
- [ ] FPFH 학습 (`0413_new_iss_fpfh_v5_lg`)
- [ ] SHOT v5 학습 (`0413_new_iss_shot_v5_lg`)
- [ ] match_recall 추이 비교 → FPFH vs SHOT 메인 트랙 결정

### Phase B — 추론 wrapper 작성
- [ ] `inference_new_iss_desc.py` 신규 작성:
  - 입력: 두 zmap 경로 (+ 옵션: scene config YAML for world transform)
  - 출력: 매칭 시각화, Transform T, RMSE (GT 있으면)
  - 내부: step 1-9 모두 포함, npz save 없음
- [ ] Two-view multiprocessing 적용 (3.5)
- [ ] 속도 측정 (현재 ~6-8s → 목표 ~3-4s per pair)

### Phase C — 실제 카메라 적응
- [ ] 카메라 spec 확보: resolution, depth 단위, intrinsic (LAT_MM/VERT_MM 등가물)
- [ ] 학습 데이터와 mm/pixel 다르면 zmap 전처리로 normalize (예: 리샘플)
- [ ] 3-5개 실제 쌍으로 sanity check → 매칭 품질 확인
- [ ] 시각화 포맷 데모용으로 정리 (4-color overlay + 3D cloud alignment)

### Phase D — 리허설 (demo 주 최소 1-2일 전)
- [ ] 전체 파이프라인 end-to-end live 실행 × 5 쌍
- [ ] 실패 케이스 fallback (저품질 매칭 시 메시지 등)
- [ ] 속도 재측정, 필요 시 3.1 (zmap 다운스케일) 적용 여부 결정

## 5. 결정 가이드

**속도 목표가 "발표 중 지연 없이 보일 정도"(pair당 ≤3s)**:
→ Phase B의 multiprocessing만 적용해도 ~3-4s 달성 가능. 추가 최적화 불필요.

**속도 목표가 "거의 실시간"(pair당 ≤1s)**:
→ 3.1 (zmap 0.5× + train 재실행) + 3.5 (multiprocessing) 조합 필요. 학습 ~몇 시간 + 재precompute ~30분 추가 소요. **diagnostic 재실행으로 pos<neg ≥ 0.55 보장 필수**.

**속도 목표가 "학습만 하고 바로 데모"**:
→ 현재 파이프라인 그대로 + multiprocessing만. ~3-4s per pair.

## 6. 참고

- 학습 파라미터/cache 구조: `docs/superpowers/specs/2026-04-13-new-dataset-iss-desc-lg-design.md`
- Descriptor 구분력 diagnostic 방법론: 메모리 `project_new_dataset_descriptor_diagnostic.md`
- SHOT v5 C++ 파라미터: `docs/superpowers/specs/C++_SHOT_zmap_params_v5.md`
- 메모리 인덱스: `MEMORY.md`
