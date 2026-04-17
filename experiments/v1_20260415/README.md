# v1 (2026-04-15) ISS + FPFH/SHOT + LightGlue

Mitsubishi zmap dataset (641장) 대상 ISS keypoint + FPFH/SHOT descriptor + LightGlue 매칭.

## 좌표계 / 파라미터

- 단위: **mm** (X=u·0.056, Y=v·0.056, Z=raw·0.0085)
- Zero-pad: (W=2432, H=3008)
- voxel=1mm (descriptor 계산용), normal_r=20mm, fpfh_r=20mm, shot_r=40mm
- ISS: dense PCD (voxel 미적용), salient=6·avg_nn, non_max=2·salient
- MAX_KEYPOINTS=512 (ISS n ≥ 512면 랜덤 subsample, 미만이면 voxel cloud에서 랜덤 패딩)

## 구조

```
experiments/v1_20260415/
  precompute/          ISS 검출 + FPFH/SHOT 캐시 생성
    helpers.py         공통 헬퍼 (좌표변환, ISS, subsample)
    iss_fpfh.py        FPFH(33D) precompute
    iss_shot.py        SHOT(352D) precompute (pybind shot_module)
  train/               학습 entry shell
    iss_fpfh.sh
    iss_shot.sh
  test/                매칭 시각화
  registration/        2-view registration (RANSAC+SVD)
  eval/                양방향 Transform RMSE 평가
  logs/                precompute 로그
  results/             test/registration/eval 결과 이미지·수치
```

패키지 코드 (건드리지 않음):
- `gluefactory/train_iss_v1_20260415.py` — 학습 루프
- `gluefactory/configs/iss_{fpfh,shot}_v1_20260415_lg.yaml` — 학습 config
- `gluefactory/datasets/mitsubishi_v1_20260415_iss_{fpfh,shot}_dataset.py` — 데이터셋
- `gluefactory/datasets/mitsubishi/iss_{fpfh,shot}_v1_20260415_cache_*/` — 캐시 출력

## 실행 (repo root 기준)

```bash
conda activate LightGlue

# 1) Precompute (한 번만)
python experiments/v1_20260415/precompute/iss_fpfh.py        # FPFH, ~65분
python experiments/v1_20260415/precompute/iss_shot.py        # SHOT, ~30분

# 2) 학습
bash experiments/v1_20260415/train/iss_fpfh.sh
bash experiments/v1_20260415/train/iss_shot.sh
# 환경변수 override 가능: GPU=1 EXP=my_exp BATCH=8 bash ...
```

## 주요 컨벤션

- **FPFH 캐시는 L2-normalize 미적용** (raw histogram). 추론 시에도 normalize 금지 — train/test 분포 일치 유지.
- **SHOT**은 PCL 구현 자체가 unit-sphere 정규화 → 별도 처리 불필요.
- 학습 체크포인트: `outputs/training/<EXP>/` (gluefactory 표준 경로, 건드리지 않음)

## 배경

- 이전 파이프라인 (`precompute_new_iss_fpfh.py`) 은 ISS를 voxel 위에서 검출 → 1mm voxel에서 n_iss ≈ 44~72로 부족. v1에서는 **dense PCD에서 ISS 검출**로 전환해 n_iss ≈ 수백~1000 확보.
- 실험 파라미터 분포: `iss_distribution_v1_20260415.json` + `measure_iss_distribution_v1.py` 참조.
