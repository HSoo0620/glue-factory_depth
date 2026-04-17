# SHOT_zmap 파라미터 설정 기록 (v5)

## 변경 사유

이전 설정(voxelSize=2.0, normalR=10.0, shotR=20.0)으로 검증한 결과 **descriptor 구분력이 거의 없었음**.
shotRadius=20mm이 너무 작아 descriptor가 충분한 기하학적 맥락을 캡처하지 못한 것으로 판단.

## 물리 해상도 (resolution_mm)

| 축 | 해상도 | 의미 |
|----|--------|------|
| lateral (x) | 0.056 mm/pixel | 픽셀당 가로 간격 |
| transport (y) | 0.056 mm/pixel | 픽셀당 세로 간격 |
| vertical (z) | 0.0085 mm/raw | raw값 1당 높이 간격 |

이미지 물리 크기: ~135 mm x ~157 mm, z 범위: 0~557 mm

## 적용 파라미터 (mm 단위)

| 파라미터 | v5 (현재) | v2 (이전) | 비고 |
|---------|-----------|-----------|------|
| voxelSize | **5.0 mm** | 2.0 mm | |
| normalRadius | **25.0 mm** | 10.0 mm | |
| shotRadius | **50.0 mm** | 20.0 mm | 물체의 ~1/3 커버 |

비율: `voxel : normal : shot = 1 : 5 : 10`

## 실행 결과 (2026-04-13)

| 항목 | v5 (현재) | v2 (이전) |
|------|-----------|-----------|
| 처리 프레임 | 641/641 | 641/641 |
| 총 처리 시간 | 109초 | 169초 |
| 프레임당 포인트 | ~980-1,100 | ~5,400-6,100 |
| descriptor 유효율 | 100% | ~99.98% |
| 총 출력 용량 | **884 MB** | 4.9 GB |
| 출력 경로 | `output_shot_zmap_v5/` | `output_shot_zmap/` |

## 커맨드라인

```
SHOT_zmap.exe [datasetDir] [outputDir] [lateralMm] [transportMm] [verticalMm] [voxelSize] [normalRadius] [shotRadius]
```

기본값: `../../0413_resample ../../output_shot_zmap_v5 0.056 0.056 0.0085 5.0 25.0 50.0`
