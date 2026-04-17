# SHOT_zmap 파라미터 설정 기록

## 배경

`0413_resample` zmap 이미지(2413x~2800, 16-bit PNG, 641장)에서 SHOT352 descriptor를 추출할 때,
기존 설정(voxelSize=1.0, 픽셀 단위)으로는 출력 용량이 **프레임당 ~5-6 GB, 총 ~3-4 TB**로 비현실적이었음.

### 원인

- Orthographic projection에서 `x=u(pixel), y=v(pixel)` 로 매핑하면 각 픽셀이 고유한 정수 좌표를 가짐
- `voxelSize=1.0`일 때 VoxelGrid가 사실상 아무 것도 다운샘플하지 못함 (모든 픽셀이 서로 다른 voxel)
- 결과: 프레임당 ~440만 포인트 x 1,420 bytes(SHOT352) = ~5.9 GB

### 비교: perspective projection (output_shot_0407)

- 5761x5761 depth map + perspective 투영 -> 3D 좌표가 뭉침 -> VoxelGrid 효과적
- 프레임당 ~34,620 포인트 = ~47 MB
- 641장 총 23 GB

## 해결: 물리 단위(mm) 변환 + 파라미터 스케일 조정

### 물리 해상도 (resolution_mm)

| 축 | 해상도 | 의미 |
|----|--------|------|
| lateral (x) | 0.056 mm/pixel | 픽셀당 가로 간격 |
| transport (y) | 0.056 mm/pixel | 픽셀당 세로 간격 |
| vertical (z) | 0.0085 mm/raw | raw값 1당 높이 간격 |

### 좌표 변환 (SHOT_zmap.cpp)

```
x = u * 0.056   (mm)
y = v * 0.056   (mm)
z = raw * 0.0085 (mm)
```

- 이미지 물리 크기: ~135 mm x ~157 mm (lateral x transport)
- z 범위: raw 0~65535 -> 0~557 mm

### 최종 적용 파라미터 (mm 단위)

| 파라미터 | 값 | 의미 |
|---------|-----|------|
| voxelSize | 2.0 mm | 다운샘플 voxel 크기 |
| normalRadius | 10.0 mm | Normal 추정 반경 |
| shotRadius | 20.0 mm | SHOT descriptor 반경 |

### 파라미터 선정 과정

1. **voxelSize=10.0mm** (비율 1:5:10 → normal=50, shot=100): 프레임당 ~280 포인트. 정밀 정합에는 너무 적음.
2. **voxelSize=0.5mm** (비율 1:5:10 → normal=2.5, shot=5.0): 프레임당 ~30,000 포인트. output_shot_0407 수준이나 voxelSize가 너무 작음.
3. **voxelSize=2.0mm** (비율 1:5:10 → normal=10, shot=20): 프레임당 ~5,400-6,100 포인트. 정밀 정합 용도로 채택.

### 파라미터 비율

`voxel : normal : shot = 1 : 5 : 10` (기존과 동일한 비율 유지)

### 주의사항

- `normalRadius`는 반드시 `voxelSize`보다 커야 함 (이웃을 찾기 위해)
- `shotRadius`는 `normalRadius`보다 커야 descriptor가 충분한 이웃을 포함함
- voxelSize만 올리고 normalRadius/shotRadius를 그대로 두면 이웃 부족으로 NaN 발생

## 실행 결과 (2026-04-13)

| 항목 | 값 |
|------|-----|
| 처리 프레임 | 641/641 (100%) |
| 총 처리 시간 | 169초 (~2분 49초) |
| 프레임당 처리 시간 | ~0.2-0.3초 |
| 프레임당 포인트 | ~5,400-6,100 |
| descriptor 유효율 | ~99.98% (NaN 거의 없음) |
| 총 출력 용량 | **4.9 GB** (1,282 파일: 641 .pcd + 641 .bin) |
| 프레임당 평균 용량 | ~7.6 MB |
| 출력 경로 | `output_shot_zmap/` |

## 커맨드라인 인자 순서

```
SHOT_zmap.exe [datasetDir] [outputDir] [lateralMm] [transportMm] [verticalMm] [voxelSize] [normalRadius] [shotRadius]
```

기본값: `../../0413_resample ../../output_shot_zmap 0.056 0.056 0.0085 2.0 10.0 20.0`
