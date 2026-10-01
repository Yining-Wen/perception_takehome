# Kc → Kd → 时间筛选 → Tcd → 时间重配对 → 轨迹

```bash
uv run python scripts/audit_ordered.py --scene scene_a
# 仅依赖顺序改变时，可复用已完成且输入未变的 RGB/depth 独立检查
uv run python scripts/audit_ordered.py --scene scene_a --reuse-intrinsics
```

输出根目录：deliverables/ordered_audit/scene_a。scale_m 固定原始值，不做尺度扫描，也不声称深度内参检查验证了绝对尺度。

scene_b 按最终范围运行（Twc 只检查 det；空间候选包含平移和联合外参）：

```bash
.venv/bin/python scripts/audit_ordered.py --scene scene_b --det-only --models principal focal rotation translation rigid --accept-principal
.venv/bin/python scripts/search_depth_pairs.py --scene scene_b --window-ms 1000
.venv/bin/python scripts/materialize_pairs.py --source data/scene_b --sync variant/scene_b/sync.json --pairing deliverables/depth_pair_search/scene_b/best_pairing.json --calib deliverables/ordered_audit/scene_b/working_calib.json --output data/repaired/scene_b
```

`--accept-principal` 显式允许主点模型在 validation 胜出并通过现有 validation/test 改善阈值后，成为条件性的工作修正；并非仅凭 RGB-only 指标修改主点，也不代表物理误差已唯一定位。缺省仍只允许通过筛选的旋转模型。`--det-only` 跳过下述扩展轨迹检查，在 `05_twc_det/summary.json` 保存结果。`--reuse-intrinsics` 只在输入和设置未变时使用。物化命令拒绝覆盖已有输出。

1. **01_rgb**：RGB-only Kc 敏感性扫描与视觉运动，未得到可靠修正则保留 Kc。
2. **02_depth**：独立 depth ICP、Kd 敏感性扫描与反向闭合；不使用给定轨迹初始化。保留 Kd。
3. **03_timestamp**：只统计日志，筛选原始 RGB/depth 时间戳相同的配对。original_only/<scene>/pairing.json 中原配正确时填 i，其余填 null。这一步不修配对。
4. **04_extrinsic**：仅原配时间一致的帧参与训练、验证和测试。比较外参旋转与主点/焦距替代解释；旋转模型只有胜出且通过筛选才采用，否则保留 Tcd。固定工作标定写入 working_calib.json。外参平移未优化或独立验证。
5. **04b_pairing_repair**：外参决定之后才找同时间曝光的 depth。候选只依据日志选取，然后用固定工作标定比较原配和新配的图像误差；无曝光填 null。不修改时间戳数值。映射在 <scene>/pairing.json，逐帧证据在 pairing.csv。
6. **05_trajectory**：RGB 运动、depth ICP 运动与给定相对位姿比较。depth 文件索引按实际曝光时间转换成 RGB 位姿时间；重复曝光或无对应位姿的运动跳过。使用前面固定的 Tcd，不再根据异常配对调整外参。

原始时间戳一致仅是日志条件，并不证明硬件真实曝光同步。边缘距离受纹理、遮挡和其他标定影响，因此重配对误差是条件性证据，不能独立验证所有参数。

内参部分复用 parameter_audit.py；空间对齐复用 check_alignment.py 的 --pairing-root。所有时间帧参与覆盖，ICP 使用 lag=3、像素 stride=8。轨迹字段为 T_world_camera，相机到世界；用户所说 Kcd 在此指 T_color_depth 刚体变换。

这次修正复用了未改变的 01_rgb 和 02_depth，重新执行时间筛选、外参拟合、重配对复核和轨迹检查。旧运行残留的 03_timestamp/mapping、03_timestamp/pairing.csv、mapping_summary.json 属于旧顺序，不用于当前运行；当前映射以 04b_pairing_repair 为准，manifest.json 记录实际顺序。
