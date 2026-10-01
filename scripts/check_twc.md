# 检查每帧 T_world_camera（Twc）

```bash
# 全量矩阵/运动检查 + 自动标记区间附近的几何/ICP检查
uv run python scripts/check_twc.py
# 仅做矩阵、时间戳、运动跳变检查，输出到单独目录
uv run python scripts/check_twc.py --basic-only --output deliverables/twc_basic/scene_a
# 额外指定关注区间中心；默认检查前后8帧，间隔1/3/5帧
uv run python scripts/check_twc.py --frames 764 --radius 15
```

默认读取 data/repaired/scene_a，包括本目录 sync.json；不会误用旧配对日志。普通 Scene 直接读取重排后的同编号图像。输入根目录可用 --scene-root 指定。

1. 全量检查有限值、齐次矩阵底行、RᵀR≈I 和 det(R)≈1（误差阈值1e-3），以及时间严格递增。
2. 计算相邻平移、旋转和速度。速度阈值为中位数+8×robust sigma，属于相对统计告警，不是机器人运动学硬限制。
3. 仅在自动告警/用户指定区间，固定全部标定参数和轨迹，用 inv(Tcd) inv(Twc_j) Twc_i Tcd 双向重投影深度，报告残差中位数/P90和重叠率。目标插值要求局部深度有效、平滑，不把大残差直接过滤掉；遮挡和动态物体也会造成残差。
4. 在相同帧对上独立执行正反向 point-to-plane ICP，恒等初始化，不用待检查的 Twc 初始化。比较独立运动与轨迹；只有收敛、重叠>50%、条件数<1000、正反闭合<5cm和2°才给出可靠性初筛标记。

默认两端 RGB-depth 曝光差都必须在0.001ms内（浮点同时间判断），并排除重复同次depth曝光。非同步帧标记 exposure_mismatch，不用它们把时间误差归给位姿。不估计尺度，不优化 Tcd，不自动修正位姿。

输出：

- poses.csv：全量矩阵及逐帧运动统计。
- local_geometry.csv：局部帧对的重投影残差、覆盖率、ICP差异及排除理由（仅有局部检查行时生成）。
- summary.json：检查范围、告警帧和质量统计。
- motion.png：速度曲线。

foreground_conflict_fraction 字段实际统计所有有效投影点中绝对深度差>15cm的比例（包含前后两个方向的冲突），不能只解释为前景遮挡。

第一步通过仅表示矩阵形式合法/未发现运动尖峰。局部几何通过不证明全局轨迹无漂移。ICP失败不证明轨迹一定错误；通过筛选也不是真值保证。全部参数、原始数据和重排数据都保持不变。
