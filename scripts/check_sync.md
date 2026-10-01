# 第一步：全量 RGB–depth 时间配对检查

运行（在仓库根目录）：

```bash
uv run python scripts/check_sync.py
# 只运行一个场景
uv run python scripts/check_sync.py --scenes scene_a
```

所有帧均参与检查，没有抽帧。默认结果在 `deliverables/sync_audit/<scene>/`。

- `frames.csv`：每帧时间差、相邻采样间隔、轨迹估计的速度、原配对与最佳候选的评分、改善量和复核标记。
- `candidates.csv`：每一个候选配对的评分、中位数、P90 和有效深度边界点数。
- `summary.json`：时间戳统计、候选帧列表、评分参数及限制。
- `timeline.png`：时间差、运动、图像误差和改善量曲线。
- `overlay_*.png`：部分候选及原始误差最大帧的叠加图；绿色为投影后的深度边界。仅图片展示数量受限，数值检查覆盖全部帧。

## 方法

1. 检查时间戳，计算 depth 时间减 RGB 时间，以及重复深度时间戳、采样缺口。时间异常本身不是标定故障结论。
2. 对每一张 RGB，从现有深度图中选择时间差在 ±0.35 秒内的候选，并始终包含原配对。按深度时间戳去重，避免把重复曝光视为不同候选。
3. 在有效深度像素之间提取跳变（大于 5 cm 且大于局部深度的 3%），仅保留前景侧；深度 0 的边界不作为有效几何边界。
4. 用给定深度内参、scale_m、T_color_depth 和 RGB 内参投影。利用全部有效深度点的 z-buffer 过滤明显遮挡。
5. 计算投影边界到 RGB Canny 边缘的距离。主评分为距离截断到 10 像素后的均值，越低越好；同时输出中位数和 P90。有效边界不足 30 点或 RGB 边缘不足 30 点时不评分。
6. 最佳候选必须比原配对时间更接近，误差至少下降 0.5 px 且下降 20%，边界点数至少为原配对的 80%，才标为 `review_candidate`。这些是可配置的初筛阈值，不是验收标准。

## 如何读结果

先打开 timeline，再用 frames.csv 定位异常区间，检查叠加图和 candidates.csv。比较整段时间内的表现，尤其关注运动片段是否重复出现改善。最佳候选是从多个候选中选出的最小值，存在选择偏差；孤立改善不证明错配。

运动速度由给定轨迹估计，仅作为诊断信息。没有做运动补偿，也没有优化空间标定，以便隔离配对变动的影响。内外参错误、RGB 纹理、弱纹理、动态物体、遮挡，以及不同候选中可见边界不同，都可能影响评分。

脚本不自动应用修正，不估计真实时钟偏移，也不修改输入数据。复核通过后，可在后续流程显式使用 RGB 帧 i 对应 depth 帧 j 的映射；RGB 位姿仍属于 RGB 时间，不应随 depth 文件索引更换。缺失观测不能靠重新索引恢复。

## 查看 RGB / depth 边缘图

```bash
# 四个场景全部帧
uv run python scripts/view_sync_edges.py
# 指定场景和帧，建议使用单独目录，避免覆盖全量 HTML 索引
uv run python scripts/view_sync_edges.py --scenes scene_a --frames 150 200 --output deliverables/sync_edges_selected
# 当前 RGB 配另一张候选深度图
uv run python scripts/view_sync_edges.py --scenes scene_a --frames 150 --depth-frame 153 --output deliverables/sync_edges_candidate
# 用统一米制色彩范围比较不同帧
uv run python scripts/view_sync_edges.py --depth-range 0.3 5 --output deliverables/sync_edges_fixed_range
```

打开 `deliverables/sync_edges/<scene>/index.html` 浏览全量图片。每张 overview 包含六个面板：RGB 原图、RGB Canny 边缘、RGB 上的投影深度边缘、彩色深度、深度原生边缘、彩色深度上的白色边缘。标题标出帧号及实际配对时间差。RGB 图默认使用原配对和原始标定。

深度显示用 TURBO 色图，近处蓝、远处红，无效值为灰色。默认用每张图有效深度的 2%–98% 分位数确定显示范围，范围之外饱和显示；标题标出具体米值，frames.json 记录范围。不同帧的同一种颜色默认不代表相同距离，需要跨帧比较时使用 `--depth-range` 固定范围。归一化仅影响显示，深度值和边缘算法没有更改。

独立 PNG 保留原生分辨率和精确二值掩码：`*_rgb_edges.png`、`*_depth_edges.png`、`*_projected_edges.png`；`*_depth_color.png` 为彩色深度。overview 为 2 倍最近邻放大后的 JPG，供浏览，不用于数值测量。深度原生边缘属于深度坐标系；需要和 RGB 边缘直接比较时看 projected_edges 或 RGB 叠加面板。
