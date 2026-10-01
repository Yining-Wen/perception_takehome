# 六步参数分析：按顺序执行

所有入口默认处理四个场景；使用 `--scenes scene_a` 可指定场景。前四步输出到 `deliverables/parameter_audit/<scene>/01_sync` 等目录；第五、六步输出到 `deliverables/parameter_audit/05_scale/<scene>` 与 `06_extrinsic/<scene>`。

```bash
uv run python scripts/audit_01_sync.py
uv run python scripts/audit_02_rgb_intrinsics.py
uv run python scripts/audit_03_depth_intrinsics.py
uv run python scripts/audit_04_trajectory.py
uv run python scripts/audit_05_depth_scale.py
uv run python scripts/audit_06_extrinsics.py
```

六个入口分别调用 `parameter_audit.py`，避免重复实现。所有检查覆盖整个序列；RGB/独立 depth 配准默认比较 i 与 i+3，深度空间采样 stride=8。没有将每个场景缩减为几十帧。最后三帧作为目标帧参与比较。第五步沿用间隔 1、5、10 帧的重投影检查。

## 1. audit_01_sync.py：只检查日志

输入 sync.json。输出每帧时间差、采样间隔、重复时间戳和倒退统计，以及 median/MAD/P95/max。不使用投影误差。时间异常不等于实际错位，不拟合时钟偏移。

`timestamps.csv`、`summary.json` 是原始事实。之前的 `check_sync.py` 是额外的跨模态对照实验，可在第六步后用来复核时间异常。

## 2. audit_02_rgb_intrinsics.py：RGB-only

ORB → Hamming 匹配 → 0.75 ratio test → F 的 RANSAC → E=KᵀFK。每个帧对至少 12 个匹配/内点。

分别扫描 RGB 共同焦距倍率 0.9/0.95/1.05/1.1 与若干 ±12 px 主点候选，记录 E 两个大奇异值的不等程度 `abs(s1-s2)/(s1+s2)`。这是内参敏感性筛查，不是完整自标定；小指标不保证 K 正确。F 在像素坐标的误差不验证 K。没有实现 bundle adjustment，也不导出“已修正”内参。

使用原始 K 将 E 投影到本质矩阵约束，再 recoverPose，输出 RGB 相对旋转、单位平移方向、正深度比例。额外报告 homography 支持比例，提示平面/低视差退化。

输出 `pairs.csv`、`intrinsic_sweep.csv`、`motions.json`、`summary.json`。按 5 秒时间块循环 train/validation/test，首尾各留 0.5 秒；跨块帧对只做 audit。独立单目没有绝对平移尺度。

## 3. audit_03_depth_intrinsics.py：Depth-only

从深度内参和固定 scale_m 重建点云，中心差分估计表面法向，屏蔽缺失和大深度跳变。使用独立 point-to-plane ICP，恒等变换初始化，不读取给定轨迹作为配准初值。

最近邻距离上限 20 cm，按法向残差保留较好的 85% 对应点，最多 20 次迭代。输出 RMSE、fitness、线性系统条件数、是否收敛。大更新/不足 30 点时标失败。原始内参另外做反向 ICP，记录前后变换闭合误差。

扫描原始、共同焦距与主点假设，固定深度比例。各模型配准独立，不能只比较成功帧的均值来接受参数：对应集合/失败率可能变化，ICP 可以吸收形变，平面存在退化。summary 仅概括，不自动选择内参。固定尺度下的敏感性也不证明物理尺度。

输出 `pairs.csv`、`motions.json`、`summary.json`。此步骤计算量最大。

## 4. audit_04_trajectory.py：运动对比

必须先运行步骤 2、3。计算给定位姿的相对运动，与视觉估计旋转/平移方向比较；深度运动通过给定 T_color_depth 转换后比较旋转和平移。

RGB 可用性初筛：正深度比例 >70%，homography 内点比例 <90%。Depth 初筛：时间差 ≤5 ms、fitness >50%、条件数 <1000、ICP 收敛、反向闭合平移 <5 cm 且旋转 <2°。这些是启发式质量筛选，不证明测量正确。

输出 `motion_comparison.csv`、`pose_matrix_checks.csv`、`summary.json`。metric_scale_ratio 是深度配准平移长度 / 给定轨迹经外参转换后的平移长度，仅在后者 >3 cm 时计算。相同 timestamp 的重复 depth 观测、退化或动态物体可能污染结果。必须结合质量列和时间序列查看，不直接修正轨迹。

## 5. audit_05_depth_scale.py：条件性尺度验证

复用 `check_geometry.py`，只用原始外参，明确不自动加载之前的旋转候选。扫描 0.8–1.2 倍 scale_m，训练/验证/测试分组并输出固定支持的双向重投影误差。

使用给定轨迹作为条件性米制参照。步骤 4 提供独立诊断，但当前没有自动将其标记转换为轨迹剔除/修正；因此只能得出“相对于给定轨迹”的尺度结论。请结合步骤 4 查看异常片段，而非视为无条件标定真值。

## 6. audit_06_extrinsics.py：跨模态外参检查

先检查步骤 5 的 summary，只有尺度候选通过筛选才读入；否则用原始标定。具体输入和来源保存在 `06_inputs/<scene>/`。

复用 `check_alignment.py` 比较旋转、主点和焦距三个独立假设。保留内参替代假设，是为了避免错误地把所有残差归给外参。前面 RGB/depth-only 扫描只是敏感性证据，所以不会擅自将最小值写入标定。

当前仅优化外参旋转，不优化外参平移，不实现 hand-eye，也不声称六自由度外参已验证。对平移应结合深度分层残差、丰富运动和额外 fixture 继续检查。所有输出均为候选，不覆盖 data/。

## 解释边界

这些脚本提供不同来源的证据，不是六次独立证明。步骤 2/3 的内参敏感性不能保证可辨识性；步骤 4 仍依赖对应相机模型；步骤 5/6 仍是条件性检验。不能因某个数值最小就接受参数，尤其不能把 ICP 的低残差等同于正确尺度或轨迹。

## 本次运行验证范围

步骤 1、2、5、6 已在四个场景运行完成。步骤 3、4 已在 scene_a 完整运行；其余场景的完整 ICP 扫描本轮未跑完，避免继续扩大验证计算。代码默认仍处理全部四个场景，按上面的命令即可执行。合成非平面点云验证了 ICP 的已知旋转和平移恢复；Python 编译检查通过。未宣称尚未运行场景的配准结果有效。
