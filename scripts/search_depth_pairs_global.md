# 窗口搜索 → 全局搜索 → flagged

```bash
.venv/bin/python scripts/search_depth_pairs_global.py --scene scene_c
.venv/bin/python scripts/materialize_filtered_pairs.py --scene scene_c --output data/repaired_filtered/scene_c
```

输入始终是原始 data/scene_c、variant/scene_c/sync.json 和冻结的 ordered_audit/scene_c/working_calib.json，不对上一次重排结果重复应用映射。默认拒绝覆盖已有输出。

1. 原始时间戳不同的 63 张 RGB 进入搜索；693 张原配同步帧作为保留锚点。
2. 先搜索 ±1000 ms，并保留原配作对照；相同曝光时间去重。
3. 最佳候选须不同于原配，平均截断边缘误差 ≤3 px，比原配改善 ≥0.5 px 且 ≥20%，领先第二名 ≥0.15 px 且 ≥第二名的 10%，边界数量 ≥原配的 80%。3 px 是新增绝对误差启发式上限，可通过 --max-score-px 修改，不是经过真值标定的概率。
4. 质量失败或与保留序列冲突的帧，搜索全体 depth 的不同曝光。使用同样阈值，不因为候选更多就放宽标准。全球候选不做相机运动补偿。
5. 最终保留序列不得复用曝光或时间倒退；迭代剔除冲突异常端点。原始同步锚点不剔除。最终序列检查若新拒绝一个尚未做过全局搜索的窗口帧，也会补做全局搜索；已经全局搜索过的失败帧不再尝试次优候选或动态规划，属于保守处理。
6. pairing.json 用 null 表示待归档。frames.csv、decisions.json 保留窗口及全局选择和拒绝原因，candidates.csv 记录所有计算的候选。仍然无法证明被接受者是真实同步。

scene_c 本次比较 39452 个不同 RGB/depth 候选：63 张异常，7 张窗口通过，56 张全局重试，0 张全局通过。最终 700 对保留、56 对 flagged。54 张无同时间曝光，另 352 和 519 因支持不足归档。全局搜索没有生成新曝光，也不能把不同时刻相似房间画面当同步证明。

## 数据布局与索引

最终目录 data/repaired_filtered/scene_c/：

- rgb/、depth/：700 对连续编号图片。
- calib.json：保留帧的原始时间和 Twc，id 重排为 0..699；空间标定沿用已冻结 Tcd 修正。
- sync.json：700 行，RGB 时间取原 RGB，depth 时间取所选原曝光；不改时钟。
- index_map.json：新编号对应原 RGB/depth 编号。
- original_to_new_rgb.json：原编号→新编号；删除行对应 null。
- flagged/：56 对未修配的原始 RGB/depth，连续编号；独立的原始空间标定 calib.json、sync.json 和带失败理由的 index_map.json。这里不应用候选修正。

父目录的普通 Scene 不加载 flagged 子目录。过滤造成的时间缺口保留，不插值、不复制相邻 Twc。variant queries/detections 仍用原 RGB 编号，后续必须借助映射转换；本脚本不重写它们或旧答案。

原始数据和 data/repaired/scene_c 旧输出保留。最新策略只适用于此处生成的 filtered 版本，scene_a/b 不变。所有复制图片逐一 SHA-256 验证。

## 其他场景与物化后校验

同一实现可用于 scene_d：

```bash
.venv/bin/python scripts/audit_ordered.py --scene scene_d --det-only --models principal focal rotation translation rigid --accept-principal
.venv/bin/python scripts/search_depth_pairs_global.py --scene scene_d
.venv/bin/python scripts/materialize_filtered_pairs.py --scene scene_d --output data/repaired_filtered/scene_d
.venv/bin/python scripts/verify_filtered_scene.py --scene scene_d
```

最后一步逐文件核对图像哈希、calib 帧及索引、sync、分区完整性、新旧索引映射、保留序列曝光顺序，并重新计算过滤后 Twc 的 det；结果写到最终目录的 verification.json。此校验不代表真实轨迹或几何精度已经验证。以上脚本均不会覆盖原始数据。

scene_a 最新重跑使用同样命令，将 --scene 改为 scene_a。原始 1252 帧，135 张接受重配对（134 窗口、1 全局阶段），112 张归档，最终 1140 对。全局阶段那一张仍选择原窗口内同时间曝光，是序列冲突解除后恢复保留，不是新增远时刻曝光。详细数据以 deliverables/depth_pair_search_global/scene_a 为准；scene_a_before_fallback_fix 是修复回退边界情况之前的中间结果。

## scene_b 最新重跑

```bash
.venv/bin/python scripts/audit_ordered.py --scene scene_b --det-only --models principal focal rotation translation rigid --accept-principal
.venv/bin/python scripts/search_depth_pairs_global.py --scene scene_b
.venv/bin/python scripts/materialize_filtered_pairs.py --scene scene_b --output data/repaired_filtered/scene_b
.venv/bin/python scripts/verify_filtered_scene.py --scene scene_b
```

1081 帧原始输入，16 张异常，6 张窗口接受、10 张全局重试仍失败；1071 对保留、10 对归档。主点修正与旧版一致；最新数据是 data/repaired_filtered/scene_b，旧 data/repaired/scene_b 不再代表最终配对策略。四场景现在均有 filtered 输出。
