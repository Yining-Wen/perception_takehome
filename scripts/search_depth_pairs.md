# 固定标定后按边缘误差搜索 depth

```bash
uv run python scripts/search_depth_pairs.py --scene scene_a --window-ms 1000
```

默认读取 ordered_audit/scene_a/working_calib.json 的固定标定，输出 depth_pair_search/scene_a。只搜索原配时间不一致的 247 张 RGB；其余帧原样保留。

对每张 RGB 搜索曝光时间前后各 1000 ms 内所有现有 depth，另始终包含原配。约 10 Hz 时通常相当于前后各 10 帧，但依据是时间而非文件编号。重复曝光去重。

每个候选投影深度前景边界到当前 RGB，计算到 Canny 边缘的截断均值（上限 10 px）。RGB 和投影 depth 边界均至少 30 点才评分。最低分胜出，分数完全相同时优先时间更近者。不要求胜出者的时间戳相同，亦不要求它比原配时间更接近。

- candidates.csv：全部候选、曝光差、边界数量、均值、中位数和 P90。
- frames.csv：原配、最低分、第二名、改善量、两名差距和序列诊断。
- best_pairing.json：每张 RGB 的最低分 depth 索引；无可评分候选为 null。
- reviewed_pairing.json：保守初筛映射；异常帧只有改善 ≥0.5 px 且 ≥20%、第一名领先第二名 ≥0.15 px 且 ≥第二名分数的 10%、边界数量至少保留 80%，并且未触发相邻曝光倒退/重复时才保留候选。其他异常帧为 null。
- search.png / overlay_*.png：误差曲线和部分对照图。

reviewed 是启发式初筛，不是真值保证。时间一致的原配没有在本脚本重做图像验证；序列检查只发现局部倒退/重复，没有做动态规划或重新选择次优候选。候选可见边界集合可能不同，80% 点数约束不能保证内容完全一致。最小值比较存在选择偏差；非零时间差的赢家可能在补偿标定残差或匹配到纹理，不能仅凭最低分确认物理同步。

原始图像、时间戳和标定不修改，也不自动替换 ordered_audit 或 Part 2 的配对映射。
