
#!/usr/bin/env python3
"""Merge my_task (10 eps) into tool_good as episodes 32-41."""
import json, shutil, os, sys
import pyarrow.parquet as pq
import pyarrow as pa

BASE = os.path.dirname(os.path.abspath(__file__))
src  = os.path.join(BASE, "data", "my_task")
dst  = os.path.join(BASE, "data", "tool_good")

with open(os.path.join(dst, "meta", "info.json")) as f:
    dst_info = json.load(f)

ep_offset    = dst_info["total_episodes"]
frame_offset = dst_info["total_frames"]
print(f"Merging {src} -> {dst}")
print(f"  ep_offset={ep_offset}, frame_offset={frame_offset}")

with open(os.path.join(src, "meta", "episodes.jsonl")) as f:
    src_episodes = [json.loads(l) for l in f]

new_total_frames = frame_offset

for ep in src_episodes:
    old_idx = ep["episode_index"]
    new_idx = old_idx + ep_offset
    ep_len  = ep["length"]

    # ── Parquet ──────────────────────────────────────────────────
    src_pq = os.path.join(src, "data", "chunk-000", f"episode_{old_idx:06d}.parquet")
    dst_pq = os.path.join(dst, "data", "chunk-000", f"episode_{new_idx:06d}.parquet")

    tbl = pq.read_table(src_pq)

    # Replace scalar index columns
    tbl = tbl.set_column(tbl.schema.get_field_index("episode_index"),
                         "episode_index",
                         pa.array([new_idx] * ep_len, type=pa.int64()))
    tbl = tbl.set_column(tbl.schema.get_field_index("frame_index"),
                         "frame_index",
                         pa.array(list(range(ep_len)), type=pa.int64()))
    tbl = tbl.set_column(tbl.schema.get_field_index("index"),
                         "index",
                         pa.array(list(range(new_total_frames,
                                             new_total_frames + ep_len)),
                                  type=pa.int64()))

    cam_col = "observation.images.cam_0"
    if cam_col in tbl.schema.names:
        new_vp = (f"videos/chunk-000/observation.images.cam_0/"
                  f"episode_{new_idx:06d}.mp4")
        old_struct = tbl.column(cam_col).combine_chunks()
        ts_vals = old_struct.field("timestamp")
        img_struct = pa.StructArray.from_arrays(
            [pa.array([new_vp] * ep_len, type=pa.string()), ts_vals],
            names=["path", "timestamp"])
        tbl = tbl.set_column(tbl.schema.get_field_index(cam_col),
                             cam_col, img_struct)

    pq.write_table(tbl, dst_pq)

    # ── Video ────────────────────────────────────────────────────
    src_vid = os.path.join(src, "videos", "chunk-000",
                           "observation.images.cam_0", f"episode_{old_idx:06d}.mp4")
    dst_vid = os.path.join(dst, "videos", "chunk-000",
                           "observation.images.cam_0", f"episode_{new_idx:06d}.mp4")
    if os.path.exists(src_vid):
        shutil.copy2(src_vid, dst_vid)
        vid_status = "video copied"
    else:
        vid_status = "NO VIDEO"

    print(f"  ep {old_idx:2d} -> {new_idx:2d}: {ep_len} frames  {vid_status}")

    # ── episodes.jsonl ───────────────────────────────────────────
    with open(os.path.join(dst, "meta", "episodes.jsonl"), "a") as f:
        f.write(json.dumps({
            "episode_index": new_idx,
            "tasks": ["tool_good demo"],
            "length": ep_len,
        }) + "\n")

    new_total_frames += ep_len

# ── Update info.json ─────────────────────────────────────────────
dst_info["total_episodes"] = ep_offset + len(src_episodes)
dst_info["total_frames"]   = new_total_frames
dst_info["total_videos"]   = ep_offset + len(src_episodes)
dst_info["splits"]         = {"train": f"0:{ep_offset + len(src_episodes)}"}
with open(os.path.join(dst, "meta", "info.json"), "w") as f:
    json.dump(dst_info, f, indent=2)

print(f"\nDone. tool_good now has {dst_info['total_episodes']} episodes, "
      f"{dst_info['total_frames']} total frames.")
