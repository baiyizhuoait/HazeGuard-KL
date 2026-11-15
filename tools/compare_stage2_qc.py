"""Compare a rebuilt 02 candidate with the published 02 table by grid and hour."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stage2_storage import digest, feature_artifact

KEY = ["station_id", "timestamp"]
SPLITS = ("train", "val", "test")
LABELS = ["target_pm25_3h", "target_level_3h"]
ELIGIBLE = "audit_model_eligible"
EPISODE_START = pd.Timestamp("2023-03-08 14:00")
UPSTREAM_COMMIT = "ec1ff0aa8a3a7bd4cfe776ddee28cf2d68853919"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require_fingerprint(root, name, fingerprints):
    data = (root / name).read_bytes()
    expected = fingerprints[name]
    stored_hash = sha256(data).hexdigest()
    if len(data) == expected["bytes"] and stored_hash == expected["sha256"]:
        return {"sha256": stored_hash, "stored_sha256": stored_hash, "line_endings": "as_stored"}
    if Path(name).suffix in (".csv", ".json"):
        crlf = data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        if len(crlf) == expected["bytes"] and sha256(crlf).hexdigest() == expected["sha256"]:
            return {"sha256": expected["sha256"], "stored_sha256": stored_hash,
                    "line_endings": "CRLF_export"}
    raise ValueError(f"Upstream fingerprint mismatch: {name}")


def key_index(frame):
    index = pd.MultiIndex.from_frame(frame[KEY])
    if not index.is_unique or frame[KEY].isna().to_numpy().any():
        raise ValueError("Expected nonnull and unique grid/hour keys")
    return index


def key_string(station, timestamp):
    return f"{station}|{timestamp.isoformat(sep=' ')}"


def key_fingerprint(frame, mask):
    index = key_index(frame.loc[mask])
    lines = sorted(key_string(*key) for key in index)
    return sha256("".join(f"{line}\n" for line in lines).encode("utf-8")).hexdigest()


def field_delta(before, after):
    am, bm = before.isna().to_numpy(), after.isna().to_numpy()
    equal = before.eq(after).fillna(False).to_numpy(dtype=bool)
    both = ~(am | bm)
    result = {
        "changed_rows": int(((am ^ bm) | (both & ~equal)).sum()),
        "became_missing": int((~am & bm).sum()),
        "became_available": int((am & ~bm).sum()),
        "changed_both_present": int((both & ~equal).sum()),
    }
    if pd.api.types.is_numeric_dtype(before):
        a, b = before.to_numpy(dtype="float64"), after.to_numpy(dtype="float64")
        finite = np.isfinite(a) & np.isfinite(b)
        result["max_abs_finite_delta"] = float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else None
    return result


def compare_columns(old, new, oi, ni, columns):
    return {c: field_delta(old[c].iloc[oi].reset_index(drop=True),
                           new[c].iloc[ni].reset_index(drop=True)) for c in columns}


def counts(frame, mask):
    return {str(role): int(n) for role, n in frame.loc[mask, "role"].value_counts().items() if n}


def describe_keys(frame, positions, reason):
    checks = [c for c in frame if c.startswith("audit_")]
    return [dict(key=key_string(row.station_id, row.timestamp), role=str(row.role),
                 audit_model_eligible=bool(getattr(row, ELIGIBLE)),
                 failed_checks=[c for c in checks if not getattr(row, c)],
                 upstream_reason=reason.get((row.station_id, row.timestamp)))
            for row in frame.iloc[positions].itertuples(index=False)]


def split_evidence(old, new, oi, ni, old_dropped, new_added, fields, reasons):
    out = {}
    original = list(old)[:35]
    spatial = [c for c in old if c.startswith(("upwind_", "far_source_", "spatial_"))]
    for split in SPLITS:
        old_split, new_split = old.split.eq(split).to_numpy(), new.split.eq(split).to_numpy()
        omitted, added = old_dropped & old_split, new_added & new_split
        shared = old_split[oi]
        i, j = oi[shared], ni[shared]
        a, b = old[ELIGIBLE].to_numpy()[i], new[ELIGIBLE].to_numpy()[j]
        receptor = old.role.eq("receptor").to_numpy()[i]
        active = old.role.eq("receptor").to_numpy() & old_split & old[ELIGIBLE].to_numpy()
        next_active = new.role.eq("receptor").to_numpy() & new_split & new[ELIGIBLE].to_numpy()
        both = receptor & a & b
        label_deltas = compare_columns(old, new, i, j, LABELS)
        out[split] = {
            "rows": {"old": int(old_split.sum()), "new": int(new_split.sum()), "common": len(i),
                     "removed_by_role": counts(old, omitted), "added_by_role": counts(new, added)},
            "eligibility": {
                "common_true_to_false_by_role": counts(old, i[a & ~b]),
                "common_false_to_true_by_role": counts(new, j[~a & b]),
                "removed_eligible_by_role": counts(old, omitted & old[ELIGIBLE].to_numpy()),
                "added_eligible_by_role": counts(new, added & new[ELIGIBLE].to_numpy()),
                "removed_receptor_keys": describe_keys(old, np.flatnonzero(omitted & old.role.eq("receptor")), reasons),
                "lost_eligible_receptor_common_keys": describe_keys(new, j[receptor & a & ~b], reasons),
                "gained_eligible_receptor_common_keys": describe_keys(old, i[receptor & ~a & b], reasons),
            },
            "labels_on_common_rows": label_deltas,
            "all_common": compare_columns(old, new, i, j, list(old.drop(columns=KEY))),
            "spatial_fields_on_common_receptors": compare_columns(old, new, i[receptor], j[receptor], spatial),
            "eligible_receptors": {
                "old": int(active.sum()), "new": int(next_active.sum()), "common": int(both.sum()),
                "old_keys_sha256": key_fingerprint(old, active),
                "new_keys_sha256": key_fingerprint(new, next_active),
                "labels": compare_columns(old, new, i[both], j[both], LABELS),
                "model_fields": compare_columns(old, new, i[both], j[both], fields),
                "excluded_upstream_fields": compare_columns(
                    old, new, i[both], j[both],
                    [c for c in original if c.startswith("b_proto_") or c.endswith("_anomaly")]),
            },
        }
    return out


def removed_source_checks(old, old_keys, listed):
    positions = old_keys.get_indexer(key_index(listed))
    removed = old.iloc[positions][["station_id", "role", "split", "audit_source_clean"]].copy()
    removed["reason"] = listed.reason.to_numpy()
    breakdown = removed.groupby(["split", "role", "reason"], observed=True)
    return [{"split": split, "role": role, "reason": reason, "rows": len(rows),
             "previously_source_clean": int(rows.audit_source_clean.sum()),
             "grid_ids": sorted(rows.station_id.unique())}
            for (split, role, reason), rows in breakdown]


def sepang_episode(old, new, oi, ni, fields):
    local = old.station_id.eq("Sepang") & old.timestamp.between(
        EPISODE_START - pd.Timedelta(hours=3), EPISODE_START + pd.Timedelta(hours=28))
    use = local.to_numpy()[oi]
    i, j = oi[use], ni[use]
    checks = ["o3", "pm2_5", "qc_interpolated", "is_missing", "audit_source_clean",
              "audit_history_complete", "audit_target_within_split_3h", "audit_target_aligned_3h",
              "audit_target_safe_3h", ELIGIBLE]
    rows = {}
    episode_end = EPISODE_START + pd.Timedelta(hours=4)
    for name, frame, ix in (("old", old, i), ("new", new, j)):
        span = frame.iloc[ix]
        focus = span.loc[span.timestamp.between(EPISODE_START, episode_end), ["timestamp", *checks]]
        rows[name] = json.loads(focus.to_json(orient="records", date_format="iso"))
    return {"timezone": "Asia/Kuala_Lumpur", "local_rows_14_to_18": rows,
            "affected_window_deltas": compare_columns(
                old, new, i, j, list(dict.fromkeys(checks + fields + LABELS)))}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--candidate", required=True, type=Path)
    cli.add_argument("--upstream", required=True, type=Path)
    cli.add_argument("--output", required=True, type=Path)
    args = cli.parse_args()
    fingerprints = load_json(args.upstream / "docs/qc_fix/fingerprints.json")
    names = ["data/candidate/haze_panel_qcfix.parquet", "docs/qc_fix/drops.parquet",
             "docs/qc_fix/rows_dropped_271.csv", "docs/qc_fix/evidence.json",
             "docs/qc_fix/gate0_report_candidate.json"]
    verified = {name: require_fingerprint(args.upstream, name, fingerprints) for name in names}
    before_input, after_input = ROOT / "data/haze_panel_clean.parquet", args.upstream / names[0]
    before_hash, after_hash = digest(before_input), verified[names[0]]["sha256"]
    assert fingerprints["data/haze_panel_clean.parquet"]["sha256"] == before_hash
    feat_audits = [load_json(root / "models/metrics_stage2_features.json") for root in (ROOT, args.candidate)]
    fields = feat_audits[0]["model_feature_columns"]
    assert fields == feat_audits[1]["model_feature_columns"]
    frames = []
    for root, source, source_hash, audit in zip((ROOT, args.candidate), (before_input, after_input),
                                               (before_hash, after_hash), feat_audits):
        path = root / "data/feat_spatiotemporal.parquet"
        assert audit["input_sha256"] == source_hash
        assert feature_artifact(path) == {k: audit[k] for k in ("output_sha256", "output_bytes", "output_parts")}
        panel = pd.read_parquet(source)
        frame = pd.read_parquet(path)
        assert frame.shape == (len(panel), len(panel.columns) + audit["new_column_count"])
        pd.testing.assert_frame_equal(frame[list(panel)], panel, check_exact=True)
        frames.append(frame)
    old, new = frames
    assert old.dtypes.equals(new.dtypes)
    old_keys, new_keys = key_index(old), key_index(new)
    old_pos = old_keys.get_indexer(new_keys)
    new_pos = new_keys.get_indexer(old_keys)
    ni = np.flatnonzero(old_pos >= 0)
    oi = old_pos[ni]
    assert np.array_equal(old.split.to_numpy()[oi], new.split.to_numpy()[ni])
    assert np.array_equal(old.role.to_numpy()[oi], new.role.to_numpy()[ni])
    listed = pd.read_csv(args.upstream / names[2], parse_dates=["timestamp"])
    dropped = pd.read_parquet(args.upstream / names[1])
    reasons = listed.set_index(KEY).reason.to_dict()
    assert set(key_index(listed)) == set(old_keys[new_pos < 0])
    dropped_reasons = dropped.set_index(KEY).reason
    assert dropped_reasons.index.is_unique
    assert all(dropped_reasons.loc[key] == reason for key, reason in reasons.items())
    candidate_gate = load_json(args.upstream / names[4])
    assert feat_audits[1]["gate0_disclosed"] == candidate_gate
    eda = [load_json(root / "models/metrics_stage2_eda.json") for root in (ROOT, args.candidate)]
    assert [a["input_sha256"] for a in eda] == [before_hash, after_hash]
    assert eda[0]["source_sha256"] == eda[1]["source_sha256"]
    old_figures, new_figures = [a["figure_sha256"] for a in eda]
    assert old_figures.keys() == new_figures.keys()
    assert all(digest(root / "figures" / name) == expected for root, figures in
               ((ROOT, old_figures), (args.candidate, new_figures)) for name, expected in figures.items())
    per_split = split_evidence(old, new, oi, ni, new_pos < 0, old_pos < 0, fields, reasons)
    rebuild = any(item["eligible_receptors"]["old_keys_sha256"] != item["eligible_receptors"]["new_keys_sha256"]
                  or any(delta["changed_rows"] for group in ("labels", "model_fields")
                         for delta in item["eligible_receptors"][group].values())
                  for item in per_split.values())
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_fingerprints": verified,
        "published": {"shape": list(old.shape), "input_sha256": before_hash,
                      "output_sha256": feat_audits[0]["output_sha256"]},
        "candidate": {"shape": list(new.shape), "input_sha256": after_hash,
                      "stage2_generator_commit": feat_audits[1]["git_commit"],
                      "output_sha256": feat_audits[1]["output_sha256"]},
        "key_definition": ["station_id", "timestamp (Asia/Kuala_Lumpur local hour)"],
        "comparison": "Exact keys and values. Both missing count as equal. Maximum delta compares finite pairs only.",
        "upstream_removed_key_reasons": {str(k): int(v) for k, v in listed.reason.value_counts().items()},
        "upstream_removals_and_previous_source_quality": removed_source_checks(old, old_keys, listed),
        "by_split": per_split,
        "sepang_o3_episode": sepang_episode(old, new, oi, ni, fields),
        "eda_figures": {name: {"old_sha256": old_figures[name], "new_sha256": new_figures[name],
                               "changed": old_figures[name] != new_figures[name]} for name in old_figures},
        "rebuild_required_if_upstream_candidate_is_adopted": rebuild,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8", newline="\n")
    print(f"Compared {len(oi):,} common grid-hours. Candidate rebuild required: {rebuild}")


if __name__ == "__main__":
    main()
