def dense_path_to_target_segments(
    dense_alignment: torch.Tensor,
    frame_scores: torch.Tensor,
    target_ids: list[int],
    blank_id: int,
    duration: float,
    blank_policy: str = "drop",
) -> list[dict]:
    """
    Recover one segment per target token from the dense CTC path.

    This intentionally does not use merge_tokens, because adjacent repeated
    phones such as "sil sil" must remain separate for timestamp comparison.
    """
    labels = dense_alignment.detach().cpu().tolist()
    probs = frame_scores.detach().cpu().tolist()
    chunks: list[tuple[int, int, int, float]] = []
    active_token = None
    active_start = None

    for frame_idx, token_id in enumerate(labels):
        if token_id == blank_id:
            if active_token is not None:
                mean_score = float(np.mean(probs[active_start:frame_idx]))
                chunks.append((active_token, active_start, frame_idx, mean_score))
                active_token = None
                active_start = None
            continue

        if active_token is None:
            active_token = token_id
            active_start = frame_idx
        elif token_id != active_token:
            mean_score = float(np.mean(probs[active_start:frame_idx]))
            chunks.append((active_token, active_start, frame_idx, mean_score))
            active_token = token_id
            active_start = frame_idx

    if active_token is not None:
        mean_score = float(np.mean(probs[active_start:len(labels)]))
        chunks.append((active_token, active_start, len(labels), mean_score))

    segments = []
    cursor = 0
    for target_index, target_id in enumerate(target_ids):
        while cursor < len(chunks) and chunks[cursor][0] != target_id:
            cursor += 1
        if cursor >= len(chunks):
            raise ValueError(
                f"could not recover target segment {target_index}/{len(target_ids)} "
                f"from {len(chunks)} nonblank chunks"
            )

        token_id, start_frame, end_frame, mean_score = chunks[cursor]
        segments.append(
            {
                "token_id": int(token_id),
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "mean_score": mean_score,
            }
        )
        cursor += 1

    apply_blank_policy_to_segments(segments, len(labels), blank_policy)

    seconds_per_frame = duration / float(len(labels))
    for segment in segments:
        segment["start"] = segment["start_frame"] * seconds_per_frame
        segment["end"] = segment["end_frame"] * seconds_per_frame

    return segments

def apply_blank_policy_to_segments(segments: list[dict], num_frames: int, blank_policy: str) -> None:
    """Attach blank gaps to neighboring phone segments in-place."""
    if blank_policy == "drop" or not segments:
        return

    if blank_policy not in {"previous", "next", "split"}:
        raise ValueError(f"Unknown blank_policy: {blank_policy}")

    original_starts = [segment["start_frame"] for segment in segments]
    original_ends = [segment["end_frame"] for segment in segments]

    if blank_policy == "previous":
        segments[0]["start_frame"] = 0
        for idx in range(len(segments) - 1):
            segments[idx]["end_frame"] = original_starts[idx + 1]
        segments[-1]["end_frame"] = num_frames
        return

    if blank_policy == "next":
        segments[0]["start_frame"] = 0
        for idx in range(1, len(segments)):
            segments[idx]["start_frame"] = original_ends[idx - 1]
        segments[-1]["end_frame"] = num_frames
        return

    # split: put the boundary in the middle of each blank gap.
    segments[0]["start_frame"] = 0
    for idx in range(len(segments) - 1):
        midpoint = int(round((original_ends[idx] + original_starts[idx + 1]) / 2.0))
        segments[idx]["end_frame"] = midpoint
        segments[idx + 1]["start_frame"] = midpoint
    segments[-1]["end_frame"] = num_frames

def native_k2_forced_align_ids(ctc_p: torch.Tensor, target_ids: list[int], blank_id: int):
    """Run native K2 CTC FA and return a dense frame-level label path."""
    return native_k2_forced_align_ids_batch(
        ctc_p=ctc_p,
        target_id_seqs=[list(target_ids)],
        input_lengths=torch.tensor([ctc_p.shape[-2]], dtype=torch.int32),
        blank_id=blank_id,
    )[0]

def native_k2_forced_align_ids_batch(
    ctc_p: torch.Tensor,
    target_id_seqs: list[list[int]],
    input_lengths: torch.Tensor,
    blank_id: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Run native K2 CTC FA for a batch and return dense paths per sample."""
    try:
        import k2
    except Exception as exc:
        raise RuntimeError(
            "Native K2 alignment backend requested, but k2 could not be imported. "
            "Please run this in the environment where k2 is installed."
        ) from exc

    target_id_seqs = [list(target_ids) for target_ids in target_id_seqs]
    if not target_id_seqs:
        return []
    if ctc_p.shape[0] != len(target_id_seqs):
        raise ValueError(
            f"ctc batch size {ctc_p.shape[0]} does not match target batch size {len(target_id_seqs)}"
        )
    if input_lengths.numel() != len(target_id_seqs):
        raise ValueError(
            f"input_lengths size {input_lengths.numel()} does not match target batch size {len(target_id_seqs)}"
        )
    if any(len(target_ids) == 0 for target_ids in target_id_seqs):
        raise ValueError("native K2 batch alignment received an empty target sequence")

    # The available k2 build may be CPU-only. Keep model inference on the
    # requested device, but run native k2 graph/intersection on CPU.
    k2_ctc_p = ctc_p.detach().to("cpu").float().contiguous()
    k2_device = torch.device("cpu")
    input_lengths = input_lengths.detach().to("cpu").long().clamp(min=1, max=k2_ctc_p.shape[-2])

    graph = k2.ctc_graph(target_id_seqs, modified=False, device=k2_device)
    graph = k2.arc_sort(graph)
    supervision_segments = torch.tensor(
        [[idx, 0, int(input_lengths[idx].item())] for idx in range(len(target_id_seqs))],
        dtype=torch.int32,
        device=k2_device,
    )
    dense_fsa_vec = k2.DenseFsaVec(k2_ctc_p, supervision_segments)

    beam_configs = [
        (20.0, 8.0, 30, 10000),
        (40.0, 20.0, 100, 30000),
        (80.0, 40.0, 200, 100000),
        (160.0, 80.0, 500, 300000),
        (320.0, 160.0, 1000, 1000000),
    ]
    last_empty = None
    for search_beam, output_beam, min_active_states, max_active_states in beam_configs:
        lattice = k2.intersect_dense_pruned(
            graph,
            dense_fsa_vec,
            search_beam=search_beam,
            output_beam=output_beam,
            min_active_states=min_active_states,
            max_active_states=max_active_states,
            frame_idx_name="frame_idx",
        )
        best_path = k2.shortest_path(lattice, use_double_scores=True)

        labels = best_path.labels.detach().long()
        frame_idx = best_path.frame_idx.detach().long()
        scores = best_path.scores.detach().float()

        outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
        empty_indices: list[int] = []
        for sample_index, target_ids in enumerate(target_id_seqs):
            sample_path = k2.index_fsa(
                best_path,
                torch.tensor([sample_index], dtype=torch.int32, device=k2_device),
            )
            sample_labels = sample_path.labels.detach().long()
            sample_frame_idx = sample_path.frame_idx.detach().long()
            sample_scores = sample_path.scores.detach().float()
            num_frames = int(input_lengths[sample_index].item())
            valid = (
                (sample_labels != -1)
                & (sample_frame_idx >= 0)
                & (sample_frame_idx < num_frames)
            )
            if not bool(valid.any()):
                empty_indices.append(sample_index)
                outputs.append(
                    (
                        torch.full((num_frames,), int(blank_id), dtype=torch.long),
                        torch.ones((num_frames,), dtype=torch.float32),
                    )
                )
                continue
            dense_labels = torch.full((num_frames,), int(blank_id), dtype=torch.long)
            dense_scores = torch.ones((num_frames,), dtype=torch.float32)
            dense_labels[sample_frame_idx[valid]] = sample_labels[valid]
            dense_scores[sample_frame_idx[valid]] = sample_scores[valid].exp()
            outputs.append((dense_labels, dense_scores))

        if empty_indices:
            last_empty = (
                f"empty best-path alignment for batch_indices={empty_indices[:8]} "
                f"with search_beam={search_beam}, output_beam={output_beam}, "
                f"min_active_states={min_active_states}, max_active_states={max_active_states}"
            )
            continue
        return outputs

    raise ValueError(
        "native K2 returned an empty best-path alignment; "
        f"batch={len(target_id_seqs)}, max_frames={int(k2_ctc_p.shape[-2])}, {last_empty}"
    )
