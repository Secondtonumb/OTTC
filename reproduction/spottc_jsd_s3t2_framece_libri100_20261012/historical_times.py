def _token_times_from_training_alignment(target_tokens, phonemes, duration, context_phone_mode):
    """Map context-phone tokens to seconds using the manifest phoneme spans."""
    duration = max(float(duration), 1e-06)
    cleaned_phonemes = []
    for item in phonemes or []:
        phone = str(item.get('phoneme', '')).strip()
        if not phone:
            continue
        start = max(0.0, min(duration, float(item.get('start', 0.0))))
        end = max(start, min(duration, float(item.get('end', start))))
        cleaned_phonemes.append({'phoneme': phone, 'start': start, 'end': end})
    starts = []
    ends = []
    phone_idx = 0
    last_end = 0.0
    silence_labels = {'sil', 'sp', 'spn', '<blank>', '*'}
    for token in target_tokens:
        center = center_phone(token, context_phone_mode)
        if phone_idx < len(cleaned_phonemes) and center == cleaned_phonemes[phone_idx]['phoneme']:
            phone = cleaned_phonemes[phone_idx]
            starts.append(phone['start'])
            ends.append(phone['end'])
            last_end = phone['end']
            phone_idx += 1
            continue
        match_idx = None
        for candidate_idx in range(phone_idx + 1, min(len(cleaned_phonemes), phone_idx + 4)):
            if center == cleaned_phonemes[candidate_idx]['phoneme']:
                match_idx = candidate_idx
                break
        if match_idx is not None:
            phone = cleaned_phonemes[match_idx]
            starts.append(phone['start'])
            ends.append(phone['end'])
            last_end = phone['end']
            phone_idx = match_idx + 1
            continue
        if center in silence_labels:
            next_start = cleaned_phonemes[phone_idx]['start'] if phone_idx < len(cleaned_phonemes) else duration
            starts.append(last_end)
            ends.append(max(last_end, next_start))
        else:
            starts.append(last_end)
            ends.append(last_end)
    return (starts, ends)
