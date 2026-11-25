import csv
import html
import json
import os

import torch
from sklearn.metrics import roc_auc_score


def detect_device():
    if torch.cuda.is_available():
        return "cuda"
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def compute_auc_score(logits, labels):
    batch_size = logits.shape[0]
    auc = roc_auc_score(labels.numpy(), logits.numpy(), average="weighted")
    return auc * batch_size


def compute_acc(scores, labels):
    predictions = torch.argmax(scores, dim=-1)
    gold_labels = labels.squeeze(-1).long()
    return int((predictions == gold_labels).sum().item())


def get_label_token_ids(tokenizer, label_0_word, label_1_word):
    clean_label_0 = label_0_word.strip()
    clean_label_1 = label_1_word.strip()
    label_0_tokens = tokenizer.encode(clean_label_0, add_special_tokens=False)
    label_1_tokens = tokenizer.encode(clean_label_1, add_special_tokens=False)
    if not label_0_tokens or not label_1_tokens:
        raise ValueError(
            f"Unable to tokenize label words: label_0={label_0_word!r}, label_1={label_1_word!r}"
        )
    if len(label_0_tokens) > 1 or len(label_1_tokens) > 1:
        print(
            "Warning: label words map to multiple tokens. "
            "Only the first token is used for evaluation."
        )
    label_0_token_id = label_0_tokens[0]
    label_1_token_id = label_1_tokens[0]
    print(
        f"Token IDs - label_0 {clean_label_0}: {label_0_token_id}, "
        f"label_1 {clean_label_1}: {label_1_token_id}"
    )
    return label_0_token_id, label_1_token_id


def evaluate_binary_generation(
    model,
    tokenizer,
    data_point,
    generation_config,
    generate_prompt_fn,
    label_0_token_id,
    label_1_token_id,
    max_new_tokens=128,
    device=None,
):
    prompt = generate_prompt_fn(data_point)
    inputs = tokenizer(prompt, return_tensors="pt")
    eval_device = device or detect_device()
    input_ids = inputs["input_ids"].to(eval_device)
    attention_mask = inputs["attention_mask"].to(eval_device)

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )

    sequence = generation_output.sequences[0]
    label_1_score = generation_output["scores"][0][0, label_1_token_id].item()
    label_0_score = generation_output["scores"][0][0, label_0_token_id].item()
    if label_1_score < -1e10:
        label_1_score = -1e10
    if label_0_score < -1e10:
        label_0_score = -1e10

    decoded_output = tokenizer.decode(sequence)
    if "### Response:" in decoded_output:
        generated_text = decoded_output.split("### Response:", 1)[1].strip().split("\n")[0]
    else:
        generated_text = decoded_output.strip().split("\n")[0]
    return generated_text, [label_0_score, label_1_score]


def get_error_type(gold_label, pred_label):
    if pred_label == -1:
        return "INVALID"
    if gold_label == 1 and pred_label == 1:
        return "TP"
    if gold_label == 0 and pred_label == 0:
        return "TN"
    if gold_label == 0 and pred_label == 1:
        return "FP"
    return "FN"


def build_case_record(data_point, label_map, generated_text, raw_scores, pred_label, error_message=""):
    score_tensor = torch.tensor(raw_scores, dtype=torch.float32)
    positive_probability = float(torch.softmax(score_tensor, dim=-1)[1].item())
    gold_label = int(data_point["label"])
    record = {
        "case_id": data_point.get("case_id") or data_point.get("path") or data_point.get("img") or "",
        "image_name": data_point.get("path") or data_point.get("img") or "",
        "image_path": data_point.get("image_path", ""),
        "text": data_point.get("text", ""),
        "meme_description": data_point.get("meme_description", ""),
        "text_description": data_point.get("text_description", ""),
        "input_text": data_point.get("input", ""),
        "instruction": data_point.get("instruction", ""),
        "gold_label": gold_label,
        "gold_label_text": label_map[gold_label],
        "pred_label": pred_label,
        "pred_label_text": label_map.get(pred_label, ""),
        "correct": gold_label == pred_label,
        "error_type": get_error_type(gold_label, pred_label),
        "positive_probability": positive_probability,
        "raw_label_0_score": float(raw_scores[0]),
        "raw_label_1_score": float(raw_scores[1]),
        "generated_text": generated_text,
        "error_message": error_message,
    }
    excluded = {
        "case_id",
        "img",
        "path",
        "image_path",
        "text",
        "meme_description",
        "text_description",
        "input",
        "instruction",
        "label",
        "output",
    }
    for key, value in data_point.items():
        if key in excluded:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            record[key] = value
        else:
            record[key] = json.dumps(value, ensure_ascii=False)
    return record


def build_invalid_case_record(data_point, label_map, error_message):
    return build_case_record(
        data_point=data_point,
        label_map=label_map,
        generated_text="",
        raw_scores=[0.0, 0.0],
        pred_label=-1,
        error_message=error_message,
    )


def run_binary_generation_case_study(
    model,
    tokenizer,
    data,
    generation_config,
    generate_prompt_fn,
    label_0_word,
    label_1_word,
    progress_every=200,
    max_new_tokens=128,
):
    outputs = collect_binary_generation_outputs(
        model=model,
        tokenizer=tokenizer,
        data=data,
        generation_config=generation_config,
        generate_prompt_fn=generate_prompt_fn,
        label_0_word=label_0_word,
        label_1_word=label_1_word,
        progress_every=progress_every,
        max_new_tokens=max_new_tokens,
    )
    score_tensor = outputs["score_tensor"]
    label_tensor = outputs["label_tensor"]
    acc = compute_acc(score_tensor, label_tensor)
    positive_scores = torch.softmax(score_tensor, dim=-1)[:, 1].unsqueeze(-1)
    auc = compute_auc_score(positive_scores, label_tensor)
    total_count = max(len(data), 1)
    return (
        outputs["invalid"],
        auc * 100.0 / total_count,
        acc * 100.0 / total_count,
        outputs["records"],
    )


def collect_binary_generation_outputs(
    model,
    tokenizer,
    data,
    generation_config,
    generate_prompt_fn,
    label_0_word,
    label_1_word,
    progress_every=200,
    max_new_tokens=128,
):
    label_0_token_id, label_1_token_id = get_label_token_ids(
        tokenizer, label_0_word=label_0_word, label_1_word=label_1_word
    )
    label_map = {0: label_0_word, 1: label_1_word}
    eval_device = detect_device()

    invalid = []
    raw_scores = []
    labels = []
    records = []

    for index, row in enumerate(data):
        labels.append([row["label"]])
        try:
            generated_text, sample_scores = evaluate_binary_generation(
                model=model,
                tokenizer=tokenizer,
                data_point=row,
                generation_config=generation_config,
                generate_prompt_fn=generate_prompt_fn,
                label_0_token_id=label_0_token_id,
                label_1_token_id=label_1_token_id,
                max_new_tokens=max_new_tokens,
                device=eval_device,
            )
            pred_label = int(torch.argmax(torch.tensor(sample_scores)).item())
            raw_scores.append(sample_scores)
            records.append(
                build_case_record(
                    data_point=row,
                    label_map=label_map,
                    generated_text=generated_text,
                    raw_scores=sample_scores,
                    pred_label=pred_label,
                )
            )
        except Exception as exc:
            case_id = row.get("path") or row.get("img") or str(index)
            print(f"Invalid: {case_id}")
            invalid.append(case_id)
            raw_scores.append([0.0, 0.0])
            records.append(
                build_invalid_case_record(
                    data_point=row,
                    label_map=label_map,
                    error_message=str(exc),
                )
            )

        if index % progress_every == 0:
            print("\tAlready finished...", index)

    score_tensor = torch.tensor(raw_scores, dtype=torch.float32)
    label_tensor = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
    return {
        "invalid": invalid,
        "score_tensor": score_tensor,
        "label_tensor": label_tensor,
        "records": records,
    }


def summarize_case_records(records):
    summary = {
        "num_cases": len(records),
        "num_correct": 0,
        "num_wrong": 0,
        "num_invalid": 0,
        "accuracy": 0.0,
        "counts": {"TP": 0, "TN": 0, "FP": 0, "FN": 0, "INVALID": 0},
    }
    for record in records:
        if record["correct"]:
            summary["num_correct"] += 1
        else:
            summary["num_wrong"] += 1
        if record["error_type"] == "INVALID":
            summary["num_invalid"] += 1
        summary["counts"][record["error_type"]] = summary["counts"].get(record["error_type"], 0) + 1
    if records:
        summary["accuracy"] = round(summary["num_correct"] / len(records), 6)
    return summary


def _write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as file_obj:
        for record in records:
            file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_csv(records, path):
    preferred_order = [
        "case_id",
        "image_name",
        "image_path",
        "text",
        "meme_description",
        "text_description",
        "gold_label",
        "gold_label_text",
        "pred_label",
        "pred_label_text",
        "correct",
        "error_type",
        "positive_probability",
        "raw_label_0_score",
        "raw_label_1_score",
        "generated_text",
        "error_message",
        "input_text",
        "instruction",
    ]
    all_keys = []
    for key in preferred_order:
        if any(key in record for record in records):
            all_keys.append(key)
    for record in records:
        for key in record.keys():
            if key not in all_keys:
                all_keys.append(key)

    with open(path, "w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(records)


def _to_rel_uri(target_path, base_dir):
    rel_path = os.path.relpath(target_path, base_dir)
    return rel_path.replace("\\", "/")


def _render_case_card(record, output_dir):
    image_html = ""
    image_path = record.get("image_path", "")
    if image_path and os.path.exists(image_path):
        rel_uri = html.escape(_to_rel_uri(image_path, output_dir))
        alt_text = html.escape(record.get("image_name", "case image"))
        image_html = (
            '<div class="preview">'
            f'<img src="{rel_uri}" alt="{alt_text}">'
            "</div>"
        )

    def text_block(title, value):
        safe_value = html.escape(str(value or ""))
        return f'<div class="field"><div class="label">{title}</div><div class="value">{safe_value}</div></div>'

    score_text = f"{record.get('positive_probability', 0.0):.4f}"
    body = [
        '<article class="card">',
        image_html,
        '<div class="content">',
        f'<div class="tag {record["error_type"].lower()}">{html.escape(record["error_type"])}</div>',
        text_block("Case ID", record.get("case_id", "")),
        text_block("Gold", record.get("gold_label_text", "")),
        text_block("Prediction", record.get("pred_label_text", "")),
        text_block("Positive Probability", score_text),
        text_block("Text", record.get("text", "")),
    ]
    if record.get("meme_description"):
        body.append(text_block("Meme Description", record["meme_description"]))
    if record.get("text_description"):
        body.append(text_block("Text Description", record["text_description"]))
    if record.get("generated_text"):
        body.append(text_block("Generated Text", record["generated_text"]))
    if record.get("error_message"):
        body.append(text_block("Error", record["error_message"]))
    body.extend(["</div>", "</article>"])
    return "\n".join(body)


def _write_html_report(records, output_dir, path, title):
    summary = summarize_case_records(records)
    sections = []
    section_order = [
        ("FP", "False Positives"),
        ("FN", "False Negatives"),
        ("TP", "True Positives"),
        ("TN", "True Negatives"),
        ("INVALID", "Invalid Cases"),
    ]
    for error_type, heading in section_order:
        selected = [record for record in records if record["error_type"] == error_type]
        if not selected:
            continue
        cards = "\n".join(_render_case_card(record, output_dir) for record in selected)
        sections.append(
            f"<section><h2>{html.escape(heading)} ({len(selected)})</h2>{cards}</section>"
        )

    counts_text = " | ".join(f"{key}: {value}" for key, value in summary["counts"].items())
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{
      font-family: "Segoe UI", Arial, sans-serif;
      margin: 24px;
      background: #f6f7fb;
      color: #1f2937;
    }}
    h1, h2 {{
      margin-bottom: 12px;
    }}
    .summary {{
      background: #ffffff;
      border-radius: 12px;
      padding: 16px 20px;
      margin-bottom: 24px;
      box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08);
    }}
    section {{
      margin-bottom: 28px;
    }}
    .card {{
      display: grid;
      grid-template-columns: minmax(240px, 320px) 1fr;
      gap: 16px;
      background: #ffffff;
      border-radius: 14px;
      padding: 16px;
      margin-bottom: 16px;
      box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08);
    }}
    .preview {{
      display: flex;
      align-items: flex-start;
      justify-content: center;
      background: #eef2ff;
      border-radius: 10px;
      min-height: 180px;
      overflow: hidden;
    }}
    .preview img {{
      width: 100%;
      height: auto;
      display: block;
    }}
    .content {{
      display: grid;
      gap: 8px;
    }}
    .field {{
      background: #f8fafc;
      border-radius: 10px;
      padding: 10px 12px;
    }}
    .label {{
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      color: #64748b;
      margin-bottom: 4px;
      letter-spacing: 0.04em;
    }}
    .value {{
      white-space: pre-wrap;
      word-break: break-word;
      line-height: 1.45;
    }}
    .tag {{
      display: inline-block;
      width: fit-content;
      padding: 6px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
    .fp, .fn {{
      background: #fee2e2;
      color: #b91c1c;
    }}
    .tp, .tn {{
      background: #dcfce7;
      color: #166534;
    }}
    .invalid {{
      background: #e5e7eb;
      color: #374151;
    }}
    @media (max-width: 900px) {{
      .card {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <div class="summary">
    <div>Total Cases: {summary["num_cases"]}</div>
    <div>Correct: {summary["num_correct"]}</div>
    <div>Wrong: {summary["num_wrong"]}</div>
    <div>Invalid: {summary["num_invalid"]}</div>
    <div>Accuracy: {summary["accuracy"]:.4f}</div>
    <div>{html.escape(counts_text)}</div>
  </div>
  {"".join(sections)}
</body>
</html>
"""
    with open(path, "w", encoding="utf-8") as file_obj:
        file_obj.write(page)


def save_case_study_records(records, output_dir, stem, title):
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, f"{stem}.jsonl")
    csv_path = os.path.join(output_dir, f"{stem}.csv")
    summary_path = os.path.join(output_dir, f"{stem}_summary.json")
    html_path = os.path.join(output_dir, f"{stem}.html")

    _write_jsonl(records, jsonl_path)
    _write_csv(records, csv_path)
    with open(summary_path, "w", encoding="utf-8") as file_obj:
        json.dump(summarize_case_records(records), file_obj, ensure_ascii=False, indent=2)
    _write_html_report(records, output_dir, html_path, title)

    return {
        "jsonl": jsonl_path,
        "csv": csv_path,
        "summary": summary_path,
        "html": html_path,
    }
