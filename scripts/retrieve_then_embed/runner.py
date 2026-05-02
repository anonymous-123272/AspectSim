from __future__ import print_function

import ast
import os
import time

import pandas as pd
from tqdm import tqdm

from retrieve_then_embed.generation import build_chat_messages, generate_with_vllm_batch
from retrieve_then_embed.parsing import extract_narrative_for_storage


def flatten_text_fragments(value):
    """Recursively flatten list-like/stringified-list values into text fragments."""
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(flatten_text_fragments(item))
        return out
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if (text.startswith("[") and text.endswith("]")) or (
            text.startswith("(") and text.endswith(")")
        ):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, (list, tuple)):
                    return flatten_text_fragments(parsed)
            except Exception:
                pass
        return [text]
    return [str(value)]


def merge_to_string(value):
    """Merge one side (pair1 or pair2) into a single readable string."""
    return ", ".join(flatten_text_fragments(value))


def parse_clean_pair_output(output):
    """
    Return (clean_value, error_text).
    - clean_value is always a list for pair1/pair2 cells.
    - if parsing fails or output is unusable, returns [] and an error message.
    """
    if output is None or (isinstance(output, str) and not output.strip()):
        return [], "empty model response"

    try:
        parsed = extract_narrative_for_storage(output)
    except Exception as error:
        return [], "parse exception: %s" % error

    if isinstance(parsed, list):
        return parsed, ""
    if isinstance(parsed, tuple):
        return list(parsed), ""
    if isinstance(parsed, str):
        text = parsed.strip()
        if not text:
            return [], ""
        # Accept stringified list/tuple only; reject raw text.
        try:
            literal = ast.literal_eval(text)
            if isinstance(literal, (list, tuple)):
                return list(literal), ""
        except Exception:
            pass
        return [], "parsed output is not a list"

    return [], "unsupported parsed type: %s" % type(parsed).__name__


def response_generation(
    aspect_data,
    column_name,
    pair_name,
    method,
    llm_info,
    prompts_yaml,
    output_file,
    batch_size=1,
):
    print("--------------------------------------")
    print("Method: %s" % method)
    print("LLM: %s" % llm_info["llm_name"])
    print("Article column: %s" % column_name)
    print("Pair: %s" % pair_name)
    print("Samples: %d" % len(aspect_data))
    print("Batch size: %d" % max(1, int(batch_size)))
    print("--------------------------------------")
    print("Output: %s" % output_file)

    sentence_list = []
    error_list = []
    total_rows = len(aspect_data)
    chunk_size = max(1, int(batch_size))
    llm_name = llm_info["llm_name"]
    error_col = pair_name + "_error"
    total_errors = 0

    def checkpoint(last_done_index):
        done = last_done_index + 1
        base = aspect_data.iloc[:done].copy()
        out_df = base[["doc_n", "aspect"]].copy()
        if "domain" in base.columns:
            out_df["domain"] = base["domain"]
        else:
            out_df["domain"] = [""] * done
        if "aspect_type" in base.columns:
            out_df["aspect_type"] = base["aspect_type"]
        else:
            out_df["aspect_type"] = [""] * done
        if "pair" in base.columns:
            out_df["pair"] = base["pair"]
        elif "gt_pair" in base.columns:
            out_df["pair"] = base["gt_pair"]
        else:
            out_df["pair"] = [""] * done
        if "label" in base.columns:
            out_df["label"] = base["label"]
        else:
            out_df["label"] = [""] * done

        if os.path.exists(output_file):
            existing_data = pd.read_excel(output_file, sheet_name=llm_name)
            if "pair1" in existing_data.columns:
                old_pair1 = existing_data["pair1"].tolist()
                out_df["pair1"] = old_pair1[:done] + [""] * max(0, done - len(old_pair1))
            if "pair2" in existing_data.columns:
                old_pair2 = existing_data["pair2"].tolist()
                out_df["pair2"] = old_pair2[:done] + [""] * max(0, done - len(old_pair2))
            if "pair1_error" in existing_data.columns:
                old_pair1_err = existing_data["pair1_error"].tolist()
                out_df["pair1_error"] = old_pair1_err[:done] + [""] * max(
                    0, done - len(old_pair1_err)
                )
            if "pair2_error" in existing_data.columns:
                old_pair2_err = existing_data["pair2_error"].tolist()
                out_df["pair2_error"] = old_pair2_err[:done] + [""] * max(
                    0, done - len(old_pair2_err)
                )

        out_df[pair_name] = sentence_list
        out_df[error_col] = error_list

        if "pair1" in out_df.columns and "pair2" in out_df.columns:
            out_df["pairs"] = [
                [merge_to_string(pair1_val), merge_to_string(pair2_val)]
                for pair1_val, pair2_val in zip(
                    out_df["pair1"].tolist(), out_df["pair2"].tolist()
                )
            ]

        with pd.ExcelWriter(output_file, engine="openpyxl", mode="w") as writer:
            out_df.to_excel(writer, index=False, sheet_name=llm_name)
        time.sleep(5)

    progress = tqdm(
        total=total_rows,
        desc="%s-%s" % (method, pair_name),
        unit="row",
        dynamic_ncols=True,
    )
    for start in range(0, total_rows, chunk_size):
        end = min(start + chunk_size, total_rows)
        messages_batch = []

        for i in range(start, end):
            row = aspect_data.iloc[i]
            aspect = row["aspect"]
            document = row[column_name]
            if pd.isna(document):
                document = ""
            else:
                document = str(document)
            messages_batch.append(
                build_chat_messages(method, document, aspect, llm_name, prompts_yaml)
            )

        outputs = generate_with_vllm_batch(
            llm_info["tokenizer"],
            llm_info["llm"],
            llm_name,
            messages_batch,
            llm_info["sampling_params"],
        )
        expected = end - start
        if len(outputs) != expected:
            print(
                "WARNING: batch returned %d outputs for %d prompts; padding."
                % (len(outputs), expected)
            )
            while len(outputs) < expected:
                outputs.append("")
            outputs = outputs[:expected]

        for j, output in enumerate(outputs):
            i = start + j
            sentence, error_text = parse_clean_pair_output(output)
            sentence_list.append(sentence)
            error_list.append(error_text)
            if error_text:
                total_errors += 1

            done = len(sentence_list)
            if done % 30 == 0 or done == total_rows:
                checkpoint(done - 1)
            progress.update(1)
            if total_errors:
                progress.set_postfix(errors=total_errors)

        time.sleep(5)
    progress.close()

    print(
        "Completed with %d parsing/model-output errors for %s."
        % (total_errors, pair_name)
    )

    return output_file

