"""将 CSV 数据清洗并转换为 RAG 知识库 Markdown 文件"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASE_DIR = Path(__file__).resolve().parent.parent
PNEUMONIA_CSV = BASE_DIR / "肺炎(1).csv"
DRUG_CSV = BASE_DIR / "药品库(1).csv"
OUTPUT_DIR = BASE_DIR / "data" / "medical_knowledge" / "generated"

# 过滤广告/噪声标题的关键词
NOISE_TITLE_KEYWORDS = (
    "排名", "排行榜", "哪家好", "前十", "公布", "排行名单",
    "口碑", "正规实力", "最有名的医院",
)

QA_PER_FILE = 80
DRUGS_PER_FILE = 60


def _clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _is_noise_row(row: pd.Series) -> bool:
    title = _clean_text(row.get("title", ""))
    subject = _clean_text(row.get("subject", ""))

    if any(kw in title for kw in NOISE_TITLE_KEYWORDS):
        return True
    if subject.startswith("医院>") and "肺炎" not in subject and "呼吸" not in subject:
        return True
    return False


def process_pneumonia_qa(max_entries: int = 3000) -> list[dict]:
    """清洗肺炎问答 CSV，生成问答对"""
    if not PNEUMONIA_CSV.exists():
        print(f"跳过：未找到 {PNEUMONIA_CSV.name}")
        return []

    print(f"正在处理 {PNEUMONIA_CSV.name} ...")
    df = pd.read_csv(PNEUMONIA_CSV, low_memory=False)
    df = df.drop_duplicates(subset=["post_id", "user_type", "content"], keep="first")
    df = df[~df.apply(_is_noise_row, axis=1)]

    questions = df[(df["user_type"] == 1) & df["title"].notna()].copy()
    answers = df[(df["user_type"] == 2) & df["content"].notna()].copy()
    answers = answers[answers["content"].str.len() >= 40]

    qa_pairs = []
    seen_posts = set()

    for post_id, q_group in questions.groupby("post_id"):
        q_row = q_group.iloc[0]
        a_group = answers[answers["post_id"] == post_id]
        if a_group.empty:
            continue

        a_row = a_group.iloc[0]
        question = _clean_text(q_row.get("title")) or _clean_text(q_row.get("content"))
        answer = _clean_text(a_row.get("content"))
        if len(question) < 4 or len(answer) < 40:
            continue

        subject = _clean_text(q_row.get("subject", "医患问答"))
        tags = _clean_text(q_row.get("tags", ""))

        qa_pairs.append({
            "post_id": post_id,
            "subject": subject,
            "tags": tags,
            "question": question,
            "answer": answer,
        })
        seen_posts.add(post_id)

    # 补充：无配对但有高质量医生回答的记录
    for _, row in answers.iterrows():
        post_id = row["post_id"]
        if post_id in seen_posts:
            continue
        content = _clean_text(row.get("content"))
        title = _clean_text(row.get("title", ""))
        if len(content) < 80:
            continue
        qa_pairs.append({
            "post_id": post_id,
            "subject": _clean_text(row.get("subject", "医患问答")),
            "tags": _clean_text(row.get("tags", "")),
            "question": title or "医学咨询",
            "answer": content,
        })

    # 优先保留呼吸/肺炎相关，再按回答长度排序
    def relevance_score(item: dict) -> tuple:
        text = f"{item['subject']} {item['tags']} {item['question']}"
        respiratory = any(kw in text for kw in ("肺炎", "呼吸", "咳嗽", "发热", "肺"))
        return (respiratory, len(item["answer"]))

    qa_pairs.sort(key=relevance_score, reverse=True)
    qa_pairs = qa_pairs[:max_entries]
    print(f"  生成 {len(qa_pairs)} 条问答知识")
    return qa_pairs


def process_drug_library(max_entries: int = 5000) -> list[dict]:
    """清洗药品库 CSV"""
    if not DRUG_CSV.exists():
        print(f"跳过：未找到 {DRUG_CSV.name}")
        return []

    print(f"正在处理 {DRUG_CSV.name} ...")
    df = pd.read_csv(DRUG_CSV, low_memory=False)
    df = df.drop_duplicates(subset=["medicine_name"], keep="first")

    drugs = []
    for _, row in df.iterrows():
        name = _clean_text(row.get("medicine_name"))
        if not name:
            continue

        indication = _clean_text(row.get("indication"))
        dosage = _clean_text(row.get("dosage"))
        contraindication = _clean_text(row.get("contraindication"))
        adr = _clean_text(row.get("adr"))
        interaction = _clean_text(row.get("interaction"))
        related = _clean_text(row.get("related_diseases"))
        comment = _clean_text(row.get("comment"))
        major = _clean_text(row.get("major_functions"))

        if not any([indication, dosage, comment, major]):
            continue

        drugs.append({
            "name": name,
            "medical_system": _clean_text(row.get("medical_system", "")),
            "related_diseases": related,
            "indication": indication or major,
            "dosage": dosage,
            "contraindication": contraindication,
            "adr": adr,
            "interaction": interaction,
            "comment": comment,
            "category": _clean_text(row.get("category", "")),
        })

    drugs.sort(key=lambda d: len(d["indication"]), reverse=True)
    drugs = drugs[:max_entries]
    print(f"  生成 {len(drugs)} 条药品知识")
    return drugs


def _format_qa_entry(item: dict, index: int) -> str:
    tags_line = f"\n**标签**: {item['tags']}" if item["tags"] else ""
    return (
        f"### 问答{index}: {item['question'][:60]}\n\n"
        f"**分类**: {item['subject']}{tags_line}\n\n"
        f"**患者问题**: {item['question']}\n\n"
        f"**医生建议**: {item['answer']}\n"
    )


def _format_drug_entry(drug: dict, index: int) -> str:
    parts = [f"### 药品{index}: {drug['name']}\n"]
    if drug["medical_system"]:
        parts.append(f"**类型**: {drug['medical_system']}")
    if drug["related_diseases"]:
        parts.append(f"**相关疾病**: {drug['related_diseases']}")
    if drug["indication"]:
        parts.append(f"**适应症**: {drug['indication']}")
    if drug["dosage"]:
        parts.append(f"**用法用量**: {drug['dosage']}")
    if drug["contraindication"]:
        parts.append(f"**禁忌**: {drug['contraindication']}")
    if drug["adr"]:
        parts.append(f"**不良反应**: {drug['adr']}")
    if drug["interaction"]:
        parts.append(f"**药物相互作用**: {drug['interaction']}")
    if drug["comment"]:
        parts.append(f"**说明**: {drug['comment']}")
    return "\n\n".join(parts) + "\n"


def _write_batches(entries: list, output_prefix: str, formatter, per_file: int, header: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # 清除旧的生成文件
    for old in OUTPUT_DIR.glob(f"{output_prefix}_*.md"):
        old.unlink()

    file_count = 0
    for i in range(0, len(entries), per_file):
        batch = entries[i : i + per_file]
        file_count += 1
        content = f"# {header}\n\n" + "".join(
            formatter(item, j + 1) + "\n" for j, item in enumerate(batch)
        )
        out_path = OUTPUT_DIR / f"{output_prefix}_{file_count:03d}.md"
        out_path.write_text(content, encoding="utf-8")

    return file_count


def main():
    parser = argparse.ArgumentParser(description="预处理 CSV 并生成 RAG 知识库")
    parser.add_argument("--max-qa", type=int, default=3000, help="最大问答条数")
    parser.add_argument("--max-drugs", type=int, default=5000, help="最大药品条数")
    args = parser.parse_args()

    qa_pairs = process_pneumonia_qa(max_entries=args.max_qa)
    drugs = process_drug_library(max_entries=args.max_drugs)

    qa_files = _write_batches(
        qa_pairs, "pneumonia_qa", _format_qa_entry, QA_PER_FILE,
        "肺炎及呼吸类医患问答知识库",
    )
    drug_files = _write_batches(
        drugs, "drug_library", _format_drug_entry, DRUGS_PER_FILE,
        "药品参考知识库",
    )

    print(f"\n完成！共生成 {qa_files + drug_files} 个 Markdown 文件")
    print(f"  问答: {len(qa_pairs)} 条 → {qa_files} 个文件")
    print(f"  药品: {len(drugs)} 条 → {drug_files} 个文件")
    print(f"输出目录: {OUTPUT_DIR}")
    print("\n下一步: python scripts/build_knowledge_base.py --force")


if __name__ == "__main__":
    main()
