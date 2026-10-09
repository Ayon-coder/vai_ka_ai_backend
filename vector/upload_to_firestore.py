"""
upload_to_firestore.py
----------------------
Uploads the updated vector JSON files to Firebase Firestore with schemas
matching the file names exactly:
1. ieee_student_branch.json  -> Collection: ieee_student_branch
2. teams_overview.json        -> Collection: teams_overview
3. team_members_details.json -> Collection: team_members_details
"""

import os
import sys
import json
import base64
import re
from typing import List, Dict, Any

# Ensure UTF-8 output on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

# Load environment
script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.abspath(os.path.join(script_dir, ".."))
dotenv_candidates = [
    os.path.join(script_dir, ".env"),
    os.path.join(backend_dir, ".env"),
]
for p in dotenv_candidates:
    if os.path.exists(p):
        load_dotenv(p)

import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1.vector import Vector
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def get_firestore_client() -> firestore.Client:
    """Initialize and return Firebase Firestore client."""
    if not firebase_admin._apps:
        creds_base64 = os.getenv("FIREBASE_CREDS_BASE64")
        if not creds_base64:
            raise ValueError("FIREBASE_CREDS_BASE64 not found in environment variables.")
        try:
            creds_json_str = base64.b64decode(creds_base64).decode("utf-8")
            creds_dict = json.loads(creds_json_str)
        except Exception as e:
            raise ValueError(f"Failed to decode FIREBASE_CREDS_BASE64: {e}")

        cred = credentials.Certificate(creds_dict)
        firebase_admin.initialize_app(cred)

    return firestore.client()


def get_embeddings_model() -> GoogleGenerativeAIEmbeddings:
    """Initialize LangChain GoogleGenerativeAIEmbeddings model with 768-dim output."""
    api_key = os.getenv("GOOGLE_API_KEY", "").split(",")[0].strip()
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")

    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
        output_dimensionality=768
    )


def slugify(text: str) -> str:
    """Convert a name into a Firestore-safe ID slug."""
    clean = re.sub(r"[^a-z0-9]+", "-", str(text or "").strip().lower()).strip("-")
    return clean or "item"


def upload_ieee_student_branch(db: firestore.Client, embeddings: GoogleGenerativeAIEmbeddings):
    """Upload ieee_student_branch.json to collection 'ieee_student_branch'."""
    file_path = os.path.join(script_dir, "ieee_student_branch.json")
    if not os.path.exists(file_path):
        print(f"[Error] File not found: {file_path}")
        return

    print("\n" + "="*60)
    print("1. Uploading ieee_student_branch.json -> Collection: 'ieee_student_branch'")
    print("="*60)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    col_ref = db.collection("ieee_student_branch")

    # Upload document overview
    doc_meta = data.get("document", {})
    if doc_meta:
        col_ref.document("report_overview").set(doc_meta, merge=True)
        print("   [+] Saved document metadata -> 'report_overview'")

    chunks = data.get("chunks", [])
    print(f"   [+] Generating embeddings for {len(chunks)} chunks...")

    embed_texts = []
    for c in chunks:
        title = c.get("title", "")
        sec = c.get("section_title", "")
        body = c.get("content", "")
        embed_texts.append(f"Title: {title}\nSection: {sec}\nContent: {body}")

    vectors = embeddings.embed_documents(embed_texts)
    print(f"   [+] Computed {len(vectors)} embeddings (768-dim).")

    batch = db.batch()
    for idx, (c, vec, text) in enumerate(zip(chunks, vectors, embed_texts), 1):
        doc_id = c.get("chunk_id") or f"chunk_{idx:02d}"
        doc_ref = col_ref.document(doc_id)
        payload = {
            "chunk_id": doc_id,
            "document_id": c.get("document_id", "ieee_student_branch_detailed_report_2026"),
            "title": c.get("title", ""),
            "section_number": c.get("section_number"),
            "section_title": c.get("section_title", ""),
            "subsection_titles": c.get("subsection_titles", []),
            "content": c.get("content", ""),
            "keywords": c.get("keywords", []),
            "metadata": c.get("metadata", {}),
            "embedding_text": text,
            "embedding": Vector(vec)
        }
        batch.set(doc_ref, payload)

    batch.commit()
    print(f"   [OK] Successfully uploaded {len(chunks)} chunks to collection 'ieee_student_branch'.")


def upload_teams_overview(db: firestore.Client, embeddings: GoogleGenerativeAIEmbeddings):
    """Upload teams_overview.json to collection 'teams_overview'."""
    file_path = os.path.join(script_dir, "teams_overview.json")
    if not os.path.exists(file_path):
        print(f"[Error] File not found: {file_path}")
        return

    print("\n" + "="*60)
    print("2. Uploading teams_overview.json -> Collection: 'teams_overview'")
    print("="*60)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    teams = data.get("team", [])
    print(f"   [+] Generating embeddings for {len(teams)} teams...")

    embed_texts = []
    for t in teams:
        name = t.get("name", "")
        desc = t.get("description", "")
        ans = t.get("answer", "")
        queries = " ".join(t.get("common_queries", []))
        kws = ", ".join(t.get("keywords", []))
        text = f"Team: {name}\nDescription: {desc}\nRole & Answer: {ans}\nCommon Queries: {queries}\nKeywords: {kws}"
        embed_texts.append(text)

    vectors = embeddings.embed_documents(embed_texts)
    print(f"   [+] Computed {len(vectors)} embeddings (768-dim).")

    col_ref = db.collection("teams_overview")
    batch = db.batch()
    for t, vec, text in zip(teams, vectors, embed_texts):
        doc_id = slugify(t.get("name", ""))
        doc_ref = col_ref.document(doc_id)
        payload = {
            "name": t.get("name", ""),
            "type": t.get("type", "team"),
            "organization": t.get("organization", "IEEE Student Branch, Academy of Technology (IEEE SB AOT)"),
            "description": t.get("description", ""),
            "keywords": t.get("keywords", []),
            "common_queries": t.get("common_queries", []),
            "answer": t.get("answer", ""),
            "embedding_text": text,
            "embedding": Vector(vec)
        }
        batch.set(doc_ref, payload)

    batch.commit()
    print(f"   [OK] Successfully uploaded {len(teams)} team overviews to collection 'teams_overview'.")


def upload_team_members_details(db: firestore.Client, embeddings: GoogleGenerativeAIEmbeddings):
    """Upload team_members_details.json to collection 'team_members_details'."""
    file_path = os.path.join(script_dir, "team_members_details.json")
    if not os.path.exists(file_path):
        print(f"[Error] File not found: {file_path}")
        return

    print("\n" + "="*60)
    print("3. Uploading team_members_details.json -> Collection: 'team_members_details'")
    print("="*60)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    org = data.get("organization", "IEEE Student Branch, Academy of Technology (IEEE SB AOT)")
    teams = data.get("teams", [])

    # Prepare team parent summaries and individual member texts
    all_members = []
    member_embed_texts = []
    team_embed_texts = []

    for t in teams:
        team_name = t.get("name", "")
        members = t.get("members", [])
        m_names = [m.get("name", "") for m in members]
        team_summary = (
            f"Team: {team_name}\nOrganization: {org}\nTotal Members: {len(members)}\n"
            f"Members: {', '.join(m_names)}\n"
        )
        team_embed_texts.append(team_summary)

        for m_idx, m in enumerate(members, 1):
            name = m.get("name", "")
            bio = m.get("bio", "")
            interests = m.get("interests", "")
            linkedin = m.get("linkedin", "")
            member_text = (
                f"Member Name: {name}\n"
                f"Team: {team_name}\n"
                f"Organization: {org}\n"
                f"Bio: {bio}\n"
                f"Interests & Skills: {interests}\n"
                f"LinkedIn Profile: {linkedin}"
            )
            all_members.append((team_name, m_idx, m))
            member_embed_texts.append(member_text)

    print(f"   [+] Generating embeddings for {len(teams)} teams and {len(all_members)} members...")
    team_vectors = embeddings.embed_documents(team_embed_texts)
    member_vectors = embeddings.embed_documents(member_embed_texts)
    print(f"   [+] Computed {len(team_vectors)} team embeddings + {len(member_vectors)} member embeddings.")

    col_ref = db.collection("team_members_details")

    # 1. Write team parent docs
    for t, vec, text in zip(teams, team_vectors, team_embed_texts):
        team_name = t.get("name", "")
        ts = slugify(team_name)
        team_doc_ref = col_ref.document(ts)
        team_payload = {
            "name": team_name,
            "team": team_name,
            "organization": org,
            "total_members": len(t.get("members", [])),
            "member_names": [m.get("name", "") for m in t.get("members", [])],
            "members": t.get("members", []),
            "embedding_text": text,
            "embedding": Vector(vec)
        }
        team_doc_ref.set(team_payload, merge=True)
        print(f"      [+] Set team doc: {ts} ({len(t.get('members', []))} members)")

    # 2. Write individual members to subcollection 'members'
    batch = db.batch()
    batch_count = 0
    total_uploaded = 0

    for (team_name, m_idx, m), vec, text in zip(all_members, member_vectors, member_embed_texts):
        ts = slugify(team_name)
        m_slug = slugify(m.get("name", ""))
        doc_id = f"{m_idx:02d}-{m_slug}"
        member_doc_ref = col_ref.document(ts).collection("members").document(doc_id)

        member_payload = {
            "id": m_idx,
            "name": m.get("name", ""),
            "team": team_name,
            "organization": org,
            "bio": m.get("bio", ""),
            "interests": m.get("interests", ""),
            "linkedin": m.get("linkedin", ""),
            "linkedin_url": m.get("linkedin", ""),
            "embedding_text": text,
            "embedding": Vector(vec)
        }
        batch.set(member_doc_ref, member_payload)
        batch_count += 1
        total_uploaded += 1

        if batch_count >= 20:
            batch.commit()
            batch = db.batch()
            batch_count = 0

    if batch_count > 0:
        batch.commit()

    print(f"   [OK] Successfully uploaded {len(teams)} teams and {total_uploaded} member documents to 'team_members_details'.")


def upload_events(db: firestore.Client, embeddings: GoogleGenerativeAIEmbeddings):
    """Upload events.json to collection 'events'."""
    file_path = os.path.join(script_dir, "events.json")
    if not os.path.exists(file_path):
        print(f"[Error] File not found: {file_path}")
        return

    print("\n" + "="*60)
    print("4. Uploading events.json -> Collection: 'events'")
    print("="*60)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    col_ref = db.collection("events")

    # Upload document overview
    doc_meta = data.get("document", {})
    if doc_meta:
        col_ref.document("events_overview").set(doc_meta, merge=True)
        print("   [+] Saved document metadata -> 'events_overview'")

    chunks = data.get("chunks", [])
    print(f"   [+] Generating embeddings for {len(chunks)} chunks...")

    embed_texts = []
    for c in chunks:
        title = c.get("title", "")
        sec = c.get("section_title", "")
        body = c.get("content", "")
        embed_texts.append(f"Title: {title}\nSection: {sec}\nContent: {body}")

    vectors = embeddings.embed_documents(embed_texts)
    print(f"   [+] Computed {len(vectors)} embeddings (768-dim).")

    batch = db.batch()
    for idx, (c, vec, text) in enumerate(zip(chunks, vectors, embed_texts), 1):
        doc_id = c.get("chunk_id") or f"chunk_{idx:02d}"
        doc_ref = col_ref.document(doc_id)
        payload = {
            "chunk_id": doc_id,
            "document_id": c.get("document_id", "ieee_sb_aot_events_activities"),
            "title": c.get("title", ""),
            "section_number": c.get("section_number"),
            "section_title": c.get("section_title", ""),
            "subsection_titles": c.get("subsection_titles", []),
            "content": c.get("content", ""),
            "keywords": c.get("keywords", []),
            "metadata": c.get("metadata", {}),
            "embedding_text": text,
            "embedding": Vector(vec)
        }
        batch.set(doc_ref, payload)

    batch.commit()
    print(f"   [OK] Successfully uploaded {len(chunks)} chunks to collection 'events'.")


def main():
    print("Initializing Firebase and Gemini Embeddings...")
    db = get_firestore_client()
    embeddings = get_embeddings_model()
    print("Connected successfully!")

    upload_ieee_student_branch(db, embeddings)
    upload_teams_overview(db, embeddings)
    upload_team_members_details(db, embeddings)
    upload_events(db, embeddings)

    print("\n" + "="*60)
    print("All vector JSON files uploaded successfully to Firestore!")
    print("Collections:")
    print("  1. ieee_student_branch   (Detailed report & chunks)")
    print("  2. teams_overview         (Team domains, queries, answers)")
    print("  3. team_members_details   (Team structures & member profiles)")
    print("  4. events                 (Student branch events & activities)")
    print("="*60)


if __name__ == "__main__":
    main()
