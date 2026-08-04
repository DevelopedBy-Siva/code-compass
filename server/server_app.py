from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import shutil
import os
from pathlib import Path
import sys
from openai import OpenAI
import json

sys.path.insert(0, str(Path(__file__).parent))

from src.rag_system import IncrementalRAGSystem
from src.database import get_db_session, DocumentVersion, DocumentChunk


client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1"
)
CLIENT_URL=os.getenv("APP_URL")

app = FastAPI(
    title="Incremental RAG API",
    description="API for document Q&A RAG System",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
       CLIENT_URL,
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_system = None


@app.on_event("startup")
def startup():
    global rag_system
    rag_system = IncrementalRAGSystem()


TEMP_UPLOAD_DIR = "./temp_uploads"
Path(TEMP_UPLOAD_DIR).mkdir(exist_ok=True)


class QueryRequest(BaseModel):
    question: str
    version_id: Optional[int] = None
    k: int = 5


class ComparisonRequest(BaseModel):
    question: str
    version_id_1: int
    version_id_2: int
    k: int = 3


@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "Document Q&A RAG API is running",
    }


@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...), doc_name: Optional[str] = Form(None)
):
    temp_file_path = None
    try:
        allowed_extensions = {".pdf", ".txt", ".docx"}
        file_ext = Path(file.filename).suffix.lower()

        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400, detail=f"File type {file_ext} not supported"
            )

        temp_file_path = Path(TEMP_UPLOAD_DIR) / file.filename
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if not doc_name:
            doc_name = Path(file.filename).stem

        result = rag_system.add_document(
            file_path=str(temp_file_path), doc_name=doc_name
        )

        temp_file_path.unlink()

        return JSONResponse(
            content={
                "success": True,
                "message": f"Document uploaded as version {result['version_number']}",
                "data": result,
            }
        )

    except Exception as e:
        if temp_file_path and temp_file_path.exists():
            temp_file_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))


def build_source_context(results):
    parts = []
    for i, r in enumerate(results, start=1):
        excerpt = r["content"]
        if len(excerpt) > 2000:
            excerpt = excerpt[:2000] + "..."
        parts.append(f"[Source {i}]\n{excerpt}")
    return "\n\n".join(parts)


def extract_document_topics(chunks: list, max_topics: int = 5) -> list:

    sample_text = "\n".join([c["content"] for c in chunks[:3]])

    try:
        prompt = f"""
Extract the main topics covered in this document.

Document sample:
{sample_text[:1000]}

Return JSON with main topics/sections:
{{
  "topics": ["Topic 1", "Topic 2", "Topic 3"]
}}

Keep topics concise (2-4 words each). Maximum {max_topics} topics.
"""

        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
            response_format={"type": "json_object"},
        )

        result = json.loads(resp.choices[0].message.content)
        return result.get("topics", [])[:max_topics]

    except Exception as e:
        words = sample_text.lower().split()
        fallback_topics = []
        policy_keywords = [
            "policy",
            "work",
            "remote",
            "vacation",
            "benefits",
            "security",
            "equipment",
            "eligibility",
        ]
        for keyword in policy_keywords:
            if keyword in words:
                fallback_topics.append(keyword.title())

        return (
            fallback_topics[:max_topics] if fallback_topics else ["General Information"]
        )


@app.post("/api/query/generate")
async def query_with_llm(query_request: QueryRequest):
    question = query_request.question.strip()

    if len(question) < 3:
        return {
            "question": question,
            "not_found": True,
            "answer": "",
            "message": "Question too short (minimum 3 characters)",
            "sources": [],
        }

    results = rag_system.query(
        question=question,
        version_id=query_request.version_id,
        k=query_request.k,
    )

    if not results:
        return {
            "question": question,
            "not_found": True,
            "answer": "",
            "message": "No content found in this document version",
            "suggestion": "Check if you selected the correct version or try searching all versions",
            "sources": [],
        }

    top_score = results[0]["similarity_score"]

    if top_score < 0.35:
        topics = extract_document_topics(results)

        return {
            "question": question,
            "not_found": True,
            "answer": "",
            "message": "No direct match for your question",
            "topics": topics,
            "suggestions": [
                "Try asking about specific topics listed above",
                "Use keywords from the document",
                (
                    f"Example: 'What is the {topics[0].lower()}?'"
                    if topics
                    else "Be more specific"
                ),
            ],
            "top_score": round(top_score, 3),
            "sources": [],
        }

    force_low_confidence = False

    if top_score < 0.4:
        filtered = results[:3]
        force_low_confidence = True
    elif top_score > 0.6:
        filtered = [r for r in results if r["similarity_score"] > 0.5][:3]
    elif top_score > 0.45:
        filtered = [r for r in results if r["similarity_score"] > 0.4][:2]
    else:
        filtered = results[:1]

    context = build_source_context(filtered)
    avg_sim = sum(r["similarity_score"] for r in filtered) / len(filtered)

    system_msg = """You are a helpful document Q&A assistant.

IMPORTANT RULES:
1. Answer using ONLY the provided context
2. If context is relevant, provide an answer even if partial
3. Only return not_found=true if context is COMPLETELY unrelated
4. For general questions (like "policy" or "document"), summarize key points

You must return valid JSON in this format:
{
  "not_found": false,
  "answer": "Your answer here",
  "confidence": "high|medium|low"
}

Only use not_found=true if truly nothing relevant exists."""

    user_prompt = f"""
Context (avg similarity: {avg_sim:.2f}):
{context}

Question: {question}

Provide a helpful answer based on the context. If the question is general, summarize the main points."""

    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=800,
            response_format={"type": "json_object"},
        )

        text = resp.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM API error: {str(e)}")

    try:
        j = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            try:
                j = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                j = {
                    "not_found": False,
                    "answer": text,
                    "confidence": "low",
                    "note": "Response format was non-standard",
                }
        else:
            raise HTTPException(
                status_code=500, detail="Failed to parse LLM response as JSON"
            )

    j["sources"] = filtered
    j["question"] = question
    j["avg_similarity"] = round(avg_sim, 3)

    if "confidence" not in j:
        if avg_sim > 0.6:
            j["confidence"] = "high"
        elif avg_sim > 0.45:
            j["confidence"] = "medium"
        else:
            j["confidence"] = "low"

    if force_low_confidence:
        j["confidence"] = "low"
        j["warning"] = "Answer based on limited context relevance"

    return j


@app.get("/api/documents")
async def list_documents():
    try:
        documents = rag_system.get_all_documents()
        return documents
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents/{doc_name}/versions")
async def get_document_versions(doc_name: str):
    try:
        versions = rag_system.get_document_versions(doc_name)
        if not versions:
            raise HTTPException(
                status_code=404, detail=f"Document '{doc_name}' not found"
            )
        return versions
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents/{doc_name}/versions/{version_id}/diff")
async def get_version_diff(doc_name: str, version_id: int):
    try:
        session = get_db_session()

        try:
            current_version = (
                session.query(DocumentVersion).filter_by(id=version_id).first()
            )

            if not current_version:
                raise HTTPException(status_code=404, detail="Version not found")

            prev_version = (
                session.query(DocumentVersion)
                .filter_by(
                    document_id=current_version.document_id,
                    version_number=current_version.version_number - 1,
                )
                .first()
            )

            if not prev_version:
                return {
                    "success": True,
                    "message": "This is the first version",
                    "is_first_version": True,
                    "current_version": current_version.version_number,
                }

            current_chunks = [chunk.content for chunk in current_version.chunks]
            prev_chunks = [chunk.content for chunk in prev_version.chunks]

            current_text = "\n\n".join(current_chunks)
            prev_text = "\n\n".join(prev_chunks)

            stats = {
                "chunks_added": len(current_chunks) - len(prev_chunks),
                "current_chunks": len(current_chunks),
                "previous_chunks": len(prev_chunks),
                "current_version": current_version.version_number,
                "previous_version": prev_version.version_number,
            }

            system_msg = """You are analyzing document changes. 
Identify what changed between two versions.
Be specific and concise.
You must respond with valid JSON only."""

            user_prompt = f"""
Previous Version:
{prev_text[:3000]}...

Current Version:
{current_text[:3000]}...

Analyze the changes and return valid JSON in this format:
{{
  "summary": "Brief overview of changes",
  "key_changes": [
    {{"type": "added|modified|removed", "description": "what changed"}},
  ],
  "impact": "low|medium|high"
}}
"""
            try:
                resp = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.1,
                    max_tokens=500,
                    response_format={"type": "json_object"},  # Now this works
                )

                llm_response = resp.choices[0].message.content.strip()

                try:
                    diff_analysis = json.loads(llm_response)
                except json.JSONDecodeError as e:
                    print(f"Failed to parse LLM response: {llm_response}")
                    diff_analysis = {
                        "summary": f"Version {current_version.version_number} has {len(current_chunks) - len(prev_chunks)} more chunks than version {prev_version.version_number}",
                        "key_changes": [
                            {
                                "type": "modified",
                                "description": f"Content updated with {abs(len(current_chunks) - len(prev_chunks))} chunk difference",
                            }
                        ],
                        "impact": "medium",
                    }

            except Exception as llm_error:
                print(f"LLM API error: {llm_error}")
                diff_analysis = {
                    "summary": "Unable to generate detailed analysis",
                    "key_changes": [
                        {
                            "type": "modified",
                            "description": f"{len(current_chunks)} chunks in current version vs {len(prev_chunks)} in previous",
                        }
                    ],
                    "impact": "unknown",
                }

            return {
                "success": True,
                "is_first_version": False,
                "stats": stats,
                "analysis": diff_analysis,
                "version_info": {
                    "current": {
                        "id": current_version.id,
                        "number": current_version.version_number,
                        "date": current_version.upload_date.isoformat(),
                    },
                    "previous": {
                        "id": prev_version.id,
                        "number": prev_version.version_number,
                        "date": prev_version.upload_date.isoformat(),
                    },
                },
            }

        finally:
            session.close()

    except HTTPException:
        raise
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to parse LLM response: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/compare/detailed")
async def compare_versions_detailed(comparison: ComparisonRequest):

    try:
        session = get_db_session()

        try:
            v1 = (
                session.query(DocumentVersion)
                .filter_by(id=comparison.version_id_1)
                .first()
            )
            v2 = (
                session.query(DocumentVersion)
                .filter_by(id=comparison.version_id_2)
                .first()
            )

            if not v1 or not v2:
                raise HTTPException(status_code=404, detail="Version not found")

            v1_chunks = [chunk.content for chunk in v1.chunks]
            v2_chunks = [chunk.content for chunk in v2.chunks]

            v1_text = "\n\n".join(v1_chunks)
            v2_text = "\n\n".join(v2_chunks)

            if comparison.question:
                results_v1 = rag_system.query(
                    question=comparison.question,
                    version_id=comparison.version_id_1,
                    k=comparison.k,
                )

                results_v2 = rag_system.query(
                    question=comparison.question,
                    version_id=comparison.version_id_2,
                    k=comparison.k,
                )

                context_v1 = "\n".join([r["content"] for r in results_v1[:2]])
                context_v2 = "\n".join([r["content"] for r in results_v2[:2]])

                system_msg = """Compare how two document versions answer the same question.
                Identify specific differences."""

                user_prompt = f"""
Question: {comparison.question}

Version {v1.version_number} says:
{context_v1}

Version {v2.version_number} says:
{context_v2}

Return JSON:
{{
  "answer_v1": "Answer from version 1",
  "answer_v2": "Answer from version 2",
  "changed": true/false,
  "differences": [
    {{"aspect": "what changed", "v1": "old value", "v2": "new value"}}
  ],
  "summary": "Overall comparison"
}}
"""
            else:
                system_msg = """Compare two document versions.
                Identify all significant changes."""

                user_prompt = f"""
Version {v1.version_number}:
{v1_text[:4000]}...

Version {v2.version_number}:
{v2_text[:4000]}...

Return JSON:
{{
  "overall_change": "high|medium|low",
  "summary": "What changed overall",
  "sections_changed": ["section 1", "section 2"],
  "key_differences": [
    {{"category": "category", "description": "what changed", "type": "added|modified|removed"}}
  ],
  "recommendations": "Who should review these changes"
}}
"""

            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=1000,
            )

            analysis = json.loads(resp.choices[0].message.content)

            return {
                "success": True,
                "question": comparison.question if comparison.question else None,
                "version_info": {
                    "version_1": {
                        "id": v1.id,
                        "number": v1.version_number,
                        "date": v1.upload_date.isoformat(),
                        "chunks": len(v1_chunks),
                    },
                    "version_2": {
                        "id": v2.id,
                        "number": v2.version_number,
                        "date": v2.upload_date.isoformat(),
                        "chunks": len(v2_chunks),
                    },
                },
                "analysis": analysis,
                "stats": {
                    "chunks_difference": len(v2_chunks) - len(v1_chunks),
                    "text_length_v1": len(v1_text),
                    "text_length_v2": len(v2_text),
                },
            }

        finally:
            session.close()

    except HTTPException:
        raise
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse LLM response")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server_app:app", host="0.0.0.0", port=8000, reload=True)
