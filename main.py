import os
import re
import json
import shutil
import logging
import uvicorn
import pdfplumber
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from typing import Optional, Dict, List, Union
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from pydantic import BaseModel

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download nltk packages for text-preprocessing
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Initialise embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Set up vector database
chroma_client = chromadb.Client(Settings(persist_directory="./judgments_db"))
collection = chroma_client.get_or_create_collection(name="judgments")

# Initialise FastAPI
app = FastAPI(title="Legal Judgment Knowledge System",
              description="Provides RESTful API Endpoints for uploading, retrieving, and searching legal judgments")

# Regex patterns for metadata extraction
CITATION_PATTERN = re.compile(r"\[\d{4}\]\s?\d*\s?[A-Z]{2,10}(\([A-Z]+\))?\s?\d{1,4}", re.MULTILINE)
PARTIES_V_PATTERN = re.compile(r"(.+?)\s+v\.?\s+(.+)",re.IGNORECASE)
PARTIES_BETWEEN_PATTERN = re.compile(r"(?i)(Between\s+)", re.MULTILINE)
CASE_NAME_PATTERN = re.compile(r"(?i)(?:Case\s+Name\s*:?)(.{0,200})")
V_PATTERN = re.compile(r"([A-Za-z\s&,.()]+)\s+v\.?\s+([A-Za-z\s&,.()]+)", re.IGNORECASE)
NUMBERED_PARTY_PATTERN = re.compile(r"^\(?\d+\)?\s*[.:]?\s*(.+)$")


def preprocess_text(text):
    """
    Pre-processing text from a legal judgment
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    legal_stopwords = {'court', 'judgment', 'case', 'v'}
    stop_words.update(legal_stopwords)
    tokens = [token for token in tokens if token not in stop_words]
    cleaned_text = ' '.join(tokens)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text


def generate_embedding(text):
    """
    Generates vector embedding for the given text.
    """
    preprocessed_text = preprocess_text(text)
    embedding = embedding_model.encode(preprocessed_text, convert_to_tensor=False)
    return embedding.tolist()


def normalize_citation(citation):
    if not citation:
        return None

    citation = re.sub(r"\s+", " ", citation).strip()
    match = re.match(r"(\[\d{4}\]\s?\d*\s?[A-Z]{2,10}(\([A-Z]+\))?\s?)(\d+)", citation)

    if match:
        prefix = match.group(1)
        number = str(int(match.group(3)))
        return f"{prefix}{number}"

    return citation


def looks_like_errata(page_text):
    """
    Check if a page is just showing errata based on word or length
    """
    text_lower = page_text.lower()

    if "errata" in text_lower or "erratum" in text_lower:
        return True

    if len(page_text.split()) < 30:
        return True

    return False


def extract_parties_from_lines(page_text):
    """
    Extracts parties and case name based on regex patterns
    """
    # Empty page
    lines = [line.strip() for line in page_text.splitlines() if line.strip()]
    if not lines:
        return None, None

    between_idx = None
    and_idx = None

    # Finds between and and words
    for i, line in enumerate(lines):
        if re.match(r"(?i)^Between$", line):
            between_idx = i
        elif re.match(r"(?i)^And$",
                      line) and between_idx is not None and i > between_idx:
            and_idx = i
            break

    # Get party names from 'between' and 'and'
    if between_idx is not None and and_idx is not None and between_idx <= and_idx:
        if between_idx + 1 < and_idx and and_idx + 1 < len(lines):
            left_party = lines[between_idx + 1]
            right_party = lines[and_idx + 1]

            # Check whether they are actually party names
            if (len(left_party) > 2 and len(right_party) > 2 and
                    not re.match(r"\[\d{4}\]", left_party) and
                    not re.match(r"\[\d{4}\]", right_party) and
                    not re.search(r"(?i)JUDGMENT$|^COURT OF|^CASE NO", left_party) and
                    not re.search(r"(?i)JUDGMENT$|^COURT OF|^CASE NO", right_party)):
                left_parties = [left_party]
                right_parties = [right_party]
                case_name = f"{left_party} v {right_party}"
                return [left_parties, right_parties], case_name

    # Check multiple lines
    if between_idx is not None and and_idx is not None and between_idx < and_idx:
        # Get the left party
        left_parties = []
        for i in range(between_idx + 1, and_idx):
            line = lines[i].strip()
            # Stop in the below case
            if (re.match(r"\[\d{4}\]", line) or
                    re.search(r"(?i)JUDGMENT|COURT OF|CASE NO", line) or
                    len(line) < 3):
                continue
            left_parties.append(line)

        # Get the right party
        right_parties = []
        for i in range(and_idx + 1, min(and_idx + 10, len(lines))):
            line = lines[i].strip()
            # Stop in the below case
            if (re.match(r"\[\d{4}\]", line) or
                    re.search(r"(?i)JUDGMENT|COURT OF|CASE NO", line) or
                    len(line) < 3):
                break
            right_parties.append(line)

        # Make the case name
        if left_parties and right_parties:
            left_str = " and ".join(left_parties)
            right_str = " and ".join(right_parties)
            case_name = f"{left_str} v {right_str}"
            return [left_parties, right_parties], case_name

    # Find the case name containing "v"
    for line in lines:
        v_match = re.search(r"([A-Za-z\s&,.()]+)\s+v\s+([A-Za-z\s&,.()]+)", line)
        if v_match:
            left_party = v_match.group(1).strip()
            right_party = v_match.group(2).strip()

            # Check whether they are really party names
            if (len(left_party) > 2 and len(right_party) > 2 and
                    not re.match(r"\[\d{4}\]", left_party) and
                    not re.match(r"\[\d{4}\]", right_party)):
                return [[left_party], [right_party]], f"{left_party} v {right_party}"

    # Else, return None, None
    return None, None


def extract_page_metadata(page_text):
    """
    Gets the metadata from a page using helper functions.
    """
    metadata = {
        "citation_id": None,
        "parties": None,
        "case_name": None
    }
    citation_match = CITATION_PATTERN.search(page_text)

    if citation_match:
        metadata["citation_id"] = normalize_citation(citation_match.group().strip())
    parties_result = extract_parties_from_lines(page_text)

    if parties_result:
        parties, case_name = parties_result
        metadata["parties"] = parties
        if case_name:
            metadata["case_name"] = case_name

    name_match = CASE_NAME_PATTERN.search(page_text)

    if name_match and not metadata.get("case_name"):
        metadata["case_name"] = name_match.group(1).strip()

    return metadata


def combine_metadata(base_metadata, new_metadata):
    for key, val in new_metadata.items():
        if val and not base_metadata.get(key):
            base_metadata[key] = val

    return base_metadata


def metadata_is_complete(metadata):
    return bool(metadata.get("citation_id") and metadata.get("parties") and metadata.get("case_name"))


def extract_metadata_from_pdf(pdf_path):
    metadata = {
        "citation_id": None,
        "parties": None,
        "case_name": None,
        "year": None,
        "chunks": [],
        "embeddings": []
    }

    # Open pdf of legal judgment
    with pdfplumber.open(pdf_path) as pdf:
        full_text = []
        for page in pdf.pages:
            text = page.extract_text() or ""
            text = text.strip()
            if looks_like_errata(text):
                continue
            full_text.append(text)
            page_md = extract_page_metadata(text)
            metadata = combine_metadata(metadata, page_md)
            if metadata_is_complete(metadata):
                break
        full_text_str = "\n".join(full_text)
        metadata["chunks"] = chunk_text(full_text_str)
        metadata["embeddings"] = [generate_embedding(chunk) for chunk in metadata["chunks"]]

    # Get the year from the citation id
    if metadata["citation_id"]:
        metadata["year"] = parse_year_from_citation(metadata["citation_id"])

    return metadata


def parse_year_from_citation(citation):
    """
    Get the year of the judgment from the citation.
    """
    match = re.search(r"\[(\d{4})\]", citation)
    return match.group(1) if match else None


def store_metadata_in_db(metadata_dict):
    """
    Store the metadata in the vector database.
    """

    citation_id = metadata_dict["citation_id"]
    if not citation_id:
        raise ValueError("No citation_id found in metadata")

    chunk_ids = [f"{citation_id}_chunk_{i}" for i in range(len(metadata_dict["chunks"]))]
    metadatas = [{
        "citation_id": citation_id,
        "case_name": metadata_dict["case_name"],
        "parties": json.dumps(metadata_dict["parties"]),
        "year": metadata_dict["year"],
        "parent_id": citation_id
    } for _ in metadata_dict["chunks"]]

    print(metadatas)

    collection.add(
        ids=chunk_ids,
        embeddings=metadata_dict["embeddings"],
        metadatas=metadatas,
        documents=metadata_dict["chunks"]
    )
    print(f"Stored {len(chunk_ids)} chunks for citation_id {citation_id}.")


def process_all_pdfs_in_dir(folder_path):
    """
    Process all the PDFs and store data in vector database.
    """
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(".pdf"):
            fpath = os.path.join(folder_path, fname)
            metadata = extract_metadata_from_pdf(fpath)
            if metadata.get("citation_id"):
                store_metadata_in_db(metadata)


def chunk_text(text, max_chunk_size=500):
    """
    Break a large text into chunks for better semantic search when searching.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0

    for word in words:
        current_size += len(word.split())
        if current_size > max_chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_size = len(word.split())
        else:
            current_chunk.append(word)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def search_judgments(query, n_results=5):
    """
    Perform search in vector database with a natural language query.
    """
    query_embedding = generate_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results * 2
    )
    judgment_results = {}
    for i, (doc_id, metadata, document) in enumerate(
            zip(results["ids"][0], results["metadatas"][0], results["documents"][0])):
        parent_id = metadata["parent_id"]
        if parent_id not in judgment_results:
            parties = json.loads(metadata["parties"]) if metadata["parties"] else None
            judgment_results[parent_id] = {
                "metadata": {
                    "citation_id": metadata["citation_id"],
                    "case_name": metadata["case_name"],
                    "parties": parties,
                    "year": metadata["year"]
                },
                "relevant_chunks": [],
                "score": float('inf')
            }

        judgment_results[parent_id]["relevant_chunks"].append({
            "chunk_id": doc_id,
            "text": document,
            "score": results["distances"][0][i]
        })

        distance = results["distances"][0][i]
        judgment_results[parent_id]["score"] = min(
            judgment_results[parent_id]["score"],
            distance
        )

    # Sort the results by score
    sorted_results = sorted(judgment_results.items(), key=lambda x: x[1]["score"])[:n_results]

    return [{"citation_id": k, **{**v, "score": str(v["score"])}} for k, v in sorted_results]


# Pydantic models for response structures
class Judgment(BaseModel):
    citation_id: str
    case_name: str
    parties: List[List[str]]
    year: Optional[str] = None
    chunks: List[str]

    class Config:
        json_schema_extra = {
            "example": {
                "citation_id": "[2019] SGHC 166",
                "case_name": "Public Prosecutor v Ewe Pang Kooi",
                "parties": [["Public Prosecutor"], ["Ewe Pang Kooi"]],
                "year": "2019",
                "chunks": ["Chunk 1 text...", "Chunk 2 text..."]
            }
        }


class JudgmentSummary(BaseModel):
    citation_id: str
    case_name: str
    parties: List[List[str]]
    year: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "citation_id": "[2019] SGHC 166",
                "case_name": "Public Prosecutor v Ewe Pang Kooi",
                "parties": [["Public Prosecutor"], ["Ewe Pang Kooi"]],
                "year": "2019"
            }
        }


class SearchResult(BaseModel):
    citation_id: str
    metadata: JudgmentSummary
    relevant_chunks: List[Dict[str, Union[str, float]]]
    score: float

    class Config:
        json_schema_extra = {
            "example": {
                "citation_id": "[2019] SGHC 166",
                "metadata": {
                    "citation_id": "[2019] SGHC 166",
                    "case_name": "Public Prosecutor v Ewe Pang Kooi",
                    "parties": [["Public Prosecutor"], ["Ewe Pang Kooi"]],
                    "year": "2019"
                },
                "relevant_chunks": [
                    {"chunk_id": "[2019] SGHC 166_chunk_0", "text": "text...", "score": 0.95}
                ],
                "score": 0.95
            }
        }


class UploadResponse(BaseModel):
    citation_id: str
    metadata: Dict[str, object]

    class Config:
        json_schema_extra = {
            "example": {
                "citation_id": "[2019] SGHC 166",
                "metadata": {
                    "citation_id": "[2019] SGHC 166",
                    "case_name": "Public Prosecutor v Ewe Pang Kooi",
                    "parties": [["Public Prosecutor"], ["Ewe Pang Kooi"]],
                    "year": "2019",
                    "chunks": ["Chunk 1 text..."],
                    "embeddings": ["embedding_vector"]
                }
            }
        }


@app.post(
    "/api/upload",
    summary="Upload a Judgment PDF",
    description="Uploads a PDF file containing a legal judgment, extracts metadata (citation ID, parties, case name, year), and stores it in the database.",
    response_model=UploadResponse,
    responses={
        200: {"description": "Successfully uploaded and processed the PDF"},
        400: {"description": "Failed to extract required metadata"},
        409: {"description": "Judgment with this citation_id already exists"}
    }
)
async def upload_pdf(file: UploadFile = File(..., description="A PDF file containing a legal judgment")):
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    metadata = extract_metadata_from_pdf(file_path)

    if not metadata.get("citation_id") or not metadata.get("parties"):
        raise HTTPException(status_code=400, detail="Failed to extract required metadata")

    existing = collection.get(where={"citation_id": metadata["citation_id"]})

    if existing["ids"]:
        raise HTTPException(status_code=409, detail="Judgment with this citation_id already exists")

    store_metadata_in_db(metadata)

    return {"citation_id": metadata["citation_id"], "metadata": metadata}


@app.get(
    "/api/judgments/{citation_id}",
    summary="Retrieve a Specific Judgment",
    description="Fetches a judgment by its citation ID, case name, parties, year, and text chunks.",
    response_model=Dict[str, Judgment],
    responses={
        200: {"description": "Judgment found and returned"},
        404: {"description": "Judgment not found in the database"},
        500: {"description": "Internal server error, e.g., database query failed"}
    }
)
async def get_judgment(citation_id: str):
    logger.info(f"Attempting to query for citation_id: {citation_id}")

    try:
        result = collection.get(where={"parent_id": citation_id}, include=["metadatas", "documents"])
        logger.info(f"Query successful, got {len(result.get('metadatas', []))} results")
    except Exception as e:
        logger.error(f"Error querying ChromaDB: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    if not result.get("metadatas") or len(result["metadatas"]) == 0:
        raise HTTPException(status_code=404, detail="Judgment not found")

    judgment = {
        "citation_id": citation_id,
        "case_name": result["metadatas"][0]["case_name"],
        "parties": json.loads(result["metadatas"][0]["parties"]),
        "year": result["metadatas"][0]["year"],
        "chunks": result["documents"]
    }

    return {"judgment": judgment}


@app.get(
    "/api/judgments",
    summary="List Judgments",
    description="Retrieves a list of all judgments or filters them based on provided metadata (e.g., citation_id, parties).",
    response_model=Dict[str, List[JudgmentSummary]],
    responses={
        200: {"description": "List of judgments returned (may be empty)"},
        400: {"description": "Invalid metadata format provided"},
        500: {"description": "Internal server error"}
    }
)
async def get_judgments(metadata: Optional[str] = Query(
    None,
    description="Optional JSON string to filter judgments (e.g., '{\"citation_id\": \"[2019] SGHC 166\"}')")
):
    try:
        if metadata is None:
            results = collection.get(include=["metadatas"])

            if not results.get("metadatas") or len(results["metadatas"]) == 0:
                return {"judgments": []}

            judgment_dict = {}

            for metadata_entry in results["metadatas"]:
                parent_id = metadata_entry["parent_id"]
                if parent_id not in judgment_dict:
                    judgment_dict[parent_id] = {
                        "citation_id": parent_id,
                        "case_name": metadata_entry["case_name"],
                        "parties": json.loads(metadata_entry["parties"]),
                        "year": metadata_entry["year"]
                    }

            return {"judgments": list(judgment_dict.values())}

        else:
            metadata_filter = {}
            try:
                metadata_filter = json.loads(metadata)
                if "citation_id" in metadata_filter:
                    metadata_filter["citation_id"] = normalize_citation(metadata_filter["citation_id"])
                if "parties" in metadata_filter:
                    metadata_filter["parties"] = json.dumps(metadata_filter["parties"])

            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid metadata format.")

            results = collection.get(where=metadata_filter, include=["metadatas"])

            if not results.get("metadatas") or len(results["metadatas"]) == 0:
                return {"judgments": []}

            judgment_dict = {}

            for metadata_entry in results["metadatas"]:
                parent_id = metadata_entry["parent_id"]
                if parent_id not in judgment_dict:
                    judgment_dict[parent_id] = {
                        "citation_id": parent_id,
                        "case_name": metadata_entry["case_name"],
                        "parties": json.loads(metadata_entry["parties"]),
                        "year": metadata_entry["year"]
                    }

            return {"judgments": list(judgment_dict.values())}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error listing judgments: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get(
    "/api/search",
    summary="Search Judgments",
    description="Searches judgments by a query string, returning the top N relevant results with metadata and relevant text chunks.",
    response_model=Dict[str, List[SearchResult]],
    responses={
        200: {"description": "Search results returned (may be empty)"},
        500: {"description": "Internal server error"}
    }
)
async def search_judgments_by_query(
    query: str = Query(..., description="The search query string to find relevant judgments"),
    n_results: Optional[int] = Query(5, description="Number of results to return, default is 5", ge=1)
):
    try:
        results = search_judgments(query, n_results)
        return {"judgments": results}
    except Exception as e:
        logger.error(f"Error for query '{query}': {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    input_dir = "sample-judgments"
    process_all_pdfs_in_dir(input_dir)
    uvicorn.run(app, host="0.0.0.0", port=8000)