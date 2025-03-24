# Legal Judgment Knowledge System

This repository contains a solution that uses an ETL pipeline to extract data from legal reports, then generates vector embeddings of the legal reports, and stores them in a vector database. 

It provides RESTful API endpoints for retrieval of data.

## Features

1. Extracts metadata (case name, parties involved, year, citation) from legal reports.
2. Stores vector embeddings in a Chroma Database.
3. Provides semantic search, metadata-based retrieval of judgments, uploading of legal judgments as RESTful API endpoints.

## Pre-requisites
- **Python**: 3.8 or higher

## Set-Up Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/psh12320/AGCInternCodingAssessment.git
cd AGCInternCodingAssessment
```

### 2. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
python main.py
```

On first run, the program processes all of the PDFs in `sample-judgments` and builds the vector database.
Afterwards, the server will run on `http://localhost:8000/`.

### 4. Visit Swagger UI to Test API Endpoints
```bash
http://localhost:8000/docs
```

All of the API Endpoints can be tested on the Swagger UI. 
The API Documentation can be viewed on Swagger UI or as shown below here as well.

## API Documentation

All endpoints are prefixed with `/api`.

### 1. Upload PDF of a Legal Judgment
**Method**: `POST`

**Path**: `/api/upload`

**Purpose**: Upload PDF file, extract metadata, generate vector embeddings, and store it in the vector database.

**Request**:
- Body: Multipart form-data
    - `file`: PDF file to upload

**Response**:
- Status: `200 OK` on success, `400 Bad Request` if extraction of metadata fails
- Body: JSON object with `citation_id` and other extracted metadata

**Example of Successful Response**:
```bash
{
  {
  "citation_id": "[2025] SGHC 48",
  "metadata": {
    "citation_id": "[2025] SGHC 48",
    "parties": [
      "Public Prosecutor",
      "Ng Soon Kiat"
    ],
    "case_name": "Between\nPublic Prosecutor\nAnd\nNg Soon Kiat",
    "year": "2025",
    "chunks": [
      "IN THE GENERAL DIVISION OF THE HIGH COURT OF THE REPUBLIC OF SINGAPORE [2025] SGHC 48 Criminal Case No 4 of 2025 Between Public Prosecutor And Ng Soon Kiat GROUNDS OF DECISION [Criminal Procedure and Sentencing — Sentencing] [Criminal Law— Offences — Rioting] [Criminal Law — Statutory offences — Misuse of Drugs Act] [Road Traffic — Offences — Drink driving] Version No 1: 21 Mar 2025 (10:51 hrs)"
    ],
    "embeddings": [
      [
        0.1,
        0.2,
        0.3
      ],
      [
        0.1,
        0.2,
        0.3
      ]
    ]
  }
}
```

The embedding values shown above are dummy values.

**Example of Unsuccessful Response**:
```bash
{
  "detail": "Failed to extract required metadata"
}
```

### 2. Retrieve all or filtered judgments
**Method**: `GET`

**Path**: `/api/judgments`

**Purpose**: Retrieve all stored judgments or retrieve filtered judgments based on provided metadata.

**Parameters**:
- `metadata` (query, optional): A JSON string specifying filters like `{"citation_id": "[2020] 1 SLR 486"}`

**Response**:
- Status: `200 OK` on success, `400 Bad Request` if invalid format of metadata fails
- Body: JSON object with `judgments` and their metadata

**Example of Successful Response**:
```bash
{
  "judgments": [
    {
      "citation_id": "[2020] 1 SLR 486",
      "case_name": "SLR PP v GCK",
      "parties": [
        [
          "SLR PP"
        ],
        [
          "GCK"
        ]
      ],
      "year": "2020"
    }
  ]
}
```

**Example of Unsuccessful Response**:
```bash
{
  "detail": "Invalid metadata format"
}
```

### 3. Retrieve a specific judgment
**Method**: `GET`

**Path**: `/api/judgments/{citation_id}`

**Purpose**: Retrieves detailed information about a judgment by using `citation_id` as an identifier

**Parameters**:
- `citation_id` (path, required): The citation ID of a judgment, for example: `[2020] 1 SLR 486`

**Response**:
- Status: `200 OK` on success, `400 Bad Request` if judgment not found
- Body: JSON object with `judgments` and their metadata

**Example of Successful Response**:
```bash
{
  "judgment": {
    "citation_id": "[2025] SGHC 45",
    "case_name": "Public Prosecutor v Muhammad Sufian bin",
    "parties": [
      [
        "Public Prosecutor"
      ],
      [
        "Muhammad Sufian bin"
      ]
    ],
    "year": "2025",
    "chunks": [
      "IN THE GENERAL DIVISION OF THE HIGH COURT OF THE REPUBLIC OF SINGAPORE [2025] SGHC 45 Magistrate’s Appeal No 9139 of 2024 Between Public Prosecutor … Appellant And Muhammad Sufian bin Hussain … Respondent GROUNDS OF DECISION [Criminal Law — Appeal] [Criminal Law — Offences — Sexual exploitation of a child] [Criminal Procedure and Sentencing — Sentencing — Persistent offenders] Version No 1: 18 Mar 2025 (14:22 hrs)"
    ]
  }
}
```

**Example of Unsuccessful Response**:
```bash
{
  "detail": "Judgment not found"
}
```

### 4. Search judgments by natural language query
**Method**: `GET`

**Path**: `/api/search`

**Purpose**: Searches judgments using natural language and finds judgments with the most semantic similarity from vector database

**Parameters**:
- `query` (query, required): A string to search, for example: "divorce cases"
- `n_results` (query, required): The number of results to return (default is 5)

**Response**:
- Status: `200 OK`
- Body: JSON object with `judgments` array containing all relevant judgments, their metadata, and similarity scores

**Example of Successful Response**:
```bash
{
  "judgments": [
    {
      "citation_id": "[2025] SGHCF 16",
      "metadata": {
        "citation_id": "[2025] SGHCF 16",
        "case_name": "XIK v XIL",
        "parties": [
          [
            "XIK"
          ],
          [
            "XIL"
          ]
        ],
        "year": "2025"
      },
      "relevant_chunks": [
        {
          "chunk_id": "[2025] SGHCF 16_chunk_0",
          "text": "IN THE FAMILY JUSTICE COURTS OF THE REPUBLIC OF SINGAPORE [2025] SGHCF 16 Divorce (Transferred) No 688 of 2023 Between XIK … Plaintiff And XIL … Defendant JUDGMENT [Family Law — Matrimonial assets — Division — Alleged dissipation of matrimonial assets] [Family Law — Matrimonial assets — Division — Whether an adverse inference should be drawn] [Family Law — Maintenance — Child] [Family Law — Maintenance — Wife] Version No 1: 03 Mar 2025 (15:16 hrs)",
          "score": 1.03329598903656
        }
      ],
      "score": 1.03329598903656
    },
    {
      "citation_id": "[2025] SGHCF 17",
      "metadata": {
        "citation_id": "[2025] SGHCF 17",
        "case_name": "XJI v XJJ",
        "parties": [
          [
            "XJI"
          ],
          [
            "XJJ"
          ]
        ],
        "year": "2025"
      },
      "relevant_chunks": [
        {
          "chunk_id": "[2025] SGHCF 17_chunk_0",
          "text": "IN THE FAMILY JUSTICE COURTS OF THE REPUBLIC OF SINGAPORE [2025] SGHCF 17 Divorce (Transferred) No 2198 of 2023 Between XJI … Plaintiff And XJJ … Defendant JUDGMENT [Family Law — Custody — Care and control] [Family Law — Custody — Access] [Family Law — Matrimonial assets — Division] [Family Law — Maintenance — Wife] [Family Law — Maintenance — Child] Version No 1: 05 Mar 2025 (16:35 hrs)",
          "score": 1.0445648431777954
        }
      ],
      "score": 1.0445648431777954
    },
    {
      "citation_id": "[2025] SGHCF 18",
      "metadata": {
        "citation_id": "[2025] SGHCF 18",
        "case_name": "XIW v XIX",
        "parties": [
          [
            "XIW"
          ],
          [
            "XIX"
          ]
        ],
        "year": "2025"
      },
      "relevant_chunks": [
        {
          "chunk_id": "[2025] SGHCF 18_chunk_0",
          "text": "IN THE FAMILY JUSTICE COURTS OF THE REPUBLIC OF SINGAPORE [2025] SGHCF 18 Divorce (Transferred) No 4047 of 2022 Between XIW … Plaintiff And XIX … Defendant JUDGMENT [Family Law — Matrimonial Assets — Division] Version No 1: 05 Mar 2025 (16:27 hrs)",
          "score": 1.0852735042572021
        }
      ],
      "score": 1.0852735042572021
    },
    {
      "citation_id": "[2025] SGHCF 15",
      "metadata": {
        "citation_id": "[2025] SGHCF 15",
        "case_name": "Kee Cheong Keng v Dinh Thi Thu Hien",
        "parties": [
          [
            "Kee Cheong Keng"
          ],
          [
            "Dinh Thi Thu Hien"
          ]
        ],
        "year": "2025"
      },
      "relevant_chunks": [
        {
          "chunk_id": "[2025] SGHCF 15_chunk_0",
          "text": "IN THE FAMILY JUSTICE COURTS OF THE REPUBLIC OF SINGAPORE [2025] SGHCF 15 Suit No 4 of 2022 Between Kee Cheong Keng … Plaintiff And Dinh Thi Thu Hien … Defendant GROUNDS OF DECISION [Family Law — Marriage — Nullity] Version No 1: 28 Feb 2025 (15:58 hrs)",
          "score": 1.1920548677444458
        }
      ],
      "score": 1.1920548677444458
    },
    {
      "citation_id": "[2025] SGHC(I) 8",
      "metadata": {
        "citation_id": "[2025] SGHC(I) 8",
        "case_name": "(1) Marketlend Pty Ltd v QBE Insurance (Singapore)",
        "parties": [
          [
            "(1) Marketlend Pty Ltd"
          ],
          [
            "QBE Insurance (Singapore)"
          ]
        ],
        "year": "2025"
      },
      "relevant_chunks": [
        {
          "chunk_id": "[2025] SGHC(I) 8_chunk_0",
          "text": "IN THE SINGAPORE INTERNATIONAL COMMERCIAL COURT OF THE REPUBLIC OF SINGAPORE [2025] SGHC(I) 8 Originating Application No 16 of 2023 Between (1) Marketlend Pty Ltd (2) Australian Executor Trustees Limited … Claimants And QBE Insurance (Singapore) Pte Ltd … Defendant Counterclaim of Defendant Between (1) QBE Insurance (Singapore) Pte Ltd … Claimant in Counterclaim And (1) Marketlend Pty Ltd (2) Australian Executor Trustees Limited … Defendants in Counterclaim Version No 2: 11 Mar 2025 (14:45 hrs)",
          "score": 1.33475923538208
        }
      ],
      "score": 1.33475923538208
    }
  ]
}
```
## Use of Generative AI

1. Used to simplify complex regex patterns
2. Sort results by score when searching judgments
3. Generate example response structure in pydantic models
4. To improve code quality
5. Gave some suggestions on how to structure API documentation for ease of reading and clarity







