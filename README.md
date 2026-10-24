# 🎬 Celebrity Face Recognition & Filmography Explorer (PFA)

An end-to-end intelligent computer vision and web application system that identifies celebrities from portrait photographs and immediately retrieves their complete filmography, characters played, movie posters, and storyline synopses.

---

## 📑 Table of Contents
1. [Overview](#-overview)
2. [Visual Tour & Screenshots](#-visual-tour--screenshots)
3. [System Architecture](#-system-architecture)
4. [Deep Dive into Components](#-deep-dive-into-components)
   - [Scraping & Data Acquisition Pipeline](#3-scraping--data-acquisition-pipeline)
   - [Face Encoding Engine (CPU & GPU)](#4-face-encoding-engine-cpu--gpu)
   - [Analytics & Maintenance Tools](#5-analytics--maintenance-tools)
5. [Data Schema & Storage Format](#-data-schema--storage-format)

---

## 🌟 Overview

The **Celebrity Face Recognition & Filmography Explorer** is designed to solve the *“Who is that actor and what movies have they starred in?”* question seamlessly. The project combines state-of-the-art facial landmark detection and 128-dimensional metric face embeddings with automated web scraping and a rich, responsive React client.

### Key Highlights
- **16,700+ Celebrities Indexed**: Pre-computed face encodings covering 38,800+ photo embeddings in a serialized vector database (`faces.json`).
- **Comprehensive Movie Database**: Hundreds of feature films cataloged across release years (2021, 2022, 2023), complete with high-resolution posters, cast lists, character mappings, and plot summaries.
- **Accurate Face Biometrics**: Powered by `dlib` and `face_recognition`, using ResNet-based deep metric learning where the Euclidean distance between faces directly reflects visual identity.
- **Interactive Carousel Interface**: Dynamic card slider presenting movie appearances, character roles, release years, and synopses with smooth animations and dynamic styling.
- **Resilient Web Harvester**: Headless/headful Playwright scraper capable of handling IMDb rate limits, human verification challenges, and Wikipedia release directories.

---

## 📸 Visual Tour & Screenshots

### 1. Web Application: Matched Celebrity Filmography (Card 1)
When an image is uploaded and recognized (e.g., actor *Seth Fuentes*), the application displays the actor's name along with an interactive carousel of movies they appeared in. Below, *Seth Fuentes* is matched and his appearance in the 2023 movie **Air** as "Restaurant Guest" is shown:

![Web App - Matched Celebrity in Air](screenshots/Screenshot%202026-10-24%20054936.png)

---

### 2. Web Application: Carousel Navigation (Card 2)
Navigating through the carousel reveals additional filmography entries. Here, *Seth Fuentes* is displayed in the 2023 DC film **Blue Beetle** playing the role of "Student":

![Web App - Matched Celebrity in Blue Beetle](screenshots/Screenshot%202026-10-24%20054949.png)

---

### 3. Web Application: Multi-Movie Filmography (Card 3)
Browsing further displays the 2023 comedy film **Bottoms**, where the actor also portrays a "Student", complete with synopsis and poster:

![Web App - Matched Celebrity in Bottoms](screenshots/Screenshot%202026-10-24%20054955.png)

---

### 4. Celebrity Reference Dataset
The automated image scraper collects high-quality portrait datasets for every actor. The directory contains over **33,045+ individual celebrity folders**:

![Celebrity Dataset Directory](screenshots/Screenshot%202026-10-24%20024008.png)

---

### 5. Movie Catalog & Metadata Store
The scraped movie catalog (organized by release year, such as `movies/2023`) contains hundreds of folders (e.g., 338 films in 2023 alone), each storing posters, cast mappings, descriptions, and metadata:

![Movie Catalog Directory](screenshots/Screenshot%202026-10-24%20024019.png)

---

## 🏗 System Architecture

```mermaid
graph TD
    User([User in Browser]) -->|Uploads Image| Frontend[React Client: Port 3000]
    Frontend -->|POST /api multipart/form-data| Backend[Flask API: Port 8000]
    
    subgraph Data Pipeline
        Wikipedia[Wikipedia Release Lists] -->|Playwright & BeautifulSoup| Scraper[movie_scraper_pipeline.py]
        IMDb[IMDb Cast & Movie Metadata] -->|Playwright & BeautifulSoup| Scraper
        Bing[Bing Image Downloader] -->|Download Actor Portraits| CelebritiesDir[(scrap/celebrities/)]
        Scraper --> MoviesDir[(scrap/movies/)]
        
        CelebritiesDir -->|HOG / CNN Face Recognition| FaceGen[generate_faces.py / GPU]
        FaceGen --> FacesJSON[(scrap/faces.json: 16,706 actors)]
    end
    
    subgraph Backend Processing
        Backend -->|Extract 128-d Embedding| DlibModel[dlib ResNet Face Model]
        DlibModel -->|Euclidean Distance Tolerance 0.5| CompareFaces[face_recognition.compare_faces]
        FacesJSON -.->|Loaded in Memory| CompareFaces
        CompareFaces -->|Identified Celebrity Name| QueryCatalog[Query movies/year/title]
        MoviesDir -.-> QueryCatalog
    end
    
    QueryCatalog -->|JSON: Celebrity Name + Base64 Posters & Roles| Backend
    Backend -->|JSON Response| Frontend
    Frontend -->|Render Animated Carousel & Info Cards| User
```


---

## 🔍 Deep Dive into Components


### 1. Scraping & Data Acquisition Pipeline
Located in [`scrap/movie_scraper_pipeline.py`](file:///F:/PFA/scrap/movie_scraper_pipeline.py):
- **Wikipedia Movie Harvesting**:
  - Scrapes `https://en.wikipedia.org/wiki/List_of_American_films_of_{year}`.
  - Parses standard Wikipedia HTML tables to extract verified film titles.
  - Caches HTML pages locally (`american_films_{year}.html`) to prevent repeat requests.
- **IMDb Playwright Browser Automation**:
  - Launches a persistent Chromium browser profile (`user_data_dir`) with customized viewport and User-Agent headers.
  - Supports importing existing cookie sessions (`cookie_file`).
  - **Bot Detection Handling**: Intelligently senses Amazon WAF security challenges (`let's confirm you are human`, `robot check`, status `202/403/429`), pauses execution, alerts the developer in the console to solve the CAPTCHA interactively in the browser, and resumes automatically.
  - Scrapes:
    - Full cast & crew with character names (`/fullcredits`).
    - Movie poster image (`photo.jpg`).
    - Official storyline/synopsis (`description.txt`).
    - JSON metadata (`metadata.json`).
- **Bing Celebrity Image Downloader**:
  - Automatically queries Bing Image Search for each cast member to download 3–5 representative portrait photographs.
  - Saves images into dedicated folders under `scrap/celebrities/{Actor_Name}/`.

---

### 2. Face Encoding Engine (CPU & GPU)

#### A. CPU Multi-Threaded Generator (`scrap/generate_faces.py`)
- Traverses `celebrities/` and detects faces using HOG (Histogram of Oriented Gradients) or CNN models.
- Rejects images that contain 0 or multiple faces to guarantee high data quality.
- Employs an incremental cache (`faces_cache.json`) checking file size and modification timestamps (`mtime_ns`) so previously computed faces are never recomputed.
- Writes atomically to `faces.json.tmp` and replaces `faces.json` safely to avoid data corruption.

#### B. GPU Batch-Accelerated Generator (`scrap/generate_faces_gpu.py`)
- Utilizes `dlib` compiled with CUDA support for high-throughput batch detection.
- Normalizes orientations via EXIF metadata using `PIL.ImageOps.exif_transpose()`.
- Standardizes images into fixed-dimension square tensors with padding to preserve aspect ratio.
- Employs cuDNN accelerated CNN face detection for processing large volumes of celebrity images.

---
---

## 💾 Data Schema & Storage Format

### 1. `faces.json` (Vector Database)
A JSON mapping of celebrity names to arrays of 128-float embedding vectors:
```json
{
  "Seth Fuentes": [
    [-0.1423, 0.0891, 0.0451, ..., -0.0129]
  ],
  "Margot Robbie": [
    [-0.0982, 0.1214, 0.0341, ..., 0.0512],
    [-0.0911, 0.1189, 0.0355, ..., 0.0488]
  ]
}
```

### 2. Movie Record (`scrap/movies/<year>/<title>/`)
- **`casts.json`**:
  ```json
  {
    "Matt Damon": "Sonny Vaccaro",
    "Ben Affleck": "Phil Knight",
    "Jason Bateman": "Rob Strasser",
    "Seth Fuentes": "Restaurant Guest"
  }
  ```
- **`metadata.json`**:
  ```json
  {
    "title": "Air",
    "year": 2023,
    "imdb_id": "tt16419074",
    "url": "https://www.imdb.com/title/tt16419074/"
  }
  ```
- **`description.txt`**: Plain text containing the plot overview.
- **`photo.jpg`**: High-resolution movie poster.

---
