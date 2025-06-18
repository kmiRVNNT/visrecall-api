from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from PIL import Image
import open_clip
import torch
import chromadb
from chromadb.config import Settings
import os 
import numpy as np
import pickle


device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-H-14')
model.to(device)


#Creating the FastAPI App
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the images directory for static file serving
app.mount("/images", StaticFiles(directory="images"), name="images")

#set the Access-Control-Allow-Origin header for static files.
class CORSMiddlewareForImages(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/images/"):
            response.headers["Access-Control-Allow-Origin"] = "*"
        return response

app.add_middleware(CORSMiddlewareForImages)

@app.get("/search")
def search_endpoint(query: str = Query(...), top_k: int = 5):
    image_results = search_images(query, top_k)
    # Convert paths to URLs
    image_urls = [f"/images/{os.path.basename(path)}" for path in image_results]
    return {"results": image_urls}

#setting up the chroma client
chroma_client = chromadb.Client(Settings(
    persist_directory="./chroma_index",  # Saves index to disk
    anonymized_telemetry=False
))

collection = chroma_client.get_or_create_collection(name="image_embeddings")


#Defining the Text -> Image Search Function
def search_images(text_query, top_k=5):
    with torch.no_grad():
        tokenized = tokenizer([text_query]).to(device)
        text_features = model.encode_text(tokenized)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    # Convert to NumPy
    query_vec = text_features.cpu().numpy().astype(np.float32)

    # Search in FAISS index
    D, I = index.search(query_vec, top_k)
    results = [image_paths[i] for i in I[0]]
    return results


#Indexing the Images
image_folder = "images"
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder)]

for idx, path in enumerate(image_paths):
    try:
        image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model.encode_image(image)
            features /= features.norm(dim=-1, keepdim=True)
        vector = features.squeeze(0).cpu().tolist()
        
        collection.add(
            ids=[f"img_{idx}"],
            embeddings=[vector],
            metadatas=[{"path": path}],
            documents=[f"Image at {path}"]
        )
    except Exception as e:
        print(f"Failed to embed {path}: {e}")

#Search Function
def search_images(text_query, top_k=5):
    with torch.no_grad():
        tokens = tokenizer([text_query]).to(device)
        text_features = model.encode_text(tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        query_vector = text_features.squeeze(0).cpu().tolist()

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k
    )

    return [metadata["path"] for metadata in results["metadatas"][0]]





if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
