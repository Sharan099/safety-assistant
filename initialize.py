"""
Initialize Safety Copilot - Download or build vector store dynamically
Downloads from Google Drive if available, otherwise builds from PDFs
Standalone script - NO Streamlit dependencies
"""
import os
from pathlib import Path
import gdown
from vector_store import SafetyVectorStore
from core_app import SafetyCopilotCore
from config import REGULATIONS_DIR, VECTOR_STORE_DIR

# Google Drive file IDs (update these after uploading to Drive)
# Get the file IDs from the Google Drive shareable links
GDRIVE_CHUNKS_JSON_ID = "YOUR_CHUNKS_JSON_FILE_ID"  # Update after uploading chunks.json
GDRIVE_CONFIG_JSON_ID = "YOUR_CONFIG_JSON_FILE_ID"  # Update after uploading config.json
GDRIVE_FAISS_INDEX_ID = "YOUR_FAISS_INDEX_FILE_ID"  # Update after uploading faiss_index.bin

# Alternative: Use folder ID if you uploaded the entire vector_store folder
GDRIVE_FOLDER_ID = "1ym0KVtO0dkxwb3mmrdpULDq0b91hCHnJ"  # From the Drive link

def download_from_drive(file_id: str, output_path: Path, description: str = "file"):
    """Download a file from Google Drive using gdown"""
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"📥 Downloading {description} from Google Drive...")
        gdown.download(url, str(output_path), quiet=False)
        if output_path.exists():
            print(f"✅ Downloaded {description} successfully!")
            return True
        else:
            print(f"⚠️  Download completed but file not found at {output_path}")
            return False
    except Exception as e:
        print(f"❌ Error downloading {description}: {e}")
        return False

def download_vector_store_from_drive():
    """Download vector store files from Google Drive"""
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    
    chunks_json = VECTOR_STORE_DIR / "chunks.json"
    config_json = VECTOR_STORE_DIR / "config.json"
    faiss_index = VECTOR_STORE_DIR / "faiss_index.bin"
    
    # Check if all files already exist
    if chunks_json.exists() and config_json.exists() and faiss_index.exists():
        print("✅ Vector store files already exist locally!")
        return True
    
    print("🔄 Attempting to download vector store from Google Drive...")
    print("   (If files are not found, will build from PDFs instead)")
    
    # Try downloading individual files if IDs are provided
    if GDRIVE_CHUNKS_JSON_ID != "YOUR_CHUNKS_JSON_FILE_ID":
        if not chunks_json.exists():
            download_from_drive(GDRIVE_CHUNKS_JSON_ID, chunks_json, "chunks.json")
        if not config_json.exists():
            download_from_drive(GDRIVE_CONFIG_JSON_ID, config_json, "config.json")
        if not faiss_index.exists():
            download_from_drive(GDRIVE_FAISS_INDEX_ID, faiss_index, "faiss_index.bin")
        
        # Check if all files downloaded successfully
        if chunks_json.exists() and config_json.exists() and faiss_index.exists():
            return True
    
    # Alternative: Try downloading from folder (if you share the folder)
    # Note: This requires the folder to be publicly accessible or using service account
    print("💡 Tip: Update the GDRIVE_*_ID variables in initialize.py with your file IDs")
    print("   Or make the folder publicly accessible and use folder download")
    
    return False

def main():
    print("🛡️ Safety Copilot Initialization")
    print("=" * 50)
    
    # Try to download from Google Drive first
    downloaded = download_vector_store_from_drive()
    
    # Check if vector store already exists (either downloaded or built previously)
    vector_store_path = VECTOR_STORE_DIR / "faiss_index.bin"
    if vector_store_path.exists() and downloaded:
        print("✅ Vector store loaded from Google Drive!")
        print(f"   Location: {vector_store_path}")
        print("\n🚀 Vector store is ready. You can run: streamlit run app.py")
        return
    
    # If not downloaded, check if it exists locally
    if vector_store_path.exists():
        print("✅ Vector store already exists locally!")
        print(f"   Location: {vector_store_path}")
        print("\n💡 To rebuild, delete the vector_store/ folder and run this script again.")
        return
    
    # If not found, build from PDFs
    print("\n📦 Vector store not found. Building from PDFs...")
    
    # Check if regulations directory exists and has PDFs
    if not REGULATIONS_DIR.exists():
        print(f"⚠️  Regulations directory not found: {REGULATIONS_DIR}")
        print("   Creating directory...")
        REGULATIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    pdf_files = list(REGULATIONS_DIR.rglob("*.pdf"))
    
    if not pdf_files:
        print(f"⚠️  No PDF files found in {REGULATIONS_DIR}")
        print("   Please add PDF documents to the regulations folder.")
        return
    
    print(f"📄 Found {len(pdf_files)} PDF file(s):")
    for pdf_file in pdf_files[:10]:  # Show first 10
        print(f"   - {pdf_file.name}")
    if len(pdf_files) > 10:
        print(f"   ... and {len(pdf_files) - 10} more")
    
    print("\n🔄 Building vector store (this may take a few minutes)...")
    print("   This is a one-time process. The vector store will be saved for future use.")
    
    try:
        # Build vector store from regulations directory
        vector_store = SafetyVectorStore.load_or_build_store(
            force_rebuild=False,  # Will build if doesn't exist
            regulations_dir=REGULATIONS_DIR,
            user_documents=None
        )
        
        # Create core app
        core = SafetyCopilotCore()
        core.set_vector_store(vector_store)
        
        stats = core.get_stats()
        print("\n✅ Initialization Complete!")
        print(f"   Documents: {stats['num_documents']}")
        print(f"   Chunks: {stats['num_chunks']}")
        print(f"   Model: {stats['embedding_model']}")
        print(f"   Vector store saved to: {VECTOR_STORE_DIR}")
        print("\n🚀 You can now run: streamlit run app.py")
        
    except Exception as e:
        print(f"\n❌ Error during initialization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

