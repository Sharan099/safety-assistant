"""
Initialize Safety Copilot - Process documents and build vector store dynamically
This script builds the vector store if it doesn't exist (for first run)
Standalone script - NO Streamlit dependencies
"""
from vector_store import SafetyVectorStore
from core_app import SafetyCopilotCore
from config import REGULATIONS_DIR, VECTOR_STORE_DIR
from pathlib import Path

def main():
    print("🛡️ Safety Copilot Initialization")
    print("=" * 50)
    
    # Check if vector store already exists
    vector_store_path = VECTOR_STORE_DIR / "faiss_index.bin"
    if vector_store_path.exists():
        print("✅ Vector store already exists!")
        print(f"   Location: {vector_store_path}")
        print("\n💡 To rebuild, delete the vector_store/ folder and run this script again.")
        return
    
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

